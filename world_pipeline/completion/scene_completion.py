"""Experimental ordinary completion: content -> boxes -> bounded visual feedback.

User-run Qwen3-VL-8B only. Reuses frozen observations and existing exporters;
does not change legacy methods, estimate a measured floor, or optimize boxes.
CPU preflight never constructs a model. Numerical checks do not certify semantics.
"""
import argparse
from copy import deepcopy
from html import escape
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..common import finite_vector, new_output, read, sha, validate_graph, write
from .audit_semantic_layout import project_aabb
from .category_bridge import MAPPING, category
from .geometry import parse_json
from .qwen_upstream import DEFAULT_MODEL, Qwen
from .semantic_layout import build_states
from .semantic_layout_visual import render
from .semantic_probe import DEFAULT_BATCH, load_cases, validate_answer

PROTOCOL = 'ordinary_content_boxes_visual_feedback_v1'


def content_prompt(case):
    # No predicted depth, solver API, category menu or furniture checklist here.
    inventory = [{k: obj[k] for k in ('id', 'category', 'description', 'visible_bbox_xyxy_norm')}
                 for obj in case['context']]
    return (
        'This photograph shows part of an indoor environment. Describe ONE plausible surrounding '
        'scene, including what may exist outside the photograph, consistent with the image. '
        'Use the room function and visible contents to choose useful additional objects. '
        'The inventory below is an approximate detector summary; the image may show objects '
        'or object parts missing from it. Preserve everything visible, including empty surfaces. '
        'Do not list a visible instance or its components as a new object. '
        'Return JSON with scene_type, scene_hypothesis, additions. Each addition represents ONE '
        'new instance and has id, name, object_type, reason, placement. '
        'name describes that instance; object_type is its ordinary object noun without color or '
        'material adjectives. reason explains its role in this particular scene. placement '
        'describes a plausible location outside the photograph, referring to inventory IDs when '
        'useful. These are hypotheses. Choose their number freely; there is no direction quota. '
        'Do not predict sizes or coordinates in this step. '
        f'The photograph covers {case["hfov"]:g} horizontal degrees. '
        '\nOBSERVED INVENTORY: '+json.dumps(inventory, ensure_ascii=False, separators=(',', ':')))


def validate_plan(plan, case, max_objects=160):
    # Existing validation checks distinct observed/new IDs and required prose.
    validate_answer(plan, case['context'])
    if len(plan['additions']) + len(case['context']) > max_objects:
        raise ValueError('Object capacity exceeded; no truncation')
    for item in plan['additions']:
        if not isinstance(item.get('object_type'), str) or not item['object_type'].strip():
            raise ValueError('Each instance needs its actual object_type noun')
    return deepcopy(plan)


def map_plan(plan, policy='dataset_exact'):
    """Map the actual noun, never ask the layout model to pick a convenient ID."""
    result = []
    for obj in plan['additions']:
        row = dict(id=obj['id'], name=obj['name'], object_type=obj['object_type'],
                   category=None, method=None, semantics_verified=False)
        if policy == 'semantic_lexicon_v1':
            from .semantic_categories import resolve_identity
            row.update(resolve_identity(obj['object_type'], obj['name']))
            result.append(row)
            continue
        if policy != 'dataset_exact':
            raise ValueError('Unknown category policy')
        try:
            row['category'], evidence = category(obj['object_type'])
            row['method'] = evidence or dict(method='exact_canonical_label')
        except ValueError as exc:
            row['unresolved_reason'] = str(exc)
        result.append(row)
    return result


def layout_prompt(case, plan):
    displayed = [{**o, 'center': [round(v, 3) for v in o['center']],
                  'full_aabb_size': [float(f'{v:.3g}') for v in o['full_aabb_size']]}
                 for o in case['context']]
    return (
        'Arrange the FIXED new instances below into ONE plausible surrounding scene compatible '
        'with the attached photograph and the observed boxes. Keep their identity and number. '
        'Return JSON {"objects":[{"id":...,"center":[x,y,z],"size":[sx,sy,sz],"reason":...}]}. '
        'Return exactly one box per new ID. center is in camera-local METRES: x right, y image-up, '
        'z forward; negative z is behind the camera. size contains FULL camera-axis AABB lengths. '
        'Use observed boxes for approximate scale (rounded for display, not measured precision). '
        'Respect the qualitative arrangements and '
        'functional relationships, leaving space to use the furniture. Choose natural locations '
        'outside the photograph, anywhere around the same camera. Moving everything far away '
        'or shrinking furniture is not a reasonable solution. '
        'Image-up is not measured gravity. Floor, walls and room extent are unknown; any support '
        'or relative height is a layout hypothesis, not a measurement. Explain important support '
        'and height assumptions in reason. Do not change observed boxes or return categories. '
        f'The input horizontal FOV is {case["hfov"]:g} degrees. '
        '\nOBSERVED BOXES: '+json.dumps(displayed, ensure_ascii=False, separators=(',', ':'))
        +'\nFIXED CONTENTS: '+json.dumps(plan, ensure_ascii=False, separators=(',', ':')))


def validate_layout(answer, plan, mappings):
    objects = answer.get('objects') if isinstance(answer, dict) else None
    if not isinstance(objects, list):
        raise ValueError('Need objects array')
    expected = {o['id'] for o in plan['additions']}
    by_id = {o['id']: o for o in mappings}
    seen, result = set(), []
    for obj in objects:
        if not isinstance(obj, dict) or not isinstance(obj.get('id'), str):
            raise ValueError('Every box needs an instance ID')
        oid = obj['id']
        if oid not in expected or oid in seen:
            raise ValueError('Unknown/duplicate ID; no objects can be added or merged')
        if (not finite_vector(obj.get('center'), 3) or not finite_vector(obj.get('size'), 3)
                or min(obj['size']) <= 0 or not isinstance(obj.get('reason'), str) or not obj['reason'].strip()):
            raise ValueError('Need finite center, positive full size and placement reason')
        if 'category' in obj:
            raise ValueError('Categories are mapped separately from layout')
        seen.add(oid)
        result.append(dict(id=oid, plan_id=oid, center=deepcopy(obj['center']),
                           size=deepcopy(obj['size']), category=by_id[oid]['category'],
                           unseen_reason='outside_fov', reason=obj['reason']))
    if seen != expected:
        raise ValueError('Every planned instance must be accounted for; no silent deletion')
    ordered = {o['id']: o for o in result}
    return [ordered[o['id']] for o in plan['additions']]


def view_check(center, size, hfov, image_size):
    """Exact AABB/centred-pinhole-cone intersection (up to floating precision).

    At the box's largest z the cone's x/y intervals are widest. Both intervals
    intersect the box iff one common point exists; no LP dependency is needed.
    Intersection describes the solid BOX proxy, not visibility of a real object.
    """
    if not 0 < hfov < 180 or min(image_size) <= 0:
        raise ValueError('Invalid pinhole camera')
    c, s = np.array(center, dtype=float), np.array(size, dtype=float)
    lo, hi = c-s/2, c+s/2
    tx = math.tan(math.radians(hfov)/2)
    ty = tx*image_size[1]/image_size[0]
    x, y = np.clip([0., 0.], lo[:2], hi[:2])
    z = max(1e-6, float(lo[2]), abs(x)/tx, abs(y)/ty)
    intersects = bool(z <= hi[2]+1e-9)
    return dict(intersects_input_box_proxy=intersects,
                frustum_witness_m=[float(x), float(y), float(z)] if intersects else None,
                camera_inside_box_proxy=bool(np.all(lo <= 0) and np.all(hi >= 0)),
                **project_aabb(center, size, hfov, image_size))


def review_boxes(items, observed, hfov, image_size):
    boxes = [dict(id=o['track_id'], center=o['center_local_m']['value'],
                  size=o['size_aabb_local_m']['value']) for o in observed['objects']]
    result = []
    for item in items:
        check = dict(id=item['id'], **view_check(item['center'], item['size'], hfov, image_size))
        overlaps = []
        for other in boxes:
            lo = np.maximum(np.array(item['center'])-np.array(item['size'])/2,
                            np.array(other['center'])-np.array(other['size'])/2)
            hi = np.minimum(np.array(item['center'])+np.array(item['size'])/2,
                            np.array(other['center'])+np.array(other['size'])/2)
            extent = hi-lo
            if np.all(extent > 1e-6):
                overlaps.append(dict(id=other['id'], intersection_volume_m3=float(np.prod(extent))))
        check['box_proxy_overlaps'] = overlaps
        check['distance_from_camera_m'] = float(np.linalg.norm(item['center']))
        result.append(check)
        boxes.append(item)
    warnings = sum(r['intersects_input_box_proxy'] or r['camera_inside_box_proxy']
                   or bool(r['box_proxy_overlaps']) for r in result)
    return dict(objects=result, objects_with_numeric_warnings=warnings,
                semantics_verified=False, physical_support_verified=False, gravity_verified=False,
                note='Solid AABB overlap/intersection requires review; it does not prove physical '
                     'collision or actual visibility. No measured floor, walls or navigability test.')


def export_states(observed, plan, items, mappings, review):
    candidates, full, _ = build_states(observed, plan, items)
    by_id = {o['id']: o for o in mappings}
    for graph in (candidates, full):
        graph.update(completion_protocol=PROTOCOL, semantics_verified=False, geometry_verified=False,
                     quality_status='requires_semantic_and_spatial_review',
                     numeric_review=review)
        if 'content_review' in plan:
            graph['content_review'] = deepcopy(plan['content_review'])
        for obj in graph['objects'][len(observed['objects']):]:
            obj['category_id'].update(source='noun_category_bridge' if obj['category_id']['valid'] else 'unknown',
                                      method=by_id[obj['track_id']]['method'])
            obj['object_type'] = by_id[obj['track_id']]['object_type']
        validate_graph(graph)
    # Recompute these copies after source metadata is attached.
    unmapped = [deepcopy(o) for o in candidates['objects'][len(observed['objects']):]
                if not o['category_id']['valid']]
    return candidates, full, unmapped


def feedback_image(image, directory):
    canvas = Image.new('RGB', (1536, 1536), 'white')
    panels = [('ORIGINAL RGB', image),
              ('BOX PROJECTION INTO RGB', directory/'local_projection.png'),
              ('CAMERA X/Z BOX MAP (NOT A FLOOR PLAN)', directory/'layout_xz.png'),
              ('360 DIRECTION MAP (NOT A GENERATED IMAGE)', directory/'erp_state_map.png')]
    for i, (label, source) in enumerate(panels):
        if isinstance(source, Path):
            with Image.open(source) as handle:
                panel = handle.convert('RGB')
        else:
            panel = source.copy()
        panel.thumbnail((752, 724))
        x, y = (i % 2)*768, (i // 2)*768
        canvas.paste(panel, (x+(768-panel.width)//2, y+32))
        ImageDraw.Draw(canvas).text((x+8, y+8), label, fill='black')
    canvas.save(directory/'feedback.png')
    return canvas


def revision_prompt(case, plan, answer, checks):
    return (layout_prompt(case, plan)
        +'\nReview the previous proposal using the attached four-panel image and computed checks. '
        'Blue boxes are observed, orange are new and red are unmapped. The shaded ERP region is '
        'the input view. Resolve places that put new objects into photographed space. Proxy '
        'overlaps can also occur for valid nesting or inflated AABBs; assess them rather than '
        'blindly separating all furniture. Keep the intended function, realistic sizes and '
        'meaningful support. Return the complete new-ID box list, changing only geometry and '
        'placement reasons. An unchanged proposal remains unresolved when checks still disagree. '
        '\nPREVIOUS BOXES: '+json.dumps(answer, ensure_ascii=False, separators=(',', ':'))
        +'\nCOMPUTED BOX CHECKS: '+json.dumps(checks, ensure_ascii=False, separators=(',', ':')))


def saved_plan(source, case, max_objects):
    directory = source/case['id']
    previous = read(directory/'input.json')
    if previous['source_sha256'] != case['source_sha256'] or previous['hfov'] != case['hfov']:
        raise ValueError('Saved plan image/observation/FOV differ')
    raw, parsed = directory/'content.txt', directory/'content.json'
    if parse_json(raw.read_text()) != read(parsed):
        raise ValueError('Saved semantic plan differs from its raw model response')
    request = directory/'content.request.json'
    if read(request)['prompt'] != content_prompt(case):
        raise ValueError('Saved content prompt differs')
    provenance = {str(p.resolve()): sha(p) for p in (raw, parsed, request, directory/'input.json')}
    return validate_plan(read(parsed), case, max_objects), provenance


def run(model, cases, output, stage='content', plans_from=None, seed=3407,
        max_tokens=4096, max_objects=160, layout_revisions=1, category_policy='dataset_exact',
        reviewed_plans_from=None):
    records = []
    for case in cases:
        directory = output/case['id']
        directory.mkdir()
        record = dict(id=case['id'], stage=stage, status='failed', semantics_verified=False,
                      geometry_verified=False, panorama_generated=False)
        try:
            observed = validate_graph(read(case['state']))
            write(directory/'input.json', case)
            write(directory/'G_obs.json', observed)
            with Image.open(case['image']) as handle:
                image = handle.convert('RGB')
            image.thumbnail((768, 768)); image.save(directory/'local.png')
            options = dict(seed=seed, sample=True, profile='official', max_tokens=max_tokens,
                           guard_repetition=True)
            if reviewed_plans_from is not None:
                from .content_review import load_reviewed_plan
                plan, provenance = load_reviewed_plan(reviewed_plans_from, case, max_objects)
                if provenance != case.get('reviewed_plan_sha256', provenance):
                    raise ValueError('Reviewed plan changed after preflight/config binding')
                write(directory/'content_source.json', provenance)
            elif plans_from is None:
                plan = validate_plan(model.ask(image, content_prompt(case), directory/'content.txt', **options),
                                     case, max_objects)
            else:
                plan, provenance = saved_plan(plans_from, case, max_objects)
                if provenance != case.get('saved_plan_sha256', provenance):
                    raise ValueError('Saved plan changed after preflight/config binding')
                write(directory/'content_source.json', provenance)
            write(directory/'semantic_plan.json', plan)
            mappings = map_plan(plan, category_policy)
            write(directory/'category_review.json', mappings)
            record.update(proposed_instances=len(plan['additions']),
                          unresolved_categories=sum(o['category'] is None for o in mappings),
                          status='content_saved_needs_review')
            if stage == 'full':
                history = []
                question, current_image = layout_prompt(case, plan), image
                # Empty content is retained and labelled; no geometry call can invent additions.
                for attempt in range(layout_revisions+1 if plan['additions'] else 1):
                    attempt_dir = directory/f'layout_{attempt:03d}'
                    attempt_dir.mkdir()
                    answer = (model.ask(current_image, question, attempt_dir/'boxes.txt', **options)
                              if plan['additions'] else dict(objects=[]))
                    write(attempt_dir/'proposal.json', answer)
                    items = validate_layout(answer, plan, mappings)
                    checks = review_boxes(items, observed, case['hfov'], image.size)
                    states = export_states(observed, plan, items, mappings, checks)
                    write(attempt_dir/'numeric_review.json', checks)
                    render(states[0], image, case['hfov'], attempt_dir, checks,
                           position_note='Qwen布局假设；程序没有移动框或证明物理/语义正确。')
                    history.append(dict(attempt=attempt, checks=checks))
                    if not checks['objects_with_numeric_warnings']:
                        break
                    current_image = feedback_image(image, attempt_dir)
                    question = revision_prompt(case, plan, answer, checks)
                candidates, full, unmapped = states
                for name, value in (('G_full_candidates.json', candidates), ('G_full_proposal.json', full),
                                    ('unmapped_candidates.json', unmapped), ('layout_history.json', history)):
                    write(directory/name, value)
                exportable = bool(plan['additions'] and not unmapped and not checks['objects_with_numeric_warnings'])
                if exportable:
                    write(directory/'G_full.json', full)
                status = ('no_additions_proposed' if not plan['additions'] else
                          'needs_geometry_review' if checks['objects_with_numeric_warnings'] else
                          'needs_category_review' if unmapped else 'numeric_checks_passed_needs_semantic_review')
                record.update(status=status,
                              selected_attempt=attempt, selection='last_model_proposal_not_quality_ranking',
                              numeric_warning_objects=checks['objects_with_numeric_warnings'],
                              adapter_subset_objects=len(full['objects']), adapter_state_exported=exportable,
                              hypothesized_instances=len(full['objects'])-len(observed['objects']),
                              observed_state_preserved=full['objects'][:len(observed['objects'])]==observed['objects'])
                render(candidates, image, case['hfov'], directory, record,
                       position_note='实验性场景假设，尚未通过语义/空间质量验证。')
        except Exception as exc:
            record.update(status='failed', error=str(exc))
        records.append(record)
        write(directory/'summary.json', record)
        write(output/'summary.json', dict(protocol=PROTOCOL, records=records, scene_gt_used=False,
                                          model_quality_validated=False))
        cards = ['<article><h2>'+escape(r['id'])+'</h2><img width="320" src="'
                 +escape(r['id'], quote=True)+'/local.png"><pre>'
                 +escape(json.dumps(r, ensure_ascii=False, indent=2))+'</pre><a href="'
                 +escape(r['id'], quote=True)+'/semantic_plan.json">Content plan</a>'
                 +(('<p><a href="'+escape(r['id'], quote=True)+'/index.html">360 box review</a></p>')
                   if 'selected_attempt' in r else '')+'</article>' for r in records]
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8">'
            '<h1>Ordinary scene completion — experimental, quality unverified</h1>'+''.join(cards))
        print(json.dumps(record, ensure_ascii=False), flush=True)
    if any(sha(path) != digest for case in cases for path, digest in case['source_sha256'].items()):
        raise ValueError('Source observation/image changed during completion')
    if any(sha(path) != digest for case in cases
           for path, digest in case.get('saved_plan_sha256', {}).items()):
        raise ValueError('Saved semantic plan changed during layout')
    if any(sha(path) != digest for case in cases
           for path, digest in case.get('reviewed_plan_sha256', {}).items()):
        raise ValueError('Reviewed semantic plan changed during layout')
    return all(r['status'] != 'failed' for r in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=DEFAULT_BATCH)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--only-id', action='append')
    parser.add_argument('--stage', choices=('content', 'full'), default='content')
    plans = parser.add_mutually_exclusive_group()
    plans.add_argument('--plans-from', type=Path)
    plans.add_argument('--reviewed-plans-from', type=Path)
    parser.add_argument('--category-policy', choices=('dataset_exact','semantic_lexicon_v1'), default='dataset_exact')
    parser.add_argument('--layout-revisions', type=int, choices=(0, 1), default=1)
    parser.add_argument('--max-objects', type=int, default=160)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.max_tokens < 1 or not 0 <= args.seed < 2**63 or not 1 <= args.max_objects <= 160:
        parser.error('Invalid token budget, seed or capacity')
    cases = load_cases(args.batch.resolve(), args.only_id)
    for case in cases:
        case.pop('prompt')  # Legacy probe prompt is not part of this experiment.
        observed = read(case['state'])
        if len(observed['objects']) > args.max_objects:
            parser.error('Observed state exceeds capacity')
        if any(not o['center_local_m']['valid'] or not o['size_aabb_local_m']['valid'] for o in observed['objects']):
            parser.error('This layout experiment requires valid frozen observed geometry')
        if args.plans_from:
            _, provenance = saved_plan(args.plans_from.resolve(), case, args.max_objects)
            case['saved_plan_sha256'] = provenance
        if args.reviewed_plans_from:
            from .content_review import load_reviewed_plan
            _, provenance = load_reviewed_plan(args.reviewed_plans_from.resolve(), case, args.max_objects)
            case['reviewed_plan_sha256'] = provenance
    code_files = [Path(__file__), Path(__file__).with_name('semantic_layout.py'),
                  Path(__file__).with_name('semantic_layout_visual.py'),
                  Path(__file__).with_name('semantic_probe.py'), Path(__file__).with_name('qwen_upstream.py'),
                  Path(__file__).with_name('category_bridge.py'), Path(__file__).with_name('semantic_categories.py'), MAPPING]
    if args.reviewed_plans_from:
        code_files.append(Path(__file__).with_name('content_review.py'))
    config = dict(protocol=PROTOCOL, cases=cases, stage=args.stage,
                  plans_from=str(args.plans_from.resolve()) if args.plans_from else None,
                  reviewed_plans_from=str(args.reviewed_plans_from.resolve()) if args.reviewed_plans_from else None,
                  model=str(args.model.resolve()), seed=args.seed, max_tokens=args.max_tokens,
                  max_objects=args.max_objects, layout_revisions=args.layout_revisions,
                  category_policy=args.category_policy,
                  code_sha256={str(p.resolve()): sha(p) for p in code_files}, scene_gt_used=False)
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed', cases=len(cases), stage=args.stage,
                              observed=[len(c['context']) for c in cases],
                              content_calls=0 if args.plans_from or args.reviewed_plans_from else len(cases),
                              maximum_layout_calls=len(cases)*(1+args.layout_revisions) if args.stage=='full' else 0,
                              model_loaded=False, gpu_started=False, scene_gt_used=False), ensure_ascii=False))
        return
    output = new_output(args.output)
    write(output/'config.json', config)
    if not run(Qwen(args.model), cases, output, args.stage, args.plans_from, args.seed,
               args.max_tokens, args.max_objects, args.layout_revisions, args.category_policy,
               args.reviewed_plans_from):
        raise SystemExit('Some cases failed; inspect raw responses and summary; do not overwrite this run')


if __name__ == '__main__':
    main()
