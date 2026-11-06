"""Isolate scene-content hypotheses from numeric layout; user runs inference.

Reuses predicted WildDet3D observations. No scene GT, layout examples, new
coordinates, automatic furniture placement, Adapter calls or image generation.
"""
import argparse
from html import escape
import json
import math
from pathlib import Path

from PIL import Image

from ..common import ROOT, new_output, read, sha, validate_graph, write
from .category_bridge import category as bridge_category
from .geometry import CANONICAL_CATEGORIES
from .qwen_upstream import DEFAULT_MODEL, Qwen

PROTOCOL = 'fixed_predicted_observation_semantic_probe_v1'
DEFAULT_BATCH = ROOT/'qwen_edit_pano/outputs/box_world_val6_run002'


def state_context(graph):
    validate_graph(graph)
    if graph['stage'] != 'prediction' or graph['scope'] != 'observed' or not graph['objects']:
        raise ValueError('Use a nonempty predicted G_obs, not scene GT/full state')
    result = []
    for obj in graph['objects']:
        if obj['local_evidence'] != 'visible' or not obj['category_id']['valid']:
            raise ValueError('Need visible observed instances with valid categories')
        result.append(dict(id=obj['track_id'], category=CANONICAL_CATEGORIES[obj['category_id']['value']],
            description=obj.get('description', ''), center=obj['center_local_m']['value'],
            full_aabb_size=obj['size_aabb_local_m']['value'],
            visible_bbox_xyxy_norm=obj['bbox2d_visible_xyxy_norm']['value']))
    return result


def prompt(context, hfov):
    # Identical text in both conditions; only the RGB attachment changes.
    return (
        'Infer ONE plausible complete surrounding indoor scene from the fixed partial observations below. '
        'An RGB photograph may also be attached as additional evidence. '
        'First understand the actual scene purpose, then propose the missing contents that make '
        'a coherent scene around this SAME camera. Unobserved space is unknown, not known empty. '
        'The eventual target is a full 360-degree panorama saved as a 2:1 ERP image. '
        'Extend the surrounding scene relevant to this view, rather than inventing a whole building. '
        'Preserve all observed instances; do not repeat them as new instances. Additional instances '
        'of an observed category are allowed if there is a scene-specific reason. '
        'Choose contents from the evidence and room function, not a generic furniture checklist. '
        'Distinguish inferred additions from observed facts. Keep visible empty areas consistent '
        'with the photograph when one is supplied. Do not require any number of additions or '
        'left/right/back coverage; a sparse completion can be reasonable. '
        'Use ordinary specific object names. No category-vocabulary restriction in this content test. '
        'Return JSON: scene_type (string), scene_hypothesis (string), additions (list). '
        'Each addition has id (a distinct new instance ID), name, reason, placement (all strings). '
        'placement describes a qualitative location in the surrounding scene and why it could '
        'be outside the photograph or hidden. Cite existing object IDs when relating to them. '
        'Mention only observed objects or explicitly listed additions as committed scene contents. '
        'Do not output new numeric positions, sizes, offsets or a relation graph. '
        'The input observed boxes are approximate predictions, not measured ground truth. '
        'Their axes are x right, y IMAGE-UP, z forward, in metres; full_aabb_size is full camera-axis '
        'box size. Image-up is not verified gravity. No floor or room bounds are measured. '
        f'The photograph covers {hfov:g} horizontal degrees. Return ONLY JSON. '
        '\nFIXED OBSERVATIONS: '+json.dumps(context, ensure_ascii=False, separators=(',', ':')))


def scene_contents_prompt(context, hfov):
    """An opt-in task clarification; keep the original probe for comparison."""
    return (
        'The photograph is a partial view of one surrounding indoor environment. Plan ONE plausible '
        'complete scene around this SAME camera, suitable for a full 360-degree panorama. '
        'Infer the scene function from the observations, then extend the environment beyond the '
        'photograph with a coherent choice of independent scene objects and their qualitative arrangement. '
        'The task is scene continuation, rather than identifying missing fittings or redecorating '
        'the objects already in the photograph. Existing objects include their normal components. '
        'The supplied inventory is a coarse detector summary: absence of a separate entry does '
        'not imply absence from the RGB image. Preserve visible fixtures and empty surfaces. '
        'New instances can share categories with observed ones, but must be distinct instances. '
        'Choose contents for this particular environment, rather than a standard list for every room. '
        'Consider the surrounding functional layout, not only accessories beside the visible objects. '
        'A hidden object may be a hypothesis, but do not use an unsupported occlusion claim to '
        'place an object where the photograph shows empty space. '
        'Choose the number of additions and their directions freely; do not fill direction quotas. '
        'Return JSON with scene_type, scene_hypothesis, additions. Each addition has a new id, '
        'name, reason and placement, all strings. name is a specific object name, not a whole sentence. '
        'reason links its role to the observed scene; placement describes its qualitative location '
        'around the camera and why it would be outside the photograph or hidden. '
        'Refer to existing instances by ID. Commit to an object only by including it in the '
        'observed inventory or additions; omit rejected proposals entirely from additions. '
        'Do not predict numeric positions, sizes or offsets in this step. '
        'Observed centers and full_aabb_size are approximate camera-axis predictions in metres, '
        'x right/y IMAGE-UP/z forward. Gravity, floor and room bounds are not measured. '
        f'The photograph covers {hfov:g} horizontal degrees; the eventual RGB output is a 2:1 ERP. '
        'An RGB image may also be attached. Return ONLY JSON. '
        '\nFIXED OBSERVATIONS: '+json.dumps(context, ensure_ascii=False, separators=(',', ':')))


def validate_answer(answer, context):
    if (not isinstance(answer, dict) or
            any(not isinstance(answer.get(k), str) or not answer[k].strip()
                for k in ('scene_type', 'scene_hypothesis')) or
            not isinstance(answer.get('additions'), list)):
        raise ValueError('Need scene_type, scene_hypothesis and additions list')
    seen = {obj['id'] for obj in context}
    mappings = []
    for obj in answer['additions']:
        if (not isinstance(obj, dict) or any(not isinstance(obj.get(k), str) or not obj[k].strip()
                for k in ('id', 'name', 'reason', 'placement')) or obj['id'] in seen):
            raise ValueError('Each addition needs a new unique ID, name, reason and placement')
        seen.add(obj['id'])
        # Missing mapping is reported, never used to reject or silently rename a semantic proposal.
        try:
            category, _ = bridge_category(obj['name'])
            mappings.append(dict(id=obj['id'], name=obj['name'], adapter_category=category))
        except ValueError:
            mappings.append(dict(id=obj['id'], name=obj['name'], adapter_category=None))
    return mappings


def load_cases(batch, only_ids, plan_style='baseline'):
    # Fixed already-observed cases; no GT, completion answers or room labels are loaded.
    case_dirs = sorted(p for p in batch.iterdir() if p.is_dir() and (p/'observer_001/G_obs.json').exists())
    if only_ids:
        found = {p.name for p in case_dirs}
        if set(only_ids) - found:
            raise ValueError('Unknown case IDs: '+str(sorted(set(only_ids)-found)))
        case_dirs = [p for p in case_dirs if p.name in only_ids]
    if not case_dirs:
        raise ValueError('No fixed observer_001/G_obs.json cases in batch')
    cases = []
    for directory in case_dirs:
        source = directory/'observer_001'
        config_path, state_path = source/'config.json', source/'G_obs.json'
        config = read(config_path)
        image_path = Path(config['image']).resolve()
        if sha(image_path) != config['image_sha256']:
            raise ValueError('Original observer image changed: '+directory.name)
        hfov = config.get('hfov', config.get('hfov_deg', 90.))
        if not isinstance(hfov, (float, int)) or not math.isfinite(hfov) or not 0 < hfov < 180:
            raise ValueError('Need a valid perspective horizontal FOV')
        context = state_context(read(state_path))
        make_prompt = {'baseline': prompt, 'scene_contents': scene_contents_prompt}[plan_style]
        cases.append(dict(id=directory.name, image=str(image_path), state=str(state_path.resolve()),
            hfov=hfov, context=context, prompt=make_prompt(context, hfov),
            source_sha256={str(p.resolve()): sha(p) for p in (config_path, state_path, image_path)}))
    return cases


def review(output, records):
    cards = []
    for record in records:
        rel = Path(record['id'])/record['mode']
        detail = output/rel/'answer.json'
        value = read(detail) if detail.exists() else record.get('error', 'pending')
        cards.append('<article><h2>'+escape(record['id']+' / '+record['mode'])+'</h2>'
            '<img src="'+escape(record['id']+'/local.png', quote=True)+'">'
            '<p>Format: '+escape(record['status'])+'; semantic quality: NOT VALIDATED</p>'
            '<pre>'+escape(json.dumps(value, ensure_ascii=False, indent=2))+'</pre></article>')
    (output/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Scene content probe</title>'
        '<style>body{font:16px sans-serif;margin:24px}article{border-top:1px solid #aaa;padding:12px}'
        'img{width:320px;max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        '<h1>Scene content hypotheses: state versus image + state</h1>'
        '<p>state includes predicted categories, boxes and descriptions. No GT/room labels/examples '
        'were supplied. JSON format and category mapping do not certify semantic correctness. '
        'This test does not create G_full or an ERP image.</p>'+''.join(cards))


def run(model, cases, output, modes, seed, max_tokens):
    records = []
    for case in cases:
        directory = output/case['id']
        directory.mkdir(exist_ok=True)
        write(directory/'input.json', case)
        with Image.open(case['image']) as handle:
            image = handle.convert('RGB')
        image.thumbnail((768, 768))
        image.save(directory/'local.png')
        for mode in modes:
            destination = directory/mode
            destination.mkdir(exist_ok=True)
            record = dict(id=case['id'], mode=mode, observed=len(case['context']),
                          status='failed', semantics_verified=False)
            try:
                answer = model.ask(image if mode == 'image_state' else None, case['prompt'],
                    destination/'answer.txt', seed=seed, sample=True, profile='official',
                    max_tokens=max_tokens, guard_repetition=True)
                mapping = validate_answer(answer, case['context'])
                write(destination/'adapter_mapping_review.json', mapping)
                record.update(status='format_passed_needs_semantic_review', additions=len(answer['additions']),
                              adapter_names_unresolved=sum(m['adapter_category'] is None for m in mapping))
            except Exception as exc:
                record['error'] = str(exc)
            records.append(record)
            write(output/'summary.json', dict(protocol=PROTOCOL, records=records, scene_gt_used=False,
                numeric_layout_generated=False, full_state_generated=False, semantics_verified=False))
            review(output, records)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    if any(sha(Path(path)) != digest for case in cases for path,digest in case['source_sha256'].items()):
        raise ValueError('Source observation or image changed during probe')
    return all(record['status'] != 'failed' for record in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=DEFAULT_BATCH)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--only-id', action='append')
    parser.add_argument('--modes', nargs='+', choices=('state', 'image_state'), default=['state', 'image_state'])
    parser.add_argument('--plan-style', choices=('baseline', 'scene_contents'), default='baseline')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.seed < 2**63 or args.max_tokens < 1 or len(set(args.modes)) != len(args.modes):
        parser.error('Invalid seed/token budget or duplicate modes')
    cases = load_cases(args.batch.resolve(), args.only_id, args.plan_style)
    config = dict(protocol=PROTOCOL, model=str(args.model.resolve()), seed=args.seed, modes=args.modes,
        max_tokens=args.max_tokens, inputs=cases, source_gt_used=False,
        code_sha256={p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name('qwen_upstream.py'))})
    if args.plan_style != 'baseline':
        config['plan_style'] = args.plan_style
    output = args.output.resolve()
    if output.exists() and (not args.resume or read(output/'config.json') != config):
        parser.error('Use a new output, or --resume with unchanged inputs/code/settings')
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed', cases=len(cases), requests=len(cases)*len(args.modes),
            observed=[len(c['context']) for c in cases], model_loaded=False, scene_gt_used=False)))
        return
    if not output.exists():
        output = new_output(output)
        write(output/'config.json', config)
    if not run(Qwen(args.model), cases, output, args.modes, args.seed, args.max_tokens):
        raise SystemExit('Some responses failed; inspect raw answers and summary.json; no automatic repair')


if __name__ == '__main__':
    main()
