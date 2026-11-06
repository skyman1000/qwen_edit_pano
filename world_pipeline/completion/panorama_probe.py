"""Small frozen-Qwen diagnostic, separate from the historical upstream runs.

Downloads and GPU execution are manual. This module does not train or repair layouts.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

from PIL import Image

from ..common import ROOT, read, rows, write
from .geometry import CANONICAL_CATEGORIES, parse_json
from .qwen_upstream import DEFAULT_MODEL, Qwen, camera_text, load_image
from .qwen_upstream_core import assemble, compact_observations
from .qwen_upstream_review import diagnostics, layout, view_diagnostics

PROTOCOL = 'panorama_unobserved_probe_v1'
DEFAULT_IDS = [
    'zsNo4HB9uLZ_ead481533f834704bd489d3d44b6a03a',
    'zsNo4HB9uLZ_ed8645386a0a4de2bfc486e0a1d02468',
    'zsNo4HB9uLZ_38e0c09ac7a748dbadea6471861b30c3',
]
NAMES = set(CANONICAL_CATEGORIES.values())


def decode_plan(answer, capacity):
    """Separate lossless extraction, protocol compliance, and scene quality.

    A bare array has unambiguous list semantics, so wrapping it loses no content.
    Missing reasoning is recorded; no reasoning or object fields are invented.
    Core identity/category/placement errors still prevent downstream use.
    """
    if isinstance(answer, list):
        plan = dict(additions=deepcopy(answer))
        representation = 'bare_array'
    elif isinstance(answer, dict) and 'additions' in answer:
        plan = deepcopy(answer)
        representation = 'object_with_additions'
    else:
        raise ValueError('Plan must be an additions object or a bare object array')
    items = plan['additions']
    if not isinstance(items, list) or len(items) > capacity:
        raise ValueError('Plan exceeds capacity or additions is not a list')
    seen, missing_reasons, groups = set(), [], defaultdict(list)
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f'Plan item {index} must be an object')
        pid = item.get('id')
        if not isinstance(pid, str) or not pid.strip() or pid in seen:
            raise ValueError('Plan needs distinct nonempty string IDs')
        seen.add(pid)
        if not isinstance(item.get('category'), str) or item['category'] not in NAMES:
            raise ValueError(f'Unsupported plan category at {pid}: {item.get("category")!r}')
        if item.get('unseen_reason') not in ('outside_fov', 'occluded', 'uncertain'):
            raise ValueError('Unknown unseen_reason')
        for key in ('description', 'placement'):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f'Plan needs nonempty {key}')
        if not isinstance(item.get('reason'), str) or not item['reason'].strip():
            missing_reasons.append(pid)
        groups[(item['category'], item['description'], item['placement'])].append(pid)
    repeats = [dict(category=key[0], description=key[1], placement=key[2], ids=ids)
               for key, ids in groups.items() if len(ids) > 1]
    diagnostic = dict(raw_representation=representation,
                      normalization='wrap_array_in_additions' if representation == 'bare_array' else 'none',
                      strict_protocol_conformant=representation == 'object_with_additions' and not missing_reasons,
                      missing_or_invalid_reason_ids=missing_reasons,
                      repeated_text_groups=repeats,
                      repeated_text_excess=sum(len(g['ids'])-1 for g in repeats),
                      at_capacity=len(items) == capacity,
                      semantic_quality='not_validated',
                      note='Lossless extraction is not protocol compliance or semantic success; repeated text is a review flag, not automatic object deduplication')
    return plan, diagnostic


def validate_plan(answer, capacity):
    """Validate core fields; protocol deviations are exposed by decode_plan."""
    return decode_plan(answer, capacity)[0]


def validate_geometry(answer, plan):
    """Keep semantic IDs/categories aligned; reject rather than silently drop items."""
    items = answer['hidden_objects']
    expected = {o['id']: o for o in plan['additions']}
    if not isinstance(items, list) or len(items) != len(expected):
        raise ValueError('Geometry count differs from semantic plan')
    actual = {}
    for item in items:
        pid = item['id']
        if pid not in expected or pid in actual:
            raise ValueError('Geometry IDs differ from semantic plan')
        if item['category'] != expected[pid]['category']:
            raise ValueError('Geometry changed a planned category')
        actual[pid] = item
    return dict(hidden_objects=[actual[o['id']] for o in plan['additions']])


def plan_prompt(observed, capacity, camera):
    return (
        'Infer a plausible object state for the complete 360-degree panorama at this same camera. '
        'The image is a partial observation. Preserve every observed instance; propose only additional '
        'objects that are not seen in it. Missing objects may lie to the sides, above, below or behind '
        'the CAMERA, or be occluded by other objects inside the input field of view. '
        'This is scene completion, not a request to redecorate visible surfaces. Do not fill clearly '
        'visible empty space or invent distant rooms. Absence from the image is not evidence that '
        'the full panorama is empty. Equally, do not force additions when none are plausible. '
        'Return one hypothesis, not a list of mutually exclusive alternatives. '
        'Use no metric coordinates at this stage. Return JSON with key additions containing objects '
        'with id (distinct string), category (one allowed canonical name), description, placement '
        '(relative to CAMERA or an observed instance), unseen_reason '
        '(outside_fov, occluded or uncertain), and reason. An empty list is allowed. '
        f'At most {capacity} additions. Do not repeat observed objects. Allowed category names: '
        + ', '.join(CANONICAL_CATEGORIES.values())
        + '. Specific names such as wardrobe or bedside lamp can appear in description; '
        'category must use the canonical name cabinet or lighting respectively.\n'
        + 'CAMERA: ' + camera
        + '\nOBSERVED: ' + json.dumps(compact_observations(observed), ensure_ascii=False)
    )


def geometry_prompt(observed, plan, camera):
    return (
        'Translate this scene hypothesis into approximate camera-local 3D boxes. '
        'Return JSON with key hidden_objects. Each entry must have id and category copied from '
        'the plan, center_local_m (three numbers) and size_aabb_local_m (three positive full lengths). '
        'Use each planned ID exactly once and no other objects. Do not output observed objects. '
        'The origin is the CAMERA. Axes: x image-right, y image-up, z forward; units metres. '
        'Behind the CAMERA means negative z; left means negative x. Behind a bed is an object-relative '
        'relation, and does not by itself mean negative z. Camera axes need not align with gravity. '
        'Sizes are full camera-axis AABB edge lengths, not half extents. '
        'Preserve the plan directions and observed evidence. outside_fov additions must lie outside '
        'the supplied pinhole view; occluded additions may be inside it only if occlusion is plausible. '
        'Do not copy an observed box as a new object. Unknown observations remain unknown. '
        'These are predictions, not measured ground truth.\nCAMERA: ' + camera
        + '\nOBSERVED: ' + json.dumps(compact_observations(observed), ensure_ascii=False)
        + '\nPLAN: ' + json.dumps(plan, ensure_ascii=False)
    )


def ask(model, image, prompt, path):
    # Unlike the historical official profile, no generated-token presence penalty;
    # deterministic decoding is intentional, so different seeds are not advertised.
    return model.ask(image, prompt, path, seed=0, sample=False,
                     profile='completion', max_tokens=4096)


def controls(model, output):
    image = Image.new('RGB', (128, 128), 'gray')
    payload = dict(hidden_objects=[dict(category='chair', center_local_m=[-1.25, -.4, -2.5],
                                       size_aabb_local_m=[.6, .9, .7])])
    copied = ask(model, image, 'Numeric protocol control. The image is irrelevant. Copy the following '
                 'JSON exactly, preserving every key, number and sign. Do not infer a scene.\n'
                 + json.dumps(payload), output/'copy.txt')
    direction = ask(model, image, 'Coordinate convention control; the gray image is irrelevant. '
                    'Camera origin, x right, y up, z forward, units metres. A point lies directly '
                    'behind the CAMERA at a distance of 2 metres, with no horizontal or vertical '
                    'offset. Return JSON with the single key center_local_m and its three coordinates.',
                    output/'direction.txt')
    result = dict(copy_exact=copied == payload,
                  behind_camera_correct=isinstance(direction, dict) and direction.get('center_local_m') == [0, 0, -2],
                  behind_camera_answer=direction,
                  behind_camera_expected=[0, 0, -2],
                  note='Protocol controls, not a test of scene geometry or completion quality')
    write(output/'controls.json', result)
    return result


def infer_plan(model, entry, observed, output, max_hidden):
    output.mkdir(parents=True, exist_ok=True)
    capacity = min(max_hidden, 160-len(observed['objects']))
    if capacity < 0:
        raise ValueError('Observed context exceeds state capacity')
    image = load_image(entry)
    camera = camera_text(image, entry) + f'Horizontal FOV={entry["hfov_deg"]} degrees. '
    write(output/'G_obs.json', observed)
    image.save(output/'local.png')
    # Always re-enter request checking, never skip just because status says completed.
    answer = ask(model, image, plan_prompt(observed, capacity, camera), output/'plan.txt')
    plan, protocol = decode_plan(answer, capacity)
    write(output/'semantic_plan.json', plan)
    write(output/'plan_protocol.json', protocol)
    return image, camera, plan


def plan_case(model, entry, observed, output, max_hidden):
    _, _, plan = infer_plan(model, entry, observed, output, max_hidden)
    result = dict(status='completed', stage='semantic_plan_only',
                  proposed_objects=len(plan['additions']),
                  protocol=read(output/'plan_protocol.json'),
                  note='No new metric boxes or G_full generated; this stage does not evaluate metric geometry')
    write(output/'status.json', result)
    return result


def replay_saved_plans(source, output):
    """Read saved raw responses on CPU into a separate review directory."""
    if (source.resolve() == output.resolve() or source.resolve() in output.resolve().parents
            or output.resolve() in source.resolve().parents):
        raise ValueError('Review output must be separate from the original run')
    config = read(source/'config.json')
    if config.get('stage') != 'plan':
        raise ValueError('Replay requires a saved semantic-only plan run')
    records = []
    for condition in config['conditions']:
        for sid in config['ids']:
            directory = source/condition/sid
            target = output/condition/sid
            target.mkdir(parents=True, exist_ok=True)
            record = dict(condition=condition, id=sid, source=str(directory.resolve()))
            try:
                observed = read(directory/'G_obs.json')
                capacity = min(config['max_hidden'], 160-len(observed['objects']))
                answer = parse_json((directory/'plan.txt').read_text())
                plan, protocol = decode_plan(answer, capacity)
                write(target/'semantic_plan.json', plan)
                write(target/'plan_protocol.json', protocol)
                record.update(extraction='core_fields_extracted', additions=len(plan['additions']),
                              protocol=protocol, semantic_quality='not_validated')
            except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
                record.update(extraction='failed', error=str(exc))
            records.append(record)
    summary = dict(stage='cpu_replay_only', source=str(source.resolve()), records=records,
                   total_cases=len(records),
                   extracted_cases=sum(r['extraction'] == 'core_fields_extracted' for r in records),
                   strict_protocol_cases=sum(r.get('protocol', {}).get('strict_protocol_conformant', False) for r in records),
                   extracted_objects=sum(r.get('additions', 0) for r in records),
                   missing_reason_objects=sum(len(r.get('protocol', {}).get('missing_or_invalid_reason_ids', [])) for r in records),
                   repeated_text_excess=sum(r.get('protocol', {}).get('repeated_text_excess', 0) for r in records),
                   note='No inference, prompt changes, invented fields, deduplication or semantic approval; original outputs unchanged')
    write(output/'summary.json', summary)
    return summary


def complete_case(model, entry, observed, output, max_hidden):
    image, camera, plan = infer_plan(model, entry, observed, output, max_hidden)
    if plan['additions']:
        answer = validate_geometry(ask(model, image, geometry_prompt(observed, plan, camera),
                                       output/'geometry.txt'), plan)
    else:
        answer = dict(hidden_objects=[])
    hidden, full = assemble(observed, answer, 160, max_hidden)
    for obj, planned in zip(hidden['objects'], plan['additions']):
        obj['completion_plan'] = deepcopy(planned)
    full['objects'] = deepcopy(observed['objects']) + deepcopy(hidden['objects'])
    for name, value in [('obs', observed), ('hidden', hidden), ('full', full)]:
        write(output/f'G_{name}.json', value)
    layout(output/'layout.png', [(observed, 'blue', 'Observed'), (hidden, 'orange', 'Hypothesis')])
    result = dict(status='completed', generated=len(hidden['objects']),
                  observed_preserved=full['objects'][:len(observed['objects'])] == observed['objects'],
                  diagnostics=diagnostics(hidden, observed),
                  view=view_diagnostics(hidden, entry['hfov_deg'], image.size),
                  note='Centre FOV and AABB flags do not verify occlusion or physical plausibility')
    write(output/'status.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['controls', 'plan', 'complete', 'replay-plan'])
    p.add_argument('--source', type=Path, help='Saved plan run for CPU-only replay-plan')
    p.add_argument('--bundle', type=Path, default=ROOT/'qwen_edit_pano/data/qwen_upstream_val12_v1')
    p.add_argument('--observer', type=Path,
                   default=ROOT/'qwen_edit_pano/outputs/qwen_world_simple_run001/observer')
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ids', nargs='+', default=DEFAULT_IDS)
    p.add_argument('--conditions', nargs='+', choices=['gt', 'pred'], default=['gt', 'pred'])
    p.add_argument('--max-hidden', type=int, default=12)
    a = p.parse_args()
    if a.stage == 'replay-plan':
        if a.source is None:
            p.error('replay-plan requires --source')
        summary = replay_saved_plans(a.source, a.output)
        print(json.dumps({k: v for k, v in summary.items() if k != 'records'}, ensure_ascii=False), flush=True)
        if summary['extracted_cases'] != summary['total_cases']:
            raise SystemExit('Some stored answers could not be extracted; inspect summary.json')
        return
    if not 0 <= a.max_hidden <= 160:
        p.error('--max-hidden must be in 0..160')
    inputs = {r['id']: r for r in rows(a.bundle/'inputs.jsonl')}
    if any(sid not in inputs for sid in a.ids) or len(set(a.ids)) != len(a.ids):
        p.error('--ids must be distinct IDs from the input manifest')
    refs = ({r['id']: r for r in rows(a.bundle/'references.jsonl')}
            if a.stage != 'controls' and 'gt' in a.conditions else {})
    # Read/validate every input before allocating the model.
    cases = []
    if a.stage != 'controls':
        for condition in a.conditions:
            for sid in a.ids:
                source = Path(refs[sid]['gt_obs']) if condition == 'gt' else a.observer/sid/'G_obs.json'
                if condition == 'pred' and read(source.parent/'status.json')['status'] not in ('completed', 'partial'):
                    raise ValueError('Predicted observer did not complete; do not reuse stale geometry')
                observed = read(source)
                compact_observations(observed)
                if observed['id'] != sid:
                    raise ValueError('Observation ID differs from requested image')
                cases.append((condition, sid, observed))
    config = dict(protocol=PROTOCOL, **{k: str(v.resolve()) if isinstance(v, Path) else v
                                      for k, v in vars(a).items() if k != 'source'})
    config['decoding'] = dict(profile='completion', sample=False, presence_penalty=0, seed=0)
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'config.json').exists() and read(a.output/'config.json') != config:
        raise ValueError('Different probe configuration: select a new output directory')
    write(a.output/'config.json', config)
    model = Qwen(a.model)
    if a.stage == 'controls':
        result = controls(model, a.output)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not (result['copy_exact'] and result['behind_camera_correct']):
            raise SystemExit('Protocol control failed; inspect controls.json. Use the plan stage to test semantics separately; coordinate failure remains recorded.')
        return
    summary = []
    for condition, sid, observed in cases:
        directory = a.output/condition/sid
        directory.mkdir(parents=True, exist_ok=True)
        try:
            run_case = plan_case if a.stage == 'plan' else complete_case
            result = run_case(model, inputs[sid], observed, directory, a.max_hidden)
        except (ValueError, KeyError, TypeError) as exc:
            result = dict(status='failed', error=str(exc))
            write(directory/'status.json', result)
        summary.append(dict(condition=condition, id=sid, **result))
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    write(a.output/'summary.json', dict(protocol=PROTOCOL, stage=a.stage, records=summary,
          note='GT-observed is diagnostic only; no GT hidden state or complete panorama enters prompts'))
    if any(row['status'] != 'completed' for row in summary):
        raise SystemExit('Some cases failed; inspect summary.json and raw responses')


if __name__ == '__main__':
    main()
