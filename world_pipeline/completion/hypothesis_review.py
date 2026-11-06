"""Single-case diagnostic: distinguish hypothetical existence from placement feasibility.

Keep the v1 evidence image, original candidates and generation settings. Assessments
are model predictions, never automatic edits or certification of a completed scene.
"""
import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path

from ..common import new_output, read, sha, write
from .content_review import DEFAULT_SOURCE, load_inputs, marked_image
from .qwen_upstream import DEFAULT_MODEL, Qwen

PROTOCOL = 'hypothesis_existence_placement_diagnostic_v1'
EXISTENCE = ('plausible', 'incompatible', 'unresolved', 'outside_scope')
PLACEMENT = ('compatible', 'conflict', 'underdetermined')
FIELDS = {'id', 'existence', 'existence_reason', 'placement', 'placement_reason', 'evidence_ids'}


def prompt(case, plan):
    references = [dict(mark=i, id=o['id'], category=o['category'], description=o['description'])
                  for i, o in enumerate(case['context'])]
    targets = [o['id'] for o in plan['additions']]
    return (
        'Task: assess proposed objects in ONE POSSIBLE completion of a partially photographed '
        'indoor scene. The completion is a hypothesis, not recovery of a uniquely known hidden room. '
        'The proposed objects are already supplied below. Assess these candidates only. '
        'Return one assessment for EACH of these target IDs: '+json.dumps(targets)+'. '
        'Left is the original RGB; right is the same RGB with predicted reference boxes. '
        'Observed references are evidence only, not targets. Their labels can be mistaken; '
        'consult RGB when using them, and leave the supplied observations unchanged. '
        'For each candidate answer TWO separate questions. '
        'existence: plausible means a reasonable optional scene hypothesis; incompatible means '
        'a specific contradiction or duplication makes this candidate unreasonable in context; '
        'unresolved means its identity or contextual fit cannot be assessed; outside_scope means '
        'it is not a distinct physical object suitable for an object state. '
        'Being absent from the photograph or reference list is not a contradiction. '
        'A plausible hypothesis need not be necessary, proven present, or the only possible choice. '
        'placement: compatible means the original qualitative placement has adequate support '
        'and no identified conflict; conflict means a specific spatial inconsistency; '
        'underdetermined means the location cannot be evaluated from the available view. '
        'A bad or unknown placement does not by itself make existence incompatible. '
        'Unseen space is unknown: do not assume an empty wall or free floor behind the camera. '
        'Do not relocate candidates during this assessment. Explain each judgment briefly, '
        'distinguishing visible evidence from scene common sense and unknown space. '
        'Return only JSON {"reviews":[{"id":<target ID>,"existence":<one of '
        +json.dumps(EXISTENCE)+'>,"existence_reason":<string>,"placement":<one of '
        +json.dumps(PLACEMENT)+'>,"placement_reason":<string>,"evidence_ids":<list of observed IDs>} ]}. '
        'evidence_ids may be empty; cite references only when relevant. '
        'Use exactly these six fields per assessment. Preserve candidate identities; '
        'report judgments without new objects, coordinates, sizes, or replacement placements. '
        '\nREFERENCE_ONLY_OBSERVATIONS: '+json.dumps(references, ensure_ascii=False, separators=(',', ':'))
        +'\nSCENE_HYPOTHESIS: '+json.dumps({k: plan[k] for k in ('scene_type', 'scene_hypothesis')}, ensure_ascii=False)
        +'\nCANDIDATES_TO_ASSESS: '+json.dumps(plan['additions'], ensure_ascii=False, separators=(',', ':'))
        +'\nOUTPUT_TARGET_IDS_ONLY: '+json.dumps(targets))


def validate_assessment(answer, plan, case):
    if not isinstance(answer, dict) or set(answer) != {'reviews'} or not isinstance(answer['reviews'], list):
        raise ValueError('Expected one JSON object containing only a reviews list')
    targets = {o['id'] for o in plan['additions']}
    observed = {o['id'] for o in case['context']}
    seen = set()
    for item in answer['reviews']:
        if not isinstance(item, dict) or set(item) != FIELDS:
            raise ValueError(f'Each assessment needs exactly these fields: {sorted(FIELDS)}')
        oid = item['id']
        if not isinstance(oid, str) or oid not in targets:
            raise ValueError(f'Unexpected assessment target {oid!r}; candidates={sorted(targets)}; '
                             'observed IDs are references only')
        if oid in seen:
            raise ValueError(f'Duplicate assessment target: {oid}')
        if item['existence'] not in EXISTENCE or item['placement'] not in PLACEMENT:
            raise ValueError(f'Invalid existence or placement judgment for {oid}')
        for field in ('existence_reason', 'placement_reason'):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f'{oid} needs a nonempty {field}')
        refs = item['evidence_ids']
        if not isinstance(refs, list) or any(not isinstance(r, str) or r not in observed for r in refs):
            raise ValueError(f'{oid}: evidence_ids must refer to actual observations')
        if len(refs) != len(set(refs)):
            raise ValueError(f'{oid}: duplicate evidence IDs')
        seen.add(oid)
    if seen != targets:
        raise ValueError(f'Missing candidate assessments: {sorted(targets-seen)}')
    return deepcopy(answer)


def run(model, entry, output, seed=3407):
    case, plan = entry['case'], entry['plan']
    folder = output/case['id']
    folder.mkdir()
    image = marked_image(case)
    image.save(folder/'evidence.png')
    write(folder/'input.json', dict(entry, evidence_image_sha256=sha(folder/'evidence.png')))
    write(folder/'original_plan.json', plan)
    write(folder/'G_obs.json', read(case['state']))
    question = prompt(case, plan)
    (folder/'prompt.txt').write_text(question)
    record = dict(id=case['id'], protocol=PROTOCOL, status='failed', semantics_verified=False,
                  geometry_generated=False, plan_modified=False, observed_state_modified=False,
                  candidates=len(plan['additions']))
    try:
        answer = model.ask(image, question, folder/'assessment.txt', seed=seed, sample=True,
                           profile='official', max_tokens=4096, guard_repetition=True)
        assessment = validate_assessment(answer, plan, case)
        write(folder/'assessment_validated.json', assessment)
        record.update(status='assessment_saved_needs_semantic_review',
                      existence_counts={v: sum(r['existence']==v for r in assessment['reviews']) for v in EXISTENCE},
                      placement_counts={v: sum(r['placement']==v for r in assessment['reviews']) for v in PLACEMENT})
    except Exception as exc:
        record['error'] = str(exc)
    if any(sha(p) != h for p, h in entry['source_sha256'].items()):
        record.update(status='failed', error='Original evidence changed during assessment')
    if read(folder/'original_plan.json') != plan or read(folder/'G_obs.json') != read(case['state']):
        record.update(status='failed', error='Saved candidate or observed state changed')
    write(folder/'summary.json', record)
    write(output/'summary.json', record)
    raw = folder/'assessment.txt'
    body = (raw.read_text() if raw.exists() else record.get('error', 'No model response'))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Hypothesis diagnostic</title>'
        '<h1>Hypothesis diagnostic — unverified</h1><img width="1000" src="'+escape(case['id'], quote=True)
        +'/evidence.png"><h2>Original candidates</h2><pre>'+escape(json.dumps(plan, ensure_ascii=False, indent=2))
        +'</pre><h2>Model assessment</h2><pre>'+escape(body)+'</pre><h2>Execution summary</h2><pre>'
        +escape(json.dumps(record, ensure_ascii=False, indent=2))+'</pre>')
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record['status'] != 'failed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--only-id', required=True, help='Exactly one pilot case; no implicit six-case run')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.seed < 2**63:
        parser.error('Invalid seed')
    if args.output.exists():
        parser.error('Output already exists; use a new directory to preserve previous results')
    if not args.model.is_dir():
        parser.error('Local model directory not found; no downloads are performed')
    entry = load_inputs(args.source.resolve(), [args.only_id])[0]
    if not entry['plan']['additions']:
        parser.error('This diagnostic needs saved candidates')
    # Render the actual input during CPU preflight, before any model loading.
    image = marked_image(entry['case'])
    question = prompt(entry['case'], entry['plan'])
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed', cases=1, requests=1,
            candidates=len(entry['plan']['additions']), target_ids=[o['id'] for o in entry['plan']['additions']],
            image_size=image.size, prompt_characters=len(question), model_loaded=False, gpu_started=False)))
        return
    output = new_output(args.output)
    code = [Path(__file__), Path(__file__).with_name('content_review.py'),
            Path(__file__).with_name('scene_completion.py'), Path(__file__).with_name('qwen_upstream.py'),
            Path(__file__).with_name('json_generation.py'), Path(__file__).parents[1]/'common.py']
    write(output/'config.json', dict(protocol=PROTOCOL, inputs=[entry], seed=args.seed,
        model=str(args.model.resolve()), code_sha256={str(p.resolve()): sha(p) for p in code}))
    if not run(Qwen(args.model), entry, output, args.seed):
        raise SystemExit('Hypothesis assessment failed; original response was preserved')


if __name__ == '__main__':
    main()
