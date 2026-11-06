"""Read-only CPU postflight: recompute inference and inspect every saved world.

This checks software/data contracts, not semantic or physical truth.
"""
import argparse
import json
import math
from pathlib import Path

from PIL import Image

from ..common import read, sha, validate_graph
from .causal_world_completion import check_hashes
from .frozen_causal_core import infer, normalize_proposal, normalized_probabilities
from .frozen_causal_layout import diagnostics
from .off_fov_completion import validate_world


def close(a, b):
    return set(a) == set(b) and all(math.isclose(a[k], b[k], rel_tol=1e-7, abs_tol=1e-9) for k in a)


def audit(output):
    output = Path(output)
    config = read(output/'config.json')
    case = config['case']; folder = output/case['id']
    check_hashes(case['source_sha256'])
    check_hashes(config['code_sha256'])
    observed = validate_graph(read(folder/'G_obs.json'))
    if observed != read(case['state']): raise ValueError('Frozen observation changed')
    domain = read(folder/'proposal_domain.json')
    args = config.get('arguments', {})
    source = folder/'proposal_repair.json' if (folder/'proposal_repair.json').is_file() else folder/'proposal.json'
    normalized = normalize_proposal(read(source), case, args.get('max_objects',256),
                                   args.get('context_budget',3), args.get('proposal_world_budget',6))
    if domain != normalized: raise ValueError('Proposal domain differs from saved raw-parsed response')
    spec = read(folder/'causal_model.json')
    bindings = [('context_prior', next(n for n in spec['nodes'] if n['id']=='Z')['cpt'][0])]
    for nid, stem in (('E','evidence'),('H','hidden')):
        for row in next(n for n in spec['nodes'] if n['id']==nid)['cpt']:
            bindings.append((stem+'_'+row['when'][0], row))
    for name, cpt in bindings:
        score = read(folder/'mechanisms'/(name+'.json'))
        if not close(score['probabilities'], cpt['probabilities']):
            raise ValueError('Saved mechanism/CPT differ: '+name)
        if score.get('rounds'):
            mixture = {sid:0. for sid in score['probabilities']}
            for row in score['rounds']:
                values = normalized_probabilities(row['log_probabilities'])
                if not all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(values,row['probabilities'])):
                    raise ValueError('Raw log score normalization differs: '+name)
                for sid, value in zip(row['state_ids'], values): mixture[sid] += value/len(score['rounds'])
            if not close(mixture,score['probabilities']): raise ValueError('Label mixture differs: '+name)
    recomputed = infer(spec); saved = read(folder/'inference_trace.json')
    for field in ('posterior_context','prior_hidden','posterior_hidden','do_evidence_control'):
        if not close(recomputed[field]['probabilities'], saved[field]['probabilities']):
            raise ValueError('Recomputed inference differs: '+field)
    for z in recomputed['context_interventions']:
        if not close(recomputed['context_interventions'][z]['probabilities'],saved['context_interventions'][z]['probabilities']):
            raise ValueError('Recomputed intervention differs: '+z)
    summary = read(folder/'summary.json'); selection = read(folder/'world_selection.json')
    probs = saved['posterior_hidden']['probabilities']
    expected = sorted(domain['worlds'],key=lambda w:(-probs[w['id']],w['id']))[:args.get('worlds',3)]
    if selection['selected'] != [w['id'] for w in expected]: raise ValueError('World ranking changed')
    if not math.isclose(selection['selected_model_mass'],sum(probs[w['id']] for w in expected),abs_tol=1e-9):
        raise ValueError('Selected probability mass changed')
    if summary.get('status') in ('failed','partial_failure'): raise ValueError('Run contains explicit failures; inspect summaries')
    rows = []
    for h, proposed in zip(summary['hypotheses'],expected):
        if h['id'] != proposed['id']: raise ValueError('Hypothesis order/identity differs')
        dest = folder/h['folder']; plan = read(dest/'semantic_plan.json')
        if plan != proposed: raise ValueError('Semantic inventory changed after ranking')
        items = read(dest/'layout_solved.json')['objects']
        full = validate_graph(read(dest/'G_full.json'))
        validate_world(observed,full,items,args.get('max_objects',256))
        if read(dest/'G_out.json')['objects'] != full['objects'][len(observed['objects']):]:
            raise ValueError('Hidden/full export differs')
        if plan['additions']:
            boxes = dest/'boxes_repair.json' if (dest/'boxes_repair.json').is_file() else dest/'boxes.json'
            raw = {o['id']:o for o in read(boxes)['objects']}
            for item in items:
                if item['size'] != raw[item['id']]['size']: raise ValueError('Solver resized a box')
                if item['proposal_center'] != raw[item['id']]['center']: raise ValueError('Proposal trace differs')
        review = diagnostics(items,observed,plan['relations'],case['hfov'],case['image_size'])
        if not review['numeric_constraints_passed']: raise ValueError('New box still intersects input cone')
        if not h['world_state_ready']: raise ValueError('Saved world-state flag contradicts geometry')
        with Image.open(dest/'world/erp_state_map.png') as image:
            if image.width != 2*image.height: raise ValueError('Direction chart is not 2:1')
        if not (dest/'world/scene_3d.html').is_file(): raise ValueError('Missing interactive visualization')
        rows.append(dict(id=h['id'],new_instances=len(items),observed_state_preserved=True,
                         outside_input_frustum=True,unmapped_retained=sum(not o['category_id']['valid'] for o in full['objects'])))
    if len(summary['hypotheses']) != len(expected): raise ValueError('Selected hypotheses missing from summary')
    if expected and (folder/'G_full.json').is_file():
        if read(folder/'G_full.json') != read(folder/summary['hypotheses'][0]['folder']/'G_full.json'):
            raise ValueError('Preferred world was replaced by another candidate')
    return dict(status='postflight_contract_passed',id=case['id'],hypotheses=rows,
                inference_recomputed=True,mechanisms_bound=True,source_hashes_unchanged=True,
                semantic_quality_verified=False,causal_gain_validated=False,gpu_started=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    try: result=audit(args.output)
    except Exception as exc:
        print(json.dumps(dict(status='postflight_failed',error=f'{type(exc).__name__}: {exc}'),ensure_ascii=False))
        raise SystemExit(1)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
