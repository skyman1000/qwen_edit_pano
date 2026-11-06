"""Read-only census of stored real replies against the repaired contracts.

No reply is sent to a model or substituted into a new GPU experiment. Replaying
old deltas verifies transport/validation compatibility, not new model quality.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from ..common import ROOT,read,write,sha
from . import joint_world_model as semantic
from .fresh_observer import load_case
from .assembly_observation import host_domains

SOURCE=ROOT/'qwen_edit_pano/outputs/local_observation_fixed_002'
BASELINE=ROOT/'qwen_edit_pano/outputs/v13_local_world_001'
FAILED=ROOT/'qwen_edit_pano/outputs/joint_world_002'


def case_for(sid):
    case=load_case(SOURCE/'scenes'/sid/'observation','qwen');obs=read(case['state'])
    boxes={o['track_id']:o for o in obs['objects']}
    for row in case['context']:
        box=boxes[row['id']];row.update(center=box['center_local_m']['value'],size=box['size_aabb_local_m']['value'])
    case['observed_support_domains']=host_domains(case,obs)
    return case,obs


def request_hypothesis(path):
    prompt=read(path)['prompt'];tail=prompt.split('\nSCENE HYPOTHESIS: ',1)[1]
    return json.JSONDecoder().raw_decode(tail.lstrip())[0]


def baseline_replies(sid):
    return sorted((BASELINE/'scenes'/sid/'completion/proposals/qwen/calls').glob('furniture_*.request.json'))


def census():
    # Hash input artifacts before and after; the evidence directories stay intact.
    inputs={p:sha(p) for root in (BASELINE,FAILED,SOURCE) for p in root.rglob('*.json')}
    suite=read(BASELINE/'suite.json')
    drift=[p for p,d in suite['code_sha256'].items() if not Path(p).is_file() or sha(p)!=d]
    scenes=[]
    for s in suite['scenes']:
        sid=s['id'];case,obs=case_for(sid);rows=[]
        for request in baseline_replies(sid):
            raw=request.with_name(request.name.replace('.request.json','.json'))
            row=dict(file=str(raw.relative_to(ROOT)),repair='_repair.' in raw.name)
            try:
                h=request_hypothesis(request)
                result=semantic.programs_answer(read(raw),case,h)
                row.update(status='accepted',legal_alternatives=len(result['valid']),invalid_siblings=result['failed'],
                    instances=[sum(len(g['objects']) for g in p['groups']) for p in result['valid']],
                    hosted=[sum(bool(o['attachment']['host']) for g in p['groups'] for o in g['objects']) for p in result['valid']])
            except (ValueError,KeyError,TypeError,FileNotFoundError) as exc:
                row.update(status='rejected',error=type(exc).__name__+': '+str(exc))
            rows.append(row)
        hypath=FAILED/'scenes'/sid/'joint_completion/calls/hypotheses.json'
        h=semantic.hypotheses_answer(read(hypath),case)
        # Container-only diagnostic adaptation for the OLD root schema, never a
        # production fallback. All instance identities, hosts and values survive.
        old=[]
        for raw in sorted(hypath.parent.glob('f_h*_roots_repair.json')):
            v=deepcopy(read(raw));groups=[]
            for i,g in enumerate(v.get('groups',[])):
                items=[]
                for o in g.get('objects',[]):
                    items.append(dict(o,reason=o.get('distinct_instance_reason'),placement=o.get('placement_hint')))
                groups.append(dict(function=g.get('function','legacy zone '+str(i)),
                    reason=g.get('reason','Legacy container adaptation for diagnosis only.'),
                    observed_refs=g.get('observed_refs',[]),new_objects=items))
            alt=dict(description=v.get('description'),groups=groups,relations=v.get('relations',[]))
            row=dict(file=str(raw.relative_to(ROOT)),container_adaptation_only=True)
            try:
                r=semantic.programs_answer(dict(alternatives=[alt]),case,h[0])
                row.update(status='accepted',instances=[sum(len(g['objects']) for g in p['groups']) for p in r['valid']])
            except (ValueError,KeyError,TypeError) as exc:row.update(status='rejected',error=str(exc))
            old.append(row)
        scenes.append(dict(scene=sid,observed=len(obs['objects']),v13_replies=rows,
            semantic_H_replay=dict(status='accepted',hypotheses=len(h),metric_fields_required=False),
            failed002_final_reply_diagnostic=old))
    changed=[str(p) for p,d in inputs.items() if sha(p)!=d]
    counts=Counter(r['status'] for s in scenes for r in s['v13_replies'])
    valid_all=all(any(r['status']=='accepted' for r in s['v13_replies']) for s in scenes)
    result=dict(scope='read_only_real_reply_contract_replay_not_fresh_Qwen_quality',
        scenes=scenes,v13_reply_counts=dict(counts),all_five_scenes_have_accepted_v13_reply=valid_all,
        original_v13_code_drift=drift,source_JSON_files_unchanged=len(inputs)-len(changed),changed_input_files=changed,
        API_used=False,GPU_inference_executed=False,new_Qwen_output_quality_verified=False)
    if drift or changed or not valid_all:raise ValueError(json.dumps(result,ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=census();a.output.parent.mkdir(parents=True,exist_ok=True);write(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='scenes'},ensure_ascii=False))


if __name__=='__main__':main()
