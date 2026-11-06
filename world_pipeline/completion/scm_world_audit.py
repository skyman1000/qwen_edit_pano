"""Read-only CPU replay of parameters, count posterior, samples and geometry."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from PIL import Image

from ..common import read
from .causal_world_completion import check_hashes
from .conditional_world_core import aggregate_rotations
from .conditional_world_geometry import compile_layout,support_plane
from .off_fov_completion import validate_world
from .scm_count_model import draw,infer,trace,validate
from .scm_mechanisms import count_states,domain_answer,registration_answer
from .scm_world_layout import instantiate,support_answer
from .scm_typed_layout import decorate_review,numeric_body,replay_typed
from .scm_typed_queries import normalized_logs
from .scm_relational_layout import decorate_joint,metric_validator,replay_relational


def close(a,b):
    if isinstance(a,bool) or isinstance(b,bool):return type(a)==type(b) and a==b
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-9)
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(close(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    return a==b


def raw(folder,stem):
    repaired=folder/(stem+'_repair.json')
    return read(repaired if repaired.exists() else folder/(stem+'.json'))


def audit_semantics(root,check_all_code=True):
    root=Path(root);config=read(root/'config.json');case=config['case'];args=config['arguments'];folder=root/case['id']
    if check_all_code:check_hashes(config['code_sha256'])
    check_hashes(case['source_sha256'])
    observed=read(folder/'G_obs.json')
    if observed!=read(case['state']):raise ValueError('G_obs changed')
    domain=domain_answer(raw(folder,'domain'),args['generic_types'])
    if domain!=read(folder/'domain.normalized.json'):raise ValueError('Domain replay differs')
    # The domain request precedes image registration and has no image input.
    dr=folder/('domain_repair.request.json' if (folder/'domain_repair.request.json').exists() else 'domain.request.json')
    if read(dr).get('text_only') is not True:raise ValueError('Generative domain must be text-only')
    registration=registration_answer(raw(folder,'registration'),case,domain)
    if registration!=read(folder/'registration.normalized.json'):raise ValueError('Observed registration replay differs')
    plane=support_plane(observed,registration['floor_anchor_ids'])
    if not close(plane,read(folder/'support_plane.json')):raise ValueError('Support plane differs')
    spec=validate(read(folder/'semantic_model.json'),registration['observed_counts'])
    scoring=args.get('mechanism_scoring')
    if scoring in ('numeric','continuation','legacy_codes'):
        expected='json_integer_v1' if scoring=='numeric' else 'sentence_v1'
        if spec.get('count_event_contract','sentence_v1')!=expected:
            raise ValueError('Count event contract differs from requested mechanism scoring')
    if (spec['backgrounds']!=domain['backgrounds'] or spec['configurations']!=domain['configurations']
        or spec['object_types']!=registration['object_types'] or spec['max_count']!=args['max_count']):
        raise ValueError('Finite model domain changed')
    checks=[('B',spec['mechanisms']['B'],spec['backgrounds'],{})]
    for b in spec['backgrounds']:
        checks.append(('K_'+b['id'],spec['mechanisms']['K'][b['id']],spec['configurations'],{'B':b['description']}))
        for k in spec['configurations']:
            for ti,t in enumerate(spec['object_types']):
                states=count_states(t,spec['max_count'],spec.get('count_event_contract','sentence_v1'))
                checks.append((f'N_{b["id"]}_{k["id"]}_t{ti:02d}',spec['mechanisms']['N'][b['id']][k['id']][t],states,
                               {'B':b['description'],'K':k['description']}))
    for name,values,states,parents in checks:
        saved=read(folder/'mechanisms'/(name+'.json'));ids=[s['id'] for s in states]
        if (saved['declared_parents']!=parents or saved['state_catalog']!=states
            or not close(saved['probabilities'],dict(zip(ids,values)))):raise ValueError('Mechanism binding differs: '+name)
        if saved.get('rounds'):
            if not close(aggregate_rotations(saved['rounds'],ids)['probabilities'],saved['probabilities']):
                raise ValueError('Cyclic score aggregation differs: '+name)
        if saved.get('candidate_sequences'):
            sequences=saved['candidate_sequences']
            if (saved.get('image_used') is not False or saved['parents']!=parents or saved['states']!=states
                or [r['id'] for r in sequences]!=ids or [r['text'] for r in sequences]!=[s['description'] for s in states]):
                raise ValueError('Continuation parent/state binding differs: '+name)
            scores=[]
            for row in sequences:
                lp=row['token_log_probabilities']
                if (not lp or len(lp)!=len(row['token_ids']) or any(not math.isfinite(v) or v>0 for v in lp)
                    or not close(sum(lp),row['log_probability'])):raise ValueError('Continuation token scores differ: '+name)
                scores.append(sum(lp))
            if not close(dict(zip(ids,normalized_logs(scores))),saved['probabilities']):
                raise ValueError('Continuation normalization differs: '+name)
    result=infer(spec,registration['observed_counts'])
    if not close(trace(result),read(folder/'inference_trace.json')):raise ValueError('Posterior replay differs')
    regimes=read(folder/'mechanism_interventions.json')['regimes']
    for k in domain['configurations']:
        expected=dict(prior=trace(infer(spec,{}, {'K':k['id']})),
                      conditioned=trace(infer(spec,registration['observed_counts'],{'K':k['id']})))
        if not close(expected,regimes[k['id']]):raise ValueError('do mechanism replay differs')
    return dict(config=config,case=case,args=args,folder=folder,observed=observed,domain=domain,registration=registration,plane=plane,spec=spec,result=result)


def audit(root):
    state=audit_semantics(root);root=Path(root)
    config,case,args,folder,observed,domain,registration,plane,spec,result=(state[k] for k in ('config','case','args','folder','observed','domain','registration','plane','spec','result'))
    from .scm_checkpoint import audit_reuse,reasoning_report
    audit_reuse(folder)
    if (folder/'causal_reasoning.json').exists() and not close(reasoning_report(spec,result),read(folder/'causal_reasoning.json')):
        raise ValueError('Causal reasoning diagnostics differ')
    summary=read(folder/'summary.json')
    if summary['status'] in ('failed','partial_failure'):raise ValueError('Run contains explicit failure; inspect traceback/raw replies')
    if len(summary['hypotheses'])!=args['worlds']:raise ValueError('Missing sampled hypotheses')
    rows=[]
    for rank,h in enumerate(summary['hypotheses']):
        dest=folder/h['folder'];sample=draw(result,args['seed']+1000+rank)
        if not close(sample,read(dest/'posterior_sample.json')):raise ValueError('Posterior sample/seed differs')
        initial=instantiate(sample,case,domain)
        if initial!=read(dest/'sampled_instances.json'):raise ValueError('Sampled identities changed')
        typed=args.get('layout_backend')=='typed';joint=args.get('layout_backend')=='joint'
        if joint:
            saved_layout=read(dest/'joint_layout.json')
            plan,answer=replay_relational(case,observed,initial,plane,saved_layout)
            for row in saved_layout['metric_records']:
                if metric_validator(raw(dest,row['stem']),saved_layout['profiles'][row['id']],plane)!=row['body']:
                    raise ValueError('Joint numeric proposal differs from LM answer')
            for row in saved_layout['profile_records']+saved_layout['support_records']:
                file=dest/(row['stem']+'.txt');req=read(file.with_suffix('.request.json'))
                if (json.loads(file.read_text())!=row['choice'] or read(file.with_suffix('.json'))!=row['choice']
                    or row['choice'] not in req['candidates']
                    or req['method'] not in ('finite_token_trie_greedy_layout_v1','program_singleton_v1')
                    or (req['method']=='program_singleton_v1' and len(req['candidates'])!=1)):
                    raise ValueError('Joint finite choice differs')
            if not close(answer,read(dest/'geometry_raw.json')):raise ValueError('Joint geometry raw replay differs')
        elif typed:
            saved_layout=read(dest/'typed_layout.json')
            plan,answer=replay_typed(case,observed,initial,plane,saved_layout)
            for row in saved_layout['metric_records']:
                if numeric_body(raw(dest,row['stem']))!=row['body']:
                    raise ValueError('Typed numeric proposal differs from LM answer')
            for row in saved_layout['support_records']:
                file=dest/(row['stem']+'.txt');req=read(file.with_suffix('.request.json'))
                if (json.loads(file.read_text())!=row['choice'] or read(file.with_suffix('.json'))!=row['choice']
                    or row['choice'] not in req['candidates'] or req['method']!='finite_token_trie_greedy_layout_v1'):
                    raise ValueError('Finite layout selection differs')
            if answer!=read(dest/'geometry_raw.json'):raise ValueError('Typed geometry raw replay differs')
        else:plan=support_answer(raw(dest,'supports'),initial,case) if initial['additions'] else initial
        if plan!=read(dest/'semantic_plan.json'):raise ValueError('Layout changed fixed sampled instances/supports')
        if Counter(o['object_type'] for o in plan['additions'])!=Counter({r['object_type']:r['outside'] for r in sample['counts'] if r['outside']}):
            raise ValueError('Exported semantic counts differ')
        mappings=read(dest/'category_mapping.json');answer=read(dest/'geometry_raw.json') if plan['additions'] else {'objects':[]}
        items,review=compile_layout(answer,plan,mappings,observed,plane,case['hfov'],case['image_size'])
        if typed:
            decorate_review(review,saved_layout)
        if joint:
            decorate_joint(review,saved_layout,items,observed)
        if not close(items,read(dest/'layout_solved.json')['objects']) or not close(review,read(dest/'numeric_review.json')):
            raise ValueError('Metric compilation differs')
        full=read(dest/'G_full.json');validate_world(observed,full,items,args['max_objects'])
        if joint and (full.get('layout_installation_roles')!=saved_layout['profiles']
            or not close(full.get('numeric_review'),review)):
            raise ValueError('Exported joint layout metadata differs')
        hidden=full['objects'][len(observed['objects']):]
        if read(dest/'G_out.json')['objects']!=hidden or len(hidden)!=sample['new_instances']:
            raise ValueError('Outside sample not completely exported')
        if h['world_state_ready']!=review['geometry_contract_passed']:raise ValueError('Numeric status differs')
        if not (dest/'world/scene_3d.html').is_file():raise ValueError('Missing 3D visual')
        with Image.open(dest/'world/erp_state_map.png') as im:
            if im.width!=2*im.height:raise ValueError('State direction map must be 2:1')
        rows.append(dict(folder=h['folder'],sampled_outside=sample['new_instances'],exported=len(hidden),
                         world_state_ready=h['world_state_ready']))
    if summary['hypotheses'][0]['world_state_ready']:
        if read(folder/'G_full.json')!=read(folder/summary['hypotheses'][0]['folder']/'G_full.json'):
            raise ValueError('Root alias does not represent first draw')
    elif (folder/'G_full.json').exists():raise ValueError('Unready first sample replaced by another world')
    return dict(status='scm_postflight_contract_passed',hypotheses=rows,source_hashes_unchanged=True,
        mechanisms_bound=True,count_posterior_recomputed=True,samples_replayed=True,geometry_recomputed=True,
        observed_state_preserved=True,semantic_quality_verified=False,causal_gain_validated=False,
        joint_metric_posterior_defined=False,gpu_started=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);args=p.parse_args()
    try:result=audit(args.output)
    except Exception as exc:
        print(json.dumps(dict(status='postflight_failed',error=f'{type(exc).__name__}: {exc}'),ensure_ascii=False));raise SystemExit(1)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
