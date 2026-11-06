"""No-training v9: whole-camera functional-spatial proposals, shared controls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from ..common import read,write,sha
from . import assembly_panorama as scm,assembly_mechanism_experiment as v8
from . import assembly_realization_experiment as v7,assembly_sdscene_experiment as previous
from .assembly_experiment import initialize

EXPERIMENTS=v8.EXPERIMENTS


def prepare(root,source,seed=3407,worlds=4,draws=4,concentration=1.,reference_concentration=1.):
    scm._positive(concentration,'concentration');scm._positive(reference_concentration,'reference concentration')
    root=previous.prepare(root,source,seed,worlds,draws,'deepseek-ai/DeepSeek-V4.1-Flash',completion_model='panorama')
    config=read(root/'context/experiment.json')
    for label in ('function_free','direct_inventory'):
        initialize(root/label,root/'context/input.png',config['hfov'],Path(config['local_model']),config['api_model'],seed,
            observations_from=root/'context',shared_observer='qwen',completion_model='panorama',worlds=worlds)
    for label in ('context',)+EXPERIMENTS:
        path=root/label/'experiment.json';c=read(path)
        c.update(sdscene_bundle=str(root/'shared'),mechanism_source='pooled' if label=='layout_control' else 'per_backend',
            mechanism_policy='direct' if label=='direct_inventory' else 'function_free' if label=='function_free' else 'functional',
            stage_seed_policy='hashed_stage_v1',training_performed=False,sampling_law='functional_spatial_program_given_partial_view',
            api_base='https://api.siliconflow.cn/v1')
        write(path,c)
    run=read(root/'run.json');run.update(version=9,completion_model='panorama',experiments=list(EXPERIMENTS),concentration=concentration,reference_concentration=reference_concentration,
        training_performed=False,proposal_domain='shared_union_of_validated_functional_spatial_hypotheses',
        spatial_policy='regions_are_sampled_states_not_quantity_targets',direct_control_scope='Direct O,D inventory/spatial-variant choice; H retained in metric context',
        budget_scope='Backends matched; functional/direct total request budgets differ',proposal_interface_revision=scm.PROPOSAL_REVISION)
    run['stage_token_budgets']['direct']=2048;run['stage_token_budgets']['mechanism']=512
    write(root/'run.json',run);return root


def propose(root,backend,p=None):return v8.propose(root,backend,p,model=scm)
def mechanisms(root,backend,p=None):return v8.mechanisms(root,backend,p,model=scm)
def pool_mechanisms(root):return v8.pool_mechanisms(root,model=scm)
def ready(root,experiment,backend):return v8.ready(root,experiment,backend,model=scm)
def block(root,experiment,backend):return v8.block(root,experiment,backend,model=scm)


def pool(root):
    run=read(root/'run.json');case=read(root/'shared/case.json');hypotheses=[];programs={};sources={};tagsets={()};provenance={}
    for backend in ('qwen','api'):
        folder=root/'proposals'/backend;provenance[backend]=read(folder/'summary.json') if (folder/'summary.json').exists() else dict(status='not_run')
        if not (folder/'programs.json').is_file():continue
        hm={h['id']:scm.profile(backend+'_'+h['id'],h['functions']) for h in read(folder/'hypotheses.json')}
        hypotheses+=list(hm.values());tagsets.update(tuple(h['functions']) for h in hm.values())
        for pid,p in read(folder/'programs.json').items():
            for assembly in sorted({g['assembly_key'] for g in p['groups'] if g['objects']}):
                groups=[deepcopy(g) for g in p['groups'] if g['assembly_key']==assembly and g['objects']]
                modes={g['realization_mode'] for g in groups};regions={g['layout_region'] for g in groups}
                if len(modes)!=1 or len(regions)!=1:raise ValueError('Module has inconsistent mode or region')
                mode=next(iter(modes));keys={o['key'] for g in groups for o in g['objects']};renamed={k:mode+'_'+k for k in keys}
                relations=[deepcopy(r) for r in p['relations'] if r['source'] in keys or r['target'] in keys]
                for g in groups:
                    for o in g['objects']:
                        o['key']=renamed[o['key']];o['attachment']['host']=renamed.get(o['attachment']['host'],o['attachment']['host'])
                    for link in g['functional_links']:
                        for key in ('source','target'):link[key]=renamed.get(link[key],link[key])
                for relation in relations:
                    for key in ('source','target'):
                        if relation[key] not in keys and not relation[key].startswith('obs_'):raise ValueError('Cross-module relation')
                        relation[key]=renamed.get(relation[key],relation[key])
                fid='a'+str(len(programs));role=groups[0]['function'];region=next(iter(regions))
                programs[fid]=dict(description=role+' / '+mode+' / camera '+region+': '+', '.join(o['variant'] for g in groups for o in g['objects']),
                    hypothesis=hm[p['hypothesis']['id']],groups=groups,relations=relations,relation_bindings=[],inventory_scope='functional_spatial_realization')
                sources[fid]=dict(backend=backend,program=pid,assembly=assembly,mode=mode,region=region)
    if not programs:raise ValueError('No legal nonempty spatial programs; no empty-world fallback')
    base=scm.build_model(hypotheses,programs);slots={};definitions={};roles=set()
    for fid,p in base['furniture_states'].items():
        if fid=='f_null':continue
        g=p['groups'][0];role=g['function'];mode=g['realization_mode'];c=role+'__'+mode
        slots.setdefault(c,['none']).append(fid);definitions[c]=dict(function=role,realization_mode=mode);roles.add(role)
    slots={c:slots[c] for c in sorted(slots)};roles=sorted(roles);recognition_tags=set(tagsets)
    tagsets.update((r,) for r in roles);tagsets.add(tuple(roles));tagsets.update(tuple(sorted(set(roles)-{r})) for r in roles)
    profiles=[scm.profile('h'+str(i),tags) for i,tags in enumerate(sorted(tagsets))];pairs=[]
    for role in roles:
        for h in profiles:
            if role in h['functions']:continue
            other=next((x for x in profiles if set(x['functions'])==set(h['functions'])|{role}),None)
            if other:pairs.append(dict(role=role,without=h['id'],with_role=other['id']))
    support={c:sorted({base['furniture_states'][s]['groups'][0]['layout_region'] for s in states if s!='none'}) for c,states in slots.items()}
    domain=dict(version=9,hypotheses=hypotheses,profiles=profiles,matched_profile_pairs=pairs,programs=base['furniture_states'],base_model=base,
        slots=slots,channel_definitions=definitions,recognition_profile_ids=[h['id'] for h in profiles if tuple(h['functions']) in recognition_tags],
        RGB_sha256=run['RGB_sha256'],G_obs_sha256=run['G_obs_sha256'],hfov=case['hfov'],image_size=case['image_size'],
        common_inventory=[dict(id=o['id'],name=o['description']) for o in case['context']],proposal_provenance=provenance,program_proposal_sources=sources,
        spatial_support=support,rear_program_proposed=any('rear' in v for v in support.values()),
        missing_rear_support_meaning='No proposed rear program does not mean rear space is empty',
        support_gaps=[dict(function=r,missing_mode=mode) for r in roles for mode in scm.MODES if r+'__'+mode not in slots])
    path=root/'shared/domain.json'
    if path.exists():raise FileExistsError('Domain already frozen')
    write(path,domain);k=run['draws'];multi=sum(len(v)>2 for v in slots.values())
    stages=dict(recognition=k,reference=k,conditional_activation=k*len(profiles),conditional_variants=k*len(profiles)*multi,direct_inventory=k)
    write(root/'call_budget.json',dict(profiles=len(profiles),channels=len(slots),draws=k,stages_per_backend=stages,
        mechanism_initial_calls_per_backend=sum(stages.values()),maximum_with_one_repair_each=2*sum(stages.values()),
        total_layout_attempts_per_backend=run['worlds']*len(EXPERIMENTS),direct_and_functional_budgets_matched=False))
    return dict(status='complete',domain_sha256=sha(path),rear_program_proposed=domain['rear_program_proposed'],spatial_support=support)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','propose','pool','mechanisms','pool-mechanisms','ready','block','report'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source-root',type=Path);p.add_argument('--backend',choices=['qwen','api'])
    p.add_argument('--experiment',choices=EXPERIMENTS);p.add_argument('--seed',type=int,default=3407);p.add_argument('--worlds',type=int,default=4);p.add_argument('--draws',type=int,default=4)
    p.add_argument('--concentration',type=float,default=1.);p.add_argument('--reference-concentration',type=float,default=1.)
    a=p.parse_args();root=a.root.resolve()
    if a.action in ('propose','mechanisms','ready','block') and a.backend is None:p.error('--backend required')
    if a.action in ('ready','block') and a.experiment is None:p.error('--experiment required')
    try:
        if a.action=='prepare':
            if a.source_root is None:p.error('--source-root required')
            result=prepare(root,a.source_root,a.seed,a.worlds,a.draws,a.concentration,a.reference_concentration)
        elif a.action=='propose':result=propose(root,a.backend)
        elif a.action=='pool':result=pool(root)
        elif a.action=='mechanisms':result=mechanisms(root,a.backend)
        elif a.action=='pool-mechanisms':result=pool_mechanisms(root)
        elif a.action=='ready':
            result=ready(root,a.experiment,a.backend)
            if not result['ready']:print(json.dumps(result));raise SystemExit(1)
        elif a.action=='block':result=block(root,a.experiment,a.backend)
        else:
            from .assembly_panorama_visual import report
            result=report(root)
        print(json.dumps(result,default=str,ensure_ascii=False))
        if isinstance(result,dict) and result.get('status') in ('failed','partial'):raise SystemExit(1)
    except Exception as exc:
        folder=root/'phase_errors';folder.mkdir(parents=True,exist_ok=True);error=dict(action=a.action,backend=a.backend,error=type(exc).__name__+': '+str(exc))
        write(folder/(a.action+'_'+str(a.backend)+'.json'),error);p.exit(2,json.dumps(error,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
