"""Local-only, plan-first conditional world generator with frozen observations.

User runs GPU. prepare/report/tests never construct a model. Old pipelines untouched.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import traceback

from PIL import Image

from ..common import ROOT, new_output, read, write, sha, field, validate_graph, ELIGIBLE
from . import panorama_plan_model as model, panorama_plan_visual as visual
from .organized_local_world import observed_source
from .fresh_observer import load_case
from .fresh_pipeline import provider
from .assembly_experiment import stage_seed
from .assembly_taxonomy import review as taxonomy_review, VOCABULARY
from .category_bridge import MAPPING
from .assembly_presentation_visual import render_world
from .scene_completion import review_boxes


def code_hashes():
    # Bind dependencies as well as the new implementation, without modifying legacy pins.
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'qwen_edit_pano/world_pipeline/common.py',
          ROOT/'qwen_edit_pano/scripts/panorama_plan.sh',VOCABULARY,MAPPING]
    return {str(p.resolve()):sha(p) for p in paths if not p.name.startswith('test_')}


def prepare(root, source, hypotheses=3, seed=3407):
    model.require(type(hypotheses) is int and 1 <= hypotheses <= 8, 'Use 1..8 alternative plans')
    source=Path(source).resolve();original=read(source/'suite.json');inputs=[]
    for s in original['scenes']:
        model.require(isinstance(s['id'],str) and re.fullmatch(r'[A-Za-z0-9_-]+',s['id']), 'Unsafe scene ID')
        obs,state=observed_source(source,s)
        inputs.append((s,obs,state))
    root=new_output(root); records=[];bindings={}
    for s,obs,state in inputs:
        dst=root/'scenes'/s['id']/'observation';dst.mkdir(parents=True)
        for name in ('input.png','experiment.json'):
            shutil.copyfile(obs/name,dst/name)
        for stage in ('observer2d','observer3d'):
            shutil.copytree(obs/'qwen'/stage,dst/'qwen'/stage)
        for p in dst.rglob('*'):
            if p.is_file():bindings[str(p)]=sha(p)
        for name in ('input.png','experiment.json','qwen/observer2d/G_2d.json','qwen/observer3d/G_obs.json'):
            bindings[str(obs/name)]=sha(obs/name)
        records.append(dict(id=s['id'],hfov=s['hfov'],observed=len(state['objects'])))
    suite=dict(kind=model.VERSION,local_model=original['local_model'],seed=seed,hypotheses=hypotheses,
               scenes=records,source_suite=str(source/'suite.json'),source_sha256=sha(source/'suite.json'),
               bindings=bindings,code_sha256=code_hashes(),API_used=False,observer_rerun=False,
               hypothesis_policy='enumerate_proposed_alternatives_once_not_IID_posterior_samples')
    write(root/'suite.json',suite);visual.report(root,suite)
    return dict(status='prepared_no_model_calls',scenes=len(records),observed=sum(s['observed'] for s in records),hypotheses=hypotheses)


def check(root):
    suite=read(root/'suite.json');model.require(suite['kind']==model.VERSION,'Wrong run type')
    model.require(suite['code_sha256']==code_hashes(),'Source changed; create a new output root')
    for p,h in {**suite['bindings'],suite['source_suite']:suite['source_sha256']}.items():
        model.require(sha(p)==h,'Frozen input changed: '+p)
    return suite


class Requests:
    """Two syntax/contract attempts, stable scene/stage seeds, failed calls not redrawn."""
    def __init__(self, get_client, folder, seed, image):
        self.get_client=get_client;self.folder=folder;self.seed=seed;self.image=image
        folder.mkdir(parents=True,exist_ok=True);self.trace=[]

    def ask(self, name, prompt, validator, max_tokens=8192):
        model.identifier(name)
        digest=hashlib.sha256(prompt.encode()).hexdigest(); cache=self.folder/(name+'.accepted.json')
        seed=stage_seed(self.seed,name)
        if cache.exists():
            old=read(cache);model.require(old['prompt_sha256']==digest and old['seed']==seed,'Cached request differs')
            for p,h in old['artifacts'].items():model.require(sha(p)==h,'Saved response changed: '+p)
            result=validator(old['raw_value']);self.trace.append(old['trace']);return result
        failed=self.folder/(name+'.failed.json')
        if failed.exists():raise ValueError('Recorded failed request, not resampled: '+str(failed))
        original=prompt
        for attempt in range(2):
            path=self.folder/(name+f'_attempt{attempt}.txt')
            print(json.dumps(dict(stage=name,attempt=attempt,seed=seed+attempt)),flush=True)
            write(self.folder/(name+f'_attempt{attempt}.input.json'),dict(prompt=prompt,seed=seed+attempt))
            try:
                raw=self.get_client().ask(self.image,prompt,path,seed=seed+attempt,max_tokens=max_tokens)
                result=validator(raw)
                trace=dict(stage=name,seed=seed,accepted_attempt=attempt,prompt_sha256=digest,
                           structural_correction=bool(attempt),probability_calibrated=False)
                artifacts={str(p):sha(p) for p in self.folder.glob(name+f'_attempt{attempt}.*')}
                write(cache,dict(prompt_sha256=digest,seed=seed,raw_value=raw,trace=trace,artifacts=artifacts))
                self.trace.append(trace);return result
            except (ValueError,KeyError,TypeError,IndexError) as exc:
                write(self.folder/(name+f'_attempt{attempt}.error.json'),dict(error=str(exc)))
                if attempt:
                    write(failed,dict(error=str(exc),attempts=2));raise ValueError(name+': '+str(exc)) from exc
                previous=path.read_text() if path.exists() else json.dumps(locals().get('raw',{}),ensure_ascii=False)
                prompt=original+'\nCorrect this JSON/contract error, preserving the intended world; no retries to maximize object count. ERROR: '+str(exc)+'\nPREVIOUS ANSWER: '+previous
        raise AssertionError('unreachable')


def compile_boxes(case, observed, plan, objects, poses):
    """Only deterministic support-height construction. No search, deletion, or relocation."""
    by_id={o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']}
    raw={r['id']:r for r in poses};items=[];additions=[];warnings=[]
    model.require(set(raw)=={o['id'] for o in objects} and len(raw)==len(poses),'Pose inventory mismatch')
    for o in model.ordered(objects,by_id):
        r=raw[o['id']];c=list(r['center_m']);size=r['full_size_m'];a=o['support'];host=a['host']
        theta=math.radians(r['yaw_deg']);co,si=abs(math.cos(theta)),abs(math.sin(theta))
        box=[co*size[0]+si*size[2],size[1],si*size[0]+co*size[2]]
        if a['kind']=='floor':c[1]=plan['room_envelope']['floor_y_m']+size[1]/2
        elif a['kind']=='top':c[1]=by_id[host]['center'][1]+by_id[host]['size'][1]/2+size[1]/2
        if host:
            parent=by_id[host]
            axes=(0,2) if a['kind']=='top' else (0,1,2)
            if any(abs(c[k]-parent['center'][k])+box[k]/2 > parent['size'][k]/2+.05 for k in axes):
                warnings.append(dict(id=o['id'],kind='support_extent_needs_review',host=host))
        if a['kind'] in ('wall','ceiling','free'):
            warnings.append(dict(id=o['id'],kind='unverified_'+a['kind']+'_support'))
        b=plan['room_envelope']['bounds_xz_m']
        if c[0]-box[0]/2<b[0] or c[0]+box[0]/2>b[1] or c[2]-box[2]/2<b[2] or c[2]+box[2]/2>b[3]:
            warnings.append(dict(id=o['id'],kind='outside_rough_envelope'))
        z=next(z for z in plan['zones'] if z['id']==o['zone_id'])['bounds_xz_m']
        if not (z[0]<=c[0]<=z[1] and z[2]<=c[2]<=z[3]):warnings.append(dict(id=o['id'],kind='outside_planned_zone'))
        item=dict(id=o['id'],center=c,size=box);items.append(item);by_id[o['id']]=item
        audit=taxonomy_review(o['category']);cid=audit['dataset_category_id']
        if cid not in ELIGIBLE:cid=None
        additions.append(dict(track_id=o['id'],category_id=field(cid,'hypothesis','taxonomy_bridge',reason='unmapped_or_structural_category'),
            raw_category=o['category'],description=o['description'],center_local_m=field(c,'hypothesis','rough_plan_layout'),
            size_aabb_local_m=field(box,'hypothesis','yaw_enclosing_AABB'),bbox2d_visible_xyxy_norm=field(None,reason='unobserved'),
            in_primary_room=field(None,reason='room_is_hypothesized'),local_evidence='unobserved',
            support=deepcopy(a),zone_id=o['zone_id'],intrinsic_size_m=size,yaw_deg=r['yaw_deg'],taxonomy_review=audit))
    numeric=review_boxes(items,observed,case['hfov'],case['image_size'])
    conflicts=[r['id'] for r in numeric['objects'] if r['intersects_input_box_proxy'] or r['camera_inside_box_proxy']]
    numeric.update(observation_conflict_ids=conflicts,layout_warnings=warnings,
        geometry_policy='diagnose_only_no_posthoc_deletion_or_position_search',
        observed_objects_unchanged=True,RGB_consistency_verified=False)
    full=deepcopy(observed);full.update(scope='full',objects=deepcopy(observed['objects'])+additions,
        model_variant=model.VERSION,room_hypothesis=deepcopy(plan['room_envelope']),
        functional_zones=deepcopy(plan['zones']),world_state_ready=False,
        world_state_ready_scope='unreviewed_rough_world_hypothesis',observation_compatible_box_proxy=not conflicts,
        semantics_verified=False,geometry_verified=False)
    validate_graph(full)
    model.require(full['objects'][:len(observed['objects'])]==observed['objects'],'Observed state changed')
    return full,numeric


def causal_trace(h, plan, objects, requests, observed_ids):
    return dict(kind='assumed_hierarchical_LM_mechanisms',hypothesis=h,
        equations=['H_candidates=f_H(O,U_H); H=H_candidates[J] with enumerated J', 'Z=f_Z(O,H,U_Z)',
                   'I_k=f_I(O,H,Z,I_<k,U_Ik)', 'X=f_X(O,Z,I,U_X)',
                   'E_i=membership(i,I)', 'S_i=E_i AND S_host(i)'],
        conditioning='O is fixed image plus predicted observed instances; conditioning is not do(O)',
        inventory_order=[z['id'] for z in plan['zones']],
        spatial_relations_are_causal_edges=False,real_world_causal_identification=False,
        probabilities_calibrated=False,hypotheses_enumerated_not_posterior_ranked=True,
        mechanism_calls=requests.trace,installation=model.installation(objects,observed_ids))


def run(root, layout=True, client_factory=provider):
    suite=check(root);client=None
    def get_client():
        nonlocal client
        if client is None:client=client_factory(suite,'qwen')
        return client
    for scene in suite['scenes']:
        base=root/'scenes'/scene['id'];obsroot=base/'observation';case=load_case(obsroot,'qwen');observed=read(case['state'])
        boxes={o['track_id']:o for o in observed['objects']}
        for row in case['context']:
            b=boxes[row['id']];row.update(center=b['center_local_m']['value'],full_aabb_size=b['size_aabb_local_m']['value'])
        with Image.open(case['image']) as im:image=im.convert('RGB').copy()
        seed=stage_seed(suite['seed'],scene['id'])
        requests=Requests(get_client,base/'calls',seed,image)
        try:
            hs=requests.ask('hypotheses',model.hypotheses_prompt(case,suite['hypotheses']),
                lambda v:model.hypotheses_answer(v,case,suite['hypotheses']))
            write(base/'hypotheses.json',hs);write(base/'scene_status.json',dict(status='hypotheses_ready'))
        except Exception as exc:
            write(base/'scene_status.json',dict(status='failed',stage='hypotheses',error=str(exc)))
            (base/'error.txt').write_text(traceback.format_exc());visual.report(root,suite);continue
        for i,h in enumerate(hs):
            folder=base/f'world_{i+1:02d}';folder.mkdir(exist_ok=True);plan=None;objects=[]
            req=Requests(get_client,folder/'calls',stage_seed(seed,h['id']),image)
            stage='blueprint'
            try:
                write(folder/'hypothesis.json',h)
                plan=req.ask('blueprint',model.blueprint_prompt(case,h),lambda v:model.blueprint_answer(v,case))
                write(folder/'blueprint.json',plan)
                write(folder/'status.json',dict(status='planning_in_progress',stage='inventory'))
                visual.page(folder,h,plan,observed,case['hfov'],objects);visual.report(root,suite)
                for zone in plan['zones']:
                    stage='inventory_'+zone['id']
                    additions=req.ask(stage,model.inventory_prompt(case,h,plan,zone,objects),
                        lambda v,z=zone:model.inventory_answer(v,case,z,objects))
                    objects.extend(additions)
                    write(folder/'inventory_partial.json',dict(objects=objects,completed_zone=zone['id']))
                write(folder/'inventory.json',dict(objects=objects,all_zones_completed=True))
                stage='semantic_review'
                audit=req.ask(stage,model.review_prompt(case,h,plan,objects),lambda v:model.review_answer(v,plan,objects,case))
                write(folder/'semantic_review.json',audit)
                ids=set(boxes);hosts=sorted({o['support']['host'] for o in objects if o['support']['host']})
                write(folder/'installation_interventions.json',dict(baseline=model.installation(objects,ids),
                    interventions={host:model.installation(objects,ids,[host]) for host in hosts}))
                write(folder/'causal_trace.json',causal_trace(h,plan,objects,req,ids))
                write(folder/'status.json',dict(status='plan_complete',observed=len(boxes),new_objects=len(objects),
                    zones=len(plan['zones']),semantic_verdict=audit['plausibility'],metric_layout='not_requested_or_pending'))
                visual.page(folder,h,plan,observed,case['hfov'],objects);visual.report(root,suite)
                if layout:
                    stage='layout';poses=[];ordered=model.ordered(objects,ids)
                    for j in range(0,len(ordered),6):
                        batch=ordered[j:j+6]
                        poses.extend(req.ask('metric_'+str(j//6),model.metric_prompt(case,plan,objects,batch,poses),
                            lambda v,b=batch:model.metric_answer(v,b),4096))
                    write(folder/'poses.json',poses)
                    full,numeric=compile_boxes(case,observed,plan,objects,poses)
                    write(folder/'numeric_review.json',numeric);write(folder/'G_full_candidate.json',full)
                    if not numeric['observation_conflict_ids']:write(folder/'G_full.json',full)
                    stage='visualization';checks=render_world(full,image,case['hfov'],folder/'views')
                    model.require(checks['labels']==len(full['objects']),'Visualization dropped instances')
                    write(folder/'causal_trace.json',causal_trace(h,plan,objects,req,ids))
                    write(folder/'status.json',dict(status='layout_complete',observed=len(boxes),new_objects=len(objects),
                        zones=len(plan['zones']),semantic_verdict=audit['plausibility'],
                        observation_conflict_ids=numeric['observation_conflict_ids'],
                        exported_observation_compatible_candidate=not numeric['observation_conflict_ids'],
                        geometry_warnings=numeric['objects_with_numeric_warnings']+len(numeric['layout_warnings']),
                        quality='unverified_world_hypothesis',API_used=False))
            except Exception as exc:
                write(folder/'status.json',dict(status='failed',stage=stage,error=type(exc).__name__+': '+str(exc),
                    plan_saved=plan is not None,partial_inventory_count=len(objects)))
                (folder/'error.txt').write_text(traceback.format_exc())
            visual.page(folder,h,plan,observed,case['hfov'],objects);visual.report(root,suite)
        check(root)
    return visual.report(root,suite)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','plan','run','report','check'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--observations-from',type=Path)
    p.add_argument('--hypotheses',type=int,default=3);p.add_argument('--seed',type=int,default=3407)
    a=p.parse_args();a.root=a.root.resolve()
    if a.action=='prepare':
        if a.observations_from is None:p.error('prepare requires --observations-from')
        result=prepare(a.root,a.observations_from,a.hypotheses,a.seed)
    elif a.action in ('run','plan'):result=run(a.root,layout=a.action=='run')
    else:
        suite=check(a.root);result=visual.report(a.root,suite) if a.action=='report' else dict(status='verified_no_model_calls')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False),flush=True)
    return 2 if result.get('execution_failures',0) else 0


if __name__=='__main__':sys.exit(main())
