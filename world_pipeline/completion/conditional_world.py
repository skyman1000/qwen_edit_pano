"""Training-free world completion v2: joint visual conditioning and support.

No new training/GPU submission. User runs actual inference. v1 kept unchanged.
Relative catalog weights and assumption sensitivity are not physical causal effects.
"""
import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path
import shutil
import traceback

from PIL import Image

from ..common import new_output,read,sha,validate_graph,write
from .batch_world_review import inventory_image
from .causal_world_completion import check_hashes,visible_inventory
from .conditional_world_core import (VisualMechanismScorer,conditional_model,contexts_answer,
                                     inventory_key,worlds_answer)
from .conditional_world_geometry import compile_layout,support_plane
from .frozen_causal_world import ask_validated,load_case,write_page
from .off_fov_completion import validate_world
from .qwen_upstream import DEFAULT_MODEL,Qwen
from .scene_completion import export_states,map_plan
from .semantic_layout_visual import render

PROTOCOL='conditional_mechanism_off_fov_world_v2'


def code_hashes():
    from .frozen_causal_world import code_hashes as original_hashes
    files=[Path(__file__),*[Path(__file__).with_name(n) for n in
        ('conditional_world_core.py','conditional_world_geometry.py','conditional_world_audit.py')]]
    return {**original_hashes(),**{str(p.resolve()):sha(p) for p in files}}


def context_prompt(case,budget):
    return ('Infer plausible EXISTING scene-configuration explanations from this partial indoor RGB. '
        'The marked image shows observed instance IDs, not new objects. Return JSON '
        '{"contexts":[{"description":...,"reason":...,"evidence_ids":[...]}],"floor_anchor_ids":[...]}. '
        f'Offer up to {budget} plausible alternative configurations/uses when ambiguous, not just a '
        'paragraph of visual style. Do not invent household members, pets or activities as facts. '
        'A configuration is an explanatory hypothesis, not a room redesign, necessity list, or an '
        'instruction to convert the room. Do not enumerate hidden objects yet. Evidence_ids refer to '
        'actual observed instances. floor_anchor_ids should identify observed furniture plausibly '
        'standing on the same floor, excluding cushions, trays and supported accessories; [] is '
        'allowed if no support can be inferred. Their predicted bases will define an explicitly '
        'approximate camera-Y-level support plane, not measured gravity or a true floor. '
        'Use short descriptions with meaningful alternative uses; do not treat simplicity as a '
        'requirement for an almost empty hidden world.\nOBSERVED: '+json.dumps(visible_inventory(case),ensure_ascii=False))


def worlds_prompt(case,context,plane,budget):
    return ('Infer complete, plausible existing OFF-CAMERA object inventories under the configuration '
        'hypothesis below. Predict what might already surround this SAME camera, not what would be '
        'useful to add to visible furniture. Fixed observations remain unchanged; the target is '
        'outside the INPUT viewing cone, not occluded additions inside it. New instances may share '
        'observed categories but must be distinct. Optional ordinary objects are allowed; no minimal '
        'function checklist, object-count quota or direction quota. Do not fabricate pets/children '
        'as an observed fact. Preserve uncertainty about room extent. '
        'Return ONLY {"worlds":[{"description":...,"additions":[{"id":"n1","name":...,'
        '"object_type":...,"reason":...,"placement":...,"support":{"kind":"floor",'
        '"target":null}}]}]}. '
        f'Generate up to {budget} different full inventory hypotheses, not fragments to concatenate. '
        'Reason must link the hidden-world hypothesis to current evidence and state uncertainty; '
        'saying only that an addition enhances comfort/design is not evidence it exists. Placement '
        'describes unphotographed space around the camera, distinct from camera-z behind a visible '
        'sofa. Support kinds: floor (shared approximate plane), object (target observed/new instance ID), '
        'unspecified (wall/hanging/unknown support, geometry later predicts signed camera Y). '
        'Use structured support IDs; check the exact named object behind each ID. A chair used at a '
        'desk is not supported on its top; object support describes actual load-bearing placement. '
        'No free relation list or numeric xyz/dimensions in this stage. Do not duplicate objects '
        'already visible in RGB even if the detector omitted them. '
        '\nCONFIGURATION HYPOTHESIS: '+json.dumps(context,ensure_ascii=False)+
        '\nFIXED OBSERVED: '+json.dumps(visible_inventory(case),ensure_ascii=False)+
        '\nAPPROXIMATE SUPPORT PLANE: '+json.dumps(plane,ensure_ascii=False)+
        f'\nInput horizontal FOV={case["hfov"]:g} degrees; behind-camera space is outside this view.')


def review_prompt(case,context,plane,proposals,budget):
    return (worlds_prompt(case,context,plane,budget)+
        '\nREVIEW the proposals below against the marked RGB and exact observed ID names. '
        'Return complete revised worlds in the SAME schema. Check: does it describe unseen surroundings '
        'rather than decoration of visible surfaces; is a suggested item already visible; are declared '
        'supports physically meaningful; are unknown room properties still hypotheses? Retain ordinary '
        'optional objects when plausible; do not reduce everything to minimum required equipment. '
        'Revise a bad support/placement before withdrawing an otherwise plausible object. This is a '
        'model critique, not independent semantic certification. Keep all original proposals in logs. '
        '\nPROPOSALS TO REVIEW: '+json.dumps(proposals,ensure_ascii=False))


def bearing_prompt(case,plan,plane):
    return ('Place exactly the proposed new OFF-CAMERA instances. Return ONLY '
        '{"objects":[{"id":...,"azimuth_deg":...,"range_m":...,"size":[sx,sy,sz],'
        '"center_y_m":...,"reason":...}]}. Azimuth is measured around the ORIGINAL camera: '
        '0=optical forward, +90=right, -90=left, +/-180=behind. range_m is horizontal sqrt(x*x+z*z), '
        'not optical z or Euclidean 3D range; range may be zero directly above/below the camera. '
        'size is FULL AABB lengths in camera x,y,z, metres. '
        'The program computes x=range*sin(azimuth), z=range*cos(azimuth). It computes camera-Y from '
        'the declared shared floor/top support; center_y_m is used ONLY for unspecified support. '
        'Positive camera-Y is above the camera, not height above floor. Do not output free xyz. '
        'Choose natural off-camera bearings and metric sizes, not positions in the observed seating '
        'area or just outside an image edge. All full boxes must miss the input cone. There is no '
        'compulsory left/right/back coverage. No program will teleport a bad location to the cone '
        'boundary, change sizes or delete objects to pass checks. Use the shared support hypothesis '
        'and ensure any object-supported footprint actually overlaps its named support. '
        'Floor/room extent/gravity are not measured. Explain significant geometric assumptions. '
        '\nPLAN: '+json.dumps(plan,ensure_ascii=False)+'\nSUPPORT PRIOR: '+json.dumps(plane,ensure_ascii=False)+
        '\nFIXED OBSERVED GEOMETRY: '+json.dumps(case['context'],ensure_ascii=False)+
        f'\nCamera centred pinhole HFOV={case["hfov"]:g}, original image width/height={case["image_size"]}.')


def diagram(spec,trace,path):
    body='<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="300"><rect width="100%" height="100%" fill="white"/>'
    for x,y,label in ((20,25,'O: RGB + fixed G_obs (common conditioning)'),
                      (35,95,'Z: configuration hypothesis'),(520,95,'H: full hidden inventory'),
                      (520,175,'Support DAG + bearing/range → full AABBs')):
        width=430 if y==95 and x==35 else 520
        body+=f'<rect x="{x}" y="{y}" width="{width}" height="42" fill="#e1edf6"/><text x="{x+8}" y="{y+27}">{escape(label)}</text>'
    body+='<path d="M295 67 L295 95 M295 67 L780 95 M555 116 L520 116 M780 137 L780 175" stroke="black" fill="none"/>'
    body+='<text x="20" y="270">Conditional mechanism hypothesis; sensitivity is not an identified physical causal effect.</text></svg>'
    path.write_text(body)


def run(model,scorer_factory,case,output,context_budget=3,per_context=2,worlds=3,
        capacity=256,seed=3407,max_tokens=6144):
    folder=output/case['id'];folder.mkdir();hypotheses=[]
    record=dict(id=case['id'],protocol=PROTOCOL,status='failed',world_state_ready=False,
        observed_instances=len(case['context']),semantics_verified=False,geometry_verified=False,
        causal_gain_validated=False,scene_gt_used=False,adapter_connected=False,panorama_generated=False,
        training_performed=False)
    options=dict(seed=seed,sample=True,profile='official',max_tokens=max_tokens,guard_repetition=True)
    try:
        check_hashes(case['source_sha256']);observed=validate_graph(read(case['state']))
        write(folder/'input.json',case);write(folder/'G_obs.json',observed)
        shutil.copyfile(case['image'],folder/'local.png')
        inventory_image(case['image'],read(case['inventory']),folder/'qwen_2d.png')
        with Image.open(case['image']) as f: original=f.convert('RGB')
        with Image.open(folder/'qwen_2d.png') as f: marked=f.convert('RGB')
        marked.thumbnail((1024,1024))
        context=ask_validated(model,marked,context_prompt(case,context_budget),folder/'contexts.txt',
            lambda v:contexts_answer(v,case,context_budget),options)
        plane=support_plane(observed,context['floor_anchor_ids'])
        write(folder/'contexts_normalized.json',context);write(folder/'support_plane.json',plane)
        catalog=[];duplicates=[];seen={}
        for ci,c in enumerate(context['contexts']):
            path=folder/('contents_'+c['id']+'.txt')
            raw=ask_validated(model,marked,worlds_prompt(case,c,plane,per_context),path,
                lambda v:worlds_answer(v,case,c['id'],per_context,capacity),
                {**options,'seed':seed+10+ci*10})
            revised=ask_validated(model,marked,review_prompt(case,c,plane,raw,per_context),
                folder/('review_'+c['id']+'.txt'),
                lambda v:worlds_answer(v,case,c['id'],per_context,capacity),
                {**options,'seed':seed+12+ci*10})
            write(folder/('reviewed_'+c['id']+'.json'),revised)
            for w in revised:
                key=inventory_key(w)
                if key in seen:
                    duplicates.append(dict(duplicate=w,represented_by=seen[key],reason='same normalized noun/count multiset; retained in provenance'))
                else:
                    seen[key]=w['id'];catalog.append(w)
        if () not in seen:
            catalog.append(dict(id='empty',scene_type='hypothesized indoor scene',
                scene_hypothesis='No additional distinct off-camera objects.',additions=[],relations=[],
                placement_dependency_order=[],proposal_context_id=None))
        write(folder/'world_catalog.json',dict(worlds=catalog,duplicates=duplicates,
            finite_domain_is_exhaustive=False,coverage_probability=None))
        mechanisms=folder/'mechanisms';mechanisms.mkdir()
        scorer=scorer_factory(marked,visible_inventory(case))
        def save(name,value):
            write(mechanisms/(name+'.json'),value)
            print(json.dumps(dict(stage='conditional_mechanism_scored',name=name,
                permutation_tv=value.get('permutation_max_total_variation'),weights=value['probabilities']),ensure_ascii=False),flush=True)
        spec,trace=conditional_model(context['contexts'],catalog,scorer,save)
        write(folder/'conditional_model.json',spec);write(folder/'inference_trace.json',trace)
        diagram(spec,trace,folder/'causal_graph.svg')
        weights=trace['world_mixture']['probabilities']
        selected=sorted(catalog,key=lambda w:(-weights[w['id']],w['id']))[:worlds]
        write(folder/'selection.json',dict(selected=[w['id'] for w in selected],weights=weights,
            omitted_catalog_weight=max(0.,1-sum(weights[w['id']] for w in selected)),
            selection_scope='catalog_relative_weights_not_coverage_probability'))
        for rank,plan in enumerate(selected):
            dest=folder/f'hypothesis_{rank+1:02d}';dest.mkdir()
            h=dict(id=plan['id'],folder=dest.name,status='failed',world_state_ready=False,
                   proposed_instances=len(plan['additions']),relative_catalog_weight=weights[plan['id']])
            try:
                write(dest/'semantic_plan.json',plan);mappings=map_plan(plan,'semantic_lexicon_v1')
                write(dest/'category_mapping.json',mappings)
                base=bearing_prompt(case,plan,plane);sizes=None;attempts=[];last=None;selected_attempt=None
                for attempt in range(2):
                    prompt=base
                    if last is not None:
                        prompt+='\nGEOMETRIC REPLAN: correct bearing/range of ALL fixed objects. Do not resize/delete them. '+json.dumps(
                            dict(fixed_sizes=sizes,previous_geometry=last[0],computed_checks=last[1]),ensure_ascii=False)
                    try:
                        if plan['additions']:
                            result=ask_validated(model,marked,prompt,dest/(f'bearing_{attempt}.txt'),
                                lambda v:compile_layout(v,plan,mappings,observed,plane,case['hfov'],case['image_size'],sizes),
                                {**options,'seed':seed+100+rank*10+attempt*2})
                        else:
                            result=compile_layout({'objects':[]},plan,mappings,observed,plane,case['hfov'],case['image_size'])
                    except ValueError as exc:
                        if last is None:raise
                        attempts.append(dict(attempt=attempt,schema_valid=False,error=str(exc)))
                        break
                    items,review=result;sizes={o['id']:o['size'] for o in items};last=result
                    selected_attempt=attempt
                    write(dest/(f'geometry_{attempt}.json'),dict(objects=items,review=review))
                    attempts.append(dict(attempt=attempt,contract_passed=review['geometry_contract_passed']))
                    if review['geometry_contract_passed']:break
                items,review=last
                full,_,unmapped=export_states(observed,plan,items,mappings,review)
                full.update(completion_protocol=PROTOCOL,adapter_connected=False,target='outside_input_frustum',
                    relative_catalog_weight=weights[plan['id']],semantic_plan_fully_exported=True,
                    input_camera=dict(model='centred_pinhole',hfov_deg=case['hfov'],image_size_wh=case['image_size']),
                    declared_support_plane=plane,world_relations=plan['relations'],causal_model_identified=False)
                for obj,item in zip(full['objects'][len(observed['objects']):],items):
                    obj.update(declared_support=item['declared_support'],bearing_deg=item['bearing_proposal_deg'],
                               horizontal_range_m=item['horizontal_range_m'])
                    obj['center_local_m'].update(source='conditional_geometric_hypothesis',method=item['center_method'])
                validate_world(observed,full,items,capacity)
                h.update(status='candidate_needs_semantic_review' if review['geometry_contract_passed'] else 'geometry_needs_review',
                    world_state_ready=review['geometry_contract_passed'],new_instances=len(items),
                    off_fov_passed=review['numeric_constraints_passed'],support_proxy_failures=review['support_proxy_failures'],
                    unverified_relations=review['unverified_relations'],unmapped_new=len(unmapped),
                    observed_state_preserved=True,position_was_projected=False)
                if not items:h['status']='empty_hidden_hypothesis_needs_review'
                full['quality_status']=h['status'];full['world_state_ready']=h['world_state_ready']
                write(dest/'G_full.json',full)
                hidden=deepcopy(full);hidden['scope']='hidden';hidden['objects']=full['objects'][len(observed['objects']):]
                validate_graph(hidden);write(dest/'G_out.json',hidden)
                write(dest/'layout_solved.json',dict(objects=items));write(dest/'numeric_review.json',review)
                write(dest/'layout_attempts.json',dict(attempts=attempts,selected_attempt=selected_attempt,
                    selection='last_complete_valid_schema_attempt_no_partial_object_selection'))
                visual=dest/'world';visual.mkdir()
                render(full,original,case['hfov'],visual,h,position_note='方位/距离+显式支撑假设；无视锥边界投影。地板和重力未测量。')
                page=visual/'index.html';page.write_text(page.read_text().replace(
                    '红色候选保留在G_full_candidates中，不进入Adapter子集G_full。','未映射对象仍全部保留在G_full；本阶段不连接Adapter。'))
            except Exception as exc:
                h.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                (dest/'error_traceback.txt').write_text(traceback.format_exc())
            write(dest/'summary.json',h);hypotheses.append(h)
        ready=[h for h in hypotheses if h['world_state_ready']]
        failed=[h for h in hypotheses if h['status']=='failed']
        record.update(status='partial_failure' if failed else 'candidates_exported_needs_review',
            world_state_ready=bool(ready),hypotheses=hypotheses,hypotheses_exported=len(hypotheses)-len(failed),
            nonempty_hypotheses=sum(h.get('new_instances',0)>0 for h in hypotheses),
            observed_state_preserved=True,probability_scope='relative_weights_in_nonexhaustive_catalog',
            geometry_unresolved=sum(h['status']=='geometry_needs_review' for h in hypotheses),
            permutation_diagnostics=dict(context=trace['context_permutation_tv'],inventory=trace['inventory_permutation_tv']))
        if hypotheses and hypotheses[0]['world_state_ready']:
            for name in ('G_full.json','G_out.json'):shutil.copyfile(folder/hypotheses[0]['folder']/name,folder/name)
            record['preferred_world']=hypotheses[0]['id']
        check_hashes(case['source_sha256'])
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',world_state_ready=False)
        (folder/'error_traceback.txt').write_text(traceback.format_exc())
    write(folder/'summary.json',record);write_page(folder,record,hypotheses)
    write(output/'summary.json',dict(protocol=PROTOCOL,records=[record],scene_gt_used=False))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><a href="'+case['id']+'/index.html">打开条件机制世界补全</a>')
    print(json.dumps(record,ensure_ascii=False),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',type=Path)
    p.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--contexts',type=int,default=3);p.add_argument('--per-context',type=int,default=2)
    p.add_argument('--worlds',type=int,default=3);p.add_argument('--max-tokens',type=int,default=6144)
    p.add_argument('--max-objects',type=int,default=256);p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if (a.output.exists() or not 1<=a.contexts<=4 or not 1<=a.per_context<=3 or
        not 1<=a.worlds<=a.contexts*a.per_context+1 or not 256<=a.max_tokens<=16384 or
        not 1<=a.max_objects<=256 or not 0<=a.seed<2**63):p.error('Need NEW output and valid inference budgets')
    case=load_case(a.case)
    if len(case['context'])>a.max_objects:p.error('Observed state exceeds capacity')
    if not a.model.is_dir():p.error('Local model missing; no automatic downloads')
    if a.preflight:
        print(json.dumps(dict(status='preflight_passed',protocol=PROTOCOL,id=case['id'],observed=len(case['context']),
            gpu_started=False,model_loaded=False,scene_gt_used=False,adapter_connected=False),ensure_ascii=False));return
    output=new_output(a.output);hashes=code_hashes()
    write(output/'config.json',dict(protocol=PROTOCOL,case=case,code_sha256=hashes,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}))
    try:model=Qwen(a.model)
    except Exception as exc:
        write(output/'summary.json',dict(status='model_load_failed',error=f'{type(exc).__name__}: {exc}'));raise
    if code_hashes()!=hashes:raise ValueError('Code changed while loading model')
    record=run(model,lambda image,obs:VisualMechanismScorer(model,image,obs),case,output,
        a.contexts,a.per_context,a.worlds,a.max_objects,a.seed,a.max_tokens)
    if record['status'] in ('failed','partial_failure'):raise SystemExit('Inspect summary/raw replies; all original candidates preserved.')


if __name__=='__main__':main()
