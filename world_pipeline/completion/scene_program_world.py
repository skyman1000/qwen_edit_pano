"""Manual-run conditional joint scene programs -> frozen G_obs + outside world.

No training, GT scene, Adapter, model download, or implicit GPU submission.
An inverse visual configuration distribution and parent-conditioned finite group
mechanisms define a conditional program model. Geometry is deterministic and
approximate; this module does NOT claim an exact generative RGB posterior.
"""
import argparse
from copy import deepcopy
from html import escape
import json
import math
from pathlib import Path
import shutil
import traceback

from PIL import Image,ImageDraw

from ..common import new_output,read,sha,validate_graph,validate_new_output,write
from .causal_world_completion import check_hashes,visible_inventory
from .conditional_world_geometry import compile_layout,support_plane
from .frozen_causal_world import ask_validated,load_case
from .off_fov_completion import validate_world
from .qwen_upstream import DEFAULT_MODEL,Qwen
from .scene_completion import export_states,map_plan
from .scene_program import (PROTOCOL,enumerate_assignments,instantiate,parameterize,
                            program_answer,registry_answer,sample_modes)
from .scene_program_geometry import compile_program
from .scene_program_scores import GroupMechanismScorer,finite_scores,verify_scores
from .semantic_layout_visual import render


def code_hashes():
    # Reused implementations are part of the source contract as well.
    from .scm_world import code_hashes as previous
    return {**previous(),**{str(p.resolve()):sha(p) for p in Path(__file__).parent.glob('scene_program*.py')}}


def observation_board(case,observed,path):
    """One global image and ID-bound object crops, not a new Observer prediction."""
    with Image.open(case['image']) as handle: original=handle.convert('RGB')
    n=len(observed['objects']); cols=3; rows=max(1,math.ceil(n/cols)); cell_w=192; cell_h=192
    height=max(800,rows*cell_h); left=768
    canvas=Image.new('RGB',(left+cols*cell_w,height),'white'); draw=ImageDraw.Draw(canvas)
    panel=original.copy();panel.thumbnail((left,740));canvas.paste(panel,((left-panel.width)//2,28))
    draw.text((8,8),'ORIGINAL INPUT; cropped IDs at right; no new detection',fill='black')
    for index,obj in enumerate(observed['objects']):
        box=obj['bbox2d_visible_xyxy_norm']['value']; x=left+(index%cols)*cell_w;y=(index//cols)*cell_h
        draw.text((x+3,y+3),obj['track_id'],fill='black')
        if box is None:
            draw.text((x+3,y+30),'No 2D crop available',fill='black');continue
        w,h=original.size; pixel=(int(box[0]*w),int(box[1]*h),max(int(box[2]*w),int(box[0]*w)+1),max(int(box[3]*h),int(box[1]*h)+1))
        crop=original.crop(pixel);crop.thumbnail((cell_w-8,cell_h-45));canvas.paste(crop,(x+4,y+28))
    canvas.save(path)
    return original,canvas


def registry_prompt(case,budgets):
    return ('Describe the fixed observations and propose alternative EXISTING surrounding-scene configurations. '
        'This is inference from a partial indoor photograph, not a request to design a minimal functional room. '
        'Return {"instances":[{"id":string,"variant":specific noun,"affordances":["top","seat"],"on_floor":boolean}],'
        '"configurations":[{"description":string,"explanation":string,"evidence_ids":[observed IDs]}]}. '
        'Use every supplied observation ID exactly once. The attached board binds crops to IDs. '
        'affordances are usable support capabilities; [] is valid, an ottoman can have both, a lamp shade is not a tabletop. '
        'on_floor means the whole object normally rests directly on the floor, excluding supported pillows and trays. '
        f'Propose 1..{budgets["configurations"]} distinguishable plausible configurations of the surrounding world, '
        'including mixed use when supported. Configurations are hypotheses, not discovered facts. '
        'Do not output hidden objects yet. Observations and their geometry will remain unchanged.\nOBSERVED: '+
        json.dumps(visible_inventory(case),ensure_ascii=False))


def program_prompt(case,registry,config,budgets):
    return ('Specify a small executable family of plausible EXISTING off-camera worlds under the configuration below. '
        'Represent optional surrounding furnishings/personal storage/decor as JOINT alternatives of functional/spatial groups. '
        'Preserve the pictured world; include plausible context beyond essential equipment. Do not furnish an ideal/minimal room. '
        'Existing observations may already satisfy roles and must never be duplicated as new objects. '
        'The program will add a zero-addition alternative to each group; do not output empty alternatives. '
        f'Use 1..{budgets["groups"]} groups and 1..{budgets["modes"]} nonempty alternatives per group; '
        f'the largest combined alternative may contain at most {budgets["max_new"]} new instances (computation budget, no minimum). '
        'Different groups should not propose duplicates of the same intended instance. Group parents are optional zero-based indices '
        'of EARLIER groups whose selected alternative changes this group, not arbitrary graph decoration. '
        'Each alternative jointly specifies a coherent set of object instances, their concrete variants, full sizes, supports and relative poses. '
        'Objects in one alternative may support/reference each other. External references may only name supplied observed IDs or camera. '
        'Return JSON {"groups":[{"name":string,"explanation":evidence and plausible hidden-context rationale,'
        '"parents":[earlier group indices],"modes":[{"description":string,"objects":[{'
        '"key":short_identifier,"name":string,"category":ordinary object noun,"variant":specific variant,'
        '"size_m":[sx,sy,sz],"anchor":camera_or_observed_ID_or_local_key,"offset_m":[dx,dy,dz],'
        '"support":{"kind":floor_or_object_or_wall_or_ceiling_or_unspecified,"target":null_or_ID_or_local_key}'
        '}]}]}]}. '
        'All offsets and FULL AABB dimensions use the ORIGINAL CAMERA AXES: x right, y image-up, z forward, metres. '
        'Negative z is behind the camera. offset_m is relative to the named anchor centre, not an absolute pose unless anchor=camera. '
        'For object support, anchor must equal its target; the program sets Y to the host AABB top and checks central XZ contact. '
        'For floor support, Y comes from a declared camera-Y level prior when available; otherwise the explicit Y proposal remains unresolved. '
        'A floor/table/wall/pendant lamp are distinct variants with correspondingly different geometry and support. '
        'Plan around the ORIGINAL input view, including plausible side/back space; all NEW complete boxes must be outside it. '
        'Do not invent measured room bounds, furniture front orientation, gravity, or unseen walls. '
        'A joint region/layout compiler can adjust positions, preserving relative organisation and fixed sizes. '
        '\nCONFIGURATION: '+json.dumps(config,ensure_ascii=False)+
        '\nOBSERVED SEMANTIC CONTEXT: '+json.dumps(registry['instances'],ensure_ascii=False)+
        '\nFROZEN OBSERVED GEOMETRY: '+json.dumps(visible_inventory(case),ensure_ascii=False)+
        f'\nCAMERA HFOV={case["hfov"]}; IMAGE SIZE={case["image_size"]}.')


def solve_assignment(case,observed,plane,program,modes,search):
    plan,bodies,groups,placements,unresolved=instantiate(program,modes,case,observed,plane)
    raw,solver=compile_program(case,observed,plan,bodies,groups,placements,**search)
    mappings=map_plan(plan,'semantic_lexicon_v1')
    items,review=compile_layout(raw,plan,mappings,observed,plane,case['hfov'],case['image_size'])
    review.update(program_solver=solver,observed_state_preserved=True,
        unresolved_installations=unresolved,position_was_projected=bool(items),sizes_changed=False,
        layout_proxy_passed=review['geometry_contract_passed'] and not solver['final_overlaps'] and
            not unresolved and solver['solver_converged'],physical_geometry_verified=False)
    return dict(plan=plan,bodies=bodies,groups=groups,placements=placements,
                geometry_raw=raw,items=items,review=review,mappings=mappings)


def world_signature(solved):
    # The target state consists of category, variant, centre and full AABB size.
    # Merge identical target worlds across different latent assignments; retain
    # ALL assignment contributions and their original masses in the trace.
    objects=[];byid={o['id']:o for o in solved['plan']['additions']}
    for item in solved['items']:
        obj=byid[item['id']]
        objects.append(json.dumps(dict(category=obj['object_type'],variant=obj['variant'],
            center=[round(v,8) for v in item['center']],size=item['size']),sort_keys=True))
    return tuple(sorted(objects))


def summarize_candidates(records):
    valid=[r for r in records if r.get('numeric_feasible') and r['model_mass']>0]
    total=sum(r['model_mass'] for r in valid)
    if total<=0: raise ValueError('No positive-mass numerically feasible complete program')
    merged={}
    for r in valid:
        signature=world_signature(r['solved'])
        entry=merged.setdefault(signature,dict(model_mass=0.,representative=r,contributors=[]))
        entry['model_mass']+=r['model_mass']
        entry['contributors'].append(dict(candidate=r['candidate'],configuration=r['configuration'],modes=r['modes'],model_mass=r['model_mass']))
    worlds=sorted(merged.values(),key=lambda w:(-w['model_mass'],w['representative']['candidate']))
    for rank,w in enumerate(worlds,1):
        w.update(rank=rank,conditional_model_weight=w['model_mass']/total)
    return worlds,total


def intervention_reports(program,tables,candidates):
    feasible={tuple(r['modes']):r.get('numeric_feasible',False) for r in candidates}
    reports=[]
    for gi,group in enumerate(program['groups']):
        for mode in group['modes']:
            assignments=enumerate_assignments(program,tables,{gi:mode['id']})
            evidence=sum(r['probability'] for r in assignments if feasible.get(tuple(r['modes']),False))
            prior_mean=sum(r['probability']*sum(len(g['modes'][v]['objects']) for g,v in zip(program['groups'],r['modes']))
                           for r in assignments)
            mean=sum(r['probability']*sum(len(g['modes'][v]['objects']) for g,v in zip(program['groups'],r['modes']))
                     for r in assignments if feasible.get(tuple(r['modes']),False))
            reports.append(dict(group=group['id'],set_mode=mode['id'],mechanism_replaced=True,
                joint_recipe=mode['objects'],expected_new_before_feasibility=prior_mean,
                feasible_mass=evidence,expected_new_given_feasible=mean/evidence if evidence else None,
                mode_marginals_before_feasibility={g['id']:[sum(r['probability'] for r in assignments if r['modes'][j]==v)
                    for v in range(len(g['modes']))] for j,g in enumerate(program['groups'])},
                forward_draw_with_same_noise=sample_modes(program,tables,3407,{gi:mode['id']})))
    return dict(reports=reports,baseline_forward_draw=sample_modes(program,tables,3407),
                interpretation='conditional_mode_mechanism_replacement_then_numeric_feasibility; '
                'not identified real-world effects or noise-abducted individual counterfactuals')


def draw_graph(folder,registry,programs):
    nodes=[dict(id='C',role='inverse configuration distribution q(C|RGB,G_obs); approximate inference'),
        dict(id='E',role='fixed RGB and frozen observed-instance evidence'),
        dict(id='W',role='deterministic relative-layout compiler, fixed observations and selected joint recipes'),
        dict(id='V',role='full-AABB original-camera frustum feasibility, NOT a calibrated visibility likelihood'),
        dict(id='camera',role='known original centred pinhole HFOV and image dimensions')]
    edges=[['W','V'],['camera','V']];context_edges=[['E','W']]
    for cid,program in programs.items():
        for gi,g in enumerate(program['groups']):
            node=cid+'.'+g['id'];nodes.append(dict(id=node,role='joint group-mode mechanism; integer states with explicit empty option'))
            edges.append(['C',node]);edges.extend([[cid+'.g'+str(p),node] for p in g['parents']])
            context_edges.append(['E',node])
            nodes.append(dict(id=node+'.objects',role='joint existence, variant, size, support and relative-pose recipe'))
            edges.append([node,node+'.objects'])
            edges.append([node+'.objects','W'])
    write(folder/'mechanism_graph.json',dict(nodes=nodes,edges=edges,
        fixed_context_edges=context_edges,inverse_inference_edges=[['E','C']],
        fixed_context=['RGB','G_obs','observed semantic registry','camera'],
        graph_is_assumed=True,observational_RGB_likelihood_defined=False,
        metric_posterior_defined=False,real_world_causal_structure_identified=False))
    (folder/'mechanism_graph.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="220">'
        '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="17">'
        '<text x="20" y="35">Conditional structural scene programs; assumed mechanisms, no training</text>'
        '<text x="20" y="90">RGB + frozen G_obs</text><text x="245" y="90">q(C | evidence)</text>'
        '<text x="460" y="90">Joint group modes</text><text x="700" y="90">Instances / variants / hosts</text>'
        '<text x="20" y="150">Relative geometry + original camera → numerical feasibility → finite conditional model weights</text>'
        '<text x="20" y="195">Inverse inference is approximate. No F/D splitting; no independent per-class counts; no RGB posterior claim.</text>'
        '<path d="M210 84 L235 84 M415 84 L450 84 M660 84 L690 84" stroke="black" stroke-width="2"/></g></svg>')


def page(folder,record):
    body=['<!doctype html><meta charset="utf-8"><title>CauPano joint scene programs</title>',
        '<h1>联合场景程序：固定局部观测 → 镜头外世界假设</h1>',
        '<p>候选按有限条件模型权重排序，并合并相同目标世界。权重未校准；不代表真实场景概率。2:1图是状态方向图。</p>',
        '<img src="mechanism_graph.svg" width="1100"><p><a href="inference.json">全部候选、可行质量与空世界质量</a> | '
        '<a href="mechanism_graph.json">实际机制图</a> | <a href="summary.json">运行状态</a></p>',
        '<img src="observation_board.png" width="1100">']
    if (folder/'inference.json').exists():
        inference=read(folder/'inference.json')
        body.append(f'<p>全部可行候选的新增期望：{inference["expected_new_instances"]:.3f}；'
            f'空世界条件质量：{inference["empty_world_conditional_mass"]:.4f}。以下保留空候选。</p>'+
            '<details><summary>展开全部不同世界的数量和权重</summary><table border="1"><tr><th>排名</th><th>新增数</th><th>条件权重</th><th>完整程序记录</th></tr>')
        for w in inference['worlds']:
            body.append(f'<tr><td>{w["rank"]}</td><td>{w["new_instances"]}</td><td>{w["conditional_model_weight"]:.6f}</td>'+
                '<td><a href="candidates/'+w['representative']+'.json">'+w['representative']+'</a></td></tr>')
        body.append('</table></details>')
    for h in record.get('worlds',[]):
        body.append('<h2>'+escape(h['folder'])+f'：新增 {h["new_instances"]}，条件权重 {h["conditional_model_weight"]:.4f}</h2>'+
            '<p>'+escape(h['status'])+'；AABB重叠对 '+str(h['remaining_AABB_overlap_pairs'])+'</p>'+
            '<a href="'+h['folder']+'/world/index.html">打开3D、原视锥投影与2:1状态方向图</a> | '+
            '<a href="'+h['folder']+'/G_full.json">G_full</a> | <a href="'+h['folder']+'/scene_program.json">生成机制与位置调整</a>')
    body.append('<pre>'+escape(json.dumps(record,ensure_ascii=False,indent=2))+'</pre>')
    (folder/'index.html').write_text(''.join(body))


def run(q,case,output,budgets,search,worlds=3,seed=3407,max_tokens=8192,fixture=None):
    folder=output/case['id'];folder.mkdir();programs={};tables={};records=[];exports=[];failures=[]
    observed=validate_graph(read(case['state']));check_hashes(case['source_sha256'])
    write(folder/'input.json',case);write(folder/'G_obs.json',observed)
    shutil.copyfile(case['image'],folder/'local.png');original,board=observation_board(case,observed,folder/'observation_board.png')
    record=dict(id=case['id'],protocol=PROTOCOL,status='failed',world_state_ready=False,
        execution_kind='cpu_fixture_not_model_prediction' if fixture else 'frozen_Qwen_conditional_scene_program',
        observed_instances=len(observed['objects']),scene_gt_used=False,training_performed=False,
        adapter_connected=False,panorama_generated=False,semantics_verified=False,geometry_verified=False,
        causal_gain_validated=False,RGB_likelihood_defined=False,joint_metric_posterior_defined=False)
    stage='observed_registry'
    try:
        options=dict(seed=seed,sample=False,profile='official',max_tokens=max_tokens,guard_repetition=True)
        registry=(registry_answer(fixture['registry'],case,budgets['configurations']) if fixture else
            ask_validated(q,board,registry_prompt(case,budgets),folder/'registry.txt',
                lambda v:registry_answer(v,case,budgets['configurations']),options))
        write(folder/'registry.json',registry)
        print(json.dumps(dict(stage='observed_registry_ready',configurations=len(registry['configurations']),observed=len(registry['instances']))),flush=True)
        plane=support_plane(observed,[r['id'] for r in registry['instances'] if r['on_floor']]);write(folder/'support_plane.json',plane)
        stage='inverse_configuration'
        if fixture:
            inverse=dict(probabilities={c['id']:1/len(registry['configurations']) for c in registry['configurations']},source='explicit_CPU_test_fixture')
        else:
            inverse=finite_scores(q,board,'Infer which proposed EXISTING surrounding-scene configuration best explains '
                'the original partial image and frozen observed instances. Unseen contents are hypotheses; do not reward desired equipment.',
                dict(observed=visible_inventory(case)),registry['configurations'])
            inverse['probabilities']={c['id']:inverse['probabilities'][str(i)] for i,c in enumerate(registry['configurations'])}
        write(folder/'configuration_scores.json',inverse)
        print(json.dumps(dict(stage='inverse_configuration_ready',weights=inverse['probabilities'])),flush=True)
        for ci,config in enumerate(registry['configurations']):
            cid=config['id'];dest=folder/cid;dest.mkdir();stage='joint_program_'+cid;local=[]
            try:
                program=(program_answer(fixture['programs'][cid],case,config,budgets['groups'],budgets['modes'],budgets['max_new']) if fixture else
                    ask_validated(q,board,program_prompt(case,registry,config,budgets),dest/'program.txt',
                        lambda v:program_answer(v,case,config,budgets['groups'],budgets['modes'],budgets['max_new']),
                        {**options,'seed':seed+100+ci}))
                write(dest/'program.json',program);programs[cid]=program
                print(json.dumps(dict(stage='joint_program_ready',configuration=cid,groups=len(program['groups']),
                                      maximum_new=program['maximum_new_instances'])),flush=True)
                mechanisms=dest/'mechanisms';mechanisms.mkdir()
                if fixture: table=fixture['tables'][cid]
                else:
                    def save_mechanism(name,value):
                        write(mechanisms/(name+'.json'),value)
                        print(json.dumps(dict(stage='group_mechanism_ready',configuration=cid,mechanism=name,
                                              probabilities=value['probabilities'])),flush=True)
                    table=parameterize(program,registry,GroupMechanismScorer(q),save_mechanism)
                enumerate_assignments(program,table);tables[cid]=table;write(dest/'tables.json',table)
                for assignment in enumerate_assignments(program,table):
                    number=len(records)+len(local);name=f'candidate_{number:04d}'
                    row=dict(candidate=name,configuration=cid,modes=assignment['modes'],
                        conditional_mode_probability=assignment['probability'],
                        model_mass=inverse['probabilities'][cid]*assignment['probability'],numeric_feasible=False)
                    try:
                        solved=solve_assignment(case,observed,plane,program,assignment['modes'],search)
                        row.update(numeric_feasible=solved['review']['geometry_contract_passed'],solved=solved)
                    except ValueError as exc:
                        row.update(error=str(exc))
                    local.append(row)
                    print(json.dumps(dict(stage='joint_candidate_compiled',configuration=cid,modes=row['modes'],
                        numeric_feasible=row['numeric_feasible'],
                        new_instances=len(row['solved']['items']) if 'solved' in row else None)),flush=True)
                write(dest/'mechanism_interventions.json',intervention_reports(program,table,local))
                records.extend(local)
            except Exception as exc:
                failures.append(dict(configuration=cid,error=f'{type(exc).__name__}: {exc}'))
                if local:write(dest/'partial_candidates_not_in_inference.json',local)
                (dest/'error_traceback.txt').write_text(traceback.format_exc())
        stage='finite_world_inference'
        candidates=folder/'candidates';candidates.mkdir()
        for row in records:write(candidates/(row['candidate']+'.json'),row)
        ranked,feasible_mass=summarize_candidates(records)
        expected=sum(w['conditional_model_weight']*len(w['representative']['solved']['items']) for w in ranked)
        inference=dict(scope='finite_conditional_program_model; NOT exact RGB posterior',
            inverse_configuration=inverse['probabilities'],feasible_model_mass=feasible_mass,
            configuration_weights_after_feasibility={cid:sum(r['model_mass'] for r in records
                if r['configuration']==cid and r.get('numeric_feasible'))/feasible_mass
                for cid in inverse['probabilities']},
            configurations_failed=failures,assignments=len(records),distinct_feasible_worlds=len(ranked),
            expected_new_instances=expected,empty_world_conditional_mass=sum(w['conditional_model_weight'] for w in ranked if not w['representative']['solved']['items']),
            metric_energy_used_as_probability=False,selection='highest aggregated conditional model weight; no nonempty rejection sampling',
            worlds=[dict(rank=w['rank'],conditional_model_weight=w['conditional_model_weight'],
                new_instances=len(w['representative']['solved']['items']),representative=w['representative']['candidate'],
                contributors=w['contributors']) for w in ranked])
        write(folder/'inference.json',inference);draw_graph(folder,registry,programs)
        stage='world_exports'
        for w in ranked[:worlds]:
            row=w['representative'];solved=row['solved'];dest=folder/f'world_{w["rank"]:02d}';dest.mkdir()
            full,_,unmapped=export_states(observed,solved['plan'],solved['items'],solved['mappings'],solved['review'])
            h=dict(folder=dest.name,new_instances=len(solved['items']),conditional_model_weight=w['conditional_model_weight'],
                status='layout_needs_review' if not solved['review']['layout_proxy_passed'] else 'candidate_needs_semantic_review',
                remaining_AABB_overlap_pairs=len(solved['review']['program_solver']['final_overlaps']),
                unmapped_new=len(unmapped),observed_state_preserved=True)
            if not solved['items']:h['status']='empty_world_alternative_needs_review'
            full.update(completion_protocol=PROTOCOL,target='outside_input_frustum',adapter_connected=False,
                declared_support_plane=plane,world_relations=solved['plan']['relations'],world_state_ready=True,
                world_state_ready_scope='numeric_export_only',configuration=row['configuration'],modes=row['modes'],
                conditional_model_weight=w['conditional_model_weight'],joint_metric_posterior_defined=False,
                causal_gain_validated=False,quality_status=h['status'])
            variants={o['id']:o['variant'] for o in solved['plan']['additions']}
            for obj in full['objects'][len(observed['objects']):]:obj['variant']=variants[obj['track_id']]
            validate_world(observed,full,solved['items'],len(observed['objects'])+budgets['max_new'])
            write(dest/'G_full.json',full);hidden=deepcopy(full);hidden['scope']='hidden';hidden['objects']=full['objects'][len(observed['objects']):]
            write(dest/'G_out.json',validate_graph(hidden));write(dest/'numeric_review.json',solved['review'])
            write(dest/'scene_program.json',dict(candidate=row['candidate'],configuration=row['configuration'],modes=row['modes'],
                contributors=w['contributors'],**solved));write(dest/'summary.json',h)
            visual=dest/'world';visual.mkdir();render(full,original,case['hfov'],visual,h,
                position_note='联合可选组态与变体/宿主/相对位置机制；三维为近似求解，未估计真实RGB后验。')
            visual_page=visual/'index.html'
            visual_page.write_text(visual_page.read_text().replace(
                '红色候选保留在G_full_candidates中，不进入Adapter子集G_full。',
                '红色未映射对象完整保留在G_full和G_out；本阶段不连接Adapter。'))
            image_path=visual/'layout_xz.png'
            # This reused renderer has an old generic "no assumed floor" title.
            # Replace only its caption in the NEW visualization with the actual
            # camera-Y-level-prior statement; original images remain untouched.
            with Image.open(image_path) as handle:panel=handle.convert('RGB')
            painter=ImageDraw.Draw(panel);painter.rectangle((0,0,panel.width,32),fill='white')
            floor_note='Y-level floor prior' if plane['floor_y_m'] is not None else 'explicit camera-Y; floor unmeasured'
            painter.text((10,10),'Camera XZ boxes; '+floor_note+'; gravity/walls unmeasured.',fill='black');panel.save(image_path)
            exports.append(h)
        if exports:
            for name in ('G_full.json','G_out.json'):shutil.copyfile(folder/exports[0]['folder']/name,folder/name)
        record.update(status='partial_programs_exported_needs_review' if failures else 'conditional_worlds_exported_needs_review',
            world_state_ready=bool(exports),worlds=exports,configuration_failures=failures,
            representative_world=exports[0]['folder'] if exports else None,
            representative_policy='highest aggregated finite conditional model weight',
            expected_new_instances=expected,empty_world_conditional_mass=inference['empty_world_conditional_mass'],
            candidate_assignments=len(records),observed_state_preserved=True,
            world_state_ready_scope='numeric_export_only')
        check_hashes(case['source_sha256'])
    except Exception as exc:
        record.update(status='failed',world_state_ready=False,error=f'{type(exc).__name__}: {exc}',failed_stage=stage)
        (folder/'error_traceback.txt').write_text(traceback.format_exc())
    write(folder/'summary.json',record);page(folder,record)
    write(output/'summary.json',dict(protocol=PROTOCOL,records=[record]))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><a href="'+case['id']+'/index.html">打开联合场景程序与全部候选</a>')
    manifest={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    write(output/'artifact_manifest.json',manifest)
    print(json.dumps(record,ensure_ascii=False),flush=True);return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',type=Path);p.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--configurations',type=int,default=3);p.add_argument('--groups',type=int,default=3)
    p.add_argument('--modes',type=int,default=2);p.add_argument('--max-new',type=int,default=24)
    p.add_argument('--worlds',type=int,default=3);p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--max-tokens',type=int,default=8192);p.add_argument('--region-keep',type=int,default=3)
    p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if (not 1<=a.configurations<=4 or not 1<=a.groups<=3 or not 1<=a.modes<=3
            or not 1<=a.max_new<=48 or not 1<=a.worlds<=8 or not 1<=a.region_keep<=5
            or not 512<=a.max_tokens<=16384 or not 0<=a.seed<2**31):p.error('Need valid computational budgets')
    try:a.output=validate_new_output(a.output)
    except (ValueError,OSError) as exc:p.error(str(exc))
    case=load_case(a.case)
    if not a.model.is_dir():p.error('Local frozen Qwen checkpoint missing')
    budgets=dict(configurations=a.configurations,groups=a.groups,modes=a.modes,max_new=a.max_new)
    search=dict(region_keep=a.region_keep,beam_width=4,rounds=8)
    if a.preflight:
        print(json.dumps(dict(status='scene_program_preflight_passed',protocol=PROTOCOL,id=case['id'],
            output=str(a.output),cwd=str(Path.cwd()),output_policy_checked=True,
            observed=len(case['context']),budgets=budgets,maximum_assignments=a.configurations*(a.modes+1)**a.groups,
            model_loaded=False,gpu_started=False,training=False,adapter_connected=False)));return
    output=new_output(a.output);write(output/'config.json',dict(protocol=PROTOCOL,case=case,
        code_sha256=code_hashes(),budgets=budgets,search=search,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}))
    q=Qwen(a.model);record=run(q,case,output,budgets,search,a.worlds,a.seed,a.max_tokens)
    if not record['world_state_ready']:raise SystemExit('No world exported; inspect summary and preserved raw answers.')


if __name__=='__main__':main()
