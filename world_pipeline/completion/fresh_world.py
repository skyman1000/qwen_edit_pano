"""Fresh joint-configuration SCM -> all selected hidden instances -> visualization."""
from copy import deepcopy
from html import escape
import json
import shutil
import traceback

from PIL import Image,ImageDraw

from ..common import read,validate_graph,write
from .conditional_world_geometry import support_plane,compile_layout
from .fresh_model import (PROTOCOL,registry_prompt,registry_answer,program_prompt,
    program_answer,geometry_state,translated_relations,build_model)
from .fresh_observer import load_case
from .fresh_layout import compile_program,review_relations
from .scene_program import program_answer as metric_program,instantiate
from .scene_program_world import observation_board
from .scene_completion import map_plan,export_states
from .off_fov_completion import validate_world
from .semantic_layout_visual import render,erp_map
from .world_scm import enumerate_model,abduct,forward,mechanism_strengths
from .world_scm_runner import ask_valid,geometry_response,layout_prompt,mechanism_prompt,mechanism_response


def solve(case,observed,plane,program,relations,search):
    plan,bodies,groups,placements,unresolved=instantiate(program,[1]*len(program['groups']),case,observed,plane)
    plan['scene_relations']=deepcopy(relations)
    raw,solver=compile_program(case,observed,plan,bodies,groups,placements,relations=relations,**search)
    mappings=map_plan(plan,'semantic_lexicon_v1')
    items,review=compile_layout(raw,plan,mappings,observed,plane,case['hfov'],case['image_size'])
    relations_review=review_relations(relations,items,observed)
    review.update(program_solver=solver,observed_state_preserved=True,unresolved_installations=unresolved,
        declared_relation_review=relations_review,physical_geometry_verified=False,
        layout_proxy_passed=bool(review['geometry_contract_passed'] and not solver['final_overlaps'] and not unresolved and solver['solver_converged']
            and all(r['ordering_or_presence_satisfied'] is not False for r in relations_review)))
    return dict(plan=plan,items=items,mappings=mappings,review=review,geometry_raw=raw)


def graph_page(out):
    (out/'mechanism_graph.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="200">
<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="19">
<text x="20" y="40">Assumed finite joint-world SCM; frozen LM mechanisms</text>
<text x="20" y="100">C: use/preferences</text><text x="310" y="100">Z: joint scene configuration</text>
<text x="770" y="65">E: positive local signature</text><text x="770" y="135">H: full hidden inventory</text>
<path d="M220 94 L300 94 M575 94 L755 60 M575 94 L755 130" stroke="black" stroke-width="2"/>
<text x="20" y="175">Inference: p(C,Z,H | E). Geometry is a separate conditional compiler, not an RGB likelihood.</text>
</g></svg>''')


def page(out,summary):
    content=['<!doctype html><meta charset="utf-8"><title>Fresh world hypotheses</title>',
        '<h1>从局部观测推断镜头外世界</h1><p>蓝色为固定观测；橙色/红色为新增假设。2:1 图是状态方向图，本阶段未生成 RGB 全景。</p>',
        '<img src="mechanism_graph.svg" width="1000"><p><a href="causal_model.json">实际条件机制</a> | '
        '<a href="inference.json">前后验与全部候选</a> | <a href="mechanism_queries.json">模型内干预诊断</a></p>',
        '<img src="observation_board.png" width="1000">']
    for world in summary.get('worlds',[]):
        folder=world['folder'];content.append('<h2>'+escape(world['configuration'])+f'；新增 {world["new_instances"]}</h2>'+
            f'<p>有限域内权重 {world["semantic_model_weight"]:.4f}；'+escape(world['status'])+'</p>'+
            '<a href="'+folder+'/world/index.html">交互3D / XZ / 局部重投影</a> | <a href="'+folder+'/G_full.json">G_full</a> | '+
            '<a href="'+folder+'/object_relations.json">对象关系与检查</a><br><img width="1000" src="'+folder+'/world/erp_state_map.png">')
    content.append('<pre>'+escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
    (out/'index.html').write_text(''.join(content))


def run(provider,root,backend,max_tokens=8192):
    recipe=read(root/'experiment.json');budgets=recipe['budgets'];case=load_case(root,backend)
    observed=validate_graph(read(case['state']));grounding=read(root/backend/'observer2d'/'grounding.json')
    out=root/backend/'completion';out.mkdir(parents=True,exist_ok=False);calls=out/'calls';calls.mkdir()
    original,board=observation_board(case,observed,out/'observation_board.png')
    write(out/'G_obs.json',observed);write(out/'case.json',case);graph_page(out)
    summary=dict(protocol=PROTOCOL,backend=backend,status='failed',world_state_ready=False,
        observed_instances=len(observed['objects']),worlds=[],failures=[],scene_gt_used=False,training_performed=False,
        adapter_connected=False,panorama_generated=False,semantics_verified=False,geometry_verified=False,
        causal_gain_validated=False,RGB_likelihood_defined=False,numeric_causal_model_shared_between_backends=False,
        shared_causal_factorization='p(C) p(Z|C) p(E|Z) p(H|Z)',group_absence_sampling=False)
    stage='start'
    def ask(name,image,prompt,validator):
        nonlocal stage
        stage=name;print(json.dumps(dict(backend=backend,stage=stage)),flush=True)
        return ask_valid(provider,image,prompt,calls/(name+'.txt'),validator,max_tokens)
    try:
        registry=ask('configurations',board,registry_prompt(case,grounding,budgets['configurations']),
            lambda v:registry_answer(v,budgets['configurations']))
        write(out/'configurations.json',registry);programs={}
        for config in registry['configurations']:
            programs[config['id']]=ask('program_'+config['id'],board,program_prompt(case,grounding,config,budgets),
                lambda v,c=config:program_answer(v,case,c,budgets))
        write(out/'joint_programs.json',programs)
        def mechanism(name,payload,validator):
            return ask('mechanism_'+name,None,mechanism_prompt(payload).replace(
                'For hidden group states, optional absence and nonempty assemblies are competing possible existing states; ',
                'Z states are COMPLETE jointly specified world configurations, not independent optional assemblies; '),
                lambda v:validator(mechanism_response(v,payload)))
        evidence=dict(description=grounding['evidence_description'],camera_hfov=case['hfov'],image_size=case['image_size'],
            signature_is_positive=True,unseen_absence_not_asserted=True)
        model=build_model(registry,programs,evidence,mechanism);write(out/'causal_model.json',model)
        prior=enumerate_model(model);posterior=enumerate_model(model,{'E':'signature'})
        ranked=sorted([dict(configuration=z,probability=p,new_instances=programs[z]['new_instances'])
            for z,p in posterior['marginals']['H'].items()],key=lambda r:(-r['probability'],r['configuration']))
        tv=sum(abs(prior['marginals']['H'][z]-p) for z,p in posterior['marginals']['H'].items())/2
        inference=dict(prior=prior,posterior=posterior,ranked=ranked,hidden_posterior_update_TV=tv,
            expected_new_instances=sum(r['probability']*r['new_instances'] for r in ranked),
            empty_world_posterior_mass=sum(r['probability'] for r in ranked if not r['new_instances']),
            evidence_selected_domain=True,probabilities_are_subjective=True,no_geometric_renormalization=True)
        write(out/'inference.json',inference)
        noise=abduct(model,{'E':'signature'},3407)
        interventions=[]
        for name in ('C','Z'):
            node=next(n for n in model['nodes'] if n['id']==name)
            for value in node['states']:
                interventions.append(dict(do={name:value},prior_intervention=enumerate_model(model,do={name:value})['marginals'],
                    same_abducted_noise_world=forward(model,noise['noise'],{name:value})))
        write(out/'mechanism_queries.json',dict(abduction=noise,interventions=interventions,strengths=mechanism_strengths(model),
            scope='model-relative assumed-mechanism diagnostics, not identified real-world effects'))
        plane=support_plane(observed,[r['id'] for r in grounding['instances'] if r['on_floor']]);write(out/'support_plane.json',plane)
        for rank,world in enumerate(ranked,1):
            if world['probability']<=0:continue
            z=world['configuration'];semantic=programs[z];domain=semantic['geometry_domain'];state=geometry_state(semantic)
            dest=out/f'world_{rank:02d}';dest.mkdir();write(dest/'semantic_program.json',semantic)
            try:
                if world['new_instances']:
                    prompt=layout_prompt(domain,state,case,plane)+'\nDECLARED JOINT OBJECT RELATIONS: '+json.dumps(semantic['relations'],ensure_ascii=False)
                    normalized,converted=ask('layout_'+z,board,prompt,lambda v:geometry_response(v,domain,state,case))
                    program=metric_program(normalized,case,semantic['configuration'],max_groups=budgets['groups'],max_modes=1,max_new=budgets['max_new'])
                else:program=dict(configuration=semantic['configuration'],groups=[]);converted={}
                stage='compile_'+z;relations=translated_relations(semantic)
                solved=solve(case,observed,plane,program,relations,recipe['layout_search'])
                write(dest/'execution.json',dict(solved=solved,geometry_conversions=converted))
                write(dest/'numeric_review.json',solved['review'])
                if not solved['review']['geometry_contract_passed']:raise ValueError('No off-frustum numeric export for the complete joint program')
                full,_,unmapped=export_states(observed,solved['plan'],solved['items'],solved['mappings'],solved['review'])
                full.update(completion_protocol=PROTOCOL,adapter_connected=False,target='outside_input_frustum',
                    semantic_model_weight=world['probability'],world_state_ready=True,world_state_ready_scope='numeric_candidate_not_certified',
                    scene_relations=deepcopy(relations),declared_support_plane=plane)
                for gi,g in enumerate(domain['groups']):
                    byid={o['track_id']:o for o in full['objects']}
                    for obj in g['modes'][1]['objects']:
                        byid[f'sp_g{gi:02d}_'+obj['key']].update(variant=obj['variant'],raw_category=obj['category'],
                            distinct_instance_reason=obj['distinct_instance_reason'])
                validate_world(observed,full,solved['items'],len(observed['objects'])+budgets['max_new'])
                write(dest/'G_full.json',full);hidden=deepcopy(full);hidden.update(scope='hidden',objects=deepcopy(full['objects'][len(observed['objects']):]))
                write(dest/'G_out.json',hidden)
                write(dest/'object_relations.json',dict(support_dependencies=solved['plan']['relations'],
                    declared_relations=relations,relation_review=solved['review']['declared_relation_review'],
                    relation_graph_is_not_the_probabilistic_causal_graph=True))
                result=dict(folder=dest.name,configuration=semantic['configuration']['description'],new_instances=len(solved['items']),
                    semantic_model_weight=world['probability'],unmapped_new=len(unmapped),observed_state_preserved=True,
                    remaining_AABB_overlap_pairs=len(solved['review']['program_solver']['final_overlaps']),
                    status='candidate_needs_semantic_review' if solved['review']['layout_proxy_passed'] else 'layout_needs_review')
                visual=dest/'world';visual.mkdir()
                if full['objects']:
                    render(full,original,case['hfov'],visual,result,position_note='完整联合场景假设；原观测固定。相机Y水平及支撑平面仅为未标定布局先验。')
                    p=visual/'index.html';p.write_text(p.read_text().replace('红色候选保留在G_full_candidates中，不进入Adapter子集G_full。',
                        '未映射对象完整保留在G_full，本阶段不连接Adapter。'))
                else:
                    erp_map(full,case['hfov'],case['image_size'],visual/'erp_state_map.png')
                    canvas=Image.new('RGB',(1000,600),'white');ImageDraw.Draw(canvas).text((20,20),'Empty observed and hypothesized inventory.',fill='black')
                    canvas.save(visual/'layout_xz.png')
                    (visual/'index.html').write_text('<meta charset="utf-8"><p>观测与新增清单均为空；保留该假设。</p><img src="erp_state_map.png">')
                summary['worlds'].append(result)
            except Exception as exc:
                summary['failures'].append(dict(configuration=z,stage=stage,semantic_model_weight=world['probability'],
                    new_instances=world['new_instances'],error=type(exc).__name__+': '+str(exc)))
                (dest/'error_traceback.txt').write_text(traceback.format_exc())
        summary.update(status='worlds_exported_needs_review' if summary['worlds'] and not summary['failures'] else
            'partial_layouts_needs_review' if summary['worlds'] else 'no_world_exported',world_state_ready=bool(summary['worlds']),
            expected_new_instances=inference['expected_new_instances'],hidden_posterior_update_TV=tv,
            exported_semantic_mass=sum(r['semantic_model_weight'] for r in summary['worlds']),
            failed_layout_semantic_mass=sum(r['semantic_model_weight'] for r in summary['failures']),
            no_geometric_renormalization=True,configurations=len(programs))
    except Exception as exc:
        summary.update(status='failed',failed_stage=stage,error=type(exc).__name__+': '+str(exc))
        (out/'error_traceback.txt').write_text(traceback.format_exc())
    summary.update(api_calls=getattr(provider,'calls',None),api_usage=getattr(provider,'usage',None))
    write(out/'summary.json',summary);page(out,summary);return summary
