"""Frozen-model conditional grammar -> structural sampling -> installed world."""
from copy import deepcopy
from html import escape
import json
import traceback

from ..common import read,write,validate_graph,sha
from . import assembly_delta as scm
from . import assembly_parametric as geometry
from .conditional_world_geometry import support_plane
from .fresh_observer import load_case
from .scene_program_world import observation_board
from .scene_completion import export_states
from .off_fov_completion import validate_world
from .semantic_layout_visual import render,erp_map
from .world_scm_runner import ask_valid
from .assembly_observation import host_domains
from .assembly_pose import fit as fit_pose,diagnostics as pose_diagnostics


def graph(out,model):
    if model.get('model_variant') in ('compositional_functional_program_scm_v12','organized_functional_program_scm_v13'):
        write(out/'mechanism_graph.json',dict(nodes=[dict(id=n['id'],parents=n['parents']) for n in model['nodes']],
            generation=model['generation_factorization'],conditional_independence=model['conditional_independence_assumed'],
            physical_causation_identified=False,probabilities_calibrated=False))
        (out/'mechanism_graph.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="250"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="18"><text x="20" y="35">v12: compose OPTIONAL COMPLETE functional programs, not one global object list</text><text x="20" y="80">O_positive -> q(H); H -> A_role ; (H,A_role) -> V_role (inventory + camera plan)</text><text x="20" y="125">V_role -> canonical E_i and R_group ; E_i -> S_i &lt;- S_host ; (O,H,V,R,S) -> X</text><text x="20" y="170">One role program can contain multiple same-function groups and ordinary belongings.</text><text x="20" y="215">J is a context coupling device. No training, minimum count, rear quota or calibrated visual posterior.</text></g></svg>''')
        if model.get('method_revision')==13:
            p=out/'mechanism_graph.svg';p.write_text(p.read_text().replace('v12:','v13:').replace('(O,H,V,R,S) -> X','(O,H,V,R,S) -> frozen dimensions -> joint-host X'))
        return
    if model.get('model_variant')=='holistic_functional_spatial_scm_v11':
        write(out/'mechanism_graph.json',dict(nodes=[dict(id=n['id'],parents=n['parents']) for n in model['nodes']],
            generation=model['generation_factorization'],fixed_context=['O_positive','D'],
            physical_causation_identified=False,probabilities_calibrated=False,training_performed=False))
        (out/'mechanism_graph.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="250"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="18"><text x="20" y="35">v11: joint functional inventory and whole-camera spatial organization</text><text x="20" y="85">O -> q(H); (H,O_positive,D) -> JOINT (B,R); B -> E -> S &lt;- S_host</text><text x="20" y="130">(O,H,B,R,E,S) -> metric X. One instance identity across different camera regions.</text><text x="20" y="175">RGB input control: same canonical local draw seeds. No rear or object-count quota.</text><text x="20" y="220">J is computational. Support is a limited box proxy; physical causation is not identified.</text></g></svg>''')
        return
    write(out/'mechanism_graph.json',dict(nodes=[dict(id=n['id'],parents=n['parents']) for n in model['nodes']],
        fixed_context=['RGB','G_obs','camera'],observation_inference='finite_domain_proposal_not_calibrated_Bayes',
        spatial_mechanisms='world_*/pose_model.json; fitted deterministic support pose equations',
        spatial_interventions='world_*/pose_interventions.json; local sensitivity without layout reoptimization',
        typed_spatial_relations_are_not_automatically_causal_edges=True,
        sampling_law=model['distribution_scope']))
    (out/'mechanism_graph.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="230">
<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="18">
<text x="20" y="30">Conditional inventory / installation SCM (assumed; no training)</text>
<text x="20" y="90">Z: activities / living pattern</text><text x="340" y="90">F: JOINT hidden world delta</text>
<text x="730" y="60">E: owned inventory</text><text x="730" y="120">S: installation</text>
<path d="M260 83 L325 83 M675 83 L720 52 M820 68 L820 102" stroke="black"/>
<text x="20" y="175">E_i → S_i ← S_host ; host removal invalidates installation, not the child's existence</text>
<text x="20" y="210">Pose equations/interventions: each world folder. Image-conditioned proposal; no calibrated RGB posterior.</text>
</g></svg>''')

    if model.get('model_variant')=='anchored_realization_scene_scm_v7':
        write(out/'mechanism_graph.json',dict(nodes=[dict(id=n['id'],parents=n['parents']) for n in model['nodes']],
            recognition='q(H|O,D): inference, not image causing scene formation',generation=model['generation_factorization'],
            joint_mixture_scope=model['joint_mixture_scope'],channel_definitions=model['channel_definitions'],
            policy=model['activation_policy'],causal_gain_validated=False))
        (out/'mechanism_graph.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="230">
<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="18">
<text x="20" y="35">v7: anchored extensions and new functional groups are separate, jointly sampled paths</text>
<text x="20" y="80">O → recognition q(H); H → joint A_channels; (H,A_channel) → V_channel → E → S ← S_host</text>
<text x="20" y="125">Observed-extension and new-group channels may coexist. J is an exact sparse predictive representation.</text>
<text x="20" y="170">(O,H,V,host pose) → dimensions/poses; hard ordering + frustum; surface-proximity optimization</text>
<text x="20" y="210">Assumed conditional SCM, not identified real-world causation. No count target or resident biography.</text>
</g></svg>''')
    if model.get('need_mechanisms') is not None:
        (out/'mechanism_graph.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="230">
<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="18">
<text x="20" y="35">Assumed need / realization mechanism; image-proposed domain; untrained</text>
<text x="20" y="90">Z → R_k (need/source) → B_k (joint realization) → E_i (shared instance) → S_i ← S_host</text>
<text x="20" y="140">E_i = OR over selected realizations using this exact instance. Observations remain fixed.</text>
<text x="20" y="180">Uniform reference laws; no calibrated RGB posterior; metric backend unchanged.</text>
</g></svg>''')
    if model.get('cue_likelihood_defined'):
        write(out/'mechanism_graph.json',dict(nodes=[dict(id=n['id'],parents=n['parents']) for n in model['nodes']],
            observation_inference=model['distribution_scope'],resident_history_inferred=False,
            sampling_law=model['sampling_policy'],parameters='functional_parameters.json',
            evidence='cue_evidence.json',marginals='probability_audit.json'))
        (out/'mechanism_graph.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="240">
<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="19">
<text x="20" y="32">Functional SCM: explicit assumed probabilities; no resident biography</text>
<text x="20" y="80">H (functional organization) → A (anchor-zone cue opportunity) → O (censored report)</text>
<text x="20" y="125">(H,Z) → R (uncertain need) → B (joint realization) → E (shared instance) → S ← S_host</text>
<text x="20" y="172">Inference: O → posterior over H → numerical R/B/E probabilities → actual inventory draws</text>
<text x="20" y="213">Prior/posterior/do(R=0): probability_audit.html; geometry conditional on sampled inventory</text>
</g></svg>''')
    if model.get('count_likelihood_defined'):
        write(out/'mechanism_graph.json',dict(nodes=[dict(id=n['id'],parents=n['parents']) for n in model['nodes']],
            observation_inference=model['distribution_scope'],sampling_policy=model['sampling_policy'],
            candidate_support=model['candidate_support'],parameters='population_parameters.json',
            evidence='count_observations.json',probabilities='probability_audit.json'))
        (out/'mechanism_graph.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="235">
<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="19">
<text x="20" y="35">Full inventory population SCM; finite conditional domain; explicit uncalibrated parameters</text>
<text x="20" y="85">H (functional organization), L (population scale) → N (full counts) → F (joint program) → E → S ← S_host</text>
<text x="20" y="132">N, Q (report process) → observed positive counts. Missing families remain unknown.</text>
<text x="20" y="178">Finite reference + functional count-density ratios; condition on reports. No unmet-demand gate.</text>
<text x="20" y="215">Joint programs preserve observed anchors. Full metric layout is conditional, not a learned RGB likelihood.</text>
</g></svg>''')
    if model.get('model_variant')=='sdscene_conditional_mechanisms_v4':
        write(out/'mechanism_graph.json',dict(nodes=[dict(id=n['id'],parents=n['parents']) for n in model['nodes']],
            recognition='q(H|O,D), empirically estimated',generation='parent-only p(F|H,D)',
            no_additional_image_or_count_likelihood=True,mechanism_source=model['mechanism_source']))
        (out/'mechanism_graph.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="220">
<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="19">
<text x="20" y="35">SD-SCM-inspired conditional scene mechanisms; empirical draws, not token likelihoods</text>
<text x="20" y="85">Recognition: observation → q(H). Generation: H → F (joint inventory) → E → S ← S_host</text>
<text x="20" y="132">F → full counts N (diagnostic only). Mechanism F sees H and fixed domain, no image or vote history.</text>
<text x="20" y="180">Metric parents: fixed observations + H + F. No additional count-likelihood reweighting.</text>
</g></svg>''')
    if model.get('model_variant')=='empirical_functional_assembly_scm_v5':
        write(out/'mechanism_graph.json',dict(nodes=[dict(id=n['id'],parents=n['parents']) for n in model['nodes']],
            recognition='q(H|O,D); pure function tags',generation=model['generation_factorization'],
            conditional_independence_assumed=model['conditional_independence_assumed'],policy=model['modular_policy']))
        (out/'mechanism_graph.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="200">
<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="19">
<text x="20" y="35">Functional assembly grammar v5: assumed mechanisms; not identified real-world causation</text>
<text x="20" y="85">O → recognition q(H); H → M_role (optional joint assembly) → E_i → S_i ← S_host</text>
<text x="20" y="130">H has canonical functional tags only. Modules include hosts + contents; absence remains possible.</text>
<text x="20" y="175">Edge ablation replaces p(M_role|H) by its uniform-profile average; no extra model calls.</text>
</g></svg>''')
    if model.get('model_variant')=='joint_activation_scene_scm_v6':
        write(out/'mechanism_graph.json',dict(nodes=[dict(id=n['id'],parents=n['parents']) for n in model['nodes']],
            recognition='O -> q(H|O,D): inference, separate from the assumed generative edges',
            generation=model['generation_factorization'], policy=model['activation_policy'],
            metric_parents=['O','H','selected_variants','host_pose'],metric_equations='world_*/pose_model.json',
            downstream_validation='E,X -> support/collision/functional-proxy diagnostics; not a parent of H',
            scene_formation_DAG_identified=False,calibrated_visual_posterior=False))
        (out/'mechanism_graph.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="250">
<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="18">
<text x="20" y="30">v6 conditional SCM (assumed): joint activation and separate conditional variants</text>
<text x="20" y="75">Inference: observed RGB / G_obs → q(H | O,D); H = uncertain organization/use program</text>
<text x="20" y="120">Generation: H → A (joint role vector); (H,A_r) → V_r → E_i → S_i ← S_host</text>
<text x="20" y="165">Metric: (O,H,V,host pose) → X ; (E,X) → geometry / support / usability diagnostics</text>
<text x="20" y="205">H-cut: root A and A→V only; preserves joint activation and every module marginal.</text>
<text x="20" y="235">No biography, measured hidden-room area, minimum object count, or identified real-world causal effect.</text>
</g></svg>''')
    if model.get('model_variant') in ('reference_shrunk_functional_scene_scm_v8','functional_spatial_program_scm_v9','context_joint_assembly_scm_v10'):
        write(out/'mechanism_graph.json',dict(nodes=[dict(id=n['id'],parents=n['parents']) for n in model['nodes']],
            recognition='q(H|O,D) is inference, not image causing scene use',generation=model['generation_factorization'],
            policy=model['mechanism_policy'],reference='D-only frozen LM draws define a shrinkage base, not new image evidence',
            direct_control='O,D directly predict joint inventory; H retained only as matched metric context',
            joint_mixture_scope=model['joint_mixture_scope'],training_performed=False,real_causal_structure_identified=False))
        description=escape(model['generation_factorization']);version='v10' if model['model_variant']=='context_joint_assembly_scm_v10' else 'v9' if model['model_variant']=='functional_spatial_program_scm_v9' else 'v8'
        (out/'mechanism_graph.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1450" height="250">
<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="17">
<text x="20" y="35">'''+version+''': frozen-model reference shrinkage and declared realizations; no training, no count goal</text>
<text x="20" y="80">'''+description+'''</text>
<text x="20" y="125">Reference: D-only draws; conditional inventory: H,D-only draws; direct control: O,D-only joint draws</text>
<text x="20" y="170">J is a computational mixture. Typed spatial relations are constraints, not causal edges.</text>
<text x="20" y="215">Host removal preserves owned children and invalidates installation. No identified real-world causal effects.</text>
</g></svg>''')
        if version=='v9':
            (out/'mechanism_graph.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1450" height="280">
<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="18">
<text x="20" y="35">v9: sampled functional-spatial realizations; no training or rear-object quota</text>
<text x="20" y="80">Functional inventory: H -> joint A; (H,A) -> V ; V -> E ; V -> R</text>
<text x="20" y="125">Installation: E -> S &lt;- S_host ; metric layout: (O,H,V,R) -> X</text>
<text x="20" y="170">Direct control chooses joint V from O,D; no H enters direct inventory. H remains in metric context.</text>
<text x="20" y="215">R is a camera-relative hypothesis preserved by the solver. Functional service does not imply near.</text>
<text x="20" y="255">J is computational. Assumed conditional SCM, not identified real-world causation.</text>
</g></svg>''')
        if version=='v10':
            context_label=escape(','.join(model['fixed_context_parents']['joint_selection']))
            write(out/'mechanism_graph.json',dict(nodes=[dict(id=n['id'],parents=n['parents']) for n in model['nodes']],
                fixed_context_parents=model['fixed_context_parents'],generation=model['generation_factorization'],policy=model['mechanism_policy'],
                observation_scope='O is fixed conditioning context; not a claim that pixels physically cause furniture',
                joint_mixture_scope=model['joint_mixture_scope'],training_performed=False,real_causal_structure_identified=False))
            (out/'mechanism_graph.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1450" height="280">
<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="18">
<text x="20" y="35">v10: observation-conditioned JOINT candidate assemblies; no one-per-function slot</text>
<text x="20" y="80">Inventory: ('''+context_label+''') -> B_joint ; B -> E ; B -> R ; E -> S &lt;- S_host</text>
<text x="20" y="125">Metric layout: (O,H,B,E,S,R) -> X. Same-purpose assemblies can coexist.</text>
<text x="20" y="170">Tag-only ablation removes joint-selection RGB input; H-cut removes inventory H edges.</text>
<text x="20" y="215">No count or rear quota. Every sampled object is retained, including blocked diagnostic worlds.</text>
<text x="20" y="255">O is fixed conditional context; J is computational. Physical causation is not identified.</text>
</g></svg>''')


def page(out,summary):
    from .assembly_review import STYLE,world_card
    lines=['<!doctype html><meta charset="utf-8"><title>Conditional assembly worlds</title>',
        STYLE,'<h1>视野外补全：已放置物体</h1>',
        '<p>蓝色为观测；橙色为新增；红色为未映射但保留的新增框。显示完整 AABB，不是 RGB 全景。</p>']
    if summary.get('model_variant') in ('joint_activation_scene_scm_v6','anchored_realization_scene_scm_v7','reference_shrunk_functional_scene_scm_v8','functional_spatial_program_scm_v9','context_joint_assembly_scm_v10','holistic_functional_spatial_scm_v11'):
        lines[-1]='<p>蓝色为观测；橙色为原始类别表内；紫色为原始名表外但可映射；红色为类别未解析。所有对象保留，图为 AABB 状态。</p>'
    if not summary['worlds']:
        lines.append('<div class="missing">尚无完成的世界状态。停止阶段：'+escape(summary.get('failed_stage','布局'))+
            '<p>'+escape(summary.get('error','请展开运行记录检查布局失败原因。'))+'</p></div>')
    for w in summary['worlds']+summary.get('blocked_worlds',[]):
        lines.append(world_card(out/w['folder'],w,out))
    lines.append('<details><summary>观测与机制记录</summary><img src="observation_board.png">')
    if (out/'mechanism_graph.svg').is_file():
        lines.append('<img src="mechanism_graph.svg"><p><a href="causal_model.json">实际生成机制</a> | <a href="mechanism_graph.json">全部节点与父变量</a></p>')
    if summary.get('cue_likelihood_defined') or summary.get('count_likelihood_defined') or summary.get('empirical_mechanisms_used'):
        lines.append('<p><a href="probability_audit.html">库存概率与观测更新对比</a>：数值概率进入实际采样；参数来源见记录，概率尚未完成校准验证。</p>')
    lines.append('<p>模型内推断不证明真实因果效应；具体概率范围见记录。</p><pre>'+
        escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></details>');(out/'index.html').write_text(''.join(lines))


def run(provider,root,backend):
    config=read(root/'experiment.json')
    model_name=config.get('completion_model','joint')
    if model_name=='needs':from . import assembly_needs as selected_scm
    elif model_name=='functional':from . import assembly_functional as selected_scm
    elif model_name=='population':from . import assembly_population as selected_scm
    elif model_name=='sdscene':from . import assembly_sdscene as selected_scm
    elif model_name=='modular':from . import assembly_modular as selected_scm
    elif model_name=='activation':from . import assembly_activation as selected_scm
    elif model_name=='realization':from . import assembly_realization as selected_scm
    elif model_name=='mechanism':from . import assembly_mechanism as selected_scm
    elif model_name=='panorama':from . import assembly_panorama as selected_scm
    elif model_name=='assemblyset':from . import assembly_set as selected_scm
    elif model_name=='holistic':from . import assembly_holistic as selected_scm
    elif model_name=='compositional':from . import assembly_compositional as selected_scm
    elif model_name=='baseline':from . import assembly_baseline as selected_scm
    elif model_name=='joint':selected_scm=scm
    else:raise ValueError('Unknown completion_model: '+str(model_name))
    selected_geometry=geometry
    if model_name=='sdscene':from . import assembly_sdscene_geometry as selected_geometry
    if model_name=='modular':from . import assembly_modular_geometry as selected_geometry
    if model_name=='activation':from . import assembly_activation_geometry as selected_geometry
    if model_name=='realization':from . import assembly_realization_geometry as selected_geometry
    if model_name=='mechanism':from . import assembly_mechanism_geometry as selected_geometry
    if model_name=='panorama':from . import assembly_panorama_geometry as selected_geometry
    if model_name=='assemblyset':from . import assembly_panorama_geometry as selected_geometry
    if model_name=='holistic':from . import assembly_holistic_geometry as selected_geometry
    if model_name=='compositional':from . import assembly_compositional_geometry as selected_geometry
    if config.get('completion_revision')=='organized_v13':
        from . import assembly_organized as selected_scm,assembly_organized_geometry as selected_geometry
    return _run(provider,root,backend,config,selected_scm,selected_geometry)


def _run(provider,root,backend,config,scm,geometry=geometry):
    case=load_case(root,backend);observed=validate_graph(read(case['state']))
    case['observed_support_domains']=host_domains(case,observed)
    grounding=read(root/backend/'observer2d/grounding.json');out=root/backend/'completion';out.mkdir(parents=True,exist_ok=False)
    calls=out/'calls';calls.mkdir();original,board=observation_board(case,observed,out/'observation_board.png')
    write(out/'G_obs.json',observed);write(out/'case.json',case)
    write(out/'observed_support_domains.json',case['observed_support_domains'])
    summary=dict(protocol=scm.PROTOCOL,response_interface=scm.INTERFACE,model_variant=scm.MODEL_VARIANT,
        geometry_parameterization=geometry.PARAMETERIZATION,backend=backend,status='failed',world_state_ready=False,worlds=[],blocked_worlds=[],failures=[],
        observed_instances=len(observed['objects']),scene_gt_used=False,training_performed=False,adapter_connected=False,
        panorama_generated=False,causal_gain_validated=False,semantics_verified=False,geometry_verified=False,
        RGB_likelihood_defined=False,posterior_ranking_used=False,
        sampling_law='uniform_reference_over_conditional_LM_proposed_grammar_alternatives',
        inventory_separate_from_installation=True,inventory_is_joint_across_scales=True,
        evidence_delivery='same_observation_board_at_hypotheses_joint_program_and_metric_stages',
        installation_node_scope='declared_host_dependency_satisfied_not_physical_verification')
    stage='start'
    def ask(name,image,prompt,validator):
        nonlocal stage
        stage=name;print(json.dumps(dict(backend=backend,stage=name)),flush=True)
        selected_provider=provider
        if name.startswith('layout_') and config.get('layout_image_delivery') is False:image=None
        if config.get('stage_seed_policy')=='hashed_stage_v1':
            from .assembly_experiment import SeededProvider,stage_seed
            selected_provider=SeededProvider(provider.inner if isinstance(provider,SeededProvider) else provider,stage_seed(config['seed'],name))
        return ask_valid(selected_provider,image,prompt,calls/(name+'.txt'),validator,config['max_tokens'])
    try:
        contents={}
        if config.get('sdscene_bundle'):
            stage='load_shared_sdscene_domain'
            if config.get('completion_model') not in ('sdscene','modular','activation','realization','mechanism','panorama','assemblyset','holistic','compositional'):raise ValueError('Shared scene bundle requires an empirical mechanism model')
            hypotheses,furnitures,reuse=scm.load_domain(case,config)
            write(out/'candidate_source.json',reuse)
            write(out/'furniture_mechanisms.json',furnitures);write(out/'scene_program_mechanisms.json',furnitures)
        elif config.get('candidate_programs_from'):
            stage='revalidate_saved_joint_candidates'
            if config.get('completion_model')!='population':raise ValueError('Saved joint candidates require population model')
            hypotheses,furnitures,reuse=scm.load_joint_domain(case,config['candidate_programs_from'],backend)
            write(out/'candidate_source.json',reuse)
            write(out/'furniture_mechanisms.json',furnitures);write(out/'scene_program_mechanisms.json',furnitures)
        elif config.get('semantic_mechanisms_from'):
            from pathlib import Path
            source=Path(config['semantic_mechanisms_from']).resolve();previous=source/backend/'completion'
            stage='candidate_reuse'
            from .assembly_functional_status import candidate_status,dependency_error
            dependency=candidate_status(source,backend)
            if not dependency['ready']:raise ValueError(dependency_error(dependency))
            if config.get('completion_model') not in ('functional','population'):raise ValueError('Frozen candidate ablation requires a supported probability model')
            if read(previous/'causal_model.json')['response_interface']!=scm.INTERFACE:raise ValueError('Frozen candidate interface differs')
            if sha(source/'input.png')!=sha(root/'input.png') or read(previous/'G_obs.json')!=observed:
                raise ValueError('Frozen candidate RGB/G_obs differs')
            old_case=read(previous/'case.json')
            if any(old_case.get(k)!=case.get(k) for k in ('hfov','image_size','context','observed_support_domains')):
                raise ValueError('Frozen candidate camera/context differs')
            hypotheses=read(previous/'hypotheses.json');furnitures=read(previous/'furniture_mechanisms.json')
            write(out/'candidate_reuse.json',dict(source=str(previous),
                sha256={name:sha(previous/name) for name in ('hypotheses.json','furniture_mechanisms.json','causal_model.json')},
                reason='same_candidate_domain_probability_ablation'))
            write(out/'furniture_mechanisms.json',furnitures);write(out/'scene_program_mechanisms.json',furnitures)
        else:
            hypotheses=ask('hypotheses',board,scm.hypotheses_prompt(case,grounding),scm.hypotheses_answer)
            write(out/'hypotheses.json',hypotheses)
            if hasattr(scm,'propose_programs'):
                furnitures=scm.propose_programs(case,hypotheses,board,ask,out)
            else:
                furnitures={}
                for z in hypotheses:
                    variants=ask('furniture_'+z['id'],board,scm.furniture_prompt(case,z),lambda v,z=z:scm.furniture_answer(v,case,z))
                    for variant in variants:
                        fid='f'+str(len(furnitures));furnitures[fid]=variant
                        write(out/'furniture_mechanisms.json',furnitures)
                        write(out/'scene_program_mechanisms.json',furnitures)
        write(out/'hypotheses.json',hypotheses)
        model=scm.build_model(hypotheses,furnitures,contents)
        if hasattr(scm,'prepare_model'):model=scm.prepare_model(model,case,dict(config,execution_backend=backend),out)
        write(out/'causal_model.json',model);graph(out,model)
        summary.update(sampling_law=model['distribution_scope'],
            sampling_policy=model.get('sampling_policy','unique_inventory_reference_candidates'),
            generation_factorization=model.get('generation_factorization'),
            need_mechanisms=len(model.get('need_mechanisms',{})),
            completion_model=config.get('completion_model','joint'),
            cue_likelihood_defined=model.get('cue_likelihood_defined',False),
            count_likelihood_defined=model.get('count_likelihood_defined',False),
            empirical_mechanisms_used=model.get('model_variant') in ('sdscene_conditional_mechanisms_v4','empirical_functional_assembly_scm_v5','joint_activation_scene_scm_v6','anchored_realization_scene_scm_v7','reference_shrunk_functional_scene_scm_v8','functional_spatial_program_scm_v9','context_joint_assembly_scm_v10','holistic_functional_spatial_scm_v11','compositional_functional_program_scm_v12','organized_functional_program_scm_v13'),
            candidate_support=model.get('candidate_support'),
            posterior_sampling_used=(model.get('cue_likelihood_defined',False) and model.get('functional_sampling')=='posterior')
                or (model.get('count_likelihood_defined',False) and model.get('population_sampling') in ('posterior','no_function')))
        if summary['empirical_mechanisms_used']:
            summary['evidence_delivery']=model.get('evidence_delivery','image_at_proposal_recognition_and_layout; inventory_mechanism_sees_H_and_fixed_domain_text_only')
        worlds=scm.sample_worlds(model,config['seed'],config['worlds']);write(out/'structural_samples.json',worlds)
        plane=support_plane(observed,[o['id'] for o in grounding['instances'] if o['on_floor']])
        if config.get('frozen_world_bundle'):
            from pathlib import Path
            plane=read(Path(config['frozen_world_bundle'])/backend/'support_plane.json')
        write(out/'support_plane.json',plane)
        summary.update(hypotheses=len(hypotheses),furniture_states=len(furnitures),structural_nodes=len(model['nodes']),
            inventory_counts={fid:sum(len(g['objects']) for g in f['groups']) for fid,f in furnitures.items()},
            proposed_sample_counts=[len(w['objects']) for w in worlds],complete_inventory_preserved=True)
        for index,world in enumerate(worlds,1):
            dest=out/f'world_{index:02d}';dest.mkdir();write(dest/'semantic_world.json',world)
            write(dest/'host_interventions.json',scm.intervention_diagnostics(model,world))
            write(dest/'semantic_review.json',scm.semantic_diagnostics(world))
            try:
                if world['objects']:
                    if hasattr(geometry,'infer_layout'):
                        layout=geometry.infer_layout(case,observed,world,plane,board,ask,dest,index)
                    else:
                        layout=ask(f'layout_{index:02d}',board,geometry.layout_prompt(case,observed,world,plane),
                            lambda v,w=world:geometry.layout_answer(v,w,case['hfov']))
                else:layout={}
                stage='compile_'+str(index);solved=geometry.compile_layout(case,observed,world,layout,plane)
                write(dest/'execution.json',dict(layout=layout,solved=solved));write(dest/'numeric_review.json',solved['review'])
                pose_model=fit_pose(world,solved['items'],observed)
                write(dest/'pose_model.json',pose_model);write(dest/'pose_interventions.json',pose_diagnostics(pose_model,solved['items']))
                if not solved['review']['geometry_contract_passed']:
                    candidate,_,_=export_states(observed,solved['plan'],solved['items'],solved['mappings'],solved['review'])
                    candidate.update(completion_protocol=scm.PROTOCOL,world_state_ready=False,
                        completion_model_variant=scm.MODEL_VARIANT,geometry_parameterization=geometry.PARAMETERIZATION,
                        quality_status='blocked_numeric_diagnostic_not_completed_world',geometry_contract_passed=False)
                    for obj in candidate['objects'][len(observed['objects']):]:obj['unseen_reason']='placement_not_verified_outside_input_view'
                    if config.get('completion_model') in ('activation','realization','mechanism','panorama','assemblyset','holistic','compositional'):
                        from .assembly_taxonomy import annotate
                        candidate=annotate(candidate,world)
                    if 'spatial_program' in world:candidate.update(spatial_program=deepcopy(world['spatial_program']),functional_links=deepcopy(world.get('functional_links',[])),unresolved_functional_links=deepcopy(world.get('unresolved_functional_links',[])))
                    if hasattr(geometry,'annotate_state'):candidate=geometry.annotate_state(candidate,layout)
                    write(dest/'G_full_candidates.json',candidate);visual=dest/'candidate_world';visual.mkdir()
                    record=dict(folder=dest.name,new_instances=len(world['objects']),counts_kind='placement_roots_and_hosted',
                        root_instances=sum(o['attachment']['host'] is None for o in world['objects']),
                        hosted_instances=sum(o['attachment']['host'] is not None for o in world['objects']),
                        state_file='G_full_candidates.json',visual_folder='candidate_world',
                        status='blocked_numeric_diagnostic',remaining_AABB_overlap_pairs=len(solved['review']['program_solver']['final_overlaps']))
                    render(candidate,original,case['hfov'],visual,record,position_note='数值约束未满足：仅展示候选，未导出完成世界。')
                    summary['blocked_worlds'].append(record)
                    raise ValueError('Hidden-world numeric contract failed (frustum, declared ordering or strict residual collision checks). Full inventory remains in the diagnostic candidate; no completed G_full exported')
                full,_,unmapped=export_states(observed,solved['plan'],solved['items'],solved['mappings'],solved['review'])
                full.update(completion_protocol=scm.PROTOCOL,target='outside_input_frustum',adapter_connected=False,
                    completion_model_variant=scm.MODEL_VARIANT,geometry_parameterization=geometry.PARAMETERIZATION,
                    world_state_ready=True,world_state_ready_scope='numeric_candidate_not_semantic_certification',
                    typed_relations=world['relations'],declared_support_plane=plane)
                if 'spatial_program' in world:full.update(spatial_program=deepcopy(world['spatial_program']),functional_links=deepcopy(world.get('functional_links',[])),unresolved_functional_links=deepcopy(world.get('unresolved_functional_links',[])))
                if model.get('candidate_support',{}).get('all_nonempty_proposals_missing'):
                    full.update(world_state_ready=False,world_state_ready_scope='observations_only_candidate_domain_empty')
                metadata={o['id']:o for o in world['objects']}
                for o in full['objects'][len(observed['objects']):]:
                    source=metadata[o['track_id']];o.update(variant=source['variant'],attachment=source['attachment'],
                        generative_presence_node=source['presence_node'],installation_node=source['installation_node'],raw_category=source['category'])
                    if 'presence_probability' in source:
                        o.update(generative_presence_probability=source['presence_probability'],
                            presence_probability_scope=model['distribution_scope'],
                            presence_probability_stage='before_geometry_no_failure_resampling')
                if config.get('completion_model') in ('activation','realization','mechanism','panorama','assemblyset','holistic','compositional'):
                    from .assembly_taxonomy import annotate
                    full=annotate(full,world)
                if hasattr(geometry,'annotate_state'):full=geometry.annotate_state(full,layout)
                validate_world(observed,full,solved['items'],len(observed['objects'])+model.get('maximum_hidden_instances',24))
                write(dest/'G_full.json',full);hidden=deepcopy(full);hidden.update(scope='hidden',objects=deepcopy(full['objects'][len(observed['objects']):]))
                write(dest/'G_out.json',hidden)
                write(dest/'object_relations.json',dict(declared_relations=world['relations'],
                    attachments=[dict(id=o['id'],**o['attachment']) for o in world['objects']],
                    presence_parents={n['id']:n['parents'] for n in model['nodes'] if n['id'] in {o['presence_node'] for o in world['objects']}},
                    installation_parents={n['id']:n['parents'] for n in model['nodes'] if n['id'] in {o['installation_node'] for o in world['objects']}}))
                record=dict(folder=dest.name,new_instances=len(world['objects']),counts_kind='placement_roots_and_hosted',
                    root_instances=sum(o['attachment']['host'] is None for o in world['objects']),
                    hosted_instances=sum(o['attachment']['host'] is not None for o in world['objects']),unmapped_new=len(unmapped),
                    observed_state_preserved=True,remaining_AABB_overlap_pairs=len(solved['review']['program_solver']['final_overlaps']),
                    status='candidate_needs_semantic_review' if solved['review']['layout_proxy_passed'] else 'layout_needs_review')
                visual=dest/'world';visual.mkdir()
                if full['objects']:
                    render(full,original,case['hfov'],visual,record,position_note='条件结构场景程序；原观测固定；支撑与安装均为相机Y水平假设。')
                    p=visual/'index.html';p.write_text(p.read_text().replace('红色候选保留在G_full_candidates中，不进入Adapter子集G_full。','未映射对象完整保留在G_full，本阶段不连接Adapter。'))
                else:
                    erp_map(full,case['hfov'],case['image_size'],visual/'erp_state_map.png')
                    (visual/'index.html').write_text('<meta charset="utf-8"><p>空候选保留。</p><img src="erp_state_map.png">')
                summary['worlds'].append(record)
            except Exception as exc:
                summary['failures'].append(dict(world=index,stage=stage,proposed_instances=len(world['objects']),error=type(exc).__name__+': '+str(exc)))
                (dest/'error_traceback.txt').write_text(traceback.format_exc())
        summary.update(status='worlds_exported_needs_review' if summary['worlds'] and not summary['failures'] else
            'partial_layouts_needs_review' if summary['worlds'] else 'no_world_exported',world_state_ready=bool(summary['worlds']))
        if model.get('candidate_support',{}).get('all_nonempty_proposals_missing'):
            summary.update(status='empty_candidate_domain_no_completion',world_state_ready=False,
                completion_outcome='observed_only_no_nonempty_proposal_support',
                error='All fixed-budget proposals were empty. This does not establish that the hidden world is empty.')
    except Exception as exc:
        summary.update(status='failed',failed_stage=stage,error=type(exc).__name__+': '+str(exc))
        (out/'error_traceback.txt').write_text(traceback.format_exc())
    summary.update(api_calls=getattr(provider,'calls',None),api_usage=getattr(provider,'usage',None))
    write(out/'summary.json',summary);page(out,summary);return summary
