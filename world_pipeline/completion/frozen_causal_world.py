"""User-run, training-free off-camera world hypotheses and visualization.

Frozen Qwen2D/WildDet3D observation reuse; no Adapter, GT scene or new training.
The first causal prior is deliberately coarse Z -> E, Z -> H, not a learned
physical SCM. Candidate-domain selection and uncalibrated mechanisms are explicit.
"""
import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path
import shutil
import traceback

from PIL import Image

from ..common import FRAME, new_output, read, sha, validate_graph, write
from .batch_world_review import inventory_image
from .causal_world_completion import check_hashes, load_pilot, visible_inventory
from .frozen_causal_core import (FrozenChoiceScorer, build_model, infer, normalize_proposal)
from .frozen_causal_layout import solve_layout
from .off_fov_completion import validate_world
from .qwen_upstream import DEFAULT_MODEL, Qwen
from .scene_completion import export_states, layout_prompt, map_plan
from .semantic_layout_visual import render
from .semantic_probe import state_context

PROTOCOL = 'frozen_causal_off_fov_world_v1'


def code_hashes():
    from .off_fov_completion import code_hashes as previous_hashes
    paths = [Path(__file__), *[Path(__file__).with_name(n) for n in
             ('frozen_causal_core.py', 'frozen_causal_layout.py', 'frozen_causal_audit.py',
              'causal_probability.py', 'region_projection.py')]]
    return {**previous_hashes(), **{str(p.resolve()): sha(p) for p in paths}}


def proposal_prompt(case, context_budget, world_budget):
    return ('Predict possible EXISTING worlds around the SAME camera from this partial indoor photograph. '
        'Only propose distinct instances OUTSIDE the INPUT CAMERA VIEW, not occluded objects inside '
        'the cone, visible objects omitted by detection, or completion of already observed instances. '
        'Keep every observed instance fixed. This is inference, not redesign or minimum functional '
        'room equipment. Optional storage, ornaments and extra furniture are allowed; do not require '
        'a missing function. There is no object-count or direction quota. '
        'Return ONLY JSON {"contexts":[{"description":...,"reason":...}],'
        '"worlds":[{"description":...,"additions":[{"id":"n1","name":...,'
        '"object_type":...,"reason":...,"placement":...}],"relations":[]} ]}. '
        f'Propose 1..{context_budget} distinct hypotheses of pre-configuration intended use/preferences '
        f'and 1..{world_budget} DIFFERENT full hidden-inventory alternatives. These are proposal budgets, '
        'not limits on how many objects a world may contain. Include plausible alternative explanations '
        'rather than only one needed accessory. Every world is an entire surrounding inventory hypothesis, '
        'not one independent fragment to concatenate with the others. No metric geometry yet. '
        'Placements refer to unphotographed surroundings, not empty visible walls or visible table tops. '
        'Object_type is a specific ordinary noun with no Adapter class restriction. IDs are unique '
        'inside each world and different from observed IDs. Each optional spatial relation has '
        'source,target (instance IDs),kind,reason; kinds near,left_of,right_of,above,below,'
        'in_front_of,behind,supported_by,on_top_of,inside. near requires positive max_distance_m. '
        'Directions mean CAMERA axis centre order (x right, y image-up, z optical forward). '
        'Furniture-facing front is unknown; do not equate it with camera z. Support/gravity/room bounds '
        'are unknown. Relations are soft spatial hypotheses, NOT causal edges. '
        f'Input horizontal FOV={case["hfov"]:g} degrees. '
        'A program will add empty and unresolved alternatives if needed; do not duplicate inventory '
        'multisets merely with a different placement or wording.\nOBSERVED: '+
        json.dumps(visible_inventory(case), ensure_ascii=False))


def ask_validated(model, image, prompt, path, validator, options):
    """One explicit schema repair; preserve raw attempts rather than fuzzy ID guessing."""
    base = prompt
    for attempt in range(2):
        target = path if attempt == 0 else path.with_name(path.stem+'_repair.txt')
        try:
            answer = model.ask(image, prompt, target, **{**options, 'seed': options['seed']+attempt})
            return validator(answer)
        except ValueError as exc:
            write(target.with_suffix('.validation.json'), dict(error=str(exc)))
            if attempt:
                raise
            raw = target.read_text() if target.exists() else '(No complete answer produced.)'
            prompt = (base+'\nSCHEMA/CONTRACT ERROR: '+str(exc)+
                '\nReturn a complete corrected JSON answer; do not change observed objects. '
                'The previous answer was:\n'+raw)


def layout_question(case, plan):
    return layout_prompt(case, plan)+(
        '\nEvery new full AABB must lie OUTSIDE the input cone. Choose reasonable metric geometry '
        'around the SAME camera, including side/back space; negative camera z is valid. '
        'This is not placement on visible surfaces. Give complete box dimensions, never shrink '
        'objects or move them arbitrarily far to hide them. A separate solver can adjust centres '
        'without a fixed movement radius; its translations will be disclosed for review. '
        'Do not output movement bounds, category IDs, visibility flags or off_fov_region. '
        'Return exactly {"objects":[{"id":...,"center":[x,y,z],"size":[sx,sy,sz],"reason":...}]}.')


def load_case(case_file=None):
    if case_file is None:
        case = load_pilot()
    else:
        # For other previously observed validation inputs. Contains RGB and
        # predicted-state paths only; no dataset reference/GT target is followed.
        source = Path(case_file).resolve(); value = read(source)
        if not isinstance(value, dict) or not {'id', 'image', 'state', 'hfov'} <= value.keys():
            raise ValueError('Case JSON needs id,image,state,hfov')
        if not isinstance(value['id'], str) or not value['id'] or Path(value['id']).name != value['id'] or value['id'] in ('.', '..'):
            raise ValueError('Case ID must be a safe directory name')
        state_path = Path(value['state']).resolve(); graph = validate_graph(read(state_path))
        if graph.get('stage') != 'prediction' or graph['scope'] != 'observed':
            raise ValueError('Need predicted observed state, not GT or completed world')
        image_path = Path(value['image']).resolve()
        case = dict(id=value['id'], image=str(image_path), state=str(state_path),
            hfov=value['hfov'], context=state_context(graph),
            inventory=str(Path(value.get('inventory', state_path)).resolve()),
            source_sha256={str(p): sha(p) for p in (source, state_path, image_path)})
        case['source_sha256'][case['inventory']] = sha(case['inventory'])
        case['observer_provenance'] = value.get('observer_provenance', 'user_supplied_predicted_state_not_reverified')
    with Image.open(case['image']) as image:
        case['image_size'] = list(image.size)
    if type(case['hfov']) not in (int, float) or not 0 < case['hfov'] < 180:
        raise ValueError('Need centred-pinhole horizontal FOV in (0,180)')
    state = validate_graph(read(case['state']))
    if not state['objects'] or any(not o[k]['valid'] for o in state['objects']
            for k in ('center_local_m', 'size_aabb_local_m')):
        raise ValueError('Need nonempty frozen observations with complete finite AABB geometry')
    return case


def model_diagram(spec, trace, output):
    # Offline SVG; all descriptions/metadata stay in JSON beside it.
    prior = next(n for n in spec['nodes'] if n['id'] == 'Z')['cpt'][0]['probabilities']
    posterior = trace['posterior_context']['probabilities']
    lines = [f'{z}: prior={prior[z]:.3f}, conditioned={posterior[z]:.3f}' for z in prior]
    height = max(330,245+22*len(lines))
    body = f'<svg xmlns="http://www.w3.org/2000/svg" width="950" height="{height}"><rect width="100%" height="100%" fill="white"/>'
    body += '<text x="20" y="28">Assumed common-cause model: Z → E and Z → H; not discovered physical causality</text>'
    for x, y, label, link in ((360, 60, 'Z: intended configuration','mechanisms/context_prior.json'),
                             (80, 150, 'E: positive inventory event','causal_model.json'),
                             (590, 150, 'H: hidden inventory bundle','inference_trace.json')):
        body += f'<a href="{link}"><rect x="{x}" y="{y}" width="270" height="45" rx="8" fill="#e4edf5"/><text x="{x+8}" y="{y+28}">{escape(label)}</text></a>'
    body += '<path d="M420 105 L215 150 M560 105 L725 150" stroke="black" fill="none"/>'
    for i, line in enumerate(lines):
        body += f'<text x="25" y="{225+22*i}">{escape(line)}</text>'
    Path(output).write_text(body+'</svg>')


def write_page(folder, record, hypotheses):
    body = ('<!doctype html><meta charset="utf-8"><title>Frozen causal world hypotheses</title>'
        '<style>body{max-width:1400px;margin:24px auto;font:16px/1.6 sans-serif}img{max-width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>局部观测 → 多个视野外世界假设</h1><p>冻结模型，无新增训练。概率仅属于候选域内的假设模型；'
        '语义、物理和因果增益尚未验证。2:1方向图不是生成的全景照片。</p>'
        '<img width="700" src="local.png"><pre>'+escape(json.dumps(record, ensure_ascii=False, indent=2))+'</pre>')
    if (folder/'qwen_2d.png').exists(): body += '<h2>冻结二维观测</h2><img src="qwen_2d.png">'
    if (folder/'wilddet_projection.png').exists(): body += '<h2>WildDet3D观测重投影</h2><img src="wilddet_projection.png">'
    if (folder/'causal_graph.svg').exists():
        body += '<h2>实际执行的机制图</h2><a href="causal_graph.svg">打开可点击机制图与对应参数</a><br><img src="causal_graph.svg">'
    for h in hypotheses:
        base = h['folder']
        body += '<h2>'+escape(h['id'])+'</h2><pre>'+escape(json.dumps(h, ensure_ascii=False, indent=2))+'</pre>'
        if (folder/base/'world/index.html').exists():
            body += f'<a href="{base}/world/index.html">打开此假设的交互3D、俯视图和数值字段</a><br><img src="{base}/world/erp_state_map.png">'
    for p in sorted(folder.glob('*.json')): body += '<p><a href="'+p.name+'">'+p.name+'</a></p>'
    (folder/'index.html').write_text(body)


def run(model, scorer, case, output, worlds=3, context_budget=3, world_budget=6,
        seed=3407, max_tokens=8192, capacity=256, sweeps=3):
    folder = output/case['id']; folder.mkdir()
    hypotheses = []
    record = dict(id=case['id'], protocol=PROTOCOL, status='failed', world_state_ready=False,
        observed_instances=len(case['context']), semantics_verified=False, geometry_verified=False,
        causal_gain_validated=False, training_performed=False, scene_gt_used=False,
        adapter_connected=False, panorama_generated=False)
    options = dict(seed=seed, sample=True, profile='official', max_tokens=max_tokens, guard_repetition=True)
    try:
        check_hashes(case['source_sha256'])
        observed = validate_graph(read(case['state']))
        write(folder/'input.json', case); write(folder/'G_obs.json', observed)
        shutil.copyfile(case['image'], folder/'local.png')
        inventory_image(case['image'], read(case['inventory']), folder/'qwen_2d.png')
        projection = Path(case['state']).with_name('projection_boxes.png')
        if projection.is_file(): shutil.copyfile(projection, folder/'wilddet_projection.png')
        with Image.open(case['image']) as handle: image = handle.convert('RGB')
        inference_image = image.copy(); inference_image.thumbnail((768, 768))
        write(folder/'inference_image.json', dict(original_size=list(image.size),
            model_size=list(inference_image.size), geometry_uses_original_aspect=True))
        domain = ask_validated(model, inference_image, proposal_prompt(case, context_budget, world_budget), folder/'proposal.txt',
            lambda value: normalize_proposal(value, case, capacity, context_budget, world_budget), options)
        write(folder/'proposal_domain.json', domain)
        scores = folder/'mechanisms'; scores.mkdir()
        # Evidence is extracted once from the fixed inventory. RGB and a
        # separate image-based room posterior are not multiplied into it.
        positive = [o['description'] for o in case['context']]
        def save_score(name, row):
            write(scores/(name+'.json'), row)
            print(json.dumps(dict(stage='mechanism_scored', name=name,
                                  probabilities=row['probabilities']), ensure_ascii=False), flush=True)
        spec = build_model(domain, positive, scorer, save_score)
        write(folder/'causal_model.json', spec)
        trace = infer(spec); write(folder/'inference_trace.json', trace)
        model_diagram(spec, trace, folder/'causal_graph.svg')
        posterior = trace['posterior_hidden']['probabilities']
        candidates = sorted(domain['worlds'], key=lambda w: (-posterior[w['id']], w['id']))[:worlds]
        selected_mass = sum(posterior[w['id']] for w in candidates)
        write(folder/'world_selection.json', dict(selected=[w['id'] for w in candidates],
            selected_model_mass=selected_mass, omitted_model_mass=max(0.,1-selected_mass),
            unresolved_world_mass=posterior['w_other'],
            selection='top_known_inventory_bundles_without_renormalization_or_geometry_cherry_picking'))
        for rank, plan in enumerate(candidates):
            print(json.dumps(dict(stage='layout_started',rank=rank+1,world=plan['id'],
                                  new_instances=len(plan['additions'])), ensure_ascii=False),flush=True)
            dest = folder/f'hypothesis_{rank+1:02d}'; dest.mkdir()
            h = dict(id=plan['id'], folder=dest.name, probability_in_declared_model=posterior[plan['id']],
                     proposed_instances=len(plan['additions']), status='failed', world_state_ready=False)
            try:
                write(dest/'semantic_plan.json', plan)
                mappings = map_plan(plan, 'semantic_lexicon_v1'); write(dest/'category_mapping.json', mappings)
                # No hidden identity/size deletion or resizing in the solver.
                if plan['additions']:
                    items, review = ask_validated(model, inference_image, layout_question(case, plan), dest/'boxes.txt',
                        lambda value: solve_layout(value, plan, mappings, observed, case['hfov'], list(image.size), sweeps),
                        {**options, 'seed': seed+100+rank*10})
                else:
                    items, review = solve_layout({'objects': []}, plan, mappings, observed,
                                                 case['hfov'], list(image.size), sweeps)
                write(dest/'layout_solved.json', dict(objects=items)); write(dest/'numeric_review.json', review)
                full, _, unmapped = export_states(observed, plan, items, mappings, review['after'])
                full.update(completion_protocol=PROTOCOL, adapter_connected=False,
                    input_camera=dict(model='centred_pinhole', hfov_deg=case['hfov'], image_size_wh=case['image_size']),
                    target='outside_input_frustum', causal_model_identified=False,
                    world_relations=deepcopy(plan['relations']), probability_in_declared_model=posterior[plan['id']],
                    semantic_plan_fully_exported=True, unmapped_candidate_ids=[o['track_id'] for o in unmapped],
                    geometry_method='frozen_Qwen_initialization_then_disclosed_off_frustum_solver')
                for obj, item in zip(full['objects'][len(observed['objects']):], items):
                    obj.update(geometry_proposal_center_m=item['proposal_center'],
                        geometry_translation_m=item['translation_m'],
                        off_fov_certificate_region=item['off_fov_certificate_region'])
                    obj['center_local_m'].update(source='geometric_hypothesis',
                        method='frozen_Qwen_initialization_then_disclosed_off_frustum_solver')
                validate_world(observed, full, items, capacity)
                h.update(status='numeric_candidate_needs_semantic_review' if items else 'empty_hidden_hypothesis_needs_review',
                    world_state_ready=review['after']['numeric_constraints_passed'],
                    new_instances=len(items), unmapped_new=len(unmapped), observed_state_preserved=True,
                    numeric_constraints_passed=review['after']['numeric_constraints_passed'],
                    soft_relation_violations=review['after']['soft_relation_violations'],
                    aabb_overlap_warnings=len(review['after']['overlaps']))
                full['quality_status'] = h['status']; full['world_state_ready'] = h['world_state_ready']
                write(dest/'G_full.json', full)
                hidden = deepcopy(full); hidden['scope'] = 'hidden'; hidden['objects'] = full['objects'][len(observed['objects']):]
                validate_graph(hidden); write(dest/'G_out.json', hidden)
                write(dest/'unmapped_candidates.json', unmapped)
                visual = dest/'world'; visual.mkdir()
                render(full, image, case['hfov'], visual, h,
                    position_note='新增框为视野外假设；几何位移已记录。蓝色为固定观测。')
                page = visual/'index.html'; page.write_text(page.read_text().replace(
                    '红色候选保留在G_full_candidates中，不进入Adapter子集G_full。',
                    '红色只表示尚无粗类别映射，全部保留在G_full；本阶段不连接Adapter。'))
            except Exception as exc:
                h.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                (dest/'error_traceback.txt').write_text(traceback.format_exc())
            write(dest/'summary.json', h); hypotheses.append(h)
        ready = [h for h in hypotheses if h['world_state_ready']]
        failed = [h for h in hypotheses if h['status'] == 'failed']
        record.update(status='partial_failure' if failed else 'hypotheses_ready_needs_review',
            world_state_ready=bool(ready), hypotheses_exported=len(ready), hypothesis_failures=len(failed),
            nonempty_hypotheses=sum(h.get('new_instances', 0)>0 for h in ready),
            unresolved_world_mass=posterior['w_other'], selected_model_mass=selected_mass,
            unresolved_alternative_is_most_likely=posterior['w_other'] >= max(posterior[w['id']] for w in domain['worlds']),
            observed_state_preserved=True, hypotheses=hypotheses,
            probability_scope=domain['probability_scope'])
        if not failed and record['nonempty_hypotheses'] == 0:
            record['status'] = 'no_hidden_prediction_needs_review'
        # The convenience export is strictly the highest-posterior known world,
        # never a lower-ranked replacement chosen because geometry looked better.
        if hypotheses and hypotheses[0]['world_state_ready']:
            for name in ('G_full.json', 'G_out.json'):
                shutil.copyfile(folder/hypotheses[0]['folder']/name, folder/name)
            record['preferred_world'] = hypotheses[0]['id']
        check_hashes(case['source_sha256'])
    except Exception as exc:
        record.update(status='failed', world_state_ready=False, error=f'{type(exc).__name__}: {exc}')
        (folder/'error_traceback.txt').write_text(traceback.format_exc())
    write(folder/'summary.json', record); write_page(folder, record, hypotheses)
    write(output/'summary.json', dict(protocol=PROTOCOL, records=[record], scene_gt_used=False))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><a href="'+case['id']+'/index.html">打开世界补全与机制诊断</a>')
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case', type=Path, help='Optional JSON: id,image,state,hfov; centred pinhole and predicted G_obs only')
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--worlds', type=int, default=3, help='Top known inventory hypotheses to lay out; not an object-count quota')
    p.add_argument('--context-budget', type=int, default=3)
    p.add_argument('--proposal-world-budget', type=int, default=6)
    p.add_argument('--max-objects', type=int, default=256)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--sweeps', type=int, default=3)
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    if (a.output.exists() or not 1 <= a.context_budget <= 6 or not 1 <= a.proposal_world_budget <= 12 or
            not 1 <= a.worlds <= a.proposal_world_budget+1 or not 1 <= a.max_objects <= 256 or
            not 256 <= a.max_tokens <= 16384 or not 0 <= a.sweeps <= 10 or not 0 <= a.seed < 2**63):
        p.error('Need a NEW output and valid finite inference budgets')
    case = load_case(a.case)
    if len(case['context']) > a.max_objects: p.error('Observed state exceeds capacity')
    if not a.model.is_dir(): p.error('Local model missing; automatic downloads disabled')
    config = dict(protocol=PROTOCOL, case=case, frame=FRAME, arguments={k: str(v) if isinstance(v, Path) else v
                  for k, v in vars(a).items()}, code_sha256=code_hashes(), training_performed=False,
                  scene_gt_used=False, adapter_connected=False)
    if a.preflight:
        print(json.dumps(dict(status='preflight_passed', id=case['id'], observed=len(case['context']),
            model_loaded=False, gpu_started=False, training_performed=False, scene_gt_used=False,
            adapter_connected=False, inference_graph='Z->E,Z->H', probability_scope='restricted_uncalibrated_semantic_prior'),
            ensure_ascii=False)); return
    output = new_output(a.output); write(output/'config.json', config)
    try:
        model = Qwen(a.model)
    except Exception as exc:
        write(output/'summary.json',dict(protocol=PROTOCOL,status='model_load_failed',error=f'{type(exc).__name__}: {exc}'))
        (output/'error_traceback.txt').write_text(traceback.format_exc())
        raise
    if code_hashes() != config['code_sha256']: raise ValueError('Code changed during model loading')
    record = run(model, FrozenChoiceScorer(model), case, output, a.worlds, a.context_budget,
                 a.proposal_world_budget, a.seed, a.max_tokens, a.max_objects, a.sweeps)
    if record['status'] in ('failed', 'partial_failure') or not record['world_state_ready']:
        raise SystemExit('Inspect NEW output summary/error_traceback and raw replies; no artifacts overwritten.')


if __name__ == '__main__':
    main()
