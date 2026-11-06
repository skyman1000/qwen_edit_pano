"""Independent validation pilot: hypothesized mechanisms -> contents -> coarse boxes.

User launches all inference. CPU preflight verifies a frozen real validation case.
No Observer/Adapter changes, GT scene loading, asset generation or automatic jobs.
"""
import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path
import shutil
import textwrap

import numpy as np
from PIL import Image

from ..common import ROOT, new_output, read, rows, sha, validate_graph, write
from .batch_world_review import inventory_image
from .category_bridge import MAPPING
from .causal_contract import (diagnostics, execute, typed_graph, validate_candidates,
                              validate_factors)
from .geometry import CANONICAL_CATEGORIES, parse_json
from .qwen_upstream import DEFAULT_MODEL, Qwen
from .scene_completion import export_states, layout_prompt, map_plan, review_boxes, validate_layout
from .semantic_layout_visual import render
from .semantic_probe import load_cases

PROTOCOL = 'hypothesized_mechanisms_hidden_world_v2'
LEGACY_PROTOCOL = 'hypothesized_mechanisms_hidden_world_v1'
# Exact audited v1 implementation, not an option to bypass arbitrary code changes.
LEGACY_CORE_SHA256 = {
    'causal_world_completion.py': 'c065e081a35273cc366bd28ee1f9a1f873b1bb68e17f6601271352d3a4ae5992',
    'causal_contract.py': 'aa2e21c946d96f0e6d7031107a0243eb38a78b20a19043147ce11bb2db7ac65b',
}
CASE_ID = 'X7HyMhZNoso_d0468b580cd146778bae5e486bf3e50e'
MANIFEST = ROOT/'research/audits/validation_hidden_world_plan_2026-10-22/evidence_manifest.json'
VOCABULARY = ROOT/'qwen_edit_pano/data/gt_world_supplement_v1/vocabulary.json'


def check_hashes(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError('Bound source changed: '+str(path))


def load_pilot():
    manifest = read(MANIFEST)
    if manifest['case_id'] != CASE_ID or manifest['split_verified_from_source_manifest'] != 'val':
        raise ValueError('Unexpected pilot manifest')
    check_hashes(manifest['source_sha256'])
    bundle = ROOT/'qwen_edit_pano/data/box_world_val6_v1'
    selection = read(bundle/'selection.json')
    if selection['split'] != 'val' or sha(selection['source_manifest']) != selection['source_manifest_sha256']:
        raise ValueError('Validation membership provenance changed')
    # Membership metadata only. Never follow GT-graph or target-ERP paths.
    members = [r for r in rows(selection['source_manifest']) if r['id'] == CASE_ID]
    if len(members) != 1 or members[0]['split'] != 'val':
        raise ValueError('Pilot is not a unique validation member')
    entry = next(r for r in rows(bundle/'inputs.jsonl') if r['id'] == CASE_ID)
    state_path = Path(manifest['observed_state'])
    case = load_cases(state_path.parents[2], [CASE_ID])[0]
    case.pop('prompt')
    config = read(state_path.with_name('config.json'))
    if (config['protocol'] != 'qwen_2d_wilddet3d_geometric_v1'
            or config['source'] != manifest['inventory_state']
            or sha(config['source']) != config['source_sha256']
            or case['image'] != entry['local_rgb'] or case['hfov'] != entry['hfov_deg']
            or entry['split'] != 'val' or sha(case['image']) != entry['image_sha256']):
        raise ValueError('Qwen 2D -> WildDet3D -> RGB/FOV provenance mismatch')
    observed, inventory = validate_graph(read(case['state'])), validate_graph(read(config['source']))
    if len(observed['objects']) != len(inventory['objects']) or len(observed['objects']) != manifest['objects']:
        raise ValueError('Frozen observed instance count changed')
    for a, b in zip(observed['objects'], inventory['objects']):
        if any(a.get(k) != b.get(k) for k in ('track_id', 'description', 'category_id', 'bbox2d_visible_xyxy_norm')):
            raise ValueError('WildDet3D changed the bound Qwen 2D identity')
        if not a['center_local_m']['valid'] or not a['size_aabb_local_m']['valid']:
            raise ValueError('Pilot requires finite frozen observed geometry')
    vocabulary = read(VOCABULARY)
    if ({int(k): v for k, v in vocabulary['categories'].items()} != CANONICAL_CATEGORIES
            or vocabulary['category_mapping_sha256'] != sha(MAPPING)):
        raise ValueError('Frozen Adapter category contract changed')
    case.update(inventory=str(Path(config['source']).resolve()), split_verified='val',
                vocabulary=str(VOCABULARY), supported_coarse_categories=len(CANONICAL_CATEGORIES))
    case['source_sha256'].update(manifest['source_sha256'])
    for path in (MANIFEST, VOCABULARY, Path(selection['source_manifest'])):
        case['source_sha256'][str(path.resolve())] = sha(path)
    return case


def visible_inventory(case):
    # Isolate object semantics from noisy estimated depth and arbitrary category IDs.
    return [{k: o[k] for k in ('id', 'category', 'description', 'visible_bbox_xyxy_norm')}
            for o in case['context']]


def context_prompt(case):
    return (
        'Infer ONE plausible surrounding indoor world from this partial photograph. '
        'This is hidden-world inference, not minimum functional room design. '
        'Describe a coherent scene use and a small set of relevant hypotheses about activities, '
        'occupant preferences, environmental configuration or context. Personal objects, '
        'decoration, storage and additional instances of an already present class are possible; '
        'no function must be missing before a hidden object can exist. Unknown space is unknown. '
        'All supplied observations stay fixed; they are approximate predictions. The image can '
        'contain objects omitted from the inventory. A missing entry is not evidence of invisibility. '
        'Separate evidence from assumptions in reason. Do not assert hidden conditions as observed facts. '
        'Do not enumerate new objects, dimensions or coordinates yet. No category menu. '
        'Return ONLY JSON with scene_type, scene_hypothesis (strings), factors (array). '
        'Each factor has id (h1, h2, ...), kind (context/activity/preference/environment), '
        'statement, assumed (boolean gate chosen for this world), evidence_ids (observed IDs or []), '
        'reason and optional parents (factor IDs, default []). '
        'Only use parents for a hypothesized enabling dependence, not for statistical co-occurrence '
        'or because one item is evidence for another. The program evaluates factor = assumed AND '
        'all(parent factors); roots equal assumed. Factors must form a DAG. This is a proposed '
        'Boolean hypothesis model, not discovered causality. Empty factors are allowed if unsupported. '
        f'The photograph covers {case["hfov"]:g} horizontal degrees. '
        '\nOBSERVED INVENTORY: '+json.dumps(visible_inventory(case), ensure_ascii=False))


def contents_prompt(case, factors):
    # Examples use real factor IDs, avoiding an invented h3 in a two-factor case.
    example_grounds = [[f['id'] for f in factors['factors'][:2]]]
    return (
        'Continue the SAME partially photographed indoor environment into ONE plausible hidden world. '
        'Use the attached RGB, fixed inventory and hypothesized factors below. Infer distinct physical '
        'object instances that could reasonably exist outside the photograph or be genuinely occluded. '
        'This is not the smallest functional room, a missing-needs checklist or a request to redesign '
        'visible furniture. Existing functions do not prohibit additional instances, books, personal '
        'objects or decoration. Choose number freely, including zero; no direction/count quota. '
        'Do not repeat visible instances, their components, or visible fixtures omitted by the detector. '
        'Use ordinary specific names with no Adapter category restriction. No numeric geometry. '
        'Return ONLY JSON {"additions":[...],"relations":[...]}. '
        'Each addition has id (new1, new2, ...), name, object_type (ordinary noun), reason, '
        'placement (qualitative location and visibility explanation), grounds and optional requires. '
        'grounds is an OR-list of AND-lists of factor IDs. Each clause requires all its factors; '
        'different clauses are independent alternatives. Only use factor IDs listed below. '
        'A syntactic example using available IDs is '+json.dumps(example_grounds)+'. '
        '[[]] means an unconditional scene-prior hypothesis. '
        'Use independent alternative grounds when appropriate; do not invent necessity for every object. '
        'requires defaults to []; it lists observed/new instance IDs whose existence this particular '
        'hypothesis really requires. It is not a list of nearby objects or ordinary co-occurrences. '
        'The program selects an instance iff some grounds clause holds and all required instances '
        'are present; instance requirements must be acyclic. Selected objects are hypotheses, not facts. '
        'Each optional relation has source, target (instance IDs), kind and reason. Allowed kinds: '
        'left_of/right_of/above/below/in_front_of/behind/near/supported_by/on_top_of/inside. '
        'Directions are relative to this CAMERA (x right, y image-up, z forward); supported_by/inside '
        'go from object to support/container. in_front_of means source has SMALLER camera z than '
        'target; behind means LARGER camera z, not the phrase behind the camera (negative z). '
        'These are camera-axis centre orders, not viewing-ray distance comparisons for every ERP '
        'direction. These are spatial hypotheses, not causal edges. '
        'near may have optional positive max_distance_m (centre distance), an assumed threshold. '
        'Support/containment placement dependencies between new instances must be acyclic. '
        'Keep claimed supports/containers among observed or explicitly proposed instances; include '
        'required supporting new instances in requires if their existence is essential to that hypothesis. '
        'Otherwise leave unsupported relations out. Floor, gravity and room bounds are unmeasured. '
        '\nOBSERVED INVENTORY: '+json.dumps(visible_inventory(case), ensure_ascii=False)
        +'\nHYPOTHESIZED FACTORS: '+json.dumps(factors, ensure_ascii=False))


def selected_plan(factors, execution):
    return dict(scene_type=factors['scene_type'], scene_hypothesis=factors['scene_hypothesis'],
                additions=deepcopy(execution['selected_additions']),
                relations=deepcopy(execution['relations']),
                hypothesized_factors=deepcopy(factors['factors']),
                factor_values=deepcopy(execution['factor_values']),
                inactive_relations_for_review=deepcopy(execution['inactive_relations']),
                placement_dependency_order=execution['placement_order'])


def boxes_prompt(case, plan):
    # Shared ordinary geometric backend; the typed constraints are explicit input.
    question = layout_prompt(case, plan)
    # Do not force all hypotheses outside FOV if a genuine occlusion was proposed.
    question = question.replace('outside the photograph, anywhere around the same camera.',
                                'compatible with the photograph, anywhere around the same camera.')
    question = question.replace('"reason":...}]}', '"reason":...,"unseen_reason":...}]}')
    return (question+'\nAdd unseen_reason to each object: outside_fov, occluded, or uncertain. '
        'outside_fov means the full box should not intersect the input camera cone. Occluded '
        'requires an explicit plausible occluder in reason; box overlap alone does not establish '
        'occlusion. Do not place objects over visible empty areas. Image-up directions are camera '
        'directions, not measured physical height. Honour fixed selected relations and support '
        'dependency order. in_front_of means smaller camera z; behind means larger camera z. '
        'factor_values gives the executed active assumptions; assumed gates '
        'alone do not determine child values. Inactive relation endpoints are NOT additional '
        'scene instances. Do not introduce or rely on them as hidden supports or occluders. '
        'Explain any required change of placement that referred to an inactive object. '
        'These are hypotheses, not solved constraints. Return only id, center, '
        'size, reason, unseen_reason per box. Do not change selection, identity or category.')


def boxes_answer(answer, plan, mappings):
    if not isinstance(answer, dict) or set(answer) != {'objects'} or not isinstance(answer['objects'], list):
        raise ValueError('Layout must contain exactly objects array')
    stripped, reasons = [], {}
    for obj in answer['objects']:
        if not isinstance(obj, dict) or set(obj) != {'id', 'center', 'size', 'reason', 'unseen_reason'}:
            raise ValueError('Layout box needs exactly id, center, size, reason, unseen_reason')
        if obj['unseen_reason'] not in ('outside_fov', 'occluded', 'uncertain'):
            raise ValueError('Invalid unseen_reason')
        reasons[obj['id']] = obj['unseen_reason']
        stripped.append({k: v for k, v in obj.items() if k != 'unseen_reason'})
    items = validate_layout(dict(objects=stripped), plan, mappings)
    for item in items:
        item['unseen_reason'] = reasons[item['id']]
    return items


def relation_checks(relations, observed, items):
    boxes = {o['track_id']: (np.array(o['center_local_m']['value'], dtype=float),
                            np.array(o['size_aabb_local_m']['value'], dtype=float))
             for o in observed['objects']}
    boxes.update({o['id']: (np.array(o['center'], dtype=float), np.array(o['size'], dtype=float)) for o in items})
    results = []
    axis_sign = {'left_of': (0, -1), 'right_of': (0, 1), 'above': (1, 1),
                 'below': (1, -1), 'in_front_of': (2, -1), 'behind': (2, 1)}
    for r in relations:
        a, sa = boxes[r['source']]; b, sb = boxes[r['target']]
        kind, passed, interpretation = r['kind'], None, 'unverified'
        if kind in axis_sign:
            axis, sign = axis_sign[kind]
            passed = bool(sign*(a[axis]-b[axis]) > 1e-6)
            interpretation = 'camera_axis_centre_order_only'
        elif kind == 'inside':
            passed = bool(np.all(a-sa/2 >= b-sb/2-1e-6) and np.all(a+sa/2 <= b+sb/2+1e-6))
            interpretation = 'AABB_containment_proxy_not_shape_containment'
        elif kind == 'near' and 'max_distance_m' in r:
            passed = bool(np.linalg.norm(a-b) <= r['max_distance_m'])
            interpretation = 'centre_distance_under_hypothesized_threshold'
        elif kind in ('supported_by', 'on_top_of'):
            interpretation = 'unverified_no_measured_gravity_support_planes_or_shapes'
        results.append(dict(**r, proxy_passed=passed, interpretation=interpretation,
                            physical_relation_verified=False))
    return results


def code_hashes():
    names = ('causal_world_completion.py', 'causal_contract.py', 'scene_completion.py',
             'semantic_probe.py', 'semantic_layout.py', 'semantic_layout_visual.py',
             'semantic_categories.py', 'category_bridge.py', 'geometry.py', 'qwen_upstream.py',
             'qwen_upstream_core.py',
             'json_generation.py', 'audit_semantic_layout.py', 'batch_world_review.py', 'box_layout.py')
    paths = [Path(__file__).with_name(n) for n in names]+[Path(__file__).parents[1]/'common.py']
    return {str(p.resolve()): sha(p) for p in paths}


def load_semantics(source, case):
    """Reuse raw-bound same-input output, including explicitly audited v1 migration.

    v2 adds on_top_of without rewriting it into a weaker predicate. Original
    model JSON and requests remain untouched even if the v1 run failed validation.
    """
    folder = source/CASE_ID
    config = read(source/'config.json')
    current_hashes = code_hashes()
    legacy_hashes = {**current_hashes, **{str(Path(__file__).with_name(n).resolve()): h
                                        for n, h in LEGACY_CORE_SHA256.items()}}
    legacy = config['protocol'] == LEGACY_PROTOCOL and config['code_sha256'] == legacy_hashes
    current = config['protocol'] == PROTOCOL and config['code_sha256'] == current_hashes
    if config['case'] != case or not (legacy or current):
        raise ValueError('Semantic source protocol/input/code binding differs')
    if read(folder/'input.json') != case:
        raise ValueError('Saved semantic input differs')
    request_protocol = config.get('semantic_request_protocol', config['protocol'])
    if request_protocol not in (LEGACY_PROTOCOL, PROTOCOL) or legacy and request_protocol != LEGACY_PROTOCOL:
        raise ValueError('Unsupported semantic request protocol')
    paths = [source/'config.json', folder/'input.json']
    values = []
    for name, prompt in [('context', context_prompt(case)), ('contents', None)]:
        raw, parsed, request = [folder/(name+suffix) for suffix in ('.txt', '.json', '.request.json')]
        value = read(parsed)
        if parse_json(raw.read_text()) != value:
            raise ValueError('Semantic parsed answer differs from original raw model response')
        expected_prompt = prompt if prompt is not None else contents_prompt(case, values[0])
        if request_protocol == LEGACY_PROTOCOL and name == 'contents':
            expected_prompt = expected_prompt.replace('/supported_by/on_top_of/inside.', '/supported_by/inside.')
        if read(request)['prompt'] != expected_prompt:
            raise ValueError('Saved semantic prompt differs')
        generation = read(folder/(name+'.generation.json'))
        if generation.get('finish_reason') in ('length', 'repeated_objects'):
            raise ValueError('Saved semantic generation did not finish normally: '+name)
        values.append(validate_factors(value, [o['id'] for o in case['context']]) if name == 'context'
                      else validate_candidates(value, values[0], [o['id'] for o in case['context']]))
        paths += [raw, parsed, request, folder/(name+'.generation.json')]
    return (*values, {str(p.resolve()): sha(p) for p in paths})


def graph_svg(graph, output):
    """Standalone SVG from actual generated nodes; titles contain edge semantics."""
    groups = [[n for n in graph['nodes'] if n['type'] == 'observed_instance'],
              [n for n in graph['nodes'] if n['type'] not in ('observed_instance', 'hypothesized_instance')],
              [n for n in graph['nodes'] if n['type'] == 'hypothesized_instance']]
    height = max(260, max(map(len, groups))*116+80)
    positions, elements = {}, []
    for column, nodes in enumerate(groups):
        for row, n in enumerate(nodes):
            positions[n['id']] = (24+column*470, 65+row*116)
    for e in graph['edges']:
        x, y = positions[e['source']]; xx, yy = positions[e['target']]
        spatial = e['type'].startswith('spatial_')
        colour = '#8c919a' if spatial else '#7356b8' if 'selection' in e['type'] else '#4f7d9d'
        elements.append(f'<path d="M {x+380} {y+38} C {x+440} {y+38}, {xx-35} {yy+38}, {xx} {yy+38}" '
                        f'fill="none" stroke="{colour}" stroke-width="1.5" marker-end="url(#arrow)" '
                        f'class="{"spatial" if spatial else "mechanism"}"><title>'
                        +escape(json.dumps(e, ensure_ascii=False))+'</title></path>')
    for nodes in groups:
        for n in nodes:
            x, y = positions[n['id']]
            colour = '#ddecf8' if n['type'] == 'observed_instance' else '#fff0d2'
            opacity = '1' if n['selected'] else '.4'
            elements.append(f'<g opacity="{opacity}"><title>'+escape(json.dumps(n, ensure_ascii=False))+'</title>'
                            +f'<rect x="{x}" y="{y}" width="380" height="88" rx="8" fill="{colour}" stroke="#64748b"/>')
            label = n['id']+' | '+n.get('label', n['type'])
            for i, line in enumerate(textwrap.wrap(label, 43)[:3]):
                elements.append(f'<text x="{x+9}" y="{y+21+21*i}" font-size="14">'+escape(line)+'</text>')
            elements.append('</g>')
    output.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{height}" viewBox="0 0 1400 {height}">'
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
        'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#737b87"/></marker></defs>'
        '<rect width="100%" height="100%" fill="white"/><text x="24" y="25" font-size="18">'
        'Observed (fixed)                         Hypothesis factors                         Candidate instances</text>'
        +''.join(elements)+'</svg>')


def report_html(folder, record):
    links = ['context.json', 'contents.json', 'mechanism_execution.json', 'selector_diagnostics.json',
             'typed_graph.json', 'category_review_all_candidates.json', 'summary.json']
    html = ('<!doctype html><meta charset="utf-8"><title>CauPano hidden-world pilot</title>'
        '<style>body{font:16px/1.6 sans-serif;margin:24px auto;max-width:1400px;padding:16px}'
        'img{max-width:100%}iframe{width:100%;height:660px;border:1px solid #ddd}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere}.note{background:#fff0d5;padding:12px}</style>'
        '<h1>真实验证图：固定观测 → 假设机制 → 隐藏对象 → 三维状态</h1>'
        '<p class="note">这是待审核的世界假设。因果图未经识别；布尔规则的敏感性不等于真实因果效应。'
        '方向图不是生成的 ERP 照片，未调用 World Adapter 或全景模型。</p><pre>'
        +escape(json.dumps(record, ensure_ascii=False, indent=2))+'</pre>'
        '<h2>原局部 RGB 与已运行的 Qwen 二维框</h2><img width="620" src="local.png">'
        '<img width="620" src="qwen_2d.png"><h2>已运行的 WildDet3D 三维重投影</h2>'
        '<img width="800" src="wilddet_projection.png">')
    if (folder/'typed_graph.svg').exists():
        html += ('<h2>模型实际输出的类型化假设图</h2><p>蓝色为固定观测，黄色为假设；淡色表示未激活。'
                 '灰色为空间边；证据边不是因果边。悬停查看边与节点信息。</p>'
                 '<object style="width:100%;min-height:450px" data="typed_graph.svg" type="image/svg+xml"></object>')
    if (folder/'world/scene_3d.html').exists():
        links += ['G_full_semantic.json', 'G_full_adapter_proposal.json', 'numeric_review.json', 'category_coverage.json']
        html += ('<h2>保留全部已选对象的世界状态</h2><p>蓝色观测／橙色新增／红色未映射。'
                 '相机 y 为图像上方向，未标定重力、地板或墙体。'
                 '<a href="world/scene_3d.html">单独打开交互三维框</a></p>'
                 '<iframe src="world/scene_3d.html"></iframe><h2>2:1 ERP 方向图（不是 RGB）</h2>'
                 '<img src="world/erp_state_map.png"><h2>相机 x/z 投影</h2><img src="world/layout_xz.png">'
                 '<h2>补全状态在输入视图的投影</h2><img src="world/local_projection.png">'
                 '<h2>Adapter 可表达的类别子集</h2><p>此视图只包含可映射对象。'
                 '完整世界与未映射对象仍保存在 G_full_semantic.json。</p>'
                 '<a href="adapter/scene_3d.html">打开 Adapter 子集三维图</a>')
    html += '<h2>实际输出与审计</h2>'+''.join('<p><a href="'+p+'">'+p+'</a></p>' for p in links if (folder/p).exists())
    for name in ('context.json', 'semantic_plan.json', 'category_coverage.json'):
        if (folder/name).exists():
            html += '<details><summary>'+name+'</summary><pre>'+escape((folder/name).read_text())+'</pre></details>'
    (folder/'index.html').write_text(html)


def run(model, case, output, stage='full', semantic_source=None, seed=3407, max_tokens=4096,
        max_objects=160, semantic_source_hashes=None):
    folder = output/case['id']; folder.mkdir()
    record = dict(id=case['id'], protocol=PROTOCOL, status='failed', semantics_verified=False,
                  geometry_verified=False, causal_gain_validated=False, panorama_generated=False,
                  observed_instances=len(case['context']), geometry_generated=False)
    try:
        check_hashes(case['source_sha256'])
        write(folder/'input.json', case)
        observed = validate_graph(read(case['state']))
        write(folder/'G_obs.json', observed)
        shutil.copyfile(case['image'], folder/'local.png')
        inventory_image(case['image'], read(case['inventory']), folder/'qwen_2d.png')
        shutil.copyfile(Path(case['state']).with_name('projection_boxes.png'), folder/'wilddet_projection.png')
        with Image.open(case['image']) as handle:
            image = handle.convert('RGB')
        image.thumbnail((768, 768))  # Same aspect/FOV; exact inference resolution is recorded.
        write(folder/'inference_image.json', dict(size=list(image.size), hfov=case['hfov'],
                                                 full_original_copy='local.png'))
        ids = [o['id'] for o in case['context']]
        options = dict(seed=seed, sample=True, profile='official', max_tokens=max_tokens, guard_repetition=True)
        if semantic_source is None:
            factors = validate_factors(model.ask(image, context_prompt(case), folder/'context.txt', **options), ids)
            contents = validate_candidates(model.ask(image, contents_prompt(case, factors), folder/'contents.txt', **options), factors, ids)
        else:
            factors, contents, source_hashes = load_semantics(semantic_source, case)
            if semantic_source_hashes is not None and source_hashes != semantic_source_hashes:
                raise ValueError('Semantic source changed after preflight')
            write(folder/'semantic_source.json', source_hashes)
            write(folder/'semantic_source_binding.json', dict(
                source_protocol=read(semantic_source/'config.json')['protocol'], target_protocol=PROTOCOL,
                original_raw_and_parsed_answers_preserved=True,
                migration='explicit_audited_v1_predicate_extension' if
                    read(semantic_source/'config.json')['protocol'] == LEGACY_PROTOCOL else 'same_code',
                predicate_rewriting=False))
            for stem in ('context', 'contents'):
                for suffix in ('.txt', '.json', '.request.json', '.generation.json'):
                    shutil.copyfile(semantic_source/case['id']/(stem+suffix), folder/(stem+suffix))
        # Keep normalized validated forms separately; never replace raw parsed model outputs.
        write(folder/'hypothesis_factors.json', factors); write(folder/'candidate_contents.json', contents)
        factual = execute(factors, contents, ids)
        plan = selected_plan(factors, factual)
        write(folder/'mechanism_execution.json', factual)
        write(folder/'semantic_plan.json', plan)
        write(folder/'selector_diagnostics.json', diagnostics(factors, contents, ids, factual))
        graph = typed_graph(factors, contents, ids, factual)
        write(folder/'typed_graph.json', graph); graph_svg(graph, folder/'typed_graph.svg')
        all_plan = dict(additions=contents['additions'])
        all_mappings = map_plan(all_plan, 'semantic_lexicon_v1')
        write(folder/'category_review_all_candidates.json', all_mappings)
        mappings = [m for m in all_mappings if m['id'] in factual['selected_ids']]
        record.update(status='semantic_hypothesis_saved_needs_review',
                      proposed_instances=len(contents['additions']), selected_instances=len(plan['additions']),
                      inactive_instances=len(contents['additions'])-len(plan['additions']),
                      inactive_relations=len(factual['inactive_relations']))
        if stage == 'full':
            answer = (model.ask(image, boxes_prompt(case, plan), folder/'boxes.txt', **options)
                      if plan['additions'] else dict(objects=[]))
            write(folder/'layout_proposal.json', answer)
            items = boxes_answer(answer, plan, mappings)
            checks = review_boxes(items, observed, case['hfov'], image.size)
            checks['relations'] = relation_checks(plan['relations'], observed, items)
            for item, check in zip(items, checks['objects']):
                check['claimed_unseen_reason'] = item['unseen_reason']
                check['visibility_verified'] = False
            checks['failed_relation_proxies'] = sum(r['proxy_passed'] is False for r in checks['relations'])
            checks['unverified_relations'] = sum(r['proxy_passed'] is None for r in checks['relations'])
            checks['inactive_relation_endpoints_require_review'] = factual['inactive_relations']
            write(folder/'numeric_review.json', checks)
            semantic, adapter, unmapped = export_states(observed, plan, items, mappings, checks)
            mapped_count = len(plan['additions'])-len(unmapped)
            capacity_ok = len(adapter['objects']) <= max_objects
            identity_loss = [m['id'] for m in mappings if m.get('identity_loss_review')]
            coverage = dict(selected_new=len(items), mapped_new=mapped_count, unmapped_new=len(unmapped),
                            mapped_new_fraction=mapped_count/len(items) if items else None,
                            identity_loss_review_ids=identity_loss,
                            unsupported_ids=[o['track_id'] for o in unmapped],
                            adapter_capacity=max_objects, adapter_objects=len(adapter['objects']),
                            capacity_ok=capacity_ok, semantic_selection_uses_category_mapping=False,
                            fine_identity_control_verified=False,
                            note='Generic/coarse category mapping does not encode the fine object name.')
            for state in (semantic, adapter):
                present = {o['track_id'] for o in state['objects']}
                kept_relations = [deepcopy(r) for r in plan['relations']
                                  if {r['source'], r['target']} <= present]
                state.update(completion_protocol=PROTOCOL, typed_hypothesis_graph=graph,
                             causal_model_identified=False, category_coverage=coverage,
                             world_relations=kept_relations,
                             relations_omitted_by_category_projection=[deepcopy(r) for r in plan['relations']
                                                                      if {r['source'], r['target']} - present],
                             state_view='complete_selected_hypothesis' if state is semantic else 'adapter_category_projection',
                             typed_graph_scope='all_candidates_with_explicit_selection_flags',
                             quality_status='proposal_requires_semantic_spatial_and_category_review')
                if state['objects'][:len(observed['objects'])] != observed['objects']:
                    raise ValueError('Observed prefix changed')
                validate_graph(state)
            # A graph can describe the open-vocabulary world but fail Adapter input coverage.
            write(folder/'G_full_semantic.json', semantic)
            write(folder/'G_full_adapter_proposal.json', adapter)
            write(folder/'unmapped_candidates.json', unmapped)
            write(folder/'category_coverage.json', coverage)
            record.update(status='world_proposal_saved_needs_review' if items else 'no_selected_additions',
                          geometry_generated=bool(items), observed_state_preserved=True,
                          numeric_warning_objects=checks['objects_with_numeric_warnings'],
                          failed_relation_proxies=checks['failed_relation_proxies'],
                          unverified_relations=checks['unverified_relations'], category_coverage=coverage,
                          adapter_input_approved=False)
            for dirname, state in [('world', semantic), ('adapter', adapter)]:
                target = folder/dirname; target.mkdir()
                render(state, image, case['hfov'], target, record,
                       position_note='新增对象与位置均为待审核的假设。此实验尚未证明因果推理收益。')
                # Existing renderer names its old export files; use this pilot's actual names.
                p = target/'index.html'
                p.write_text(p.read_text().replace('G_full_candidates', 'G_full_semantic.json')
                             .replace('子集G_full', '子集 G_full_adapter_proposal.json'))
        check_hashes(case['source_sha256'])
        if semantic_source is not None:
            check_hashes(source_hashes)
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    write(folder/'summary.json', record)
    report_html(folder, record)
    write(output/'summary.json', dict(protocol=PROTOCOL, records=[record], scene_gt_used=False,
                                      model_quality_validated=False))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>CauPano pilot</h1><pre>'
        +escape(json.dumps(record, ensure_ascii=False, indent=2))+'</pre><a href="'+case['id']+'/index.html">完整可视化与诊断</a>')
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record['status'] != 'failed'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=('semantic', 'full'), default='full')
    p.add_argument('--semantic-from', type=Path, help='Reuse this protocol\'s raw-bound semantic output')
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--max-objects', type=int, default=160)
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    if not 0 <= a.seed < 2**63 or not 1 <= a.max_objects <= 160 or a.max_tokens < 1:
        p.error('Invalid seed, token budget or Adapter capacity')
    case = load_pilot()
    if len(case['context']) > a.max_objects:
        p.error('Fixed observations alone exceed the selected Adapter capacity')
    if not a.model.is_dir():
        p.error('Local model snapshot not found; no downloads are performed')
    destination = a.output.resolve()
    roots = [ROOT/'qwen_edit_pano/data', ROOT/'qwen_edit_pano/outputs', ROOT/'research/audits']
    if not any(destination.is_relative_to(r) and destination != r for r in roots):
        p.error('Output must be a new directory under project data, outputs or research/audits')
    if a.output.exists():
        p.error('Choose a NEW output directory; existing results are never overwritten')
    source = a.semantic_from.resolve() if a.semantic_from else None
    if source:
        _, _, provenance = load_semantics(source, case)
        # Keep provenance outside case: exact evidence equality must survive semantic -> layout.
    source_config = read(source/'config.json') if source else None
    config = dict(protocol=PROTOCOL, case=case, stage=a.stage, model=str(a.model.resolve()),
                  seed=a.seed, max_tokens=a.max_tokens, max_objects=a.max_objects,
                  semantic_source=str(source) if source else None,
                  semantic_request_protocol=(source_config.get('semantic_request_protocol', source_config['protocol'])
                                             if source_config else PROTOCOL),
                  semantic_source_sha256=provenance if source else None,
                  code_sha256=code_hashes(), scene_gt_used=False)
    if a.preflight:
        print(json.dumps(dict(status='preflight_passed', case_id=case['id'], split=case['split_verified'],
                              observed_instances=len(case['context']), coarse_classes=len(CANONICAL_CATEGORIES),
                              semantic_calls=0 if source else 2, maximum_layout_calls=int(a.stage == 'full'),
                              model_loaded=False, gpu_started=False, scene_gt_used=False), ensure_ascii=False))
        return
    output = new_output(a.output)
    write(output/'config.json', config)
    # Bind the reused source one more time before costly model construction.
    if source:
        check_hashes(provenance)
    # Recovery/inspection of already generated semantics needs no weights or GPU.
    model = None if source and a.stage == 'semantic' else Qwen(a.model)
    if not run(model, case, output, a.stage, source, a.seed, a.max_tokens, a.max_objects,
               provenance if source else None):
        raise SystemExit('Pilot failed; inspect case summary.json and original raw answers. Use a NEW output for retry.')


if __name__ == '__main__':
    main()
