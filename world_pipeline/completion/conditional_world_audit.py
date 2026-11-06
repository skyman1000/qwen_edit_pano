"""Read-only CPU contract audit; does not certify semantic or causal validity."""
import argparse
import json
import math
from pathlib import Path

from PIL import Image

from ..common import read
from .causal_probability import CausalGraph
from .causal_world_completion import check_hashes
from .conditional_world_core import aggregate_rotations
from .conditional_world_geometry import compile_layout
from .off_fov_completion import validate_world


def equal_probs(a,b):
    return set(a)==set(b) and all(math.isclose(a[k],b[k],abs_tol=1e-8,rel_tol=1e-7) for k in a)


def audit(root):
    root=Path(root);config=read(root/'config.json');case=config['case'];folder=root/case['id']
    check_hashes(case['source_sha256']);check_hashes(config['code_sha256'])
    observed=read(folder/'G_obs.json')
    if observed!=read(case['state']):raise ValueError('Frozen observations changed')
    spec=read(folder/'conditional_model.json');model=CausalGraph(spec);trace=read(folder/'inference_trace.json')
    if not equal_probs(model.query('H')['probabilities'],trace['world_mixture']['probabilities']):
        raise ValueError('Conditional mixture differs on recomputation')
    checks=[('context_given_observations',model.nodes['Z']['cpt'][0]['probabilities'])]
    checks.append(('direct_inventory_given_observations',trace['direct_inventory_baseline']))
    checks += [('inventory_given_'+r['when'][0],r['probabilities']) for r in model.nodes['H']['cpt']]
    for name,probs in checks:
        saved=read(folder/'mechanisms'/(name+'.json'))
        if not equal_probs(saved['probabilities'],probs):raise ValueError('CPT/mechanism binding differs: '+name)
        if saved.get('rounds'):
            result=aggregate_rotations(saved['rounds'],list(probs))
            if not equal_probs(result['probabilities'],probs):raise ValueError('Balanced score aggregation differs: '+name)
    catalog=read(folder/'world_catalog.json')['worlds'];selection=read(folder/'selection.json')
    weights=model.query('H')['probabilities']
    expected=sorted(catalog,key=lambda w:(-weights[w['id']],w['id']))[:config['arguments']['worlds']]
    if selection['selected']!=[w['id'] for w in expected]:raise ValueError('Selection/ranking changed')
    summary=read(folder/'summary.json');rows=[]
    if summary['status'] in ('failed','partial_failure'):raise ValueError('Run has explicit failures; inspect traceback')
    if len(summary['hypotheses'])!=len(expected):raise ValueError('Selected worlds missing')
    for h,plan in zip(summary['hypotheses'],expected):
        dest=folder/h['folder'];savedplan=read(dest/'semantic_plan.json')
        if savedplan!=plan:raise ValueError('Plan identity changed after ranking')
        full=read(dest/'G_full.json');items=read(dest/'layout_solved.json')['objects']
        validate_world(observed,full,items,config['arguments'].get('max_objects',256))
        if read(dest/'G_out.json')['objects']!=full['objects'][len(observed['objects']):]:raise ValueError('Hidden export differs')
        attempts=read(dest/'layout_attempts.json');attempt=attempts['selected_attempt']
        if plan['additions']:
            raw=dest/f'bearing_{attempt}_repair.json'
            if not raw.exists():raw=dest/f'bearing_{attempt}.json'
            recomputed,review=compile_layout(read(raw),plan,read(dest/'category_mapping.json'),observed,
                read(folder/'support_plane.json'),case['hfov'],case['image_size'])
            for a,b in zip(items,recomputed):
                if a['id']!=b['id'] or a['size']!=b['size'] or any(abs(x-y)>1e-8 for x,y in zip(a['center'],b['center'])):
                    raise ValueError('Bearing/support compilation differs')
            if h['world_state_ready']!=review['geometry_contract_passed']:raise ValueError('Readiness flag differs')
        if not (dest/'world/scene_3d.html').is_file():raise ValueError('Missing 3D view')
        with Image.open(dest/'world/erp_state_map.png') as image:
            if image.width!=2*image.height:raise ValueError('State chart must be 2:1')
        rows.append(dict(id=h['id'],new_instances=len(items),world_state_ready=h['world_state_ready'],
                         off_fov_passed=h['off_fov_passed'],position_was_projected=False))
    return dict(status='postflight_contract_passed',hypotheses=rows,source_hashes_unchanged=True,
        conditional_model_recomputed=True,geometry_recomputed=True,
        semantic_quality_verified=False,causal_gain_validated=False,gpu_started=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    try:result=audit(a.output)
    except Exception as exc:
        print(json.dumps(dict(status='postflight_failed',error=f'{type(exc).__name__}: {exc}'),ensure_ascii=False));raise SystemExit(1)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
