"""CPU audit: source/input hashes, raw proposals, mechanism bindings, finite
inference, fixed observations, exports and selected geometric compilations.
Does not certify semantic quality or real causal effects.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from PIL import Image

from ..common import read,sha,validate_graph
from .causal_world_completion import check_hashes,visible_inventory
from .conditional_world_geometry import support_plane
from .geometry import parse_json
from .scene_program import enumerate_assignments,parameterize,program_answer,registry_answer
from .scene_program_scores import verify_scores
from .scene_program_world import (intervention_reports,solve_assignment,summarize_candidates)
from .scm_world_audit import close


def bind_raw(folder,stem,validator,expected):
    candidates=[folder/(stem+'.txt'),folder/(stem+'_repair.txt')]
    for path in candidates[::-1]:
        if not path.exists():continue
        try:normalized=validator(parse_json(path.read_text()))
        except (ValueError,KeyError,TypeError):continue
        if close(normalized,expected):return
    raise ValueError('No preserved raw proposal binds to '+stem)


def audit(output,allow_fixture=False):
    output=Path(output);config=read(output/'config.json');case=config['case'];folder=output/case['id']
    record=read(folder/'summary.json');fixture=record['execution_kind']=='cpu_fixture_not_model_prediction'
    if fixture and not allow_fixture:raise ValueError('CPU fixture is not a model prediction; use --allow-fixture explicitly')
    if not record['world_state_ready']:raise ValueError('No numeric world export; inspect original failure')
    check_hashes(config['code_sha256']);check_hashes(case['source_sha256'])
    manifest=read(output/'artifact_manifest.json')
    actual={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    if actual!=manifest:raise ValueError('Output artifact manifest changed')
    observed=validate_graph(read(case['state']))
    if observed!=read(folder/'G_obs.json'):raise ValueError('Frozen observation changed')
    budgets=config['budgets'];registry=read(folder/'registry.json')
    if not fixture:bind_raw(folder,'registry',lambda v:registry_answer(v,case,budgets['configurations']),registry)
    plane=support_plane(observed,[r['id'] for r in registry['instances'] if r['on_floor']])
    if not close(plane,read(folder/'support_plane.json')):raise ValueError('Floor prior changed')
    inverse=read(folder/'configuration_scores.json')
    if not fixture:
        numerical=deepcopy(inverse);numerical['probabilities']={str(i):inverse['probabilities'][c['id']] for i,c in enumerate(registry['configurations'])}
        verify_scores(numerical)
        if inverse['alternatives']!=registry['configurations'] or inverse['parents']!=dict(observed=visible_inventory(case)):
            raise ValueError('Inverse configuration evidence/catalog changed')
    candidates=[read(p) for p in sorted((folder/'candidates').glob('candidate_*.json'))]
    programs={};tables={};failures={r['configuration'] for r in record['configuration_failures']}
    for c in registry['configurations']:
        cid=c['id'];dest=folder/cid
        if cid in failures:continue
        program=read(dest/'program.json');programs[cid]=program;table=read(dest/'tables.json');tables[cid]=table
        if not fixture:
            bind_raw(dest,'program',lambda v:program_answer(v,case,c,budgets['groups'],budgets['modes'],budgets['max_new']),program)
            class Reader:
                def __init__(self):self.index=0
                def score(self,description,parents,states):
                    # Match each saved mechanism to the exact loop row, including
                    # its parent assignment and complete joint-mode domain.
                    all_records=sorted((dest/'mechanisms').glob('*.json'))
                    matched=[]
                    for path in all_records:
                        r=read(path);base=deepcopy(parents);choices=base.pop('alternatives')
                        alternatives=[dict(description=o['joint_recipe']['description'],objects=o['joint_recipe']['objects']) for o in choices]
                        if r['field']==description and r['parents']==base and r['alternatives']==alternatives:matched.append(r)
                    if len(matched)!=1:raise ValueError('Missing/ambiguous conditional mechanism binding')
                    r=matched[0]
                    if r['image_used']:raise ValueError('Group mechanism unexpectedly received image')
                    verify_scores(r);return r
            recomputed=parameterize(program,registry,Reader(),lambda *args:None)
            if not close(recomputed,table):raise ValueError('Conditional tables differ from saved event scores')
        expected=enumerate_assignments(program,table);local=[r for r in candidates if r['configuration']==cid]
        if len(local)!=len(expected):raise ValueError('Missing complete program assignments')
        for a,row in zip(expected,local):
            if a['modes']!=row['modes'] or not math.isclose(a['probability'],row['conditional_mode_probability'],abs_tol=1e-10):
                raise ValueError('Finite assignment differs')
            mass=inverse['probabilities'][cid]*a['probability']
            if not math.isclose(mass,row['model_mass'],abs_tol=1e-10):raise ValueError('Joint conditional mass differs')
        if not close(intervention_reports(program,table,local),read(dest/'mechanism_interventions.json')):
            raise ValueError('Mechanism intervention replay differs')
    ranked,total=summarize_candidates(candidates);inference=read(folder/'inference.json')
    if not math.isclose(total,inference['feasible_model_mass'],abs_tol=1e-10):raise ValueError('Feasible mass differs')
    expected_new=sum(w['conditional_model_weight']*len(w['representative']['solved']['items']) for w in ranked)
    empty=sum(w['conditional_model_weight'] for w in ranked if not w['representative']['solved']['items'])
    if not math.isclose(expected_new,inference['expected_new_instances'],abs_tol=1e-10) or not math.isclose(empty,inference['empty_world_conditional_mass'],abs_tol=1e-10):
        raise ValueError('Aggregate world statistics differ')
    for exported,w in zip(record['worlds'],ranked):
        dest=folder/exported['folder'];row=w['representative'];cid=row['configuration']
        replay=solve_assignment(case,observed,plane,programs[cid],row['modes'],config['search'])
        if not close(replay,row['solved']):raise ValueError('Selected joint geometry replay differs')
        full=validate_graph(read(dest/'G_full.json'));hidden=validate_graph(read(dest/'G_out.json'))
        n=len(observed['objects'])
        if full['objects'][:n]!=observed['objects'] or hidden['objects']!=full['objects'][n:]:raise ValueError('Observation/export mismatch')
        if len(hidden['objects'])!=len(replay['items']):raise ValueError('An instantiated object was dropped')
        for obj,item in zip(hidden['objects'],replay['items']):
            if obj['track_id']!=item['id'] or obj['size_aabb_local_m']['value']!=item['size'] or not close(obj['center_local_m']['value'],item['center']):
                raise ValueError('Exported geometry differs')
        if not math.isclose(full['conditional_model_weight'],w['conditional_model_weight'],abs_tol=1e-10):raise ValueError('Exported model weight differs')
        with Image.open(dest/'world/erp_state_map.png') as image:
            if image.width!=2*image.height:raise ValueError('State direction map must be 2:1')
    if len(record['worlds'])!=min(config['arguments']['worlds'],len(ranked)):raise ValueError('Missing ranked exports')
    for name in ('G_full.json','G_out.json'):
        if read(folder/name)!=read(folder/record['representative_world']/name):raise ValueError('Representative alias differs')
    return dict(status='scene_program_contract_passed',execution_kind=record['execution_kind'],
        worlds=[r['new_instances'] for r in record['worlds']],assignments=len(candidates),
        observed_state_preserved=True,source_hashes_unchanged=True,mechanisms_replayed=True,
        selected_geometry_replayed=True,semantic_quality_verified=False,causal_gain_validated=False,
        exact_RGB_posterior=False,gpu_started=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--allow-fixture',action='store_true');a=p.parse_args()
    print(json.dumps(audit(a.output,a.allow_fixture),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
