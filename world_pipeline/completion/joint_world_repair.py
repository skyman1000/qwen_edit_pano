"""Explicit migration of frozen observations/proposals and verified metric caches.

Source artifacts are read only. New acceptance rules and available-proposal
conditioning are recorded in a fresh run; only missing metric worlds use Qwen.
"""
from copy import deepcopy
import json
from pathlib import Path

from ..common import read,write,sha
from . import joint_world_model as semantic,joint_world_geometry as geometry
from .assembly_organized_geometry import dimensions_answer


def signature(world):
    return json.dumps({k:world[k] for k in ('hypothesis','configuration','objects','relations')},sort_keys=True,ensure_ascii=False)


def prepare_repair(root,source):
    from . import joint_world as runner
    source=Path(source).resolve();root=Path(root).resolve()
    if source==root or source in root.parents:raise ValueError('Repair output must be separate from its source')
    suite=runner.adapter.check_run(source)
    # The old joint entry changed intentionally. The adapter verifies all frozen
    # observation bindings/core sources; selected historical artifacts are pinned
    # below rather than overwriting their old code binding to bypass checking.
    for scene in suite['scenes']:
        p=source/'scenes'/scene['id']/'joint_completion'
        for name in ('domain.json','empirical_tables.json'):
            if not (p/name).is_file():raise ValueError('Source has no reusable '+str(p/name))
    result=runner.prepare(root,source,suite['worlds'],suite['draws'],suite['seed'])
    bindings={}
    for scene in suite['scenes']:
        p=source/'scenes'/scene['id']/'joint_completion'
        files=[p/'domain.json',p/'empirical_tables.json']
        files += [f for f in p.glob('world_*/*.json') if f.name in ('semantic_world.json','dimension_stage.json','continuous_placement.json','support_plane.json')]
        for f in files:bindings[str(f.relative_to(source))]=sha(f)
        dest=root/'scenes'/scene['id']/'joint_completion';dest.mkdir(exist_ok=True)
        d=read(p/'domain.json');tables=read(p/'empirical_tables.json')
        write(dest/'domain.json',d);write(dest/'hypotheses.json',d['hypotheses']);write(dest/'empirical_tables.json',tables)
        failures=d.get('proposal_failures',[])
        write(root/'phase_records'/(scene['id']+'_joint_propose.json'),dict(scene=scene['id'],phase='propose',
            status='complete_with_warnings' if failures else 'complete',proposal_failures=failures,
            source='verified_saved_semantic_proposals',model_called=False))
        if semantic.has_valid_program(d):
            raw=semantic.compile_model(d['hypotheses'],d['programs'],tables)
            model,coverage=semantic.available_proposal_model(raw)
            write(dest/'causal_model_unconditioned.json',raw);write(dest/'causal_model.json',model);write(dest/'proposal_coverage.json',coverage)
            record=dict(scene=scene['id'],phase='mechanisms',status='complete_with_warnings' if coverage['excluded_reference_mass']>1e-9 else 'complete',
                proposal_coverage=coverage,source='same_saved_H_weights_conditioned_on_available_semantic_proposals',model_called=False)
        else:record=dict(scene=scene['id'],phase='mechanisms',status='blocked_proposal_domain',model_called=False)
        write(root/'phase_records'/(scene['id']+'_joint_mechanisms.json'),record)
    write(root/'repair_source_bindings.json',dict(source=str(source),files=bindings))
    meta=read(root/'joint_world_run.json');meta.update(repair_source=str(source),
        repair_source_bindings_sha256=sha(root/'repair_source_bindings.json'),
        repair_scope='same_observations_and_semantic_proposals_new_approximate_geometry_and_available_support_conditioning')
    write(root/'joint_world_run.json',meta);runner.check(root)
    adapter_meta=read(root/'local_world_run.json');adapter_meta['old_candidates_reused']=True
    write(root/'local_world_run.json',adapter_meta)
    return dict(result,status='prepared_repair_no_model_calls',source=str(source),cached_metrics_verified_on_use=True)


def cached_geometry(out,world,index):
    root=out.parents[2];meta_path=root/'joint_world_run.json'
    if not meta_path.is_file():return None
    meta=read(meta_path)
    if not meta.get('repair_source'):return None
    manifest=root/'repair_source_bindings.json'
    if sha(manifest)!=meta['repair_source_bindings_sha256']:raise ValueError('Repair source bindings changed')
    bindings=read(manifest);source=Path(bindings['source']);files=bindings['files']
    def checked(path):
        key=str(path.relative_to(source))
        if key not in files or sha(path)!=files[key]:raise ValueError('Saved metric artifact changed: '+str(path))
        return read(path)
    p=source/'scenes'/out.parent.name/'joint_completion'
    candidates=sorted(p.glob('world_*/semantic_world.json'),key=lambda f:(f.parent.name!=f'world_{index:02d}',str(f)))
    for candidate in candidates:
        prior=checked(candidate)
        if signature(prior)!=signature(world):continue
        folder=candidate.parent
        names=('dimension_stage.json','support_plane.json','continuous_placement.json')
        if any(not (folder/name).is_file() for name in names):continue
        dim,plane,raw=[checked(folder/name) for name in names]
        dims=dimensions_answer(dict(objects=list(dim['dimensions'].values())),world)
        geometry.placements_answer(dict(placements=list(raw['poses'].values())),world['objects'])
        case_observation=root/'scenes'/out.parent.name/'observation/qwen/observer3d/G_obs.json'
        obs=read(case_observation)
        if any(o['attachment']['kind']=='floor' for o in world['objects']) and plane.get('floor_y_m') is None:continue
        if plane.get('floor_y_m') is not None:
            if not semantic.finite(plane['floor_y_m']) or plane['floor_y_m']>=0:continue
            if plane.get('anchor_ids'):
                computed=geometry.floor_answer(dict(anchor_ids=plane['anchor_ids'],floor_y_m=None,basis=plane['basis']),obs)
                if abs(computed['floor_y_m']-plane['floor_y_m'])>1e-8:raise ValueError('Saved floor differs from its frozen anchors')
        expected={o['group'] for o in world['objects']}
        if set(raw['origins'])!=expected:continue
        for gid,origin in raw['origins'].items():
            geometry.origins_answer(dict(groups=[dict(id=gid,center_xz_m=origin['center_xz_m'])]),world,90,[gid])
        return dims,plane,deepcopy(raw),str(folder)
    return None
