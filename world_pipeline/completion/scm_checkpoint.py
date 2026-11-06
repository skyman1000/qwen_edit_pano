"""Validated semantic-stage reuse and honest, read-only model diagnostics."""
from copy import deepcopy
from pathlib import Path
import shutil

from ..common import read,sha,write
from .causal_world_completion import check_hashes
from .scm_count_model import infer,trace


SEMANTIC_CODE={'scm_count_model.py','scm_mechanisms.py','scm_typed_queries.py',
               'conditional_world_core.py','qwen_upstream.py','common.py','conditional_world_geometry.py'}


def validate_checkpoint(root,case,max_count,generic_types,mechanism_scoring=None):
    from .scm_world_audit import audit_semantics
    root=Path(root);config=read(root/'config.json')
    hashes={p:h for p,h in config['code_sha256'].items() if Path(p).name in SEMANTIC_CODE}
    if {Path(p).name for p in hashes}!=SEMANTIC_CODE:raise ValueError('Checkpoint lacks semantic implementation provenance')
    check_hashes(hashes)
    state=audit_semantics(root,check_all_code=False);old=state['case'];args=state['args']
    if args.get('mechanism_scoring')=='numeric':
        extra={p:h for p,h in config['code_sha256'].items() if Path(p).name=='scm_numeric_mechanisms.py'}
        if len(extra)!=1:raise ValueError('Numeric checkpoint lacks scorer provenance')
        check_hashes(extra);hashes.update(extra)
        if state['spec'].get('count_event_contract')!='json_integer_v1':raise ValueError('Numeric count event contract differs')
    if (args.get('mechanism_scoring') not in ('continuation','numeric')
        or (mechanism_scoring is not None and args.get('mechanism_scoring')!=mechanism_scoring)
        or args['max_count']!=max_count or args['generic_types']!=generic_types
        or case['id']!=old['id'] or case['context']!=old['context'] or case['hfov']!=old['hfov']
        or case['image_size']!=old['image_size'] or sha(Path(case['image']))!=sha(Path(old['image']))
        or read(case['state'])!=state['observed']):
        raise ValueError('Checkpoint belongs to different observation/domain/scoring settings')
    state['semantic_code_sha256']=hashes
    return state


def copy_checkpoint(root,case,dest,max_count,generic_types):
    state=validate_checkpoint(root,case,max_count,generic_types);source=state['folder'];files=[]
    for file in source.iterdir():
        if file.is_file() and (file.name.startswith(('domain.','domain_repair.','registration.','registration_repair.'))
            or file.name in ('semantic_model.json','score_diagnostics.json','support_plane.json')):
            files.append(file)
    files+=sorted((source/'mechanisms').glob('*.json'))
    copied={}
    for file in files:
        rel=file.relative_to(source);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise ValueError('Refuse to overwrite an existing checkpoint destination')
        shutil.copyfile(file,target);copied[str(rel)]=sha(target)
    write(dest/'reused_semantics.json',dict(source=str(Path(root).resolve()),source_files_sha256={str(f.resolve()):sha(f) for f in files},
        copied_files_sha256=copied,semantic_code_sha256=state['semantic_code_sha256'],
        posterior_recomputed=True,geometry_or_layout_reused=False))
    return state


def audit_reuse(folder):
    file=folder/'reused_semantics.json'
    if not file.exists():return
    value=read(file);check_hashes(value['source_files_sha256']);check_hashes(value['semantic_code_sha256'])
    for rel,digest in value['copied_files_sha256'].items():
        if sha(folder/rel)!=digest:raise ValueError('Copied semantic checkpoint changed: '+rel)


def reasoning_report(spec,result):
    posterior=trace(result);alternative=deepcopy(spec)
    alternative['mechanisms']['B']=[1/len(spec['backgrounds'])]*len(spec['backgrounds'])
    sensitivity=trace(infer(alternative,result['observed']))
    return dict(actual_graph=spec['graph'],registered_counts=result['observed'],
        count_event_contract=spec.get('count_event_contract','sentence_v1'),
        inference='P(B,K,N,F,D | recorded category counts), then sample outside instances',
        generative_parameters='frozen parent-only LM finite text mechanisms; not empirical room frequencies',
        RGB_used_in_count_likelihood=False,centers_and_sizes_used_in_count_likelihood=False,
        activities_or_functional_groups_explicitly_modeled=False,
        support_graph_role='conditional layout dependencies; not object-existence causality',
        camera_pose_likelihood_defined=False,real_world_causal_structure_identified=False,
        view_and_detection_separately_identified=False,
        observation_model_limitation='Outside-count allocation still depends on assumed F/D priors; no geometric count likelihood.',
        background_prior=posterior['background_prior'],background_posterior=posterior['background_posterior'],
        background_update_TV=posterior['background_total_variation'],
        expected_outside_instances=posterior['completion_diagnostics']['expected_outside_instances'],
        equal_background_prior_sensitivity=dict(background_posterior=sensitivity['background_posterior'],
            expected_outside_instances=sensitivity['completion_diagnostics']['expected_outside_instances'],
            used_for_world_generation=False,interpretation='assumption sensitivity, not a corrected/calibrated model'),
        causal_gain_validated=False)
