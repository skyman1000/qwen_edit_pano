"""v13: the v12 complete-program SCM with explicit organization hypotheses.

No count or directional quotas; probability mechanisms remain empirical and
uncalibrated. Frozen-world replay is a geometry experiment, not new inference.
"""
from copy import deepcopy
from html import escape
import json
from pathlib import Path

from ..common import read,write,sha
from . import assembly_compositional as base

PROTOCOL=base.PROTOCOL
INTERFACE='organized_complete_programs_balanced_presence_v13'
MODEL_VARIANT='organized_functional_program_scm_v13'
KIND='organized_functional_program'
build_model=base.build_model
load_domain=base.load_domain
semantic_diagnostics=base.semantic_diagnostics
active_world=base.active_world
marginals=base.marginals
intervention_diagnostics=base.intervention_diagnostics


def organization_profiles(roles):
    """A finite hypothesis family, not inferred resident activities/history."""
    values=[dict(id='h0',functions=[],primary_role=None,
        description='Mixed functional organization; no particular use is given priority.',
        assumptions=['A candidate world organization, not measured intent, activity or biography.'])]
    for role in sorted(roles):
        values.append(dict(id='h'+str(len(values)),functions=[role],primary_role=role,
            description='This candidate world is organized with '+role+' as an emphasized use, while other functions may coexist.',
            assumptions=['An uncertain generative hypothesis about this world, not an observed fact.',
                'No object, extra independent zone, quantity, or sector follows deterministically from emphasis.']))
    return values


def recognition_prompt(d,aliases):
    return ('Sample one FUNCTIONAL ORGANIZATION hypothesis for one complete existing world compatible with the fixed partial observation. '
        'This is uncertain inference, not a statement that a resident actually performs an activity. '
        'Do not infer biography, walls, or an object inventory. Mixed organization is allowed. '
        'Emphasis is not exclusive: several functions may coexist. '
        'Return exactly {"choice":"an allowed alias"}.\nFIXED_POSITIVE_CONTEXT: '+json.dumps(d['positive_context'],ensure_ascii=False)+
        '\nALLOWED_CHOICES: '+json.dumps([dict(choice=a,profile=next(h for h in d['profiles'] if h['id']==v)) for a,v in aliases.items()],ensure_ascii=False))


def presence_prompt(d,h,role,aliases):
    descriptions={
        '1':'One of the listed complete ROLE programs belongs to this world\'s additional hidden inventory. Fixed observed instances are unchanged; other roles remain optional.',
        '0':'None of the listed complete ROLE programs belongs to this world\'s additional hidden inventory. Fixed observed instances are unchanged; other roles remain optional.'}
    return ('Choose the inventory state for ONE sampled complete existing world, conditioned on the observation and organization hypothesis. '
        'Decide which state this world takes, not whether an object COULD occur somewhere and not whether it is strictly necessary. '
        'Absence refers only to an additional hidden ROLE program, never to the photographed objects. '
        'Presence may extend an observed activity without a second independent activity zone. '
        'Neither organization emphasis nor lack of a tag determines the answer. '
        'Both inventory states are legal; their frequencies must follow the condition, not a balancing rule. '
        'Return exactly {"choice":"an allowed alias"}, no explanation or object list. '
        '\nH: '+json.dumps(h,ensure_ascii=False)+'\nROLE: '+role+
        '\nFIXED_POSITIVE_CONTEXT: '+json.dumps(d['positive_context'],ensure_ascii=False)+
        '\nROLE_PROGRAMS: '+json.dumps(base.program_menu(d,role),ensure_ascii=False)+
        '\nALLOWED_CHOICES: '+json.dumps([dict(choice=a,state=descriptions[v]) for a,v in aliases.items()],ensure_ascii=False))


def variant_prompt(d,h,role,aliases):
    return base.variant_prompt(d,h,role,aliases).replace(
        'Use the COMPLETE spatial reasons, placements and fixed predicted observation geometry.',
        'H specifies this candidate world\'s functional organization, not just a list of possible affordances. '
        'Use it together with COMPLETE spatial reasons, placements and fixed predicted observation geometry. '
        'Emphasis is soft; other functions may coexist. Do not force an inventory or sector from a tag.')


def compile_model(d,t,policy='functional'):
    m=base.compile_model(d['base_model'],d,t,policy)
    m.update(model_variant=MODEL_VARIANT,response_interface=INTERFACE,method_revision=13,
        organization_hypotheses_are_assumed=True,mechanism_estimator='counterbalanced_empirical_choices_with_unchanged_smoothing',
        geometry_scope='frozen_dimensions_then_joint_host_placement; box_proxy_only')
    return m


def prepare_model(unused,case,config,out):
    bundle=Path(config['sdscene_bundle']);d=read(bundle/'domain.json')
    if config.get('frozen_world_bundle'):
        replay=Path(config['frozen_world_bundle'])/config['execution_backend'];m=read(replay/'model.json')
        m['_frozen_worlds']=read(replay/'worlds.json')
        m.update(model_variant=MODEL_VARIANT,response_interface=INTERFACE,replay_only=True,
            replay_source=config['frozen_world_source'],method_revision=13,
            sampling_policy='exact_frozen_source_world_replay_no_new_inventory_draws',
            distribution_scope='frozen_v12_worlds_geometry_control_not_new_conditional_inference',
            probability_audit_source='source_v12_model_not_new_mechanisms')
    else:
        source='pooled' if config['mechanism_source']=='pooled' else config['execution_backend'];t=read(bundle/'mechanisms'/(source+'.json'))
        if t['kind']!=KIND or t['domain_sha256']!=sha(bundle/'domain.json'):raise ValueError('Organized mechanism/domain mismatch')
        m=compile_model(d,t,config['mechanism_policy'])
    if out is not None:
        write(out/'empirical_mechanisms.json',m['empirical_tables']);law=marginals(m);write(out/'probability_audit.json',law)
        write(out/'functional_probability_interventions.json',{h['id']:marginals(m,h['id']) for h in m['profiles']})
        (out/'probability_audit.html').write_text('<meta charset="utf-8"><h1>完整程序的实际条件概率</h1><p>概率进入采样；全真和无 H 效应均原样保留。概率未校准。</p><pre>'+escape(json.dumps(law,ensure_ascii=False,indent=2))+'</pre>')
    return m


def sample_worlds(m,seed=3407,count=4):
    if '_frozen_worlds' in m:
        worlds=m['_frozen_worlds']
        if count!=len(worlds):raise ValueError('Replay must retain every planned source world, exactly once')
        return deepcopy(worlds)
    return base.sample_worlds(m,seed,count)
