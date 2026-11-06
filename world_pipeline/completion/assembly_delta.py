"""Observed anchors + a joint, all-scale hidden-world delta.

The finite-domain mechanism is proposed by a frozen LM, then sampled by the
explicit reference law. This does not identify the real room causal model.
"""
from copy import deepcopy
import json
from collections import defaultdict

from . import assembly_model as typed
from . import assembly_inventory as inventory
from .scene_program import text
from .world_scm import validate,forward
from .assembly_observation import check_observed_attachments

PROTOCOL=inventory.PROTOCOL
INTERFACE='observed_anchors_joint_world_delta_v2'
MODEL_VARIANT='anchored_joint_delta'
hypotheses_prompt=typed.hypotheses_prompt
hypotheses_answer=typed.hypotheses_answer


def furniture_prompt(case,hypothesis):
    schema=dict(key='unique_instance_id',category='bookcase',variant='freestanding wooden bookcase',
        reason='evidence-compatible reason for this separate unseen instance',
        placement='qualitative surrounding position',attachment=dict(kind='floor',host=None),affordances=[])
    return ('Complete the UNOBSERVED surrounding world around the FIXED OBSERVED ANCHORS. '
        'Use the attached original-view observation board as evidence; this request has no previous chat memory. '
        'The observed inventory is already part of the complete world; refer to its exact obs IDs when describing an activity area. '
        'Do not regenerate the photographed seating, pillows, lamp or other known instances under new IDs. '
        'A separate same-category unseen instance is possible only when the hypothesis calls for that additional instance. '
        'Infer an already inhabited surrounding world compatible with the partial photograph, not a minimal functional room or a design checklist. '
        'Return 1..2 concrete alternative JOINT hidden-world programs. Each program includes the complete additional inventory '
        'at all scales: furniture, independent belongings, supported contents and plausible fixtures. '
        'Consider surrounding activity/storage areas and architectural features as uncertain possibilities; no type is mandatory. '
        'Do not split furniture selection from small-object generation. All hosts and their contents belong to the SAME alternative. '
        'Return {"alternatives":[{"description":"fill with a concrete surrounding-world hypothesis",'
        '"groups":[{"function":"fill with the concrete activity or storage area","reason":"evidence and uncertainty",'
        '"observed_refs":["an exact obs ID if this area uses a known anchor"],"new_objects":['+json.dumps(schema)+']}],"relations":[]}]}. '
        'An observed_refs array may be empty. It does not create an object or authorize changing the observation. '
        'new_objects is ONLY the hidden-world delta, never a restatement of observed anchors or integral parts. '
        'Capacity: up to 4 groups and 24 new instances per complete alternative, no minimum or category quotas. Keep records concise. '
        'Do not expand a list by repeating the same remote/mug/book with a new number; multiple real instances are possible, '
        'but need a distinct device, collection, user or placement hypothesis. Stop when this concrete world is described. '
        'attachment.kind: floor/top/seat/shelf/hang/wall/ceiling/free. top/seat/shelf/hang require an exact host key '
        'in THIS alternative or an observed ID; other kinds use host=null. Hosted items inherit the host placement group. '
        'If using an observed host, consult OBSERVED SUPPORT DOMAINS. An empty candidate list for that attachment kind '
        'means its entire possible contact envelope lies inside the input frustum. That support hypothesis cannot produce '
        'a wholly outside-frustum object. A nonempty list is only a necessary condition, not an actual support or free-space measurement. '
        'affordances is a subset of top,seat,shelf,hanging, describing usable surfaces offered to other items, often []. '
        'Wall means physically mounted; a bookshelf standing near a wall uses floor. '
        'category is the actual concrete noun (bookcase, mug, television); never furniture, decor, everyday belonging or another broad observed label. '
        'The bookcase in the schema is only a syntax example, not a required object. No Adapter class restriction. No dimensions or coordinates. '+typed.RELATION_CONTRACT+
        'left/right are original-camera X; in_front_of means GREATER camera Z and behind means SMALLER camera Z. '
        'A TV facing a sofa does not imply either camera-depth ordering. Describe that functional intent in placement instead. '+
        '\nSCENE HYPOTHESIS: '+json.dumps(hypothesis,ensure_ascii=False)+
        '\nOBSERVED SUPPORT DOMAINS: '+json.dumps(case.get('observed_support_domains',{}),ensure_ascii=False)+
        '\nFIXED OBSERVED ANCHORS (already exist; immutable): '+json.dumps(case['context'],ensure_ascii=False))


def furniture_answer(value,case,hypothesis):
    rows=value.get('alternatives');observed={o['id'] for o in case['context']};result=[]
    if not isinstance(rows,list) or not 1<=len(rows)<=2:raise ValueError('alternatives must contain 1..2 complete hidden-world deltas')
    for ai,alt in enumerate(rows):
        if not isinstance(alt,dict) or not isinstance(alt.get('groups'),list) or len(alt['groups'])>4:
            raise ValueError(f'alternatives[{ai}] needs groups: at most 4 activity/spatial assemblies')
        groups=[];raw_objects=[];declared_group={}
        for gi,g in enumerate(alt['groups']):
            if not isinstance(g,dict):raise ValueError('Each assembly must be an object')
            gid=f'g{gi}';refs=g.get('observed_refs',[])
            if not isinstance(refs,list) or any(not isinstance(v,str) or v not in observed for v in refs) or len(set(refs))!=len(refs):
                raise ValueError(f'alternatives[{ai}].groups[{gi}].observed_refs must use distinct exact IDs from '+str(sorted(observed)))
            objects=g.get('new_objects')
            if not isinstance(objects,list):raise ValueError('Use new_objects for the hidden delta; observed anchors belong in observed_refs')
            groups.append(dict(key=gid,function=text(g.get('function'),'group.function'),reason=text(g.get('reason'),'group.reason'),
                observed_refs=list(refs),objects=[]))
            for obj in objects:
                if not isinstance(obj,dict):raise ValueError('new_objects records must be objects')
                category=obj.get('category');variant=obj.get('variant',category)
                bound=dict(key=obj.get('key'),category=category,variant=variant,name=obj.get('name',variant),
                    distinct_instance_reason=obj.get('reason'),placement_hint=obj.get('placement'),
                    attachment=obj.get('attachment'),affordances=obj.get('affordances',[]))
                if not isinstance(bound['key'],str):raise ValueError('Each new object needs a string key')
                raw_objects.append(bound);declared_group[bound['key']]=gid
        objects=typed.objects_answer(raw_objects,observed,max_objects=24,path=f'alternatives[{ai}].new_objects',host_scope='THIS ALTERNATIVE')
        check_observed_attachments(objects,case.get('observed_support_domains',{}))
        assigned={oid:'g_obs' for oid in observed};bygroup={g['key']:g for g in groups}
        for obj in typed.order_objects(objects,observed):
            host=obj['attachment']['host'];gid=assigned[host] if host else declared_group[obj['key']]
            if gid not in bygroup:
                g=dict(key=gid,function='Unphotographed extent of a fixed observed host.',reason='Explicit attachment hypothesis.',observed_refs=[],objects=[])
                groups.append(g);bygroup[gid]=g
            obj['semantic_group']=declared_group[obj['key']];obj['group']=gid;assigned[obj['key']]=gid
            bygroup[gid]['objects'].append(obj)
        relations,bindings=typed.relations_answer(alt.get('relations',[]),observed|{o['key'] for o in objects},objects,
            path=f'alternatives[{ai}].relations')
        result.append(dict(description=text(alt.get('description'),'alternative.description'),hypothesis=deepcopy(hypothesis),
            groups=groups,relations=relations,relation_bindings=bindings,inventory_scope='all_scale_hidden_delta'))
    return result


def semantic_diagnostics(world):
    """Describe proposal risks; no counts, identities, or categories are edited."""
    buckets=defaultdict(list);broad=[]
    for o in world['objects']:
        buckets[(o['variant'].strip().casefold(),o['attachment']['host'])].append(o)
        if o['category'].strip().casefold() in {'furniture','decor','everyday belonging','decoration','object','fixture'}:
            broad.append(dict(id=o['id'],category=o['category'],variant=o['variant']))
    repeats=[dict(variant=k[0],host=k[1],count=len(rows),
        instances=[dict(id=o['id'],reason=o['distinct_instance_reason']) for o in rows])
        for k,rows in buckets.items() if len(rows)>1]
    return dict(repeated_variant_on_same_host=sorted(repeats,key=lambda r:-r['count']),
        broad_category_labels=broad,objects_edited=False,semantics_verified=False,
        interpretation='Repeated categories can be real collections. This report is evidence for review, not duplicate removal.')


def build_model(hypotheses,programs,unused_contents=None):
    # Reuse the verified inventory/installation equations. A single empty
    # contents mechanism contributes no objects and has no downstream parents;
    # removing it preserves the distribution exactly, without renormalization.
    empty={fid:[dict(description='All contents are already jointly generated.',objects=[],relations=[])] for fid in programs}
    model=inventory.build_model(hypotheses,programs,empty)
    model['nodes']=[n for n in model['nodes'] if not n['id'].startswith('T_')]
    model.update(model_variant=MODEL_VARIANT,response_interface=INTERFACE,contents_mechanisms={},
        scene_program_states=deepcopy(programs),generation_factorization='Z -> joint_scene_delta_F -> E_i -> S_i <- S_host')
    return validate(model)


active_world=inventory.active_world
sample_worlds=inventory.sample_worlds


def intervention_diagnostics(model,world):
    report=inventory.intervention_diagnostics(model,world);queries=[]
    # Test the generating mechanism as well as installation gates. This is a
    # model-relative inventory intervention, not an RGB-consistent alternate
    # history or a generated coordinate counterfactual.
    for z in next(n['states'] for n in model['nodes'] if n['id']=='Z'):
        if z==world['structural_state']['Z']:continue
        after=active_world(model,forward(model,world['noise'],{'Z':z}))
        queries.append(dict(do={'Z':z},before_program=world['structural_state']['F'],after_program=after['structural_state']['F'],
            before_inventory=[dict(id=o['id'],name=o['name']) for o in world['objects']],
            after_inventory=[dict(id=o['id'],name=o['name']) for o in after['objects']],
            same_exogenous_noise=True,model_called=False,coordinate_counterfactual_generated=False,
            compatibility_with_actual_RGB_not_claimed=True))
    report['configuration_queries']=queries
    return report
