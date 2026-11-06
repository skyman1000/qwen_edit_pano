"""Finite, explicitly assumed causal prior. No training or image likelihood claims.

The small first model is Z -> E, Z -> H: configuration intent/context causes
both positive visible inventory evidence and an off-camera inventory bundle.
Conditional independence given Z is an approximation, not discovered causality.
Image-conditioned proposal selection is disclosed; probabilities are restricted
to that proposal domain, not calibrated probabilities of real rooms.
"""
from collections import Counter
from copy import deepcopy
import json
import math

from .causal_probability import CausalGraph


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Need nonempty text: '+label)
    return value.strip()


def normalize_proposal(answer, case, capacity=256, context_budget=3, world_budget=6):
    if not isinstance(answer, dict):
        raise ValueError('Need proposal object')
    contexts, worlds = answer.get('contexts'), answer.get('worlds')
    if not isinstance(contexts, list) or not 1 <= len(contexts) <= context_budget:
        raise ValueError('contexts must be a nonempty array within the proposal budget')
    if not isinstance(worlds, list) or not 1 <= len(worlds) <= world_budget:
        raise ValueError('worlds must be a nonempty array within the proposal budget')
    contexts = [dict(id=f'z{i+1}', description=text(c.get('description'), 'context.description'),
                     reason=text(c.get('reason'), 'context.reason')) for i, c in enumerate(contexts)
                if isinstance(c, dict)]
    if len(contexts) != len(answer['contexts']):
        raise ValueError('Every context must be an object')
    if len({c['description'].casefold() for c in contexts}) != len(contexts):
        raise ValueError('Duplicate context descriptions')
    contexts.append(dict(id='z_other', description='Another plausible indoor configuration not described by the explicit alternatives.',
                         reason='Unresolved context coverage; not a verified room label.'))
    observed = {o['id'] for o in case['context']}
    result = []
    kinds = {'near', 'left_of', 'right_of', 'above', 'below', 'in_front_of', 'behind',
             'supported_by', 'on_top_of', 'inside'}
    for wi, world in enumerate(worlds):
        if not isinstance(world, dict) or not isinstance(world.get('additions'), list):
            raise ValueError('Every world needs an additions array')
        if len(world['additions'])+len(observed) > capacity:
            raise ValueError('World capacity exceeded; no truncation')
        aliases, additions = {}, []
        for oi, obj in enumerate(world['additions']):
            if not isinstance(obj, dict):
                raise ValueError('Every addition must be an object')
            alias = text(obj.get('id'), 'addition.id')
            if alias in aliases or alias in observed:
                raise ValueError('Duplicate new ID or observed-instance reuse: '+alias)
            oid = f'w{wi+1:02d}_n{oi+1:03d}'
            aliases[alias] = oid
            additions.append(dict(id=oid, **{k: text(obj.get(k), 'addition.'+k)
                for k in ('name', 'object_type', 'reason', 'placement')}))
        relations = []
        if not isinstance(world.get('relations', []), list):
            raise ValueError('relations must be an array')
        for raw in world.get('relations', []):
            if not isinstance(raw, dict):
                raise ValueError('relation must be an object')
            a, b, kind = raw.get('source'), raw.get('target'), raw.get('kind')
            if (not isinstance(a, str) or not isinstance(b, str) or
                    a not in observed | aliases.keys() or b not in observed | aliases.keys() or
                    a == b or kind not in kinds):
                raise ValueError('Unknown/self relation endpoint or unsupported kind')
            row = dict(source=aliases.get(a, a), target=aliases.get(b, b), kind=kind,
                       reason=text(raw.get('reason'), 'relation.reason'))
            if kind == 'near':
                distance = raw.get('max_distance_m')
                if type(distance) not in (int, float) or not math.isfinite(distance) or distance <= 0:
                    raise ValueError('near needs a positive hypothesized max_distance_m')
                row['max_distance_m'] = distance
            relations.append(row)
        result.append(dict(id=f'w{wi+1:02d}', scene_type='hypothesized indoor scene',
            scene_hypothesis=text(world.get('description'), 'world.description'),
            additions=additions, relations=relations))
    # Inventory alternatives with identical multisets must not gain extra mass
    # merely by being proposed twice. Spatial variants belong in layout sampling.
    signatures = [tuple(sorted(Counter(o['object_type'].casefold().strip()
                  for o in w['additions']).items())) for w in result]
    if len(set(signatures)) != len(signatures):
        raise ValueError('Duplicate hidden inventory bundles; propose distinct inventories')
    if () not in signatures:
        result.append(dict(id='w_empty', scene_type='hypothesized indoor scene',
            scene_hypothesis='No additional distinct objects outside the input view.', additions=[], relations=[]))
    return dict(contexts=contexts, worlds=result, unresolved_world='w_other',
                probability_scope='image_proposed_finite_domain_not_calibrated_real_world',
                proposal_selection_bias=True)


def bundle_description(world):
    # Spatial locations, observed IDs, reasons and image evidence must not leak
    # into the hidden-inventory mechanism. Layout uses those separately.
    counts = Counter(o['object_type'].strip().casefold() for o in world['additions'])
    return json.dumps(dict(hidden_object_counts=dict(sorted(counts.items()))), ensure_ascii=False)


def normalized_probabilities(values):
    if not values or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError('Need finite candidate log probabilities')
    peak = max(values)
    weights = [math.exp(x-peak) for x in values]
    total = sum(weights)
    return [w/total for w in weights]


def continuation_logprob(predictions, continuation):
    import torch
    if predictions.ndim != 2 or predictions.shape[0] != len(continuation) or not continuation:
        raise ValueError('Need one next-token prediction per continuation token')
    targets = torch.tensor(continuation, device=predictions.device, dtype=torch.long)
    return float(predictions.float().log_softmax(-1).gather(-1, targets[:, None]).sum().item())


def shifted_token_logprob(logits, prefix_length, continuation):
    """Correct next-token alignment, including the chosen common terminator.

    Only the tiny continuation slice is converted to float32; the whole prompt
    vocabulary tensor is not duplicated. Works with CPU synthetic torch logits.
    """
    if (prefix_length < 1 or not continuation or logits.ndim != 3 or logits.shape[0] != 1 or
            logits.shape[1] != prefix_length+len(continuation)):
        raise ValueError('Logit/prefix/continuation alignment differs')
    return continuation_logprob(logits[0, prefix_length-1:prefix_length+len(continuation)-1], continuation)


class FrozenChoiceScorer:
    """Text-only teacher-forced scoring on the already loaded frozen Qwen.

    Two cyclic label assignments reduce (not eliminate) choice-label bias.
    Each mapping defines a normalized finite mechanism; we mix these two
    distributions explicitly rather than claiming calibrated text likelihoods.
    """
    def __init__(self, qwen):
        self.qwen = qwen

    def score(self, description, parents, states):
        q = self.qwen
        tokenizer = q.processor.tokenizer
        eos = tokenizer.eos_token_id
        if type(eos) is not int or not 2 <= len(states) <= 26:
            raise ValueError('Need tokenizer EOS and 2..26 finite states')
        rounds = []
        for offset in (0, 1):
            ordered = states[offset:]+states[:offset]
            labels = [chr(65+i) for i in range(len(states))]
            prompt = ('Specify one value of this conditional mechanism in a hypothetical indoor-world model. '
                'Use only the supplied parent state; do not infer any unprovided photograph or observations. '
                'The alternatives define a restricted model, not measured population frequencies. '
                'Answer exactly one option letter and end the response.\nMECHANISM: '+description+
                '\nPARENTS: '+json.dumps(parents, ensure_ascii=False)+'\nOPTIONS:\n'+
                '\n'.join(label+': '+state['description'] for label, state in zip(labels, ordered)))
            message = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
            prefix = q.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            logs = []
            token_ids = []
            for label in labels:
                continuation = tokenizer.encode(label, add_special_tokens=False)+[eos]
                ids = q.torch.tensor([prefix_ids+continuation], device=q.model.device)
                positions = q.torch.arange(len(prefix_ids)-1, len(prefix_ids)+len(continuation)-1,
                                           device=q.model.device)
                with q.torch.inference_mode():
                    # Local Qwen3-VL supports a tensor index here. Materialize
                    # only continuation logits, not prompt_length*vocab logits.
                    output = q.model(input_ids=ids, attention_mask=q.torch.ones_like(ids),
                                     use_cache=False, logits_to_keep=positions)
                    score = continuation_logprob(output.logits[0], continuation)
                logs.append(score); token_ids.append(continuation)
                del output, ids
            probs = normalized_probabilities(logs)
            rounds.append(dict(prompt=prompt, offset=offset, prefix_tokens=len(prefix_ids),
                state_ids=[s['id'] for s in ordered], labels=labels, continuation_token_ids=token_ids,
                log_probabilities=logs, probabilities=probs))
        probability = {s['id']: 0. for s in states}
        for row in rounds:
            for sid, p in zip(row['state_ids'], row['probabilities']):
                probability[sid] += p/len(rounds)
        return dict(probabilities=probability, rounds=rounds,
            source='frozen_Qwen_text_only_two_label_mixture', calibrated=False,
            training_performed=False, parents=deepcopy(parents))


def build_model(domain, positive_inventory, scorer, save_score):
    """Execute Z -> E and Z -> H, with no Boolean assumed/ground gates."""
    zstates = [dict(id=c['id'], description=c['description']) for c in domain['contexts']]
    hstates = [dict(id=w['id'], description=bundle_description(w)) for w in domain['worlds']]
    hstates.append(dict(id='w_other', description='A different hidden object inventory outside the listed alternatives.'))
    def score(name, description, parents, states):
        result = scorer.score(description, parents, states)
        if set(result['probabilities']) != {s['id'] for s in states}:
            raise ValueError('Scorer returned incomplete/extra state IDs')
        values = list(result['probabilities'].values())
        if any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in values) or not math.isclose(sum(values), 1., abs_tol=1e-9):
            raise ValueError('Mechanism probabilities must be finite, nonnegative and normalized')
        save_score(name, {**result, 'description':description, 'states':states, 'parents':deepcopy(parents)})
        return result['probabilities']
    prior = score('context_prior', 'Prior over pre-configuration intended scene use and household configuration. '
        'No local evidence has been supplied to this mechanism.', {}, zstates)
    erows, hrows = [], []
    for context in domain['contexts']:
        z = context['id']; parents = {'configuration_context': context['description']}
        evidence = score('evidence_'+z, 'Whether a room inventory includes AT LEAST these distinct instances: '+
            json.dumps(positive_inventory, ensure_ascii=False)+
            '. This is only positive presence evidence; all other objects are unknown. '
            'It is not a camera visibility, detector or RGB likelihood model.', parents,
            [dict(id='present', description='The inventory includes at least those instances.'),
             dict(id='not_present', description='The inventory does not include all those instances.')])
        hidden = score('hidden_'+z, 'Distribution over off-camera object inventory bundles under the parent configuration. '
            'Predict what could already exist, not minimum equipment or improvements. Optional decoration, '
            'storage and additional instances are permitted; no function implies mandatory equipment.', parents, hstates)
        erows.append(dict(when=[z], probabilities=evidence))
        hrows.append(dict(when=[z], probabilities=hidden))
    spec = dict(parameter_source='frozen_LM_elicited_prior_not_trained_or_calibrated',
        identified_from_data=False, graph_assumption='configuration_common_cause_with_conditional_independence',
        conditioning_scope='positive_inventory_event_only_not_RGB_posterior',
        proposal_selection_bias=True, nodes=[
            dict(id='Z', states=[s['id'] for s in zstates], parents=[], cpt=[dict(when=[], probabilities=prior)]),
            dict(id='E', states=['present', 'not_present'], parents=['Z'], cpt=erows),
            dict(id='H', states=[s['id'] for s in hstates], parents=['Z'], cpt=hrows)])
    CausalGraph(spec)
    return spec


def infer(spec):
    model = CausalGraph(spec)
    evidence = {'E': 'present'}
    before = model.query('H')
    posterior = model.query('H', evidence=evidence)
    context = model.query('Z', evidence=evidence)
    interventions = {z: model.query('H', interventions={'Z': z})
                     for z in model.nodes['Z']['states']}
    intervention_rows = [r['probabilities'] for r in interventions.values()]
    variation = max((.5*sum(abs(a[s]-b[s]) for s in a)
                     for a in intervention_rows for b in intervention_rows), default=0.)
    update = .5*sum(abs(posterior['probabilities'][s]-before['probabilities'][s])
                    for s in before['probabilities'])
    return dict(prior_hidden=before, posterior_hidden=posterior, posterior_context=context,
                context_interventions=interventions, observed_evidence=evidence,
                do_evidence_control=model.query('H', interventions={'E':'present'}),
                intervention_is_model_diagnostic=True, individual_counterfactual_computed=False,
                causal_gain_validated=False, semantics_verified=False,
                unresolved_world_mass=posterior['probabilities']['w_other'],
                mechanism_signal=dict(context_intervention_max_total_variation=variation,
                    evidence_hidden_update_total_variation=update,
                    warning='Small or zero differences indicate weak/no effective mechanism signal; changes do not prove causal validity.'),
                note='Model-implied inference in an image-proposed finite domain, not calibrated physical-room probabilities.')
