"""Aligned numeric finite events, including multimodal inverse configuration.

Counterbalance the association of integer codes to semantic alternatives. This
reduces a particular code association; it does not calibrate the frozen model.
"""
from copy import deepcopy
import json
import math

from .scm_typed_queries import normalized_logs


def aligned_events(q, image, prompt, count):
    tok = q.processor.tokenizer; eos = tok.eos_token_id
    if type(eos) is not int or not 1 <= count <= 12: raise ValueError('Need numeric events and single EOS')
    content = ([dict(type='image')] if image is not None else []) + [dict(type='text',text=prompt)]
    prefix = q.processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
    kwargs = dict(text=[prefix],return_tensors='pt')
    if image is not None: kwargs['images'] = [image]
    encoded = q.processor(**kwargs).to(q.model.device); encoded.pop('token_type_ids',None)
    rows = []
    for code in range(count):
        ids = tok.encode(str(code),add_special_tokens=False)+[eos]
        suffix = q.torch.tensor([ids],dtype=encoded.input_ids.dtype,device=q.model.device)
        inputs = dict(encoded)
        inputs['input_ids'] = q.torch.cat((encoded.input_ids,suffix),dim=1)
        inputs['attention_mask'] = q.torch.cat((encoded.attention_mask,q.torch.ones_like(suffix)),dim=1)
        with q.torch.inference_mode():
            result = q.model(**inputs,use_cache=False,logits_to_keep=len(ids)+1)
            logits = result.logits[0,-len(ids)-1:-1].float()
            values = logits.log_softmax(-1).gather(1,suffix[0,:,None]).squeeze(1).tolist()
        rows.append(dict(code=code,token_ids=ids,token_log_probabilities=values,log_probability=sum(values)))
        del result,logits,suffix,inputs
    return dict(prompt=prompt,events=rows,image_used=image is not None,
                probabilities=normalized_logs([r['log_probability'] for r in rows]))


def finite_scores(q,image,field,parents,alternatives):
    if not alternatives: raise ValueError('Need finite alternatives')
    rows = []; n = len(alternatives)
    for shift in range(n):
        order = [(i+shift)%n for i in range(n)]
        prompt = (field+'\nCONTEXT / DECLARED PARENTS: '+json.dumps(parents,ensure_ascii=False,sort_keys=True)+
                  '\nALTERNATIVES: '+json.dumps([dict(index=i,value=alternatives[j]) for i,j in enumerate(order)],ensure_ascii=False)+
                  '\nReturn only the JSON integer index of one alternative. No explanation.')
        result = aligned_events(q,image,prompt,n)
        result['code_to_alternative'] = order; rows.append(result)
    # Average log probabilities across code rotations, then normalize. The
    # declared mechanism IS this logarithmic pool; no empirical calibration.
    pooled = [sum(math.log(max(r['probabilities'][r['code_to_alternative'].index(j)],1e-300))
                  for r in rows)/n for j in range(n)]
    probs = normalized_logs(pooled)
    return dict(probabilities={str(i):p for i,p in enumerate(probs)},rotations=rows,
        alternatives=deepcopy(alternatives),parents=deepcopy(parents),field=field,
        source='cyclic_numeric_aligned_events_with_eos_v1',calibrated=False,
        image_used=image is not None,full_scene_posterior=False)


class GroupMechanismScorer:
    def __init__(self,q): self.q = q

    def score(self,description,parents,states):
        parents = deepcopy(parents); choices = parents.pop('alternatives')
        alternatives = []
        for choice in choices:
            recipe = choice['joint_recipe']
            alternatives.append(dict(description=recipe['description'],objects=recipe['objects']))
        return finite_scores(self.q,None,description,parents,alternatives)


def verify_scores(record):
    n = len(record['alternatives']); pooled = [0.]*n
    if len(record['rotations']) != n: raise ValueError('Missing code rotations')
    for shift,row in enumerate(record['rotations']):
        if row['code_to_alternative'] != [(i+shift)%n for i in range(n)]: raise ValueError('Code association changed')
        if len(row['events']) != n: raise ValueError('Missing numeric events')
        values = []
        for code,event in enumerate(row['events']):
            lp = event['token_log_probabilities']
            if (event['code'] != code or not lp or len(lp) != len(event['token_ids']) or
                    any(not math.isfinite(v) or v>0 for v in lp) or
                    not math.isclose(sum(lp),event['log_probability'],abs_tol=1e-7)):
                raise ValueError('Invalid aligned event record')
            values.append(sum(lp))
        ps = normalized_logs(values)
        if any(not math.isclose(a,b,abs_tol=1e-9) for a,b in zip(ps,row['probabilities'])):
            raise ValueError('Numeric event normalization changed')
        for code,j in enumerate(row['code_to_alternative']): pooled[j] += math.log(max(ps[code],1e-300))/n
    expected = normalized_logs(pooled)
    if any(not math.isclose(record['probabilities'][str(i)],p,abs_tol=1e-9) for i,p in enumerate(expected)):
        raise ValueError('Pooled mechanism changed')
    return expected
