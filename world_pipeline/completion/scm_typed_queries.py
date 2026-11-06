"""Finite generation and correctly aligned continuation likelihoods. No new deps."""
from copy import deepcopy
import json
import math

from ..common import write


def normalized_logs(scores):
    if not scores or any(not math.isfinite(s) for s in scores):raise ValueError('Need finite continuation scores')
    a=[math.exp(s-max(scores)) for s in scores];total=sum(a)
    return [v/total for v in a]


class TokenTrie:
    def __init__(self,sequences,prompt_length):
        self.root={};self.prompt_length=prompt_length
        for seq in sequences:
            node=self.root
            for token in seq:node=node.setdefault(token,{})

    def __call__(self,batch_id,input_ids):
        node=self.root
        for token in input_ids[self.prompt_length:].tolist():
            if token not in node:raise ValueError('Generated token left the declared finite domain')
            node=node[token]
        if not node:raise ValueError('Generation continued after finite choice EOS')
        return list(node)


def finite_choice(q,image,prompt,candidates,path):
    """Greedy trie-constrained layout proposal, NOT a normalized SCM mechanism.

    IDs and possible support targets are serialized by the program. The model
    can select one complete candidate only; it cannot rewrite the inventory.
    """
    if not candidates or len({json.dumps(c,sort_keys=True) for c in candidates})!=len(candidates):
        raise ValueError('Need distinct finite choices')
    tok=q.processor.tokenizer;eos=tok.eos_token_id
    if type(eos) is not int:raise ValueError('Need a single tokenizer EOS')
    texts=[json.dumps(c,ensure_ascii=False,separators=(',',':')) for c in candidates]
    seqs=[tok.encode(t,add_special_tokens=False)+[eos] for t in texts]
    if len({tuple(s) for s in seqs})!=len(seqs):raise ValueError('Tokenized finite choices collide')
    content=([dict(type='image')] if image is not None else [])+[dict(type='text',text=prompt+
        '\nSelect exactly one complete JSON value from this finite domain:\n'+'\n'.join(texts))]
    prefix=q.processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
    inputs=dict(text=[prefix],return_tensors='pt')
    if image is not None:inputs['images']=[image]
    encoded=q.processor(**inputs).to(q.model.device);encoded.pop('token_type_ids',None)
    length=encoded.input_ids.shape[1];trie=TokenTrie(seqs,length)
    write(path.with_suffix('.request.json'),dict(method='finite_token_trie_greedy_layout_v1',prompt=prompt,
        candidates=candidates,image_used=image is not None,semantic_validity_verified=False))
    with q.torch.inference_mode():
        result=q.model.generate(**encoded,do_sample=False,temperature=None,top_p=None,top_k=None,
            repetition_penalty=1.,min_new_tokens=0,min_length=0,num_beams=1,num_return_sequences=1,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos,
            max_new_tokens=max(map(len,seqs)),eos_token_id=eos,
            prefix_allowed_tokens_fn=trie)
    answer=result[0,length:].tolist();raw=q.processor.batch_decode(result[:,length:],skip_special_tokens=True)[0]
    path.write_text(raw)
    write(path.with_suffix('.generation.json'),dict(generated_tokens=len(answer),ended_with_eos=bool(answer and answer[-1]==eos),
        method='finite_token_trie_greedy_layout_v1'))
    if answer not in seqs:raise ValueError('Incomplete finite choice; raw reply preserved')
    value=deepcopy(candidates[seqs.index(answer)]);write(path.with_suffix('.json'),value);return value


class ContinuationMechanismScorer:
    """Domain-restricted *whole sequence* probabilities, conditioned on parents.

    No option codes or candidate list is included in the LM context. Include EOS
    so a short candidate is not a prefix event of another one. Sum aligned token
    log probabilities; never average by length or gather from the final logit.
    This is an uncalibrated finite text model, not empirical room statistics.
    """
    def __init__(self,qwen):self.qwen=qwen

    def score(self,description,parents,states):
        if not states or len({s['id'] for s in states})!=len(states):raise ValueError('Need distinct finite states')
        q=self.qwen;tok=q.processor.tokenizer;eos=tok.eos_token_id
        prompt=('Complete one field of a hypothetical EXISTING indoor household world using only its '
            'declared parent values and general knowledge. This is not an equipment recommendation. '
            'Objects may be absent or repeated. Return the field value only.\nFIELD: '+description+
            '\nPARENTS: '+json.dumps(parents,ensure_ascii=False,sort_keys=True))
        messages=[dict(role='user',content=[dict(type='text',text=prompt)])]
        prefix=q.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        encoded=q.processor(text=[prefix],return_tensors='pt').to(q.model.device);encoded.pop('token_type_ids',None)
        if encoded.input_ids.shape[0]!=1 or type(eos) is not int:raise ValueError('Need one text prefix and EOS')
        length=encoded.input_ids.shape[1];rows=[]
        for state in states:
            ids=tok.encode(state['description'],add_special_tokens=False)+[eos]
            suffix=q.torch.tensor([ids],device=q.model.device,dtype=encoded.input_ids.dtype)
            full=q.torch.cat((encoded.input_ids,suffix),dim=1)
            mask=q.torch.ones_like(full)
            with q.torch.inference_mode():
                result=q.model(input_ids=full,attention_mask=mask,use_cache=False,
                    logits_to_keep=len(ids)+1)
                # Returned logits start at prefix's last position. Position j
                # predicts candidate token j, including the final EOS.
                logits=result.logits[0,-len(ids)-1:-1].float()
                values=logits.log_softmax(-1).gather(1,suffix[0,:,None]).squeeze(1).tolist()
            rows.append(dict(id=state['id'],text=state['description'],token_ids=ids,
                token_log_probabilities=values,log_probability=sum(values)))
            del result,full,mask,suffix,logits
        probabilities=normalized_logs([r['log_probability'] for r in rows])
        return dict(probabilities=dict(zip([s['id'] for s in states],probabilities)),
            candidate_sequences=rows,prompt=prompt,parents=deepcopy(parents),states=deepcopy(states),
            rounds=[],image_used=False,calibrated=False,option_order_used=False,
            source='parent_only_aligned_complete_continuation_with_eos_v1',
            scoring='sum_autoregressive_token_log_probabilities_including_eos_no_length_normalization')
