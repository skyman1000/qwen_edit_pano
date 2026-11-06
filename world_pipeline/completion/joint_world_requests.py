"""Phase-specific bounded correction and lossless numeric representation handling."""
from copy import deepcopy
import json
import math
from pathlib import Path
import re

from ..common import write
from .assembly_experiment import SeededProvider,stage_seed

SCALARS={'floor_y_m','yaw_deg','range_m','surface_fraction','hanging_drop_m','host_offset_y_m'}
VECTORS={'center_xz_m','root_offset_m','host_offset_uv','full_size_m','bounds_xz_m'}
NUMBER=re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z')


def numeric_fields(value):
    """Only explicit numeric metric fields; never interpret IDs, units or prose."""
    actions=[]
    def number(v,path):
        if isinstance(v,str) and NUMBER.fullmatch(v) and math.isfinite(float(v)):
            actions.append(dict(path=path,raw=v,number=float(v)));return float(v)
        return v
    def visit(v,path=''):
        if isinstance(v,list):return [visit(x,path+'/'+str(i)) for i,x in enumerate(v)]
        if not isinstance(v,dict):return v
        out={}
        for k,x in v.items():
            p=path+'/'+k
            if k in SCALARS:out[k]=number(x,p)
            elif k in VECTORS and isinstance(x,list):out[k]=[number(y,p+'/'+str(i)) for i,y in enumerate(x)]
            else:out[k]=visit(x,p)
        return out
    return visit(deepcopy(value)),actions


def asker(provider,config,calls):
    calls=Path(calls);calls.mkdir(parents=True,exist_ok=True)
    def ask(name,image,prompt,validator,max_tokens=8192):
        semantic=name=='hypotheses' or name.startswith('programs_')
        metric=not semantic and not name.startswith(('recognition_','program_'))
        dimensions='_sizes_' in name
        selected=SeededProvider(provider,stage_seed(config['seed'],name))
        for attempt in range(2):
            path=calls/(name+('_repair' if attempt else '')+'.txt')
            print(json.dumps(dict(stage=name,attempt=attempt,correction_scope='semantic_proposal' if semantic else 'frozen_metric' if metric else 'categorical_choice')),flush=True)
            try:
                raw=selected.ask(image,prompt,path,max_tokens=max_tokens,profile='completion',sample=True,seed=3407+attempt)
                if not isinstance(raw,dict):raise ValueError('Top-level named JSON object required')
                value,actions=numeric_fields(raw) if metric else (raw,[])
                try:result=validator(value)
                except (KeyError,TypeError,IndexError) as exc:raise ValueError('Required JSON field invalid: '+str(exc)) from exc
                write(path.with_suffix('.accepted.json'),dict(value=value,numeric_normalizations=actions,semantic_identity_changes=False if metric else 'new_candidate_not_frozen'))
                return result
            except ValueError as exc:
                write(path.with_suffix('.validation.json'),dict(error=str(exc),semantic_proposal=semantic))
                if attempt:raise ValueError(name+': '+str(exc)) from exc
                previous=path.read_text() if path.exists() else json.dumps(locals().get('raw',{}),ensure_ascii=False)
                prompt+='\nREJECTED ANSWER:\n'+previous+'\nVALIDATION ERROR: '+str(exc)
                if name=='hypotheses':
                    prompt+='\nCorrect the semantic H records only: description, functions and assumptions. Describe concrete plausible unobserved surroundings, not copied schema phrases. No inventory or metric coordinates. Preserve observed facts.'
                elif semantic:
                    prompt+=('\nPropose a COMPLETE corrected hidden DELTA. Candidate identities are NOT frozen. '
                        'Never redeclare or rename observed objects as additions: observed IDs belong only in observed_refs or host references. '
                        'Replace invalid proposals with an evidence-compatible hypothesis; do not merely rename visible instances. '
                        'New hosts and children may coexist in this same alternative; every host must be an EXACT declared key or observed ID. '
                        'Do not invent floor/wall/ceiling host IDs. Preserve the fixed observations; do not drop a bad object just to pass. '
                        'No metric coordinates in semantic hypotheses. Return the complete requested JSON schema.')
                elif metric:
                    prompt+='\nCorrect only numeric fields/container syntax. Inventory, exact IDs and hosts are frozen. Do not omit, add or rename objects. Use JSON numbers, not units or descriptive strings.'
                    prompt+=(' Dimensions are being inferred in this stage; correct invalid full_size_m values.' if dimensions else
                        ' Previously accepted dimensions are frozen; do not resize objects.')
                    if name=='floor' or name.endswith('_floor'):
                        prompt+=' If FLOOR_CHOICES is supplied, return one exact choice ID and concrete basis; do not return anchor_ids or predict heights. Otherwise select defensible observed floor-anchor IDs or state a finite unmeasured floor hypothesis. An unresolved floor blocks this world; do not fabricate measured evidence.'
                else:prompt+='\nReturn one exact allowed choice ID with a short reason.'
    return ask
