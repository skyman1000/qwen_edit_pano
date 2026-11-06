"""One generation, then at most one bounded field patch; immutable journals."""
import hashlib
import json

from PIL import Image

from ...common import read,sha
from ..assembly_experiment import stage_seed
from ..candidate_repair.requests import Requests,RecordedFailure,save
from .contract import apply_patch,repair_paths


class PatchRequests(Requests):
    def ask(self,name,prompt,validator,budget=8192):
        receipt=self.folder/(name+'.receipt.json')
        binding=dict(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),image_sha256=sha(self.board),
            seed=self.seed,max_tokens=budget,attempts=2,repair='bounded_fields_preserve_inventory_v1')
        if receipt.exists():
            old=read(receipt)
            if old['binding']!=binding or any(sha(p)!=h for p,h in old['artifacts'].items()):raise ValueError('Saved proposal request binding changed')
            if old['status']=='failed':raise RecordedFailure(old['error'])
            if hashlib.sha256(json.dumps(old['value'],sort_keys=True).encode()).hexdigest()!=old['value_sha256']:
                raise ValueError('Accepted proposal value changed')
            return old['value']
        errors=[];draft=None
        for attempt in range(2):
            path=self.folder/(name+f'_attempt{attempt}.txt')
            if path.exists() or path.with_suffix('.request.json').exists():raise RecordedFailure('Interrupted request without receipt; preserve it and use a new run')
            try:
                if attempt:
                    patch_prompt=('Repair only the explicit fields of this JSON proposal. Return exactly '
                        '{"edits":[{"path":"one allowed JSON pointer","value":null,"reason":"specific correction"}]}. '
                        'Never delete objects, change identities/categories, remove relations, or rewrite valid fields '
                        'unrelated to the reported error. An empty edits list leaves the same proposal unchanged. '
                        'Correct physical installation explicitly; never substitute near for inside or rename an unknown host. '
                        '\nERROR: '+errors[-1]['error']+'\nALLOWED_PATHS: '+json.dumps(sorted(repair_paths(draft)))+
                        '\nCOMPLETE_ORIGINAL_PROPOSAL: '+json.dumps(draft,ensure_ascii=False)+'\nORIGINAL_TASK: '+prompt)
                with Image.open(self.board) as im:
                    raw=self.client().ask(im.convert('RGB'),prompt if not attempt else patch_prompt,path,
                        seed=stage_seed(self.seed,name+':attempt:'+str(attempt)),max_tokens=budget if not attempt else 3072,
                        profile='completion',sample=True)
                if not isinstance(raw,dict):raise ValueError('Return a JSON object')
                if not attempt:draft=raw
                else:
                    draft=apply_patch(draft,raw)
                    save(self.folder/(name+'_attempt1_applied.json'),draft)
                value=validator(draft)
            except Exception as exc:
                error=dict(attempt=attempt,error=type(exc).__name__+': '+str(exc));errors.append(error)
                save(path.with_suffix('.error.json'),error)
                # A malformed JSON has no stable inventory to patch. Do not
                # replace it by a fresh draw under the name of repair.
                if not isinstance(exc,ValueError) or attempt or not isinstance(draft,dict) or not repair_paths(draft):
                    self.finish(receipt,binding,name,dict(status='failed',error=error['error'],errors=errors));raise
                continue
            digest=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
            self.finish(receipt,binding,name,dict(status='accepted',value=value,value_sha256=digest,errors=errors))
            return value
