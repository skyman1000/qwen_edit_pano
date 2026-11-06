"""CPU verification of v3 counts, instance edits and export contracts."""
import argparse
import json
from pathlib import Path

from ..common import read
from .conditional_world_audit import audit as contract_audit
from .conditional_world_core import worlds_answer
from .surrounding_inventory import normalize,expand_answer,review_answer


def raw_answer(folder,stem):
    repaired=folder/(stem+'_repair.json')
    return read(repaired if repaired.exists() else folder/(stem+'.json'))


def audit(root):
    root=Path(root);result=contract_audit(root);config=read(root/'config.json')
    case=config['case'];folder=root/case['id'];args=config['arguments']
    contexts=read(folder/'contexts_normalized.json')['contexts']
    capacity=args.get('max_objects',256);budget=args['per_context'];reconstructed=[];counts=[]
    for c in contexts:
        cid=c['id']
        if args.get('proposal_mode','expanded')=='legacy':
            initial=worlds_answer(raw_answer(folder,'contents_'+cid),case,cid,budget,capacity)
            revised=worlds_answer(raw_answer(folder,'review_'+cid),case,cid,budget,capacity)
            for w in revised:
                before=next((r for r in initial if r['id']==w['id']),None)
                counts.append(dict(id=w['id'],initial=len(before['additions']) if before else None,
                                  expanded=None,reviewed=len(w['additions'])))
            reconstructed+=revised;continue
        blueprint=read(folder/f'blueprint_{cid}.normalized.json')
        initial=normalize(raw_answer(folder,'contents_'+cid),case,cid,blueprint['groups'],budget,capacity)
        for plan in initial:
            plan['extent_assumption']=blueprint['extent_assumption']
            if plan!=read(folder/f'initial_{plan["id"]}.json'):raise ValueError('Initial plan replay differs')
            stem=plan['id']
            expanded=expand_answer(raw_answer(folder,'expand_'+stem),plan,case,cid,capacity)
            revised=review_answer(raw_answer(folder,'review_'+stem),expanded,case,cid,capacity)
            if expanded!=read(folder/f'expanded_{stem}.json') or revised!=read(folder/f'reviewed_{stem}.json'):
                raise ValueError('Expansion or edit-ledger replay differs: '+stem)
            counts.append(dict(id=stem,initial=len(plan['additions']),expanded=len(expanded['additions']),
                reviewed=len(revised['additions']),withdrawn=sum(d['action']=='withdraw' for d in revised['review_ledger'])))
            reconstructed.append(revised)
    catalog=read(folder/'world_catalog.json')['worlds']
    actual=[w for w in catalog if w.get('proposal_context_id') is not None]
    if actual!=reconstructed:raise ValueError('Catalog lost, combined or altered complete alternatives')
    coverage=read(folder/'proposal_coverage.json')
    if coverage['stage_counts']!=counts:raise ValueError('Stage counts differ from replayed raw replies')
    result.update(instance_edit_ledger_replayed=True,category_deduplication=False,
                  alternatives_unioned=False,stage_counts=counts)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);args=p.parse_args()
    try:result=audit(args.output)
    except Exception as exc:
        print(json.dumps(dict(status='postflight_failed',error=f'{type(exc).__name__}: {exc}'),ensure_ascii=False))
        raise SystemExit(1)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
