"""Reconcile visible detections without erasing valid proposals on duplicates.

Exact copies are collapsed losslessly. Overlap only requests an explicit
identity prediction; it never automatically suppresses a distinct instance.
This module does not infer hidden objects, metric boxes, or causal mechanisms.
"""
import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path

from PIL import ImageDraw

from ..common import read,write
from .fresh_observer import PROMPT,answer,state_from_answer
from .world_scm_runner import ask_valid


def unique_inventory(value):
    clean=answer(value);seen={};unique=[];duplicates=[]
    for index,row in enumerate(clean['objects']):
        key=json.dumps(row,sort_keys=True,ensure_ascii=False)
        if key in seen:duplicates.append(dict(raw_index=index,identical_to_raw_index=seen[key]))
        else:seen[key]=index;unique.append(deepcopy(row))
    return dict(objects=unique,evidence_description=clean['evidence_description']),duplicates


def iou(a,b):
    area=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    total=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/total if total>0 else 0.


def identity_candidates(objects):
    pairs=[]
    for j,b in enumerate(objects):
        for i,a in enumerate(objects[:j]):
            if a['category'].strip().lower()==b['category'].strip().lower() and iou(a['bbox_1000'],b['bbox_1000'])>=.8:
                pairs.append([f'p{i:03d}',f'p{j:03d}'])
    return pairs


def identity_answer(value,ids,pairs=None):
    rows=value.get('groups') if isinstance(value,dict) else None
    if not isinstance(rows,list):raise ValueError('groups must partition every requested proposal ID')
    used=[];kept=[]
    for r in rows:
        if not isinstance(r,dict) or set(r)!={'members','representative_id'} or not isinstance(r['members'],list) or not r['members'] or any(not isinstance(i,str) for i in r['members']):raise ValueError('Each group needs members and representative_id')
        if not isinstance(r['representative_id'],str) or r['representative_id'] not in r['members']:raise ValueError('Representative must belong to its physical-instance group')
        used+=r['members'];kept.append(r['representative_id'])
    if len(set(used))!=len(used) or set(used)!=set(ids):raise ValueError('Use every requested proposal ID once; no invented IDs, omitted boxes or duplicated membership')
    if pairs is not None:
        adjacent={i:{i} for i in ids}
        for a,b in pairs:adjacent[a].add(b);adjacent[b].add(a)
        for r in rows:
            reachable=set();todo=[r['members'][0]]
            while todo:
                i=todo.pop()
                if i in reachable:continue
                reachable.add(i);todo+=list(adjacent[i]-reachable)
            if not set(r['members'])<=reachable:raise ValueError('Do not merge unrelated overlap components')
    return dict(groups=deepcopy(rows),keep_ids=kept)


class CountedProvider:
    def __init__(self,inner,out,initial_calls):self.inner=inner;self.calls=0;self.out=out;self.initial_calls=initial_calls
    def ask(self,*args,**kwargs):
        self.calls+=1;write(self.out/'reconciliation_usage.json',dict(source_chunk_calls=self.initial_calls,reconciliation_calls=self.calls,actual_calls=self.initial_calls+self.calls))
        return self.inner.ask(*args,**kwargs)


def reconcile(provider,root,out,backend,image,config,accepted,raw,initial_calls,max_tokens):
    candidates,duplicates=unique_inventory(dict(objects=accepted+answer(raw)['objects'],evidence_description=raw['evidence_description']))
    write(out/'reconciliation_candidates.json',dict(**candidates,exact_duplicate_copies=duplicates,exported_as_observation=False))
    client=CountedProvider(provider,out,initial_calls)
    prompt=(PROMPT+'\nWHOLE-IMAGE INVENTORY REVIEW. The previous incremental response repeated object records. '
        'Below are PROPOSALS, not accepted observations. Inspect the original RGB photograph again. '
        'Return the COMPLETE visible inventory across the whole image, including missed object types. '
        'Correct inaccurate boxes and floor flags, merge proposals only when they describe the SAME physical instance. '
        'Distinct nearby objects remain separate. Parts are not additional whole objects. '
        'This is not a request for additional objects or workflow status. Do not return done or an empty status message. '
        'Return only objects and evidence_description. No target count and no hidden objects. '
        '\nPROPOSED_VISIBLE_INVENTORY: '+json.dumps(candidates['objects'],ensure_ascii=False))
    def validate_review(value):
        clean,dups=unique_inventory(value)
        if not clean['objects'] and candidates['objects']:raise ValueError('Contradictory empty inventory: a duplicate correction cannot erase all visible proposals; provide the complete reviewed inventory')
        return dict(value=clean,exact_duplicate_copies=dups)
    result=ask_valid(client,image,prompt,out/'whole_image_review.txt',validate_review,max_tokens)
    reviewed=result['value'];pairs=identity_candidates(reviewed['objects']);identity=None
    if pairs:
        ids=sorted({i for pair in pairs for i in pair});records={f'p{i:03d}':row for i,row in enumerate(reviewed['objects'])}
        identity_prompt=('Review VISIBLE INSTANCE IDENTITY in the original image. These overlapping proposal boxes may describe '
            'one physical object or distinct objects; overlap alone is not a reason to merge. '
            'Return {"groups":[{"members":["exact proposal IDs"],"representative_id":"one member ID"}]}. '
            'Partition EVERY requested ID exactly once. A group means the SAME physical instance, '
            'and its representative is the most accurate supplied visible rectangle. Distinct objects use singleton groups. '
            'Do not delete a physical object, invent a new box, infer hidden objects, or change a coordinate. '
            '\nREQUESTED_IDS: '+json.dumps(ids)+'\nOVERLAP_PAIRS_FOR_REVIEW: '+json.dumps(pairs)+
            '\nPROPOSALS: '+json.dumps({i:records[i] for i in ids},ensure_ascii=False))
        identity=ask_valid(client,image,identity_prompt,out/'instance_identity_review.txt',lambda v:identity_answer(v,ids,pairs),min(max_tokens,4096))
        keep=set(identity['keep_ids'])
        reviewed['objects']=[row for i,row in enumerate(reviewed['objects']) if f'p{i:03d}' not in ids or f'p{i:03d}' in keep]
    write(out/'reconciliation_result.json',dict(reviewed_inventory=reviewed,review_exact_duplicates=result['exact_duplicate_copies'],
        overlap_pairs=pairs,identity_review=identity,automatic_overlap_suppression=False,identity_is_model_prediction=True,
        source_candidate_count=len(candidates['objects']),calls=client.calls))
    state,mapping=state_from_answer(reviewed,config,backend)
    # Publish only after both whole-image and identity responses validate.
    write(out/'G_2d.json',state);write(out/'labels.json',mapping)
    write(out/'grounding.json',dict(instances=[dict(id=o['track_id'],variant=r['name'],on_floor=r['on_floor']) for o,r in zip(state['objects'],reviewed['objects'])],evidence_description=reviewed['evidence_description']))
    preview=image.copy();draw=ImageDraw.Draw(preview)
    for o in state['objects']:
        b=o['bbox2d_visible_xyxy_norm']['value'];box=[b[0]*image.width,b[1]*image.height,b[2]*image.width,b[3]*image.height]
        draw.rectangle(box,outline='#287db5',width=2);draw.text(tuple(box[:2]),o['track_id']+' '+o['description'],fill='#287db5')
    preview.save(out/'boxes2d.png')
    summary=dict(status='completed',objects=len(state['objects']),backend=backend,request_interface='chunked_with_whole_image_reconciliation_v2',
        actual_calls=initial_calls+client.calls,maximum_calls=20,source_chunk_calls=initial_calls,reconciliation_calls=client.calls,
        partial_inventory_exported=False,exact_duplicate_copies_collapsed=len(duplicates),identity_review_performed=bool(pairs),
        automatic_overlap_suppression=False,visual_quality_verified=False)
    write(out/'summary.json',summary);return summary


def preview_suite(root):
    root=Path(root).resolve();suite=read(root/'suite.json')
    html='<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;margin:24px}img{max-width:900px;width:100%}td,th{padding:8px;border:1px solid #bbb}</style><h1>局部可见物体：2D观察器验收</h1><p>本页只展示模型预测的可见实例，不是隐藏世界或3D结果。先核对漏检、重复、框和类别；没有调用API或WildDet3D。</p>'
    for s in suite['scenes']:
        out=root/'scenes'/s['id']/'observation/qwen/observer2d';summary=read(out/'summary.json') if (out/'summary.json').exists() else dict(status='not_run')
        html+='<h2>'+escape(s['id'])+'</h2><pre>'+escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'
        if (out/'boxes2d.png').exists():html+='<img src="'+str((out/'boxes2d.png').relative_to(root))+'">'
        else:html+='<img src="'+str((out.parent.parent/'input.png').relative_to(root))+'">'
        if (out/'reconciliation_candidates.json').exists():html+='<p><a href="'+str((out/'reconciliation_candidates.json').relative_to(root))+'">保留的候选与精确重复记录</a></p>'
        if (out/'G_2d.json').exists():
            state=read(out/'G_2d.json');html+='<table><tr><th>ID</th><th>类别</th><th>可见框</th></tr>'
            for o in state['objects']:html+='<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in (o['track_id'],o['raw_category'],o['bbox2d_visible_xyxy_norm']['value']))+'</tr>'
            html+='</table>'
    page=root/'observation_index.html';page.write_text(html);return page


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();print(preview_suite(a.root))
