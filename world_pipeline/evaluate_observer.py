"""CPU category-aware IoU matching on explicitly reviewed GT. Not COCO AP."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import read, rows, sha, write, new_output, validate_graph


def iou(a, b):
    inter = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0, min(a[3],b[3])-max(a[1],b[1]))
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/area if area > 0 else 0.


def match(gt, pred, threshold=.5):
    scores = np.zeros((len(gt),len(pred)))
    allowed = np.zeros_like(scores, dtype=bool)
    for i,g in enumerate(gt):
        for j,p in enumerate(pred):
            score = iou(g['bbox2d_visible_xyxy_norm']['value'],p['bbox2d_visible_xyxy_norm']['value'])
            allowed[i,j] = g['category_id']['value'] == p['category_id']['value'] and score >= threshold
            # Maximize valid match COUNT before optimizing IoU; unmatched boxes contribute zero.
            scores[i,j] = (min(len(gt),len(pred))+1+score) if allowed[i,j] else 0
    if not len(gt) or not len(pred):
        return []
    ii,jj = linear_sum_assignment(-scores)
    return [(int(i),int(j)) for i,j in zip(ii,jj) if allowed[i,j]]


def metrics(counts):
    tp,fp,fn = (counts.get(k,0) for k in ('tp','fp','fn'))
    return dict(counts, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iou',type=float,default=.5)
    args=p.parse_args()
    if not 0 < args.iou <= 1:
        p.error('IoU threshold must be in (0,1]')
    complete=read(args.gt/'EXPORT_COMPLETE.json')
    if complete['gt_manifest_sha256'] != sha(args.gt/'gt_manifest.jsonl'):
        raise ValueError('GT manifest changed since export')
    config=read(args.predictions/'run_config.json')
    read(args.predictions/'COMPLETE.json')
    if config['inputs_sha256'] != sha(args.gt/'observer_inputs.jsonl'):
        raise ValueError('Observer predictions use different Local input manifest')
    if config['vocabulary_sha256'] != sha(args.gt/'vocabulary.json'):
        raise ValueError('Observer vocabulary differs from GT')
    decisions=read(args.review)
    selected=set(config['selected_ids'])
    records,skipped=[],[]
    totals={s:Counter() for s in ('train','val')}
    classes={s:{} for s in ('train','val')}
    for row in rows(args.gt/'gt_manifest.jsonl'):
        sid=row['id']
        if sid not in selected:
            continue
        if decisions.get(sid,{}).get('decision') != 'approved' or row['review_flags']:
            skipped.append(dict(id=sid,reason='not_approved_or_numerical_flags'))
            continue
        gtpath=args.gt/row['directory']/'G_obs.json'
        if sha(gtpath) != row['gt_sha256']['G_obs']:
            raise ValueError(f'GT graph changed: {sid}')
        gtgraph=validate_graph(read(gtpath))
        if gtgraph['id'] != sid or gtgraph['split'] != row['split']:
            raise ValueError('GT identity/split mismatch')
        gt=gtgraph['objects']
        status=read(args.predictions/sid/'status.json')
        if status['id'] != sid:
            raise ValueError('Prediction status identity mismatch')
        pred=[]
        if status['status']=='parsed':
            predgraph=validate_graph(read(args.predictions/sid/'G_obs.json'))
            if (predgraph['id'] != sid or predgraph['split'] != row['split'] or
                    predgraph['vocabulary_sha256'] != gtgraph['vocabulary_sha256']):
                raise ValueError('Prediction identity/split/vocabulary mismatch')
            pred=predgraph['objects']
        matches=match(gt,pred,args.iou)
        gi={i for i,j in matches};pj={j for i,j in matches}
        counts=Counter(tp=len(matches),fp=len(pred)-len(matches),fn=len(gt)-len(matches),
                       images=1,invalid_predictions=int(status['status']!='parsed'),
                       unknown_category_predictions=sum(o['category_id']['value'] is None for o in pred))
        totals[row['split']].update(counts)
        for i,o in enumerate(gt):
            category=str(o['category_id']['value'])
            classes[row['split']].setdefault(category,Counter()).update({'tp':int(i in gi),'fn':int(i not in gi)})
        for j,o in enumerate(pred):
            if j not in pj:
                classes[row['split']].setdefault(str(o['category_id']['value']),Counter()).update({'fp':1})
        records.append(dict(id=sid,split=row['split'],**metrics(counts),
            matches=[dict(gt_id=gt[i]['track_id'],pred_id=pred[j]['track_id']) for i,j in matches]))
    if not records:
        raise ValueError('No approved samples. Inspect review.html and set reviewed IDs to approved; numerical flags cannot be overridden here.')
    output=new_output(args.output)
    write(output/'metrics.json',dict(protocol='category_aware_maximum_cardinality_IoU_matching_not_AP',
        iou=args.iou, by_split={s:metrics(c) for s,c in totals.items()},
        by_class={s:{k:metrics(c) for k,c in cc.items()} for s,cc in classes.items()},
        samples=records,skipped=skipped,review_sha256=sha(args.review),
        gt_manifest_sha256=sha(args.gt/'gt_manifest.jsonl'),
        prediction_config_sha256=sha(args.predictions/'run_config.json'),
        limitations=['Review-selected small sample; not representative benchmark.',
                    'GT excludes structural/unlabeled categories and tiny instances; mesh missingness can affect scores.',
                    'Invalid model JSON counted as missed GT, reported separately; FP count is unknowable for unparsable text.',
                    'No room-type accuracy without explicit vocabulary mapping; no metric 3D predictions/metrics.']))
    print({s:metrics(c) for s,c in totals.items()})


if __name__=='__main__':
    main()
