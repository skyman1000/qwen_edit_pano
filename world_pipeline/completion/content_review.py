"""Review saved content hypotheses against RGB; no new Observer or xyz requests.

Adapt claim checking and visual marks, not a full Woodpecker/SoM reproduction.
Review decisions remain model predictions, never certified semantic truth.
"""
import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..common import ROOT, new_output, read, sha, validate_graph, write
from .qwen_upstream import DEFAULT_MODEL, Qwen
from .scene_completion import PROTOCOL as SOURCE_PROTOCOL, map_plan, saved_plan, validate_plan

PROTOCOL = 'saved_content_evidence_review_v1'
DEFAULT_SOURCE = ROOT/'qwen_edit_pano/outputs/ordinary_completion_content_val6_run001'
DECISIONS = ('retain', 'relocate', 'defer', 'withdraw')


def load_inputs(source, only_ids=None):
    config = read(source/'config.json')
    if config.get('protocol') != SOURCE_PROTOCOL or config.get('stage') != 'content':
        raise ValueError('Use the saved ordinary content experiment, not GT or numerical layouts')
    selected = [c for c in config['cases'] if only_ids is None or c['id'] in only_ids]
    if not selected or only_ids and set(only_ids)-{c['id'] for c in selected}:
        raise ValueError('No cases or unknown case IDs')
    result = []
    for case in selected:
        observed = validate_graph(read(case['state']))
        if observed['stage'] != 'prediction' or observed['scope'] != 'observed':
            raise ValueError('Use frozen predicted observations')
        if read(source/case['id']/'input.json') != case:
            raise ValueError('Saved input differs from run configuration')
        plan, provenance = saved_plan(source, case, 160)
        hashes = {**case['source_sha256'], **provenance, str((source/'config.json').resolve()): sha(source/'config.json')}
        generation = source/case['id']/'content.generation.json'
        hashes[str(generation.resolve())] = sha(generation)
        if any(sha(p) != digest for p, digest in hashes.items()):
            raise ValueError('Source image/state/answer changed')
        result.append(dict(case=deepcopy(case), plan=plan, source_sha256=hashes))
    return result


def marked_image(case):
    with Image.open(case['image']) as handle:
        original = handle.convert('RGB')
    original.thumbnail((768, 768))
    marked = original.copy(); draw = ImageDraw.Draw(marked)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    for index, obj in enumerate(case['context']):
        x1,y1,x2,y2 = [v*s for v,s in zip(obj['visible_bbox_xyxy_norm'],
                                        [original.width,original.height]*2)]
        draw.rectangle((x1,y1,x2,y2), outline='#168bc2', width=2)
        label = str(index)
        y1 = min(y1, max(0,original.height-24))
        text_box = draw.textbbox((x1,y1), label, font=font)
        draw.rectangle(text_box, fill='white')
        draw.text((x1,y1), label, fill='#005582', font=font)
    canvas = Image.new('RGB', (original.width*2,original.height+32), 'white')
    canvas.paste(original,(0,32));canvas.paste(marked,(original.width,32))
    title = ImageDraw.Draw(canvas)
    title.text((8,6),'Original RGB',fill='black',font=font)
    title.text((original.width+8,6),'Predicted inventory marks (not GT)',fill='black',font=font)
    return canvas


def prompt(case, plan):
    inventory = [dict(mark=i,id=o['id'],category=o['category'],description=o['description'])
                 for i,o in enumerate(case['context'])]
    return (
        'Review this proposed continuation of a partial indoor photograph. Left: original RGB; '
        'right: the same image with numbered PREDICTED inventory boxes. '
        'The inventory can contain mistakes; use the RGB to check its descriptions. '
        'For each proposed instance separately assess whether its existence is a reasonable '
        'scene hypothesis and whether its stated location is compatible with the photographed '
        'surfaces and objects. Unseen does not mean absent: do not reject an object merely '
        'because it is outside the photograph. If the object is plausible but its location '
        'conflicts with visible space, relocate the same instance rather than deleting it. '
        'Keep meaningful functional relationships. Preserve visible fixtures and empty surfaces. '
        'This object-state task needs distinct physical scene instances; scenery, atmosphere '
        'and assumed components are not automatically additional objects. '
        'Return JSON with scene_type, scene_hypothesis, reviews. Each review has id, decision, '
        'reason, evidence_ids. decision is retain, relocate, defer, or withdraw. '
        'retain keeps the original instance and location; relocate additionally supplies a '
        'replacement placement string describing a plausible unobserved location. defer '
        'preserves an unresolved hypothesis for review. withdraw requires a specific content '
        'reason, such as duplication or being outside the object-state scope, not just a bad '
        'location. evidence_ids is a list of observed IDs relevant to the assessment, and may '
        'be empty when no reference is needed. State uncertainty honestly; plausible does '
        'not mean observed. Address every proposed ID once; do not introduce new objects, '
        'change their names/types, or predict numeric boxes. '
        '\nPREDICTED MARK REGISTRY: '+json.dumps(inventory,ensure_ascii=False,separators=(',',':'))
        +'\nORIGINAL PROPOSAL: '+json.dumps(plan,ensure_ascii=False,separators=(',',':')))


def apply_review(answer, plan, case):
    if (not isinstance(answer,dict) or not isinstance(answer.get('reviews'),list)
            or any(not isinstance(answer.get(k),str) or not answer[k].strip()
                   for k in ('scene_type','scene_hypothesis'))):
        raise ValueError('Need scene_type, scene_hypothesis and reviews')
    source={o['id']:o for o in plan['additions']};observed={o['id'] for o in case['context']}
    ids=[]
    for item in answer['reviews']:
        if not isinstance(item,dict) or not isinstance(item.get('id'),str):
            raise ValueError('Each review needs an existing candidate ID')
        ids.append(item['id'])
    unknown=set(ids)-set(source)
    if unknown:
        missing=set(source)-set(ids)
        mistaken_observations=unknown & observed
        detail=(f'; observed IDs used as review targets: {sorted(mistaken_observations)}'
                if mistaken_observations else '')
        raise ValueError(f'Review target ID mismatch: unexpected={sorted(unknown)}; '
                         f'missing candidates={sorted(missing)}{detail}. '
                         'Review additions only; observed IDs belong in evidence_ids')
    seen=set(); patches={}
    for item in answer['reviews']:
        oid=item['id']
        if oid in seen:
            raise ValueError(f'Duplicate review target ID: {oid}')
        if item.get('decision') not in DECISIONS:
            raise ValueError(f'Invalid review decision for {oid}: {item.get("decision")!r}; '
                             f'expected one of {DECISIONS}')
        if not isinstance(item.get('reason'),str) or not item['reason'].strip():
            raise ValueError('Every decision needs its own reason')
        refs=item.get('evidence_ids')
        if not isinstance(refs,list) or any(not isinstance(r,str) or r not in observed for r in refs):
            raise ValueError('Evidence IDs must refer to actual observed instances')
        if item['decision']=='relocate':
            if not isinstance(item.get('placement'),str) or not item['placement'].strip():
                raise ValueError('Relocation needs a replacement qualitative placement')
        elif item.get('placement') not in (None,'',source[oid]['placement']):
            raise ValueError('Only relocate may change placement')
        if (any(k in item and item[k]!=source[oid][k] for k in ('name','object_type'))
                or any(k in item for k in ('category','center','size'))):
            raise ValueError('Review cannot replace identity or generate geometry')
        patches[oid]=deepcopy(item);seen.add(oid)
    if seen!=set(source):
        raise ValueError('Every original hypothesis needs an explicit decision; no silent deletion; '
                         f'missing candidates={sorted(set(source)-seen)}')
    revised=deepcopy(plan)
    revised.update(scene_type=answer['scene_type'],scene_hypothesis=answer['scene_hypothesis'],additions=[])
    withdrawn=[];deferred=[]
    for obj in plan['additions']:
        change=patches[obj['id']]
        if change['decision'] in ('withdraw','defer'):
            (withdrawn if change['decision']=='withdraw' else deferred).append(dict(original=deepcopy(obj),review=change))
        else:
            obj=deepcopy(obj)
            if change['decision']=='relocate':obj['placement']=change['placement']
            revised['additions'].append(obj)
    revised['content_review']=dict(protocol=PROTOCOL,semantics_verified=False,
        withdrawn_ids=[o['original']['id'] for o in withdrawn],deferred_ids=[o['original']['id'] for o in deferred],
        decisions=deepcopy(answer['reviews']))
    validate_plan(revised,case)
    return revised,withdrawn,deferred


def load_reviewed_plan(source, case, max_objects=160):
    folder=source/case['id']; config=read(source/'config.json')
    if config.get('protocol')!=PROTOCOL:
        raise ValueError('Different content review protocol')
    matches=[entry for entry in config['inputs'] if entry['case']['id']==case['id']]
    original_case={k:v for k,v in case.items() if k not in ('reviewed_plan_sha256','saved_plan_sha256')}
    if len(matches)!=1 or matches[0]['case']!=original_case:
        raise ValueError('Reviewed plan uses different original evidence')
    entry=matches[0]
    if any(sha(p)!=h for p,h in entry['source_sha256'].items()):
        raise ValueError('Original content review sources changed')
    evidence_input=read(folder/'input.json')
    if (evidence_input['case']!=entry['case'] or evidence_input['plan']!=entry['plan']
            or sha(folder/'evidence.png')!=evidence_input['evidence_image_sha256']):
        raise ValueError('Review image or its original evidence binding changed')
    from .geometry import parse_json
    raw,parsed=folder/'review.txt',folder/'review.json'
    if parse_json(raw.read_text())!=read(parsed):
        raise ValueError('Review differs from raw model response')
    revised,withdrawn,deferred=apply_review(read(parsed),entry['plan'],case)
    if revised!=read(folder/'semantic_plan.json') or withdrawn!=read(folder/'withdrawn_hypotheses.json') or deferred!=read(folder/'deferred_hypotheses.json'):
        raise ValueError('Saved reviewed plan differs from the actual decisions')
    if deferred:
        raise ValueError('Deferred hypotheses remain unresolved; do not silently omit them from layout')
    validate_plan(revised,case,max_objects)
    paths=(source/'config.json',raw,parsed,folder/'semantic_plan.json',folder/'review.request.json',
           folder/'withdrawn_hypotheses.json',folder/'deferred_hypotheses.json',folder/'input.json',folder/'evidence.png')
    if read(folder/'review.request.json')['prompt']!=prompt(case,entry['plan']):
        raise ValueError('Review request is not bound to the original proposal')
    return revised,{**entry['source_sha256'],**{str(p.resolve()):sha(p) for p in paths}}


def run(model, inputs, output, seed=3407):
    records=[]
    for entry in inputs:
        case,plan=entry['case'],entry['plan']
        folder=output/case['id'];folder.mkdir()
        record=dict(id=case['id'],status='failed',semantics_verified=False,geometry_generated=False)
        try:
            image=marked_image(case);image.save(folder/'evidence.png')
            write(folder/'input.json',dict(entry,evidence_image_sha256=sha(folder/'evidence.png')))
            write(folder/'original_plan.json',plan)
            answer=model.ask(image,prompt(case,plan),folder/'review.txt',seed=seed,sample=True,
                             profile='official',max_tokens=4096,guard_repetition=True)
            revised,withdrawn,deferred=apply_review(answer,plan,case)
            for name,value in (('semantic_plan.json',revised),('withdrawn_hypotheses.json',withdrawn),
                               ('deferred_hypotheses.json',deferred),
                               ('category_review.json',map_plan(revised,'semantic_lexicon_v1'))):
                write(folder/name,value)
            record.update(status='review_saved_needs_semantic_review',original_instances=len(plan['additions']),
                          retained_instances=len(revised['additions']),withdrawn=len(withdrawn),deferred=len(deferred))
        except Exception as exc:record['error']=str(exc)
        records.append(record);write(folder/'summary.json',record)
        write(output/'summary.json',dict(protocol=PROTOCOL,records=records,model_quality_verified=False))
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Content review (unverified)</h1>'
            +''.join('<article><h2>'+escape(r['id'])+'</h2><img width="900" src="'
              +escape(r['id'],quote=True)+'/evidence.png"><pre>'
              +escape(json.dumps(read(output/r['id']/'semantic_plan.json') if (output/r['id']/'semantic_plan.json').exists() else r,
                                 ensure_ascii=False,indent=2))+'</pre></article>' for r in records))
        print(json.dumps(record,ensure_ascii=False),flush=True)
    if any(sha(p)!=h for entry in inputs for p,h in entry['source_sha256'].items()):
        raise ValueError('Source data changed during review')
    return all(r['status']!='failed' for r in records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--only-id',action='append')
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--seed',type=int,default=3407)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if not 0<=args.seed<2**63:parser.error('Invalid seed')
    inputs=load_inputs(args.source.resolve(),args.only_id)
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed',cases=len(inputs),requests=len(inputs),
                              candidates=sum(len(e['plan']['additions']) for e in inputs),
                              model_loaded=False,gpu_started=False)))
        return
    output=new_output(args.output)
    code=[Path(__file__),Path(__file__).with_name('scene_completion.py'),
          Path(__file__).with_name('semantic_categories.py'),Path(__file__).with_name('qwen_upstream.py')]
    write(output/'config.json',dict(protocol=PROTOCOL,inputs=inputs,seed=args.seed,model=str(args.model.resolve()),
                                   code_sha256={str(p.resolve()):sha(p) for p in code}))
    if not run(Qwen(args.model),inputs,output,args.seed):raise SystemExit('Some reviews failed; inspect original answers')


if __name__=='__main__':main()
