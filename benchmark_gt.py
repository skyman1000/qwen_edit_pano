"""Matched native / panorama / full-GT-state benchmark; reuse existing RGB and metric records."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from .common import ROOT, atomic_json, read_jsonl, sample_seed
from .benchmark import OUT as BASE_OUT, EVAL_PY, evaluation_options

CHECKPOINT=ROOT/'qwen_edit_pano/outputs/object_world_id_A_dual_full_v1/run-1792820440973994739/checkpoint-step010000'
OUT=ROOT/'qwen_edit_pano/outputs/native_pano_gt_A10000_val923_v1'
ARMS=('native','finetuned','gt_state')


def read(path):return json.loads(path.read_text())


def jsonl(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))


def prepare():
    from .world_pipeline.object_world.data import load_pack,identity
    cfg=read(CHECKPOINT/'config.json');marker=read(CHECKPOINT/'COMPLETE.json')
    if marker['step']!=10000 or cfg['group']!='A' or cfg['condition']!='full' or cfg['pano_lora_scale']!=.5:
        raise ValueError('Expected completed A10000 full-state checkpoint at panorama scale0.5')
    pack,records=load_pack(cfg['pack'])
    if cfg['pack_plan']!=pack:raise ValueError('Training pack changed')
    base_plan=read(BASE_OUT/'plan.json');base_sources=read(BASE_OUT/'GENERATION_COMPLETE.json')['sources']
    paired={r['id']:r for r in read_jsonl(Path(base_plan['manifest']))}
    val={r['id']:r for r in records if r['split']=='val'}
    ids=[sid for sid in base_plan['ids'] if sid in val]
    if len(ids)!=923 or len(val)!=923:raise ValueError('Expected complete 923-image GT validation subset')
    for key,base_key in [('base_snapshot','model'),('base_checkpoint','checkpoint'),('height','height'),('prompt','prompt')]:
        if pack[key]!=base_plan[base_key]:raise ValueError('Base generation setting differs: '+key)
    for sid in ids:
        compact=identity(val[sid]);original=paired[sid]
        for key in ('scene_id','source_view_id'):
            if compact[key]!=original[key]:raise ValueError('Sample identity differs: '+sid)
        for key in ('local_condition','image','target_alignment'):
            for field,value in compact[key].items():
                if original[key][field]!=value:raise ValueError('Source/alignment differs: '+sid+'/'+key+'/'+field)
        for arm in ('native','finetuned'):
            if sid not in base_sources[arm]:raise ValueError('Missing previously completed baseline image: '+sid)
    excluded=sorted(set(base_plan['ids'])-set(ids))
    plan=dict(ids=ids,samples=923,houses=sorted({val[sid]['scene_id'] for sid in ids}),arms=list(ARMS),
              checkpoint=str(CHECKPOINT),checkpoint_step=10000,pack=cfg['pack'],
              base_checkpoint=pack['base_checkpoint'],model=pack['base_snapshot'],prompt=pack['prompt'],
              lora_scale=.5,height=1024,width=2048,steps=28,cfg=4.,seed=0,
              seed_mode='per-id',world_cfg='same full GT state in positive and negative CFG branches',
              baseline_run=str(BASE_OUT),excluded_from_original925=excluded,
              note='GT-available validation subset selected by pack availability, not generated quality. Oracle, not deployable predicted-state generation.')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (OUT/'plan.json').exists() and read(OUT/'plan.json')!=plan:raise ValueError('Frozen benchmark plan changed')
        atomic_json(OUT/'plan.json',plan)
        for name in ('generation','manifests'):(OUT/name).mkdir(exist_ok=True)
    return plan,val


def collect(shard,plan):
    from PIL import Image
    expected=set(plan['ids'][shard::2]);found={}
    for run in sorted((OUT/'generation').glob(f'shard{shard}_attempt[0-9][0-9][0-9]')):
        if not (run/'run_config.json').is_file():continue
        cfg=read(run/'run_config.json')
        for key,value in dict(checkpoint=str(CHECKPOINT),seed=0,seed_mode='per-id',steps=28,cfg=4.,modes=['correct'],oracle=True,split='val').items():
            if cfg.get(key)!=value:raise ValueError('GT generation setting differs: '+key)
        if not set(cfg['ids'])<=expected:raise ValueError('Shards overlap or unexpected image IDs')
        for sid in cfg['ids']:
            d=run/sid
            if not all((d/name).is_file() for name in ('correct.json','correct.png','gt_erp.png','local.png')):continue
            meta=read(d/'correct.json')
            if meta['seed']!=sample_seed(0,sid,'per-id') or meta['world_source']!=sid or not meta['oracle']:
                raise ValueError('GT state/sample/seed mismatch: '+sid)
            with Image.open(d/'correct.png') as im:
                if im.size!=(2048,1024):raise ValueError('Wrong ERP dimensions')
            found.setdefault(sid,dict(image=str(d/'correct.png'),gt=str(d/'gt_erp.png'),local=str(d/'local.png')))
    return found


def generate(shard,dry=False):
    plan,_=prepare()
    with (OUT/f'.shard{shard}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        found=collect(shard,plan)
        missing=[sid for sid in plan['ids'][shard::2] if sid not in found]
        if missing:
            number=1
            while (OUT/'generation'/f'shard{shard}_attempt{number:03d}').exists():number+=1
            run=OUT/'generation'/f'shard{shard}_attempt{number:03d}'
            selection=run.with_suffix('.json')
            atomic_json(selection,dict(ids=missing))
            command=[sys.executable,'-u','-m','qwen_edit_pano.world_pipeline.id_binding_world.inference',
                     '--checkpoint',str(CHECKPOINT),'--output',str(run),'--split','val',
                     '--ids-file',str(selection),'--limit',str(len(missing)),
                     '--steps','28','--cfg','4','--seed','0','--seed-mode','per-id','--modes','correct']
            print(dict(shard=shard,reused=len(found),generate=len(missing),modes=['correct'],seed_mode='per-id'),flush=True)
            if dry:
                print(command);return
            subprocess.run(command,cwd=ROOT,check=True)
        if dry:return
        found=collect(shard,plan)
        if set(found)!=set(plan['ids'][shard::2]):raise ValueError('GT shard incomplete')
        atomic_json(OUT/f'SHARD{shard}_COMPLETE.json',dict(shard=shard,samples=len(found)))


def manifests():
    plan,_=prepare()
    gt={**collect(0,plan),**collect(1,plan)}
    if set(gt)!=set(plan['ids']):raise ValueError(f'GT generation incomplete: {len(gt)}/923')
    old=read(BASE_OUT/'GENERATION_COMPLETE.json')['sources']
    sources={arm:{sid:old[arm][sid] for sid in plan['ids']} for arm in ('native','finetuned')}
    sources['gt_state']=gt
    for arm in ARMS:
        jsonl(OUT/'manifests'/f'{arm}.jsonl',[dict(id=sid,image=sources[arm][sid]['image'],prompt=plan['prompt'],seed=sample_seed(0,sid,'per-id')) for sid in plan['ids']])
    jsonl(OUT/'manifests/reference.jsonl',[dict(id=sid,image=sources['finetuned'][sid]['gt']) for sid in plan['ids']])
    atomic_json(OUT/'GENERATION_COMPLETE.json',dict(samples=923,arms=list(ARMS),sources=sources,
                state_conditioning=dict(native=False,finetuned=False,gt_state=True)))
    # Local/Cube records are per image: filter the completed 925-image scores,
    # rather than regenerate or rerun perceptual models for baseline arms.
    for arm in ('native','finetuned'):
        target=OUT/'paired_metrics_512_cube'/arm
        target.mkdir(parents=True,exist_ok=True)
        cache=target/'per_sample.jsonl'
        if not cache.exists():
            old_cache=BASE_OUT/'paired_metrics_512_cube'/arm/'per_sample.jsonl'
            rows=[r for r in read_jsonl(old_cache) if r['id'] in set(plan['ids'])]
            if len(rows)!=1846:raise ValueError('Prior Local/Cube records incomplete')
            jsonl(cache,rows)
            atomic_json(target/'REUSED_RECORDS.json',dict(source=str(old_cache),samples=923,records=1846))
    print(OUT/'manifests',flush=True)


def evaluate(arm,tag='evaluation',dry=False):
    if not (OUT/'GENERATION_COMPLETE.json').exists():raise ValueError('Run manifests after both GT shards finish')
    target=OUT/(tag+'_preflight' if dry else tag)/arm
    options=evaluation_options()
    options[options.index('--reference-manifest')+1]=str(OUT/'manifests/reference.jsonl')
    options[options.index('--reference-label')+1]='923 common GT-available polished validation ERP; 11 houses; three matched arms; Oracle GT state only for gt_state'
    if target.exists():
        if not dry and read(target/'summary.json').get('complete'):print('Reuse completed 11-metric report:',target,flush=True)
        else:raise ValueError(f'Preserve incomplete report {target}; use --eval-tag evaluation_retry1')
    else:
        subprocess.run([str(EVAL_PY),'-u','-m','evaluation.evaluate','--generated-manifest',str(OUT/'manifests'/f'{arm}.jsonl'),
                        '--output',str(target),*options,*(['--dry-run'] if dry else [])],cwd=ROOT/'DiT360',check=True)
    if not dry:
        subprocess.run([str(EVAL_PY),'-u','-m','qwen_edit_pano.benchmark_reference_metrics',
                        '--benchmark',str(OUT),'--arm',arm,'--device','cuda:0'],cwd=ROOT,check=True)


def summarize(tag):
    plan=read(OUT/'plan.json')
    main={a:read(OUT/tag/a/'summary.json') for a in ARMS}
    paired={a:read(OUT/'paired_metrics_512_cube'/a/'summary.json') for a in ARMS}
    for arm in ARMS:
        if not main[arm]['complete'] or main[arm]['n_generated_images']!=923 or main[arm]['n_reference_images']!=923:
            raise ValueError('Incomplete/common-subset evaluation: '+arm)
        if not paired[arm]['complete'] or paired[arm]['samples']!=923:raise ValueError('Incomplete paired metrics: '+arm)
        if main[arm]['package_versions']!=main['native']['package_versions']:raise ValueError('Metric environment differs')
        for key,value in main['native']['config'].items():
            if key not in ('generated_manifest','output') and main[arm]['config'][key]!=value:raise ValueError('Metric protocol differs: '+key)
    higher={'is','cs','qaquality','qaaesthetic'};rows=[]
    for metric in main['native']['selected_metrics']:
        if any(main[a]['results'][metric]['status']!='ok' for a in ARMS):raise ValueError('Metric failed: '+metric)
        rows.append(dict(metric=metric,direction='higher' if metric in higher else 'lower',
                         **{a:main[a]['results'][metric]['value'] for a in ARMS}))
    for label,key in [('Local','local_diagnostic'),('Cube','cube_mean'),('Nonfront','nonfront_mean'),('Back','back')]:
        for metric in ('lpips','ssim','dists'):
            rows.append(dict(metric=f'{label}/{metric}',direction='higher' if metric=='ssim' else 'lower',
                             **{a:paired[a]['means'][key][metric] for a in ARMS}))
    with (OUT/'comparison_three_methods.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(OUT/'comparison_three_methods.json',dict(samples=923,world_checkpoint=plan['checkpoint'],results=rows,
                note='Native vs panorama vs panorama+GT-state Oracle. GT-state minus panorama isolates the added state pathway under matched generation. No predicted-state deployment claim.'))
    text=['| Metric | Direction | Native Edit | Panorama | Panorama + GT state |','|---|---|---:|---:|---:|']
    for r in rows:text.append(f"| {r['metric']} | {r['direction']} | {r['native']:.4f} | {r['finetuned']:.4f} | {r['gt_state']:.4f} |")
    (OUT/'comparison_three_methods.md').write_text('\n'.join(text)+'\n')
    print(OUT/'comparison_three_methods.md',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','generate','manifests','evaluate','summarize'))
    p.add_argument('--shard',type=int,choices=(0,1),default=0)
    p.add_argument('--arm',choices=ARMS,default='gt_state')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--eval-tag',default='evaluation')
    args=p.parse_args()
    if Path(args.eval_tag).name!=args.eval_tag or args.eval_tag in ('.','..','generation','manifests'):p.error('Invalid eval-tag')
    if args.action=='prepare':
        plan,_=prepare();print({k:plan[k] for k in ['samples','houses','checkpoint','excluded_from_original925']})
    elif args.action=='generate':generate(args.shard,args.dry_run)
    elif args.action=='manifests':manifests()
    elif args.action=='evaluate':evaluate(args.arm,args.eval_tag,args.dry_run)
    else:summarize(args.eval_tag)


if __name__=='__main__':main()
