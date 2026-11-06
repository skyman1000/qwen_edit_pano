"""Native Edit vs epoch015/LoRA0.5 on the full paired validation split; no World state."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from .common import ROOT, atomic_json, read_jsonl, sample_seed, sha256

OUT = ROOT / 'qwen_edit_pano/outputs/native_vs_pano_epoch015_scale05_val925_v1'
MANIFEST = ROOT / 'qwen_edit_pano/data/paired_full_v1/val.jsonl'
CHECKPOINT = ROOT / 'qwen_edit_pano/outputs/paired_full_63637/checkpoint-epoch015'
EVAL_PY = ROOT / 'DiT360/.venv/bin/python'
ARMS = ('native', 'finetuned')


def read(path):
    return json.loads(path.read_text())


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))


def prepare():
    rows = read_jsonl(MANIFEST)
    train = read_jsonl(MANIFEST.with_name('train.jsonl'))
    if len(rows) != 925 or len({r['id'] for r in rows}) != 925:
        raise ValueError('Expected the complete 925-image validation manifest')
    houses = {r['scene_id'] for r in rows}
    if len(houses) != 11 or houses & {r['scene_id'] for r in train}:
        raise ValueError('Validation split/house separation changed')
    if not (CHECKPOINT/'COMPLETE.json').is_file():
        raise ValueError('Checkpoint incomplete')
    cfg = read(CHECKPOINT/'pano_config.json')
    plan = dict(ids=[r['id'] for r in rows], samples=925, houses=sorted(houses), arms=list(ARMS),
                manifest=str(MANIFEST), manifest_sha256=sha256(MANIFEST),
                checkpoint=str(CHECKPOINT), model=cfg['base_snapshot'], prompt=cfg['prompt'],
                steps=28, cfg=4., seed=0, lora_scale=.5, height=1024, width=2048,
                conditioning='Local RGB + fixed ordinary task prompt; no GT state, no World Adapter',
                comparison='native: no LoRA/no panorama padding; finetuned: epoch015/LoRA0.5 + panorama padding',
                reference='925 paired polished ERP GT images; 11 official validation houses; GT read after generation',
                seeds='one deterministic per-ID seed; one generation per image per arm; no multi-seed sweep')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'plan.json'
    with (OUT/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and read(path) != plan:
            raise ValueError('Existing benchmark settings differ; preserve this run and use a new version')
        atomic_json(path, plan)
        for name in ('generation', 'manifests'):
            (OUT/name).mkdir(exist_ok=True)
    return plan, rows


def collect(arm, plan, rows):
    """Reuse only completed sample records from this benchmark, including interrupted attempts."""
    from PIL import Image
    expected = {r['id']:r for r in rows}
    found = {}
    for run in sorted((OUT/'generation').glob(arm+'_attempt[0-9][0-9][0-9]')):
        config_path, subset = run/'run_config.json', run.with_suffix('.jsonl')
        if not config_path.is_file() or not subset.is_file():
            continue
        cfg = read(config_path)
        wanted = dict(base_snapshot=plan['model'], prompt=plan['prompt'], steps=28, cfg=4., seed=0,
                      height=1024, width=2048, lora_scale=None if arm=='native' else .5,
                      checkpoint='None' if arm=='native' else str(CHECKPOINT),
                      padding='none' if arm=='native' else 'persistent_target_only')
        if any(cfg.get(k) != v for k,v in wanted.items()):
            raise ValueError('Generation settings differ: '+str(run))
        if cfg['manifest_sha256'] != sha256(subset):
            raise ValueError('Attempt manifest changed: '+str(subset))
        for row in read_jsonl(subset):
            sid = row['id']
            if row != expected.get(sid):
                raise ValueError('Attempt row differs from frozen validation data: '+sid)
            directory = run/sid
            if not (directory/'sample.json').is_file():
                continue
            meta = read(directory/'sample.json')
            if meta['seed'] != sample_seed(0,sid,'per-id') or meta['output_size'] != [2048,1024]:
                raise ValueError('Wrong sample seed/size: '+sid)
            files = dict(image=str(directory/'generated_erp.png'), gt=str(directory/'gt_erp.png'),
                         local=str(directory/'input.png'))
            if not all(Path(f).is_file() for f in files.values()):
                continue
            with Image.open(files['image']) as im:
                if im.size != (2048,1024):
                    raise ValueError('Wrong generated dimensions: '+sid)
            found.setdefault(sid, files)
    return found


def generate(arm):
    plan, rows = prepare()
    with (OUT/f'.{arm}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        found = collect(arm, plan, rows)
        missing = [r for r in rows if r['id'] not in found]
        if missing:
            number = 1
            while (OUT/'generation'/f'{arm}_attempt{number:03d}').exists():
                number += 1
            run = OUT/'generation'/f'{arm}_attempt{number:03d}'
            subset = run.with_suffix('.jsonl')
            jsonl(subset, missing)
            command = [sys.executable, '-u', '-m', 'qwen_edit_pano.paired_inference',
                       '--local-files-only', '--model', plan['model'], '--prompt', plan['prompt'],
                       '--manifest', str(subset), '--output', str(run), '--limit', str(len(missing)),
                       '--steps', '28', '--cfg', '4', '--seed', '0']
            if arm == 'finetuned':
                command += ['--checkpoint', str(CHECKPOINT), '--lora-scale', '0.5']
            print(f'{arm}: reuse {len(found)}, generate {len(missing)} missing images', flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        found = collect(arm, plan, rows)
        if set(found) != set(plan['ids']):
            raise ValueError(f'Incomplete {arm}: {len(found)}/925')
        atomic_json(OUT/f'{arm.upper()}_COMPLETE.json', dict(arm=arm, samples=len(found)))


def manifests():
    plan, rows = prepare()
    sources = {arm:collect(arm,plan,rows) for arm in ARMS}
    if any(set(source) != set(plan['ids']) for source in sources.values()):
        raise ValueError('Finish generation for both arms before evaluation')
    for arm in ARMS:
        jsonl(OUT/'manifests'/f'{arm}.jsonl', [dict(id=sid, image=sources[arm][sid]['image'],
              prompt=plan['prompt'], seed=sample_seed(0,sid,'per-id')) for sid in plan['ids']])
    jsonl(OUT/'manifests/reference.jsonl', [dict(id=sid, image=sources['finetuned'][sid]['gt']) for sid in plan['ids']])
    atomic_json(OUT/'GENERATION_COMPLETE.json', dict(samples=925, arms=list(ARMS), sources=sources,
                                                   world_state_used=False))
    print(OUT/'manifests', flush=True)


def evaluation_options():
    return ['--reference-manifest', str(OUT/'manifests/reference.jsonl'),
        '--reference-label', '925 aligned polished ERP; 11 held-out official validation houses; no GT state conditioning',
        '--metrics', 'all', '--device', 'cuda:0', '--batch-size', '1',
        '--clip-model', 'openai/clip-vit-base-patch16',
        '--clip-model-path', '/data-nfs/gpu1-2/u13529658780/.cache/huggingface/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3',
        '--qalign-model-path', '/data-nfs/gpu1-2/u13529658780/.cache/huggingface/hub/models--q-future--one-align/snapshots/dcc603b95aa0ebd82afa696d4a1e20d11fc80ddb',
        '--clip-crop-fraction', '.125', '--face-size', '512',
        '--panfusion-root', str(ROOT/'benchmark_assets/PanFusion'),
        '--faed-weights', str(ROOT/'benchmark_assets/weights/faed.ckpt'),
        '--faed-height', '512', '--faed-preprocess', 'pil-bicubic',
        '--places-weights', str(ROOT/'benchmark_assets/weights/resnet18_places365.pth.tar'),
        '--places-arch', 'resnet18', '--is-splits', '1']


def evaluate(arm, dry=False, tag='evaluation'):
    if not (OUT/'GENERATION_COMPLETE.json').is_file():
        raise ValueError('Run manifests after both generation arms complete')
    target = OUT/(tag+'_preflight' if dry else tag)/arm
    if target.exists():
        if not dry and read(target/'summary.json').get('complete'):
            print('Completed 11-metric report exists:', target, flush=True)
        else:
            raise ValueError(f'Preserve incomplete report {target}; retry with --eval-tag evaluation_retry1')
    else:
        subprocess.run([str(EVAL_PY), '-u', '-m', 'evaluation.evaluate',
            '--generated-manifest', str(OUT/'manifests'/f'{arm}.jsonl'), '--output', str(target),
            *evaluation_options(), *(['--dry-run'] if dry else [])], cwd=ROOT/'DiT360', check=True)
    if not dry:
        subprocess.run([str(EVAL_PY), '-u', '-m', 'qwen_edit_pano.benchmark_reference_metrics',
                        '--benchmark', str(OUT), '--arm', arm, '--device', 'cuda:0'], cwd=ROOT, check=True)


def summarize(tag):
    reports = {a:read(OUT/tag/a/'summary.json') for a in ARMS}
    if any(not r.get('complete') or r['n_generated_images'] != 925 or r['n_reference_images'] != 925
           for r in reports.values()):
        raise ValueError('Both 925-image evaluations must finish successfully')
    for key,value in reports['native']['config'].items():
        if key not in ('generated_manifest','output') and value != reports['finetuned']['config'][key]:
            raise ValueError('Evaluation setting differs: '+key)
    if reports['native']['package_versions'] != reports['finetuned']['package_versions']:
        raise ValueError('Evaluation environments differ')
    rows = []
    higher = {'is','cs','qaquality','qaaesthetic','ssim'}
    for metric in reports['native']['selected_metrics']:
        a,b = [reports[arm]['results'][metric] for arm in ARMS]
        if a['status'] != 'ok' or b['status'] != 'ok':
            raise ValueError('Metric failed: '+metric)
        rows.append(dict(metric=metric, direction='higher' if metric in higher else 'lower',
                         native=a['value'], finetuned=b['value'], delta=b['value']-a['value']))
    paired = {a:read(OUT/'paired_metrics_512_cube'/a/'summary.json') for a in ARMS}
    if any(not p.get('complete') for p in paired.values()):
        raise ValueError('Paired perceptual metrics incomplete')
    for region in ('cube_mean','nonfront_mean','local_diagnostic'):
        for metric in ('lpips','ssim','dists'):
            a,b = [paired[arm]['means'][region][metric] for arm in ARMS]
            rows.append(dict(metric=region+'/'+metric, direction='higher' if metric=='ssim' else 'lower',
                             native=a, finetuned=b, delta=b-a))
    with (OUT/'comparison.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(OUT/'comparison.json', dict(samples=925, checkpoint=str(CHECKPOINT), lora_scale=.5,
                 world_state_used=False, results=rows,
                 note='Validation development comparison; not independent final test. Local/cube similarity is not overall visual quality.'))
    print(OUT/'comparison.csv', flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare','generate','manifests','evaluate','summarize'))
    p.add_argument('--arm', choices=ARMS, default='native')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--eval-tag', default='evaluation')
    args=p.parse_args()
    if Path(args.eval_tag).name != args.eval_tag or args.eval_tag in ('.','..','generation','manifests'):
        p.error('eval-tag must be a new report directory name')
    if args.action=='prepare':
        plan,_=prepare();print(dict(samples=plan['samples'],houses=len(plan['houses']),arms=ARMS,world_state_used=False))
    elif args.action=='generate':generate(args.arm)
    elif args.action=='manifests':manifests()
    elif args.action=='evaluate':evaluate(args.arm,args.dry_run,args.eval_tag)
    else:summarize(args.eval_tag)


if __name__=='__main__':
    main()
