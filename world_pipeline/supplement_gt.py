"""Freeze successful old sources plus three official-split replacement houses.

No retry of old failed IDs, no quality relaxation, no implicit GT approval.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
from .common import ROOT, read, rows, sha
from .checkpoint_io import atomic_json, atomic_rows, output_lock
from .expand_gt_sources import ARCHIVES, paired_selection
from .build_encoder_sources import LeanBuilder

DATA=ROOT/'qwen_edit_pano/data'
OLD=DATA/'gt_sources_14train3val_repaired_v1'
READY=DATA/'paired_gt_repaired_retained_v1'
PAIRS=DATA/'paired_gt_supplement_v1'
WORK=DATA/'gt_sources_supplement_v1'
ADDED={'train':['759xd9YjKW5','EDJbREhghzL'],'val':['8194nk5LbLH']}


def plan():
    inputs=[OLD/'BUILD_RESULT.json',OLD/'SOURCE_RUN.json',
            DATA/'paired_gt_14train3val_v1/train.jsonl',DATA/'paired_gt_14train3val_v1/val.jsonl',
            DATA/'paired_full_v1/train.jsonl',DATA/'paired_full_v1/val.jsonl']
    signature={str(p):sha(p) for p in inputs}
    if PAIRS.exists():
        r=read(PAIRS/'selection.json')
        if r['inputs']!=signature or r['added_houses']!=ADDED:
            raise ValueError('Frozen selection inputs changed; do not overwrite')
        for split in ('train','val'):
            if sha(PAIRS/f'{split}.jsonl')!=r['manifest_hashes'][split]:
                raise ValueError('Frozen manifest changed')
        if (PAIRS/'added_scans.txt').read_text().splitlines()!=sorted(sum(ADDED.values(),[])):
            raise ValueError('Download list changed')
        print(r['counts']);return
    if not READY.exists():
        subprocess.run([sys.executable,'-m','qwen_edit_pano.world_pipeline.prepare_source_ready_pairs',
            '--sources',str(OLD),'--pairs',str(DATA/'paired_gt_14train3val_v1'),
            '--output',str(READY)],check=True,cwd=ROOT)
    report=read(READY/'selection.json')
    for p,h in report['protected_files'].items():
        if sha(p)!=h:raise ValueError('Retained source selection changed')
    original=read(OLD/'SOURCE_RUN.json')['rows']
    houses={r['scene_id'] for r in original}
    selected={}
    for split in ('train','val'):
        if set(ADDED[split])&houses:raise ValueError('Replacement house is not new')
        full=rows(DATA/f'paired_full_v1/{split}.jsonl')
        additions=[r for r in full if r['scene_id'] in ADDED[split]]
        if {r['scene_id'] for r in additions}!=set(ADDED[split]):raise ValueError('Wrong replacement split')
        kept=rows(READY/f'{split}.jsonl')
        done={sid for v in read(OLD/'BUILD_RESULT.json')['houses'].values() for sid in v.get('completed',[])}
        if {r['id'] for r in kept}!={r['id'] for r in original if r['split']==split and r['id'] in done}:
            raise ValueError('Retained manifest does not match successful source IDs')
        canonical={r['id']:r for r in full}
        if any(r!=canonical.get(r['id']) for r in kept):raise ValueError('Retained pair changed')
        selected[split]=sorted(kept+additions,key=lambda r:r['id'])
    PAIRS.mkdir()
    for split,items in selected.items():atomic_rows(PAIRS/f'{split}.jsonl',items)
    paired_selection(PAIRS)  # Independent official split / identity checks.
    (PAIRS/'added_scans.txt').write_text('\n'.join(sorted(sum(ADDED.values(),[])))+'\n')
    r=dict(inputs=signature,added_houses=ADDED,counts={s:len(v) for s,v in selected.items()},
        houses={s:sorted({r['scene_id'] for r in v}) for s,v in selected.items()},
        manifest_hashes={s:sha(PAIRS/f'{s}.jsonl') for s in selected},
        excluded_old_ids=report['excluded'],ready_for_training=False,
        rule='Keep all successful old source IDs; add 61+69 train and 19 val candidates. No quality guarantee.')
    atomic_json(PAIRS/'selection.json',r);print(r['counts'])


def build():
    plan()
    args=SimpleNamespace(work_root=WORK,reuse_root=OLD,
        raw_root=ROOT/'benchmark_assets/Matterport3D_raw/v1/scans',threads=4)
    selected=paired_selection(PAIRS)
    chain=[];root=OLD
    while root not in chain and (root/'SOURCE_RUN.json').exists():
        chain.append(root);cfg=read(root/'SOURCE_RUN.json')
        root=Path(cfg['reuse_root'])
    WORK.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for p in chain:stack.enter_context(output_lock(p))
        stack.enter_context(output_lock(WORK))
        for p in chain:
            for f,h in read(p/'SOURCE_RUN.json')['code'].items():
                if sha(f)!=h:raise ValueError(f'Old source code changed: {f}')
        files=[Path(__file__),Path(__file__).with_name('build_encoder_sources.py'),Path(__file__).with_name('lean_world.py')]
        signature=dict(rows=selected,reuse_root=str(OLD),raw_root=str(args.raw_root),
            original_result_sha256=sha(OLD/'BUILD_RESULT.json'),
            code={str(f):sha(f) for f in files},profile='retained_plus_new_houses_v1')
        marker=WORK/'SOURCE_RUN.json'
        if marker.exists() and read(marker)!=signature:raise ValueError('Build signature changed')
        atomic_json(marker,signature)
        builder=LeanBuilder(args)
        # Completed attempts, including quality failures, are never rerun.
        results=read(WORK/'build_progress.json') if (WORK/'build_progress.json').exists() else {}
        for house in sorted({r['scene_id'] for r in selected}):
            prior=results.get(house,{})
            if prior.get('status')=='FAILED_CANONICAL':continue
            missing=[a for a in ARCHIVES if not (args.raw_root/house/(a+'.zip')).exists()]
            if missing:
                results[house]=dict(status='MISSING_ARCHIVES',missing=missing)
            else:
                try:index=builder.canonical(house)
                except (ValueError,OSError,RuntimeError,KeyError) as exc:
                    results[house]=dict(status='FAILED_CANONICAL',error=str(exc))
                else:
                    failures=list(prior.get('failures',[]));failed={r['id'] for r in failures};done=[]
                    for row in (r for r in selected if r['scene_id']==house):
                        if row['id'] in failed:continue
                        try:builder.sample(row,index);done.append(row['id'])
                        except (ValueError,OSError,RuntimeError,KeyError,subprocess.CalledProcessError) as exc:
                            failures.append(dict(id=row['id'],error=str(exc)))
                        results[house]=dict(status='PARTIAL' if failures else 'REVIEW_REQUIRED',completed=done,failures=failures)
                        atomic_json(WORK/'build_progress.json',results)
                        print(f'{house}: complete={len(done)} failed={len(failures)}',flush=True)
            atomic_json(WORK/'build_progress.json',results)
        complete=all(v['status']=='REVIEW_REQUIRED' for v in results.values())
        atomic_json(WORK/'BUILD_RESULT.json',dict(all_requested_built=complete,ready_for_training=False,houses=results))
        print('Build pass finished. Quality failures retained in BUILD_RESULT.json; run select to export successes.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','build']);a=p.parse_args()
    plan() if a.action=='plan' else build()


if __name__=='__main__':main()
