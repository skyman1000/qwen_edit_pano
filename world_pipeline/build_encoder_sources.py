"""Resume into a NEW source root, reusing old stages and building lean worlds.

Run only after stopping the old writer. Never edits old build code or outputs.
Quality failures remain failures, not automatically excluded/approved.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from .common import ROOT, read, sha
from .checkpoint_io import atomic_json, output_lock
from .expand_gt_sources import Builder, paired_selection, ARCHIVES
from .lean_world import VERSION


class LeanBuilder(Builder):
    def run(self, module, arguments, label):
        if module != 'data.world_state_builder':
            return super().run(module, arguments, label)
        # Legacy caller supplies --threads, but the catalog performs no raycasts.
        args = list(map(str, arguments))
        pos = args.index('--threads')
        del args[pos:pos+2]
        log = self.logs/(label+'_lean.log')
        print(f'Lean source catalog: {label}', flush=True)
        with log.open('a') as handle:
            subprocess.run([sys.executable, '-m', 'qwen_edit_pano.world_pipeline.lean_world', *args],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True,
                env=dict(os.environ, OMP_NUM_THREADS=str(self.args.threads)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, default=ROOT/'qwen_edit_pano/data/paired_gt_14train3val_v1')
    p.add_argument('--reuse-root', type=Path, default=ROOT/'qwen_edit_pano/data/gt_sources_14train3val_v1')
    p.add_argument('--work-root', type=Path, default=ROOT/'qwen_edit_pano/data/gt_sources_14train3val_encoder_v1')
    p.add_argument('--raw-root', type=Path, default=ROOT/'benchmark_assets/Matterport3D_raw/v1/scans')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    for k in ('pairs','reuse_root','work_root','raw_root'):
        setattr(args, k, getattr(args,k).resolve())
    if (args.work_root == args.reuse_root or args.work_root in args.reuse_root.parents or
            args.reuse_root in args.work_root.parents or not args.work_root.is_relative_to(ROOT/'qwen_edit_pano/data')
            or args.work_root == ROOT/'qwen_edit_pano/data' or args.threads < 1):
        p.error('Use a separate data work root and positive thread count')
    if not (args.reuse_root/'SOURCE_RUN.json').exists():
        p.error('Expected existing source build provenance')
    selected = paired_selection(args.pairs)
    old = read(args.reuse_root/'SOURCE_RUN.json')
    if old['rows'] != selected or Path(old['raw_root']).resolve() != args.raw_root:
        raise ValueError('Old build selection/raw root mismatch')
    # Avoid taking a snapshot of a directory that is still being written.
    with output_lock(args.reuse_root):
        for name, digest in old['code'].items():
            if sha(name) != digest:
                raise ValueError(f'Legacy source code changed: {name}')
        args.work_root.mkdir(parents=True, exist_ok=True)
        with output_lock(args.work_root):
            signature = dict(profile=VERSION, rows=selected, reuse_root=str(args.reuse_root),
                raw_root=str(args.raw_root), old_signature_sha256=sha(args.reuse_root/'SOURCE_RUN.json'),
                code={str(f):sha(f) for f in [Path(__file__), Path(__file__).with_name('lean_world.py')]})
            marker = args.work_root/'SOURCE_RUN.json'
            if marker.exists() and read(marker) != signature:
                raise ValueError('Lean build signature changed; use a new work root')
            atomic_json(marker, signature)
            builder = LeanBuilder(args)
            results = {}
            for house in sorted({r['scene_id'] for r in selected}):
                group = [r for r in selected if r['scene_id']==house]
                missing = [a for a in ARCHIVES if not (args.raw_root/house/(a+'.zip')).is_file()]
                if missing:
                    results[house] = dict(status='MISSING_ARCHIVES', missing=missing)
                else:
                    try:
                        index = builder.canonical(house)
                    except (ValueError,OSError,RuntimeError,KeyError) as exc:
                        results[house] = dict(status='FAILED_CANONICAL', error=str(exc))
                    else:
                        done, failures = [], []
                        for row in group:
                            try:
                                builder.sample(row,index)
                                done.append(row['id'])
                            except (ValueError,OSError,RuntimeError,KeyError,subprocess.CalledProcessError) as exc:
                                failures.append(dict(id=row['id'],error=str(exc)))
                            results[house] = dict(status='PARTIAL' if failures else 'REVIEW_REQUIRED',
                                                  completed=done,failures=failures)
                            atomic_json(args.work_root/'build_progress.json',results)
                atomic_json(args.work_root/'build_progress.json',results)
            complete = all(v['status']=='REVIEW_REQUIRED' for v in results.values())
            atomic_json(args.work_root/'BUILD_RESULT.json',dict(all_requested_built=complete,
                profile=VERSION,ready_for_training=False,houses=results))
            if not complete:
                raise SystemExit('Construction finished with unresolved quality failures; inspect BUILD_RESULT.json. Do not blindly retry.')


if __name__ == '__main__':
    main()
