"""Run existing inventory -> WildDet3D -> program completion on frozen RGB cases.

No algorithm changes. Each stage is a separate process in its existing env,
so Qwen and WildDet3D do not occupy GPU memory together. Run GPU action manually.
"""
import argparse
import importlib.metadata
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from ..common import ROOT, read, rows, sha, write, new_output, validate_graph
from qwen_edit_pano.common import atomic_json
from .qwen_upstream import DEFAULT_MODEL
from .placement_program import PROTOCOL

QWEN_PY = Path('/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python')
BOX_PY = Path('/data-nfs/gpu1-2/u13529658780/.conda/envs/wilddet3d/bin/python')
DEFAULT_MANIFEST = ROOT/'qwen_edit_pano/data/box_world_val6_v1/inputs.jsonl'
STAGES = ('inventory', 'observer', 'completion')
MODULE = 'qwen_edit_pano.world_pipeline.completion.'


def inputs(path):
    cases = rows(path)
    if not cases:
        raise ValueError('Empty validation manifest')
    for case in cases:
        if (not re.fullmatch(r'[A-Za-z0-9_-]+', case['id']) or case['split'] != 'val'
                or not math.isfinite(case['hfov_deg']) or not 1 <= case['hfov_deg'] <= 160):
            raise ValueError('Need safe distinct validation IDs and perspective FOV')
        if sha(case['local_rgb']) != case['image_sha256']:
            raise ValueError('Validation RGB changed: '+case['id'])
    return cases


def configuration(manifest, cases):
    # Pin inference code/parameters for resumable runs. Review rendering may be
    # regenerated independently; no success is inferred from existing filenames.
    names = ('batch_world.py', 'semantic_world.py', 'single_image_world.py', 'box_observer.py',
             'placement_program.py', 'qwen_upstream.py', 'qwen_upstream_core.py',
             'scene_geometry.py', 'box_layout.py', 'geometry.py', 'category_bridge.py')
    parent = Path(__file__).parent
    return dict(protocol='frozen_rgb_inventory_wilddet_program_batch_v1',
        manifest=str(Path(manifest).resolve()), manifest_sha256=sha(manifest),
        image_sha256={c['id']:c['image_sha256'] for c in cases},
        code_sha256={name:sha(parent/name) for name in names},
        program_protocol=PROTOCOL, qwen_model=str(DEFAULT_MODEL),
        seed=3407, max_observed=32, max_additions=8,
        qwen_python=str(QWEN_PY), wilddet_python=str(BOX_PY), scene_gt_used=False)


def command(stage, case, destination, previous):
    common = ['--image', case['local_rgb'], '--hfov', str(case['hfov_deg']), '--output', str(destination)]
    if stage == 'inventory':
        return [str(QWEN_PY), '-u', '-m', MODULE+'semantic_world', '--inventory-only', *common]
    if stage == 'observer':
        return [str(BOX_PY), '-u', '-m', MODULE+'box_observer',
                '--source', str(previous['inventory']/'G_obs.json'), *common]
    return [str(QWEN_PY), '-u', '-m', MODULE+'semantic_world', '--completion-mode', 'program',
            '--reuse-state', str(previous['observer']/'G_obs.json'), *common]


def execute(cmd, log):
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    with log.open('w') as handle:
        with subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                handle.write(line)
                handle.flush()
            return process.wait()


def verify_stage(directory, stage, case, previous):
    summary = read(directory/'summary.json')
    config = read(directory/'config.json')
    if summary['status'] != 'completed' or config['image_sha256'] != case['image_sha256']:
        raise ValueError('Stage incomplete or bound to another image')
    graph = validate_graph(read(directory/'G_obs.json'))
    if graph['stage'] != 'prediction' or graph['scope'] != 'observed':
        raise ValueError('Expected predicted observed state')
    if stage == 'inventory':
        if not summary.get('phase') == 'inventory_only':
            raise ValueError('Inventory stage must not estimate Qwen 3D geometry')
    elif stage == 'observer':
        if not all(o['center_local_m']['valid'] and o['size_aabb_local_m']['valid']
                   and o['center_local_m']['method'] == 'wilddet3d_camera_aabb_center' for o in graph['objects']):
            raise ValueError('WildDet3D did not supply every observed numeric box')
        if config['source_sha256'] != sha(previous['inventory']/'G_obs.json'):
            raise ValueError('Observer inventory binding mismatch')
    else:
        full = validate_graph(read(directory/'G_full.json'))
        if (config['protocol'] != PROTOCOL or full['scope'] != 'full' or full['stage'] != 'prediction'
                or graph != read(previous['observer']/'G_obs.json')
                or full['objects'][:len(graph['objects'])] != graph['objects']):
            raise ValueError('Completion must use this observation unchanged and the frozen program')
    names = ['config.json', 'summary.json', 'G_obs.json', 'local.png']
    if stage == 'completion':
        names += ['G_full.json', 'placement_program.json', 'completion_conversion.json']
    return {name:sha(directory/name) for name in names}


def run(manifest, output, launch=execute):
    cases = inputs(manifest)
    expected = configuration(manifest, cases)
    output = Path(output).resolve()
    if output.exists():
        if read(output/'run_config.json') != expected:
            raise ValueError('Batch inputs/code changed; use a new output directory')
        progress = read(output/'progress.json')
    else:
        output = new_output(output)
        write(output/'run_config.json', expected)
        progress = {}
        write(output/'progress.json', progress)
    for case in cases:
        print('\nValidation case: '+case['id'], flush=True)
        case_dir = output/case['id']
        case_dir.mkdir(exist_ok=True)
        record = progress.setdefault(case['id'], {})
        previous = {}
        for stage in STAGES:
            old = record.get(stage)
            if old and old['status'] == 'completed':
                directory = output/old['directory']
                if verify_stage(directory, stage, case, previous) != old['artifacts_sha256']:
                    raise ValueError('Saved stage changed; do not reuse: '+str(directory))
                previous[stage] = directory
                print('Reusing completed '+stage, flush=True)
                continue
            attempt = 1
            while (case_dir/f'{stage}_{attempt:03d}').exists() or (case_dir/f'{stage}_{attempt:03d}.log').exists():
                attempt += 1
            directory = case_dir/f'{stage}_{attempt:03d}'
            log = case_dir/(directory.name+'.log')
            cmd = command(stage, case, directory, previous)
            info = dict(status='running', directory=str(directory.relative_to(output)),
                        log=str(log.relative_to(output)), command=cmd)
            record[stage] = info
            atomic_json(output/'progress.json', progress)
            print(stage+'; log='+str(log), flush=True)
            try:
                code = launch(cmd, log)
                if code != 0:
                    raise RuntimeError(f'{stage} subprocess exit code {code}; inspect {log}')
                info['artifacts_sha256'] = verify_stage(directory, stage, case, previous)
                info['status'] = 'completed'
                previous[stage] = directory
            except Exception as exc:
                info.update(status='failed', error=str(exc))
                print('Case stage failed: '+str(exc), flush=True)
                atomic_json(output/'progress.json', progress)
                break  # Keep this failure; continue the other selected images.
            atomic_json(output/'progress.json', progress)
    from .batch_world_review import review
    summary = review(manifest, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'run', 'review'))
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'review':
        from .batch_world_review import review
        print(review(args.manifest, args.output))
        return
    cases = inputs(args.manifest)
    for python in (QWEN_PY, BOX_PY):
        if not python.is_file():
            raise ValueError('Missing existing environment: '+str(python))
    if not DEFAULT_MODEL.is_dir() or not list(DEFAULT_MODEL.glob('*.safetensors')):
        raise ValueError('Missing local Qwen weights')
    for package in ('plotly', 'Pillow', 'transformers'):
        importlib.metadata.version(package)
    if args.action == 'preflight':
        if args.output.exists() and read(args.output/'run_config.json') != configuration(args.manifest,cases):
            raise ValueError('Existing batch has different inputs/code')
        check = ('from qwen_edit_pano.world_pipeline.completion.box_observer import '
                 'preflight,DEFAULT_REPO,DEFAULT_CHECKPOINT; print(preflight(DEFAULT_REPO,DEFAULT_CHECKPOINT))')
        subprocess.run([str(BOX_PY), '-c', check], cwd=ROOT, check=True,
                       env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
        print(dict(status='preflight_passed', cases=len(cases), models_loaded=False, gpu_executed=False))
    else:
        result = run(args.manifest, args.output)
        print(result, flush=True)
        if result['failed_cases']:
            sys.exit(1)


if __name__ == '__main__':
    main()
