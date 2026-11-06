"""CPU-only restart planning; accept only checkpoints committed by COMPLETE."""
import argparse
from pathlib import Path
import time

from ..build_polished_world import read, atomic_json
from ..common import ROOT

FILES = ('config.json', 'world.safetensors', 'optimizer.pt', 'COMPLETE.json')
INITIAL = ROOT / 'qwen_edit_pano/outputs/object_world_id_ab_A_v1/checkpoint-step001000'


def latest(root, initial=INITIAL):
    candidates = [Path(initial)] + [p.parent for p in Path(root).glob('run-*/checkpoint-step*/COMPLETE.json')]
    complete = []
    for directory in candidates:
        if not all((directory / name).is_file() for name in FILES):
            continue
        try:
            step = read(directory / 'COMPLETE.json')['step']
            if type(step) is not int or step < 1 or directory.name != f'checkpoint-step{step:06d}':
                continue
        except (ValueError, KeyError, OSError):
            continue
        complete.append((step, str(directory.resolve())))
    if not complete:
        raise ValueError('No complete A checkpoint with World and optimizer found')
    step, directory = max(complete)
    directory = Path(directory)
    config = read(directory / 'config.json')
    source = read(Path(initial) / 'config.json')
    if config != source or config['group'] != 'A' or config['binding_weight'] != 0:
        raise ValueError('Selected checkpoint is not the unchanged A configuration')
    return directory, step


def prepare(root, steps, initial=INITIAL):
    root = Path(root).resolve()
    allowed = ROOT / 'qwen_edit_pano/outputs'
    if not root.is_relative_to(allowed) or root == allowed:
        raise ValueError('Use a dedicated run root below qwen_edit_pano/outputs')
    root.mkdir(parents=True, exist_ok=True)
    checkpoint, step = latest(root, initial)
    if step >= steps:
        return None, checkpoint, step
    output = root / f'run-{time.time_ns()}'
    atomic_json(root / 'NEXT_RUN.json', dict(output=str(output), resume=str(checkpoint),
        start_step=step, total_steps=steps, group='A', world_size=2, global_accumulation=4,
        evaluation=False, note='User-run only; no GPU or submission from this planner'))
    return output, checkpoint, step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Positive total steps required')
    output, checkpoint, step = prepare(args.root, args.steps)
    print('DONE' if output is None else str(output))
    print(checkpoint)
    print(step)


if __name__ == '__main__':
    main()
