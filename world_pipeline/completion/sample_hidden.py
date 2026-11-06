"""Sample a task-trained completion checkpoint from GT or predicted G_obs."""
import argparse
import copy
from pathlib import Path

import torch

from ..common import read, rows, write
from .hidden_data import VERSION
from .hidden_model import build, generate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path('qwen_edit_pano/third_party/M3DLayout-code'))
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--observed', type=Path, help='Observer G_obs.json, not observed.json')
    source.add_argument('--val-data', type=Path, help='Prepared completion data; uses only observed objects')
    p.add_argument('--per-building', type=int, default=0, help='0: all val; positive: deterministic diagnostic subset')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    meta = state['metadata']
    if meta['version'] != VERSION:
        raise ValueError('Expected CauPano task-trained checkpoint')
    model, _ = build(a.repo, a.device, state['config'])
    model.load_state_dict(state['model'], strict=True)
    if a.observed:
        inputs = [read(a.observed)]
    else:
        if read(a.val_data / 'metadata.json') != meta:
            raise ValueError('Validation data differs from checkpoint')
        inputs, counts = [], {}
        for scene in sorted(rows(a.val_data / 'val.jsonl'), key=lambda x: x['id']):
            building = scene['building']
            if a.per_building and counts.get(building, 0) >= a.per_building:
                continue
            counts[building] = counts.get(building, 0) + 1
            full = read(scene['graph'])
            observed = copy.deepcopy(full)
            observed.update(scope='observed', relations=[])
            observed['objects'] = [o for o in observed['objects'] if o['local_evidence'] == 'visible']
            inputs.append(observed)
    a.output.mkdir(parents=True, exist_ok=False)
    for observed in inputs:
        for seed in a.seeds:
            graph, summary = generate(model, observed, meta, seed, a.device)
            folder = a.output / observed['id'] / f'seed{seed:03d}'
            folder.mkdir(parents=True)
            # Save the exact input, enabling independent preservation/evaluation checks.
            write(folder / 'G_obs.json', observed)
            write(folder / 'G_full.json', graph)
            summary.update(id=observed['id'], checkpoint=str(a.checkpoint.resolve()),
                           checkpoint_step=state['step'], input_kind='gt' if a.val_data else 'observer')
            write(folder / 'summary.json', summary)
            print(f"{observed['id']} seed={seed}: {summary}", flush=True)


if __name__ == '__main__':
    main()
