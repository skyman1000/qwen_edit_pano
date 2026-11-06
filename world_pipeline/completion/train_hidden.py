"""Train visible-conditioned hidden completion; independent of Qwen/Adapter."""
import argparse
import json
import random
from pathlib import Path

import torch

from ..common import read, rows, write
from .hidden_data import VERSION
from .hidden_model import batch, build, initialize, losses, objective


@torch.no_grad()
def validate(model, scenes, meta, device, batch_size):
    model.eval()
    totals = {}
    pending = []

    def consume(examples):
        sample = batch(examples, meta, device)
        pred = model(sample)
        label, position, size = losses(pred, sample)
        real = 1 - sample['class_labels_tr'][..., -1]
        loss = (label + (position + size) * real).flatten().tolist()
        for (scene, k), value in zip(examples, loss):
            b = scene['building']
            totals[b] = totals.get(b, 0.) + value / (len(scene['hidden']) + 1)

    for scene in scenes:
        for k in range(len(scene['hidden']) + 1):
            pending.append((scene, k))
            if len(pending) == batch_size:
                consume(pending)
                pending = []
    if pending:
        consume(pending)
    by_building = {b: value / sum(s['building'] == b for s in scenes) for b, value in totals.items()}
    return dict(loss=sum(totals.values()) / len(scenes), by_building=by_building)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('qwen_edit_pano/data/hidden_completion_v1'))
    p.add_argument('--repo', type=Path, default=Path('qwen_edit_pano/third_party/M3DLayout-code'))
    p.add_argument('--output', type=Path, required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--init-official', type=Path, help='Optional compatible-weight initialization, not task checkpoint')
    group.add_argument('--resume', type=Path)
    p.add_argument('--steps', type=int, default=20000, help='Total optimizer steps, including resumed steps')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--validate-every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    meta = read(a.data / 'metadata.json')
    if meta['version'] != VERSION:
        raise ValueError('Wrong completion data contract')
    train, val = rows(a.data / 'train.jsonl'), rows(a.data / 'val.jsonl')
    state = torch.load(a.resume, map_location='cpu', weights_only=False) if a.resume else None
    if state and state['metadata'] != meta:
        raise ValueError('Resume data differs from checkpoint')
    model, config = build(a.repo, a.device, state['config'] if state else None)
    init = initialize(model, a.init_official) if a.init_official else dict(mode='scratch')
    optimizer = torch.optim.Adam((v for v in model.parameters() if v.requires_grad), lr=a.lr)
    start, best = 0, float('inf')
    a.output.mkdir(parents=True, exist_ok=bool(a.resume))
    if state:
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        rng.setstate(state['rng'])
        torch.set_rng_state(state['torch_rng'])
        if a.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        start, best, init = state['step'], state['best'], state['initialization']
    write(a.output / 'run.json', dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                                    metadata=meta, config=config, initialization=init))
    for step in range(start + 1, a.steps + 1):
        model.train()
        examples = []
        for _ in range(a.batch_size):
            scene = rng.choice(train)
            examples.append((scene, rng.randrange(len(scene['hidden']) + 1)))
        sample = batch(examples, meta, a.device)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(sample), sample)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps(dict(step=step, train_loss=loss.item())), flush=True)
        if step % a.validate_every == 0 or step == a.steps:
            metrics = validate(model, val, meta, a.device, a.batch_size)
            improved = metrics['loss'] < best
            best = min(best, metrics['loss'])
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), metadata=meta,
                         config=config, step=step, best=best, initialization=init, rng=rng.getstate(),
                         torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else [])
            torch.save(state, a.output / 'last.pt')
            if improved:
                torch.save(state, a.output / 'best.pt')
            record = dict(step=step, validation=metrics)
            with (a.output / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
