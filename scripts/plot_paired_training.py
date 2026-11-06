"""Plot sampled training logs, keeping the latest record per optimizer step."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in (args.run / 'train_log.jsonl').read_text().splitlines() if s.strip()]
    latest = {r['step']: r for r in rows}
    records = [latest[s] for s in sorted(latest)]
    config = json.loads((args.run / 'training_config.json').read_text())
    keys = ['total', 'flow', 'cube', 'yaw']
    values = np.array([[r['loss_last_microbatch_across_ranks'], r['rank0_last_microbatch']['flow'],
                        r['rank0_last_microbatch']['cube'], r['rank0_last_microbatch']['yaw_resampled']]
                       for r in records], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite logged loss')
    steps = np.array([r['step'] for r in records])
    epochs = np.array([r['epoch'] + 1 for r in records])
    lr = np.array([r['lr'] for r in records])
    summaries = []
    for epoch in sorted(set(epochs)):
        v = values[epochs == epoch]
        item = dict(epoch=int(epoch), logged_samples=len(v))
        for j, key in enumerate(keys):
            item[key + '_mean'] = float(v[:, j].mean())
            item[key + '_median'] = float(np.median(v[:, j]))
        summaries.append(item)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'epoch_sampled_losses.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for j, key in enumerate(keys):
        ax = axes.flat[j]
        ax.plot(steps, values[:, j], color='steelblue', alpha=.2, linewidth=.5, label='Logged last microbatch')
        # Restart smoothing at the geometry-loss transition, avoiding a mixed objective.
        for mask in [epochs <= config['geometry_start_epoch'], epochs > config['geometry_start_epoch']]:
            xs, ys = steps[mask], values[mask, j]
            if len(ys):
                window = min(50, len(ys))
                ax.plot(xs[window-1:], np.convolve(ys, np.ones(window)/window, mode='valid'),
                        color='darkorange', linewidth=1.5)
        ax.set_title(key + ' (orange: 50-record moving mean)')
        ax.set_xlabel('Optimizer step'); ax.grid(alpha=.2)
    ax = axes.flat[4]
    for key in keys:
        ax.plot([s['epoch'] for s in summaries], [s[key+'_mean'] for s in summaries], label=key, marker='.', markersize=4)
    ax.axvline(config['geometry_start_epoch'] + .5, color='gray', linestyle='--', label='Geometry loss activated')
    ax.set_title('Epoch means of LOGGED samples, not full epoch averages')
    ax.set_xlabel('Epoch (1-based)'); ax.legend(); ax.grid(alpha=.2)
    ax = axes.flat[5]
    ax.plot(steps, lr)
    ax.axvline(config['computed_warmup_steps'], linestyle='--', color='gray', label='Warmup ends')
    ax.set_title('Learning rate'); ax.set_xlabel('Optimizer step'); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Paired panorama training | training-only diagnostics, no validation loss\n'
                 'Latest duplicate step retained; first two epochs exclude geometry terms from total loss')
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ['png', 'pdf']:
        fig.savefig(args.output / ('training_curves.' + ext), dpi=160)
    plt.close(fig)
    complete = [json.loads(f.read_text()) for f in sorted(args.run.glob('checkpoint-epoch*/COMPLETE.json'))]
    report = dict(raw_records=len(rows), unique_steps=len(records), duplicate_records=len(rows)-len(records),
                  step_rollbacks=sum(b['step'] <= a['step'] for a,b in zip(rows, rows[1:])),
                  all_logged_losses_finite=True, checkpoints=complete,
                  best_sampled_training_total_epoch_after_geometry=min(
                      (s for s in summaries if s['epoch'] > config['geometry_start_epoch']), key=lambda s:s['total_mean'])['epoch'],
                  epochs=summaries,
                  limitation='Sampled last-microbatch training losses only; cannot establish best validation checkpoint or LoRA inference scale.')
    (args.output / 'training_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('epochs','checkpoints')}, indent=2))
    for s in summaries:
        print(s['epoch'], ' '.join(f'{k}={s[k+"_mean"]:.6f}' for k in keys))


if __name__ == '__main__':
    main()
