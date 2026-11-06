"""No-training comparison: text only, boxes+room hint, boxes+local caption."""
import argparse
import csv
from pathlib import Path

from ..common import rows, read, write
from .complete import main as complete, load_pretrained


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--observations', type=Path, required=True)
    p.add_argument('--captions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path('qwen_edit_pano/third_party/M3DLayout-code'))
    p.add_argument('--weights', type=Path, default=Path('qwen_edit_pano/third_party/M3DLayout-code/weights/autoregressive_59000.pth'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--size-convention', choices=['full', 'half'], default='full')
    p.add_argument('--camera-height', type=float, default=1.5)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    entries = rows(a.inputs)
    # Check only required inputs before expensive model loading.
    for entry in entries:
        folder = a.observations/entry['id']
        observation = read(folder/'observed.json')
        read(folder/'G_obs.json')
        caption = (a.captions/entry['id']/'description.txt').read_text().strip()
        if not caption or Path(observation['image']).resolve() != Path(entry['local_rgb']).resolve():
            raise ValueError(f"{entry['id']}: caption missing or observation image differs")
        if read(a.captions/entry['id']/'status.json')['reached_token_limit']:
            raise ValueError(f"{entry['id']}: caption was truncated; review before comparison")
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output/'experiment.json', {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()})
    runtime = load_pretrained(a.repo, a.weights, a.device)
    summaries = []
    for entry in entries:
        sample_id = entry['id']
        for arm in ('text_only', 'boxes_room', 'boxes_caption'):
            destination = a.output/sample_id/arm
            args = ['--observed', str(a.observations/sample_id/'observed.json'),
                    '--repo', str(a.repo), '--weights', str(a.weights), '--output', str(destination),
                    '--device', a.device, '--size-convention', a.size_convention,
                    '--camera-height', str(a.camera_height), '--max-objects', '120',
                    '--max-new-objects', '120', '--seeds', *map(str, a.seeds)]
            if arm != 'boxes_room':
                args += ['--text-file', str(a.captions/sample_id/'description.txt')]
            if arm == 'text_only':
                args += ['--mode', 'text-only']
            complete(args, runtime=runtime)
            for seed in a.seeds:
                s = read(destination/f'seed{seed:03d}'/'summary.json')
                summaries.append(dict(id=sample_id, arm=arm, seed=seed, observed=s['observed'],
                    context=s.get('context_objects', 0), generated=s['generated'], ended=s['ended'],
                    omitted_context=';'.join(s.get('omitted_context_categories', [])),
                    skipped_generated=len(s['skipped'])))
            with (a.output/'summary.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
                writer.writeheader(); writer.writerows(summaries)


if __name__ == '__main__':
    main()
