"""Historical pretrained-continuation diagnostic. Main task: train_hidden/sample_hidden."""
import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np

from .geometry import (CATEGORY_IDS, SIGNS, official_stats, plot_layout, read,
                       upright_box, world_object, write, yaw_matrix)


def load_pretrained(repo, weights, device):
    import torch
    import yaml
    sys.path.insert(0, str(repo.resolve()))
    from scene_synthesis.networks.autoregressive import build_network
    stats = official_stats(repo)
    config = yaml.safe_load((repo/'config/m3dlayout_autoregressive.yaml').read_text())
    network = build_network(7+len(stats['class_labels']), len(stats['class_labels']), config,
        device=device, n_dependency_classes=config['dataset']['sample_num_points']+1)[0]
    state = torch.load(weights, map_location='cpu', weights_only=True)
    network.load_state_dict({k: v for k, v in state.items() if not k.endswith('position_ids')})
    return network.eval(), stats


def main(argv=None, runtime=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observed', type=Path, required=True, help='observe output/<id>/observed.json')
    p.add_argument('--repo', type=Path, default=Path('qwen_edit_pano/third_party/M3DLayout-code'))
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
    p.add_argument('--max-new-objects', type=int, default=32)
    p.add_argument('--max-objects', type=int, default=128, help='Match the World training Bundle')
    p.add_argument('--camera-height', type=float, default=1.5)
    p.add_argument('--scene-offset-xz', type=float, nargs=2, default=[0., 0.])
    p.add_argument('--scene-yaw-deg', type=float, default=0.)
    p.add_argument('--size-convention', choices=['full', 'half'], default='full')
    texts = p.add_mutually_exclusive_group()
    texts.add_argument('--text', help='Default: one room-type sentence; empty string disables text')
    texts.add_argument('--text-file', type=Path)
    p.add_argument('--mode', choices=['completion', 'text-only'], default='completion')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args(argv)

    import torch
    repo = args.repo.resolve()
    observation = read(args.observed)
    observed_graph = read(args.observed.with_name('G_obs.json'))
    text = (args.text_file.read_text().strip() if args.text_file else
            args.text if args.text is not None else f"The room is a {observation['room_type']}.")
    device = args.device
    network, stats = runtime if runtime is not None else load_pretrained(repo, args.weights, device)
    classes = stats['class_labels']
    args.output.mkdir(parents=True, exist_ok=True)

    bounds = {key: np.asarray(stats['bounds_'+key], dtype=np.float32).reshape(2, -1)
              for key in ('translations', 'sizes', 'angles')}

    def normalize(values, key):
        lo, hi = bounds[key]
        return (values - (hi+lo)/2) / ((hi-lo)/2 + 1e-8)

    def denormalize(values, key):
        lo, hi = bounds[key]
        return values * ((hi-lo)/2 + 1e-8) + (hi+lo)/2

    if args.mode == 'text-only':
        write(args.output/'config.json', dict(mode=args.mode, text=text, seeds=args.seeds,
            frame='official_scene_frame_not_registered_to_input_camera',
            size_convention=args.size_convention, max_objects=args.max_objects,
            weights=str(args.weights.resolve()), repo=str(repo)))
        for seed in args.seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            directory = args.output/f'seed{seed:03d}'; directory.mkdir(parents=True, exist_ok=True)
            with torch.inference_mode():
                result = network.generate_boxes(torch.ones(1, 1, 64, 64, device=device),
                    max_boxes=args.max_objects, device=device, text=text)
            raw = {k: v.detach().cpu().numpy()[0] for k, v in result.items()}
            write(directory/'raw_normalized.json', {k: v.tolist() for k, v in raw.items()})
            decoded = {k: denormalize(raw[k], k) for k in ('translations', 'sizes', 'angles')}
            native, preview, ended = [], [], False
            for i in range(1, len(raw['class_labels'])):
                category = classes[int(raw['class_labels'][i].argmax())]
                if category == 'end':
                    ended = True
                    break
                if category == 'start':
                    continue
                center, size = decoded['translations'][i], decoded['sizes'][i]
                angle = float(decoded['angles'][i, 0])
                native.append(dict(category=category, center_scene=center.tolist(),
                                   size_native=size.tolist(), yaw=angle))
                full_size = size * (2 if args.size_convention == 'half' else 1)
                corners = center + (SIGNS * full_size/2) @ yaw_matrix(angle).T
                preview.append(dict(category=category, center_local=center.tolist(),
                                    corners_local=corners.tolist(), observed=False))
            write(directory/'layout_scene.json', dict(frame='official_scene_frame', text=text, objects=native))
            write(directory/'summary.json', dict(observed=0, generated=len(native), ended=ended,
                                                 limit_reached=not ended, skipped=[], mode=args.mode))
            plot_layout(directory/'layout_topdown.png', preview, frame='scene')
            print(f'text-only seed {seed}: {len(native)} objects -> {directory}', flush=True)
        return

    # Explicit exploratory placement of the camera in the pretrained layout frame.
    # It is not a recovered room pose or a GT-assisted alignment.
    rotation = yaw_matrix(np.deg2rad(args.scene_yaw_deg))
    offset = np.array([args.scene_offset_xz[0], args.camera_height, args.scene_offset_xz[1]])
    factor = 2. if args.size_convention == 'half' else 1.
    context, omitted = [], []
    for obj in observation['boxes']:
        if obj['category'] not in classes:
            omitted.append(obj['category'])
            continue
        corners_scene = np.asarray(obj['corners_local']) @ rotation.T + offset
        center, size, angle = upright_box(corners_scene)
        context.append((classes.index(obj['category']), center, size/factor, [angle]))
    labels = np.eye(len(classes), dtype=np.float32)[[o[0] for o in context]]
    boxes = {'class_labels': torch.as_tensor(labels[None], device=device)}
    for key, position, width in [('translations', 1, 3), ('sizes', 2, 3), ('angles', 3, 1)]:
        values = np.asarray([o[position] for o in context], dtype=np.float32).reshape(-1, width)
        boxes[key] = torch.tensor(normalize(values, key)[None], dtype=torch.float32, device=device)
    budget = min(args.max_new_objects, args.max_objects-len(observation['boxes']))
    if budget < 0:
        raise ValueError('Observed objects already exceed --max-objects')
    write(args.output/'config.json', dict(
        observed=str(args.observed.resolve()), repo=str(repo), weights=str(args.weights.resolve()),
        seeds=args.seeds, text=text, max_new_objects=budget, max_objects=args.max_objects,
        camera_height=args.camera_height, scene_offset_xz=args.scene_offset_xz,
        scene_yaw_deg=args.scene_yaw_deg, size_convention=args.size_convention,
        context_objects=len(context), omitted_context_categories=omitted,
        bridge='exploratory_camera_placement_y_up; yaw_only_enclosure_of_observed_boxes',
        category_mapping=CATEGORY_IDS))

    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        directory = args.output/f'seed{seed:03d}'; directory.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            result = network.complete_scene(boxes, torch.ones(1, 1, 64, 64, device=device),
                                            max_boxes=budget, device=device, text=text or None)
        raw = {k: v.detach().cpu().numpy()[0] for k, v in result.items()}
        write(directory/'raw_normalized.json', {k: v.tolist() for k, v in raw.items()})
        decoded = {k: denormalize(raw[k], k) for k in ('translations', 'sizes', 'angles')}
        full = copy.deepcopy(observed_graph)
        full.update(scope='full', stage='prediction', ready_for_training=False)
        layout = copy.deepcopy(observation['boxes'])
        skipped, ended = [], False
        # complete_scene returns START + all context + newly sampled tokens.
        for index in range(1+len(context), len(raw['class_labels'])):
            category = classes[int(raw['class_labels'][index].argmax())]
            if category == 'end':
                ended = True
                break
            if category not in CATEGORY_IDS:
                skipped.append(dict(index=index, category=category, reason='outside_encoder_vocabulary'))
                continue
            center_scene = decoded['translations'][index]
            size = decoded['sizes'][index] * factor
            angle = float(decoded['angles'][index, 0])
            if not (np.isfinite(center_scene).all() and np.isfinite(size).all() and np.isfinite(angle)) or min(size) <= 0:
                skipped.append(dict(index=index, category=category, reason='invalid_geometry'))
                continue
            corners_scene = center_scene + (SIGNS * size/2) @ yaw_matrix(angle).T
            center = (center_scene-offset) @ rotation
            corners = (corners_scene-offset) @ rotation
            track_id = f'generated_{index:04d}'
            full['objects'].append(world_object(track_id, category, center, corners, 'completion', False))
            layout.append(dict(track_id=track_id, category=category, center_local=center.tolist(),
                               corners_local=corners.tolist(), observed=False))
        write(directory/'G_full.json', full)
        write(directory/'layout.json', layout)
        write(directory/'summary.json', dict(observed=len(observation['boxes']),
            generated=len(layout)-len(observation['boxes']), ended=ended, skipped=skipped,
            limit_reached=not ended, context_objects=len(context), omitted_context_categories=omitted))
        plot_layout(directory/'layout_topdown.png', layout)
        print(f"seed {seed}: {len(full['objects'])} total objects -> {directory}", flush=True)


if __name__ == '__main__':
    main()
