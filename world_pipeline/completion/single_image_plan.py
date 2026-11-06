"""Manual, bounded unseen-object planning from a saved single-image observation.

This diagnostic does not estimate geometry or produce an executable G_full.
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from ..common import new_output, read, sha, validate_graph, write
from .geometry import CANONICAL_CATEGORIES
from .panorama_probe import decode_plan
from .qwen_upstream import DEFAULT_MODEL, Qwen, camera_text


def semantic_context(observed):
    """Whitelist image-grounded fields; do not pass unverified metric estimates."""
    context = []
    for obj in observed['objects']:
        category = CANONICAL_CATEGORIES.get(obj['category_id']['value'])
        if category is None:
            raise ValueError('Unsupported observed category: '+obj['track_id'])
        context.append(dict(id=obj['track_id'], category=category,
                            description=obj.get('description'),
                            bbox_2d=obj['bbox2d_visible_xyxy_norm']['value']))
    return context


def prompt_for(context, camera, capacity):
    return (
        'Construct ONE plausible hypothesis for the contents of the current room around this camera, '
        'including regions outside the photograph. This is hypothetical scene completion: '
        'an object need not be visible to be proposed. Do not merely search the image for missed detections. '
        'The visible inventory below is fixed. Propose only distinct additional physical objects '
        'with no visible evidence, either outside the camera field of view or occluded. '
        'Preserve the photographed scene and visible empty space. Stay in the current room; '
        'do not enumerate distant rooms, restate visible objects, or move an observed instance. '
        'Prefer useful major scene objects over lists of generic decorations. '
        'Give a coherent, restrained arrangement, not a union of mutually exclusive possibilities. '
        'For each proposal, explain both why it fits this scene and why it can be unseen. '
        'Use directions relative to the CAMERA or a specified observed ID. Behind CAMERA and '
        'behind an object are different relations. No metric coordinates, sizes, rotations or visible boxes. '
        'Return ONLY JSON with key additions containing objects with id (unique string), '
        'category (one allowed canonical name), description, placement, '
        'unseen_reason (outside_fov or occluded), and reason. '
        'These are hypotheses, never statements that the real room certainly contains these objects. '
        f'At most {capacity} additions, not a target count; an empty array is allowed. '
        'Allowed categories: '+', '.join(CANONICAL_CATEGORIES.values())
        +'\nCAMERA: '+camera+'\nVISIBLE INVENTORY (2D xyxy normalized to 0..1): '
        +json.dumps(context, ensure_ascii=False))


def validate_unseen(answer, capacity):
    plan, diagnostic = decode_plan(answer, capacity)
    for obj in plan['additions']:
        if obj['unseen_reason'] not in ('outside_fov', 'occluded'):
            raise ValueError('Unseen plan requires an explicit outside_fov or occluded hypothesis')
        if obj.get('role', 'unseen') != 'unseen' or obj.get('bbox_2d') is not None:
            raise ValueError('Visible recovery is a different task; do not relabel it as unseen')
        if any(obj.get(key) is not None for key in ('center_local_m', 'size_aabb_local_m', 'bbox_3d')):
            raise ValueError('This diagnostic requests semantics only, not geometry')
    return plan, diagnostic


def run_plans(model, image, context, camera, output, seeds, capacity=8):
    prompt = prompt_for(context, camera, capacity)
    (output/'prompt.txt').write_text(prompt)
    write(output/'observed_inventory.json', context)
    results = []
    for seed in seeds:
        directory = output/f'seed{seed}'
        directory.mkdir()
        try:
            answer = model.ask(image, prompt, directory/'plan.txt', seed=seed, sample=True,
                               profile='official', guard_repetition=True, max_tokens=2048)
            plan, diagnostic = validate_unseen(answer, capacity)
            write(directory/'plan_decoded.json', plan)
            write(directory/'diagnostic.json', diagnostic)
            result = dict(seed=seed, status='parsed', proposals=len(plan['additions']),
                          semantic_quality='not_validated')
        except (ValueError, KeyError, TypeError) as exc:
            result = dict(seed=seed, status='failed', error=str(exc))
        results.append(result)
        write(directory/'status.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = dict(protocol='single_image_unseen_semantic_plans_v1', results=results,
                   geometry_estimated=False, observed_geometry_used=False,
                   plans_are_alternatives_not_merged=True, semantic_quality='not_validated',
                   note='Nonempty output alone is not successful scene completion; inspect each proposal')
    write(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Saved single_image_world run directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--seeds', nargs='+', type=int, default=[3407, 3408, 3409])
    parser.add_argument('--max-additions', type=int, default=8)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not 1 <= len(args.seeds) <= 3:
        parser.error('Use one to three distinct seeds for this bounded diagnostic')
    if not 1 <= args.max_additions <= 12:
        parser.error('--max-additions must be between 1 and 12')
    if args.output.exists():
        parser.error('Output exists; use a new directory')
    if not args.model.is_dir() or not list(args.model.glob('*.safetensors')):
        parser.error('Local model weights not found; no download will be attempted')
    config = read(args.source/'config.json')
    observed = validate_graph(read(args.source/'G_obs.json'))
    context = semantic_context(observed)
    with Image.open(args.source/'local.png') as source:
        image = source.convert('RGB')
    camera = camera_text(image, dict(hfov_deg=config['hfov_deg']))
    camera += f' Horizontal FOV={config["hfov_deg"]} degrees.'
    if args.preflight:
        print(f'PREFLIGHT PASSED: {len(context)} visible instances, {len(args.seeds)} short requests; no model loaded')
        return
    output = new_output(args.output)
    write(output/'config.json', dict(source=str(args.source.resolve()),
          source_graph_sha256=sha(args.source/'G_obs.json'),
          source_image_sha256=sha(args.source/'local.png'), model=str(args.model.resolve()),
          hfov_deg=config['hfov_deg'], seeds=args.seeds, max_additions=args.max_additions,
          decoding_profile='official', scene_gt_used=False, observed_geometry_used=False))
    result = run_plans(Qwen(args.model), image, context, camera, output, args.seeds, args.max_additions)
    if any(row['status'] == 'failed' for row in result['results']):
        raise SystemExit('Some plans failed; inspect raw answers and per-seed status.json')


if __name__ == '__main__':
    main()
