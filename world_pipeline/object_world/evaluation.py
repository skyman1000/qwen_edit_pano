"""CPU-only fixed image panel and GT-based object review eligibility.

These are diagnostic filters, NOT training-token filters or LACONIC SOC.
No amodal visibility ratio can be recovered from a visible instance mask.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from ..build_polished_world import read, atomic_json
from .data import DEFAULT_PACK, identity, load_pack

PANEL_VERSION = 'object_world_fixed_images_v1'
POLICY = dict(minimum_supported_pixels=64, minimum_sphere_fraction=.001,
              minimum_support_retention=.5, excluded_categories=[39, 40],
              amodal_visibility_ratio='unavailable; support retention is NOT occlusion visibility',
              purpose='GT-only diagnostic eligibility; never filter training tokens or global loss')


def select_images(records, split, limit):
    """Exactly limit IMAGES, preferring distinct houses without expanding them."""
    candidates = [r for r in records if r['split'] == split]
    if limit < 1 or len(candidates) < limit:
        raise ValueError('Requested image count is unavailable')
    selected = []; houses = set()
    for row in candidates:
        if row['scene_id'] not in houses:
            selected.append(row); houses.add(row['scene_id'])
        if len(selected) == limit:
            return selected
    chosen = {r['id'] for r in selected}
    return (selected + [r for r in candidates if r['id'] not in chosen])[:limit]


def circular_box(mask):
    """Minimal horizontal circular span; prevents seam objects becoming full ERP crops."""
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    columns = np.unique(x); width = mask.shape[1]
    gaps = np.diff(np.r_[columns, columns[0] + width])
    gap = int(np.argmax(gaps))
    start = int(columns[(gap + 1) % len(columns)])
    span = int(width - gaps[gap] + 1)
    return dict(x_start=start, width=span, y_start=int(y.min()), height=int(y.max()-y.min()+1))


def object_report(row, names, instance, instance_valid, support, policy=POLICY):
    if instance.ndim != 2 or instance_valid.shape != instance.shape or support.shape != instance.shape:
        raise ValueError('Expected matching ERP instance and validity maps')
    if instance_valid.dtype != bool or support.dtype != bool or np.any(support & ~instance_valid):
        raise ValueError('Loss support must be a subset of valid instance pixels')
    h, w = instance.shape
    latitude = np.pi/2 - (np.arange(h)+.5)/h*np.pi
    longitude = (np.arange(w)+.5)/w*2*np.pi-np.pi
    area = np.broadcast_to(np.cos(latitude)[:, None], (h, w))
    # The actual face2 Local is a square 90-degree pinhole view at yaw/pitch 0.
    x = np.cos(latitude)[:, None]*np.sin(longitude)[None, :]
    y = np.broadcast_to(np.sin(latitude)[:, None], (h, w))
    z = np.cos(latitude)[:, None]*np.cos(longitude)[None, :]
    local = (z > 0) & (np.abs(x) <= z) & (np.abs(y) <= z)
    result = []
    for obj in row['objects']:
        full = (instance == int(obj['instance_id'])) & instance_valid
        mask = full & support
        count = int(mask.sum()); visible = int(full.sum())
        sphere = float(area[mask].sum()/area.sum())
        retention = count/max(visible, 1)
        reasons = []
        if count < policy['minimum_supported_pixels']: reasons.append('small_supported_pixel_count')
        if sphere < policy['minimum_sphere_fraction']: reasons.append('small_spherical_area')
        if retention < policy['minimum_support_retention']: reasons.append('insufficient_supported_fraction')
        if obj['category_id'] in policy['excluded_categories']: reasons.append('generic_semantics')
        box = circular_box(mask)
        # Large enclosing crops are not useful object-specific semantic evidence.
        if box is not None and box['width'] > w//2: reasons.append('crop_spans_over_180_degrees')
        result.append(dict(instance_id=obj['instance_id'], category_id=obj['category_id'],
            name=names[str(obj['category_id'])], eligible=not reasons, reasons=reasons,
            supported_pixels=count, visible_instance_pixels=visible,
            support_retention=retention, sphere_fraction=sphere,
            outside_local_fraction=float(area[mask & ~local].sum()/max(float(area[mask].sum()), 1e-12)),
            circular_crop=box))
    return dict(id=row['id'], mask_shape=[h, w], objects=result,
                eligible_count=sum(o['eligible'] for o in result),
                exclusion_reasons=dict(Counter(reason for o in result for reason in o['reasons'])))


def make_panel(plan, records, split='val', count=6, seed=0):
    selected = select_images(records, split, count)
    candidates = [r for r in records if r['split'] == split]
    samples = []
    for index, row in enumerate(selected):
        other = next((r for r in candidates if r['scene_id'] != row['scene_id']), None)
        if other is None: raise ValueError('Shuffle requires another building in the same split')
        with np.load(row['object_region_supervision']['path'], allow_pickle=False) as masks:
            report = object_report(row, plan['names'], masks['instance'], masks['instance_valid'], masks['object_loss_valid'])
        samples.append(dict(id=row['id'], scene_id=row['scene_id'], seed=seed+index,
                            source=identity(row), shuffled_source=identity(other), object_review=report))
    return dict(version=PANEL_VERSION, split=split, count=count, seed=seed,
        policy=POLICY, samples=samples, selection='fixed images; no per-building expansion',
        quality_claim='small diagnostic panel, not full validation or a SOC score')


def resolve_panel(panel, records, split, count, seed):
    if (panel['version'] != PANEL_VERSION or panel['split'] != split or panel['count'] != count
            or panel['seed'] != seed or len(panel['samples']) != count or panel['policy'] != POLICY):
        raise ValueError('Panel protocol/count/seed differs from requested generation')
    by_id = {r['id']: r for r in records if r['split'] == split}
    selected = []; ids = set()
    for index, entry in enumerate(panel['samples']):
        row = by_id[entry['id']]; other = by_id[entry['shuffled_source']['id']]
        if (row['id'] in ids or identity(row) != entry['source'] or identity(other) != entry['shuffled_source']
                or row['scene_id'] != entry['scene_id'] or row['scene_id'] == other['scene_id']
                or entry['seed'] != seed+index or entry['object_review']['id'] != row['id']):
            raise ValueError('Panel sample identity/donor/seed differs')
        ids.add(row['id']); selected.append(row)
    return selected


def generation_budget(count, modes, steps):
    if count < 1 or steps < 1 or not modes or len(set(modes)) != len(modes):
        raise ValueError('Invalid generation budget or duplicate modes')
    return dict(images=count, modes=list(modes), generated_panoramas=count*len(modes),
                denoising_steps_per_panorama=steps, total_denoising_iterations=count*len(modes)*steps,
                note='Iterations are sampler steps, not Transformer forward calls; CFG may add work')


def write_crop_review(directory, entry, modes, maximum=6):
    """Bounded CPU contact sheet, same GT-defined crops for every condition."""
    from PIL import Image, ImageDraw, ImageOps
    objects = sorted((o for o in entry['object_review']['objects'] if o['eligible']),
        key=lambda o: (-o['outside_local_fraction'], -o['sphere_fraction'], o['instance_id']))[:maximum]
    if not objects:
        return
    files = ['gt_erp'] + list(modes)
    h, w = entry['object_review']['mask_shape']
    sheet = Image.new('RGB', (200*len(files), 170*len(objects)), 'white')
    draw = ImageDraw.Draw(sheet)
    for column, name in enumerate(files):
        with Image.open(directory/(name+'.png')) as image:
            array = np.asarray(image.convert('RGB'))
        sy, sx = array.shape[0]/h, array.shape[1]/w
        for index, obj in enumerate(objects):
            box = obj['circular_crop']; start = int(np.floor(box['x_start']*sx))
            width = max(1, int(np.ceil(box['width']*sx)))
            top = int(np.floor(box['y_start']*sy)); bottom = int(np.ceil((box['y_start']+box['height'])*sy))
            crop = Image.fromarray(array[top:bottom, (np.arange(width)+start) % array.shape[1]])
            crop = ImageOps.contain(crop, (196, 125))
            sheet.paste(crop, (200*column, 170*index+40))
            draw.text((200*column+2, 170*index+2), name, fill='black')
            draw.text((200*column+2, 170*index+18), f"{obj['name']} / {obj['instance_id']}", fill='black')
    sheet.save(directory/'object_review.jpg')
    atomic_json(directory/'object_review.json', dict(objects=objects,
        note='GT-selected circular ERP crops for visual review, not perspective crops or automatic SOC'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack', type=Path, default=DEFAULT_PACK)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    plan, records = load_pack(args.pack)
    panel = make_panel(plan, records, count=args.count, seed=args.seed)
    resolve_panel(panel, records, 'val', args.count, args.seed)
    if args.output.exists() and read(args.output) != panel:
        raise ValueError('Existing panel differs; use a new filename')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, panel)
    print(dict(panel=str(args.output), **generation_budget(args.count, ['baseline','correct','shuffled'], 28),
        eligible_objects=[s['object_review']['eligible_count'] for s in panel['samples']], GPU=False), flush=True)


if __name__ == '__main__': main()
