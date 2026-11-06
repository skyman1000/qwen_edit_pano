"""Small geometry/JSON helpers shared by the two manual inference commands."""
import ast
import itertools
import json
import re
from pathlib import Path

import numpy as np

from ..common import FRAME, SCHEMA, field

SIGNS = np.array(list(itertools.product((-1., 1.), repeat=3)))

# Explicit many-to-one experimental bridge to the existing mpcat40 vocabulary.
# rug maps to structural floor (2), which the current object encoder excludes.
CATEGORY_GROUPS = {
    3: 'chair armchair chinese_chair dining_chair dressing_chair lounge_chair',
    4: 'door glass_panel_door lite_door louver_door panel_door',
    5: 'table coffee_table console_table corner_side_table desk dining_table dressing_table round_end_table side_table',
    6: 'picture wall_art', 7: 'cabinet children_cabinet kitchen_cabinet wardrobe wine_cabinet',
    8: 'cushion', 9: 'window',
    10: 'sofa chaise_longue_sofa l_shaped_sofa lazy_sofa loveseat_sofa multi_seat_sofa',
    11: 'bed kids_bed', 12: 'curtain', 13: 'chest_of_drawers nightstand',
    14: 'plant large_plant_container plant_container', 15: 'sink standing_sink',
    16: 'stairs', 18: 'toilet', 19: 'stool', 20: 'towel', 21: 'mirror',
    22: 'tv_monitor tv monitor', 23: 'shower', 24: 'column', 25: 'bathtub',
    26: 'counter kitchen_space', 27: 'fireplace',
    28: 'lighting ceiling_lamp desk_lamp floor_lamp pendant_lamp', 29: 'beam',
    30: 'railing', 31: 'shelving bookshelf cell_shelf large_shelf shelf',
    32: 'blinds', 33: 'gym_equipment', 34: 'seating', 35: 'board_panel',
    36: 'furniture tv_stand',
    37: 'appliances beverage_fridge dishwasher microwave oven', 38: 'clothes',
    39: 'objects balloon book book_column book_stack bottle bowl box can chopsticks clutter cup decoration food_bag food_box fork fruit_container hardware jar knife nature_shelf_trinkets pan plate pot spoon trashcan vase wineglass',
    40: 'misc',
}
CATEGORY_IDS = {name: cid for cid, names in CATEGORY_GROUPS.items() for name in names.split()}
CANONICAL_CATEGORIES = {cid: names.split()[0] for cid, names in CATEGORY_GROUPS.items()}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def parse_json(text):
    value = text.strip()
    if value.startswith('```'):
        value = re.sub(r'^```(?:json)?\s*', '', value, flags=re.IGNORECASE)
        if value.endswith('```'):
            value = value[:-3].strip()
    return json.loads(value)


def observer_categories(response):
    """Normalize model vocabulary at the text/structured-state boundary."""
    if not isinstance(response, dict) or not isinstance(response.get('categories'), list):
        raise ValueError('Expected an object with a categories list')
    aliases = {'appliance': 'appliances', 'object': 'objects',
               'stair': 'stairs', 'blind': 'blinds'}
    names, changes, skipped = [], [], []
    for raw in response['categories']:
        name = re.sub(r'[\s-]+', '_', raw.strip().lower().rstrip('.')) if isinstance(raw, str) else ''
        name = aliases.get(name, name)
        if name not in CATEGORY_IDS and name.endswith('s') and name[:-1] in CATEGORY_IDS:
            name = name[:-1]
        if name not in CATEGORY_IDS:
            skipped.append(dict(raw=raw, reason='unknown_category'))
            continue
        name = CANONICAL_CATEGORIES[CATEGORY_IDS[name]]
        if name != raw:
            changes.append(dict(raw=raw, category=name))
        if name not in names:
            names.append(name)
    room = response.get('room_type')
    return names, room if isinstance(room, str) and room.strip() else 'unknown', changes, skipped


def observer_boxes(response, category, limit, seen):
    """Accept a single box or box list; report malformed model items individually."""
    if isinstance(response, dict):
        response = [response] if 'bbox_3d' in response else response.get('objects', response.get('boxes'))
    if not isinstance(response, list):
        return [], [dict(category=category, reason='invalid_box_response')]
    accepted, skipped = [], []
    for index, item in enumerate(response):
        bbox = item.get('bbox_3d') if isinstance(item, dict) else None
        if (not isinstance(bbox, list) or len(bbox) != 9
                or any(type(v) not in (int, float) for v in bbox)
                or not np.isfinite(bbox).all() or min(bbox[3:6]) <= 0):
            skipped.append(dict(category=category, index=index, reason='invalid_3d_box'))
            continue
        key = (CATEGORY_IDS[category], *bbox)
        if key in seen:
            skipped.append(dict(category=category, index=index, reason='exact_duplicate'))
            continue
        if len(accepted) == limit:
            skipped.append(dict(category=category, index=index, reason='instance_cap'))
            continue
        seen.add(key)
        accepted.append(bbox)
    return accepted, skipped


def official_stats(repo):
    # Read the literal stats without importing Gradio, diffusion or the renderer.
    tree = ast.parse((Path(repo) / 'gradio_demo.py').read_text())
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'DATASET_STATS_AR' for t in node.targets))


def yaw_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def qwen_corners(bbox, angle_unit='normalized'):
    """Qwen camera x-right/y-down/z-forward -> CauPano image-up frame.

    Official example applies x, then y, then z rotations. Normalized angles
    use pi radians per unit; --angle-unit allows an explicit alternate format.
    """
    values = np.asarray(bbox, dtype=float)
    center, size = values[:3], values[3:6]
    x, y, z = values[6:] * {'normalized': np.pi, 'radians': 1., 'degrees': np.pi/180}[angle_unit]
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
    ry = yaw_matrix(y)
    rz = np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
    rotation = rz @ ry @ rx
    flip = np.array([1., -1., 1.])
    corners = (center + (SIGNS * size / 2) @ rotation.T) * flip
    return center * flip, corners


def upright_box(corners):
    """Enclose 9-DoF corners by a yaw-only box, choosing minimum xz area."""
    corners = np.asarray(corners)
    angles = np.linspace(-np.pi/2, np.pi/2, 181)[:-1]
    sizes = np.array([np.ptp(corners @ yaw_matrix(a), axis=0) for a in angles])
    index = np.argmin(sizes[:, 0] * sizes[:, 2])
    return corners.mean(axis=0), sizes[index], float(angles[index])


def world_object(track_id, category, center, corners, source, visible):
    return dict(track_id=track_id,
                category_id=field(CATEGORY_IDS[category], source, 'category_bridge_v1'),
                center_local_m=field(np.asarray(center).tolist(), source, 'camera_local_metres'),
                size_aabb_local_m=field(np.ptp(corners, axis=0).tolist(), source, 'full_corner_aabb'),
                bbox2d_visible_xyxy_norm=field(None, reason='not_estimated'),
                in_primary_room=field(None, reason='not_estimated'),
                local_evidence='visible' if visible else 'unknown')


def graph(sample_id, split, scope, objects, vocabulary_hash):
    return dict(schema_version=SCHEMA, frame=FRAME, id=sample_id, split=split,
                stage='prediction', scope=scope, vocabulary_sha256=vocabulary_hash,
                objects=objects, relations=[], ready_for_training=False)


def plot_layout(path, objects, frame='local'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 9))
    for obj in objects:
        corners = np.asarray(obj['corners_local'])
        lo, hi = corners.min(0), corners.max(0)
        color = 'tab:blue' if obj['observed'] else 'tab:orange'
        ax.add_patch(Rectangle((lo[0], lo[2]), hi[0]-lo[0], hi[2]-lo[2],
                               fill=False, edgecolor=color, linewidth=1.2))
        ax.text(obj['center_local'][0], obj['center_local'][2], obj['category'],
                fontsize=7, color=color)
    ax.plot(0, 0, 'k^', label='camera, forward +z' if frame == 'local' else 'scene origin (not camera)')
    ax.relim(); ax.autoscale_view(); ax.margins(.15)
    ax.set_aspect('equal'); ax.set_xlabel(f'{frame} x (m)'); ax.set_ylabel(f'{frame} z (m)')
    ax.set_title(f'Blue: observed; orange: generated; outlines: {frame} AABB')
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
