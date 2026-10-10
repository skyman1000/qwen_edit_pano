"""Deterministic, explicitly lossy GT-object-to-text projections. No LLM."""
from collections import Counter
import math
from .common import validate_graph

VERSION = 'gt_object_text_v1'
DIRECTIONS = ('front', 'front-right', 'right', 'back-right', 'back', 'back-left', 'left', 'front-left')


def direction(center):
    x, _, z = center
    if math.hypot(x, z) < 1e-6:
        return 'at-camera-horizontal-position'
    return DIRECTIONS[int(math.floor((math.degrees(math.atan2(x, z)) % 360 + 22.5) / 45)) % 8]


def spatial_text(graph, vocabulary, format='spatial'):
    validate_graph(graph)
    if graph['stage'] != 'gt' or graph['scope'] != 'full':
        raise ValueError('Requires actual GT G_full')
    if graph['vocabulary_sha256'] != vocabulary['category_mapping_sha256']:
        raise ValueError('Vocabulary mismatch')
    if format not in ('spatial', 'metric'):
        raise ValueError('Unknown text format')
    groups = Counter()
    records = []
    for obj in graph['objects']:
        cat = obj['category_id']['value']
        name = vocabulary['categories'].get(str(cat), 'unclassified object').replace('_', ' ')
        center = obj['center_local_m']['value']
        size = obj['size_aabb_local_m']['value']
        if center is None:
            bearing, distance = 'position-unknown', 'distance-unknown'
        else:
            bearing = direction(center)
            radius = math.hypot(center[0], center[2])
            distance = 'near' if radius < 2 else ('mid' if radius < 4 else 'far')
        groups[(bearing, name, distance)] += 1
        def vector(v):
            return 'unknown' if v is None else '(' + ','.join(f'{x:.2f}' for x in v) + ')'
        records.append(f'{name}: center={vector(center)}; extent={vector(size)}.')
    # No source instance IDs, room codes, captions, appearance, relationships or
    # unsupported orientations are inserted. Every object contributes once.
    if format == 'spatial':
        lines = []
        order = {name: n for n, name in enumerate((*DIRECTIONS, 'at-camera-horizontal-position', 'position-unknown'))}
        for bearing in sorted({k[0] for k in groups}, key=order.__getitem__):
            items = [f'{count} x {name} ({distance})' for (b, name, distance), count in sorted(groups.items()) if b == bearing]
            lines.append(bearing + ': ' + '; '.join(items) + '.')
        legend = ('Camera-relative layout: front is the input viewing direction; right/left follow the input image. '
                  'Near/mid/far mean horizontal distance <2m, 2-4m, and >=4m from the camera. '
                  'Directions use eight 45-degree sectors.')
        fields = ['category', 'object_count', 'center-derived horizontal direction and distance band']
    else:
        lines = sorted(records)
        legend = ('Camera-relative coordinates in metres: x=image-right, y=image-up, z=input-view-forward. '
                  'Center is (x,y,z); extent is the full camera-axis AABB size, not semantic orientation. '
                  'Image-up is not necessarily gravity-up.')
        fields = ['category', 'object_count', 'center_local_m rounded to 0.01m', 'size_aabb_local_m rounded to 0.01m']
    text = (legend + '\nUse this annotated layout to guide the surroundings while preserving the visible input scene. '
            'This is a partial annotation, not an exhaustive inventory; unknown does not mean absent. '
            'Do not draw text or labels in the image.\n' + '\n'.join(lines or ['No annotated objects; layout unknown.']))
    return text, dict(version=VERSION, format=format, object_count=len(graph['objects']),
                      used_fields=fields, omitted_fields=['room', 'relations', 'appearance', 'instance IDs', 'visibility flags'],
                      note='Spatial format additionally omits size, vertical position and precise metric geometry; no object truncation')


def shuffle_sources(ids, seed):
    import random
    ids = sorted(ids)
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError('Shuffle requires >=2 unique same-split samples')
    random.Random(seed).shuffle(ids)
    return {sid: ids[(n + 1) % len(ids)] for n, sid in enumerate(ids)}
