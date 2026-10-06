"""Small versioned review contract. No model imports or implicit downloads."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'caupano.local_world.v0.1-pilot1'
FRAME = 'local_view_x_right_y_image_up_z_forward_metres'
ELIGIBLE = set(range(3, 41)) - {17}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    result = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    ids = [r['id'] for r in result]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate IDs: {path}')
    return result


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_rows(path, values):
    Path(path).write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values))


def new_output(path):
    path = Path(path).resolve()
    allowed = [ROOT / 'qwen_edit_pano/data', ROOT / 'qwen_edit_pano/outputs', ROOT / 'research/audits']
    if not any(path.is_relative_to(p) and path != p for p in allowed):
        raise ValueError('Use a NEW directory below qwen_edit_pano/data, outputs, or research/audits')
    path.mkdir(parents=True, exist_ok=False)
    return path


def field(value, source='gt_derived', method=None, reason=None):
    return dict(value=value, valid=value is not None,
                source=source if value is not None else 'unknown', method=method,
                confidence=None, confidence_kind=None,
                unknown_reason=reason if value is None else None)


def finite_vector(value, n):
    return (isinstance(value, list) and len(value) == n and
            all(type(v) in (int, float) and math.isfinite(v) for v in value))


def validate_graph(graph):
    if graph['schema_version'] != SCHEMA or graph['frame'] != FRAME:
        raise ValueError('Schema/frame mismatch')
    if graph['scope'] not in ('observed', 'full', 'hidden'):
        raise ValueError('Unknown scope')
    seen = set()
    for obj in graph['objects']:
        if not isinstance(obj['track_id'], str) or obj['track_id'] in seen:
            raise ValueError('Invalid/duplicate track ID')
        seen.add(obj['track_id'])
        for key in ('category_id', 'center_local_m', 'size_aabb_local_m', 'bbox2d_visible_xyxy_norm', 'in_primary_room'):
            f = obj[key]
            if type(f['valid']) is not bool or f['valid'] != (f['value'] is not None):
                raise ValueError(f'Invalid missingness: {key}')
        category = obj['category_id']['value']
        if category is not None and (type(category) is not int or category not in ELIGIBLE):
            raise ValueError('Invalid object category')
        for key in ('center_local_m', 'size_aabb_local_m'):
            v = obj[key]['value']
            if v is not None and (not finite_vector(v, 3) or (key.startswith('size') and min(v) <= 0)):
                raise ValueError(f'Invalid geometry: {key}')
        box = obj['bbox2d_visible_xyxy_norm']['value']
        if box is not None and (not finite_vector(box, 4) or not
                               (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
            raise ValueError('Invalid normalized visible bbox')
        if obj['local_evidence'] not in ('visible', 'unobserved', 'unknown'):
            raise ValueError('Invalid local evidence')
    return graph
