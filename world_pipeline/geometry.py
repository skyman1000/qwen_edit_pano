"""CPU geometry, explicit image registration basis, direct mesh visibility."""
import itertools
from pathlib import Path

import numpy as np

from qwen_pano.caupano.world_view import erp_rays, local_rays, rotation_y, sample_erp
from .common import read, sha


def alignment_basis(alignment):
    # np.roll output(lon) = source(lon - 2*pi*roll). Mirror happens BEFORE roll.
    mirror = np.diag([-1., 1., 1.]) if alignment['mirror'] else np.eye(3)
    return mirror @ rotation_y(-360 * alignment['roll_fraction'])


def box_geometry(obj, origin, basis):
    axes = np.asarray(obj['axes_world'], dtype=float)
    radii = np.asarray(obj['radii'], dtype=float)
    center = np.asarray(obj['center_world'], dtype=float)
    if (axes.shape != (3, 3) or radii.shape != (3,) or center.shape != (3,) or
            not all(np.isfinite(v).all() for v in (axes, radii, center)) or min(radii) <= 0 or
            not np.allclose(axes @ axes.T, np.eye(3), atol=1e-3)):
        raise ValueError('Invalid source OBB')
    signs = np.array(list(itertools.product([-1, 1], repeat=3)))
    corners_world = center + (signs * radii) @ axes
    corners = (corners_world - origin) @ basis
    local = (center - origin) @ basis
    size = 2 * radii @ np.abs(axes @ basis)
    if not np.allclose(np.ptp(corners, axis=0), size, atol=1e-6):
        raise ValueError('OBB corner/AABB disagreement')
    if not np.allclose(local @ basis.T + origin, center, atol=1e-6):
        raise ValueError('Coordinate roundtrip failed')
    return local.tolist(), size.tolist()


def visible_box(mask):
    y, x = np.where(mask)
    if not len(x):
        return None
    h, w = mask.shape
    return [float(x.min()/w), float(y.min()/h), float((x.max()+1)/w), float((y.max()+1)/h)]


def ncc(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a = a - a.mean(axis=(0, 1), keepdims=True)
    b = b - b.mean(axis=(0, 1), keepdims=True)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a*b)/denominator) if denominator > 1e-8 else None


def view_scores(first, second):
    scores = []
    for yaw, pitch in [(0, 0), (90, 0), (180, 0), (270, 0), (0, 90), (0, -90)]:
        c, s = np.cos(np.deg2rad(pitch)), np.sin(np.deg2rad(pitch))
        rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
        rays = local_rays(128) @ (rotation_y(yaw) @ rx).T
        scores.append(dict(yaw=yaw, pitch=pitch, ncc=ncc(sample_erp(first, rays), sample_erp(second, rays))))
    return scores


class MeshScene:
    """All region triangles remain occluders, including unknown semantics."""
    def __init__(self, links_path, threads=2):
        import open3d as o3d
        self.o3d, self.threads = o3d, threads
        links_path = Path(links_path)
        links = read(links_path)
        region_dir = Path(links['region_directory'])
        index = read(region_dir / 'region_index.json')
        if index['category_mapping_sha256'] != links['category_mapping_sha256']:
            raise ValueError('Mesh/object vocabularies differ')
        for path in (region_dir/'region_validation.json', links_path.parent/'object_link_validation.json'):
            report = read(path)
            if report.get('errors') or report['scan_id'] != links['scan_id'] or not report['status'].startswith('PASS'):
                raise ValueError(f'Invalid source validation: {path}')
        files = {r['region_id']: r['array_file'] for r in links['region_files']}
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=threads)
        self.labels, self.source_sha256 = {}, {}
        for region in index['regions']:
            mesh_path = region_dir / region['array_file']
            ids_path = links_path.parent / files[region['region_id']]
            for p in (mesh_path, ids_path):
                self.source_sha256[str(p)] = sha(p)
            with np.load(mesh_path, allow_pickle=False) as data:
                xyz = np.ascontiguousarray(data['vertices_world'], dtype='float32')
                triangles = np.ascontiguousarray(data['triangles'], dtype='uint32')
            with np.load(ids_path, allow_pickle=False) as data:
                ids = data['face_instance_id'].copy()
                valid = data['face_instance_valid'].copy()
            if len(ids) != len(triangles):
                raise ValueError('Face/label length mismatch')
            gid = self.scene.add_triangles(o3d.core.Tensor(xyz), o3d.core.Tensor(triangles))
            self.labels[gid] = (ids, valid)

    def cast(self, origin, basis, rays):
        directions = rays @ basis.T
        packed = np.concatenate([np.broadcast_to(origin, directions.shape), directions], axis=-1).astype('float32')
        hit = self.scene.cast_rays(self.o3d.core.Tensor(packed), nthreads=self.threads)
        depth = hit['t_hit'].numpy()
        gids, tids = hit['geometry_ids'].numpy(), hit['primitive_ids'].numpy()
        hit_valid = np.isfinite(depth) & (depth > 0)
        instance = np.full(depth.shape, -1, dtype='int32')
        for gid, (ids, valid) in self.labels.items():
            selected = hit_valid & (gids == gid)
            indices = tids[selected].astype('int64')
            instance[selected] = np.where(valid[indices], ids[indices], -1)
        return dict(instance=instance, depth=np.where(hit_valid, depth, 0).astype('float32'),
                    hit_valid=hit_valid, instance_valid=instance > 0)
