"""CPU-only MP3D object/mesh sources and face2-to-world registration.

No content hashes, full legacy world export, sensor depth, or sensor normals.
All triangles remain occluders, including those without a usable object label.
"""
import csv
import io
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

from qwen_pano.caupano.data.matterport.camera_parser import member_index, parse_conf
from qwen_pano.caupano.data.matterport.region_parser import read_region_ply
from qwen_pano.caupano.data.matterport.semseg_json import load_semseg_json


def image_from_zip(archive, member):
    with Image.open(io.BytesIO(archive.read(member))) as image:
        return np.asarray(image.convert('RGB')).copy()


def unit(values):
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm < 1e-10):
        raise ValueError('Zero ray or OBB axis')
    return values / norm


def rotation_y(degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def view_rays(points, size, yaw):
    xy = (np.asarray(points, dtype=float) + .5) / size * 2 - 1
    rays = np.column_stack([xy[:, 0], -xy[:, 1], np.ones(len(xy))])
    return unit(rays) @ rotation_y(yaw).T


def fit_basis(local, world):
    # Local x-right/y-up/z-forward and source world have different handedness.
    # Do not force determinant +1.
    u, _, vt = np.linalg.svd(local.T @ world)
    return (u @ vt).T


def angular_errors(local, world, basis):
    return np.degrees(np.arccos(np.clip(np.sum((local @ basis.T) * world, axis=1), -1., 1.)))


def fit_registration_partition(a, b, groups, observations, trials, parity):
    """Estimate from one yaw partition; check a disjoint partition with unchanged gates."""
    fit_groups = np.array([r['yaw_index'] % 2 == parity for r in observations])
    fit = fit_groups[groups]
    candidates = np.flatnonzero(fit)
    if len(candidates) < 12:
        raise ValueError('Insufficient fit-side matches')
    rng, best = np.random.default_rng(0), np.zeros(len(a), dtype=bool)
    for _ in range(trials):
        sample = rng.choice(candidates, 4, replace=False)
        basis = fit_basis(a[sample], b[sample])
        keep = (angular_errors(a, b, basis) < 2.5) & fit
        if keep.sum() > best.sum():
            best = keep
    if best.sum() < 12:
        raise ValueError('No reliable registration consensus')
    basis = fit_basis(a[best], b[best])
    errors = angular_errors(a, b, basis)
    held = (~fit) & (errors < 2.5)
    covered = len(set(groups[held].tolist()))
    report = dict(fit_yaw_parity=parity, fit_inliers=int(best.sum()),
                  heldout_inliers=int(held.sum()), heldout_observations=covered,
                  heldout_median_degrees=float(np.median(errors[held])) if held.any() else None)
    if best.sum() < 60 or held.sum() < 30 or covered < 3:
        raise ValueError('Weak held-out registration: '+str(report))
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6):
        raise ValueError('Nonorthogonal registration basis')
    return basis, report


class MinimalHouse:
    def __init__(self, scan, mapping_file, threads=4):
        self.scan = Path(scan)
        self.house = self.scan.name
        self.threads = threads
        self.issues = []
        with Path(mapping_file).open() as handle:
            mapping = {int(r['index']): int(r['mpcat40index'])
                       for r in csv.DictReader(handle, delimiter='\t')}
        # Parse only panorama positions, object OBBs and their class mapping.
        categories, raw_objects, self.origins = {}, [], {}
        with ZipFile(self.scan/'house_segmentations.zip') as archive:
            names = member_index(archive)
            text = archive.read(names[self.house+'.house']).decode()
            if text.splitlines()[0].split() != ['ASCII', '1.1']:
                raise ValueError('Unsupported house format')
            for line in text.splitlines()[1:]:
                p = line.split()
                if not p:
                    continue
                if p[0] == 'C':
                    categories[int(p[1])] = int(p[2])
                elif p[0] == 'P' and len(p) == 13:
                    self.origins[p[1]] = np.array(p[5:8], dtype=float)
                elif p[0] == 'O':
                    raw_objects.append(p)
            self.repairs = []
            name = names[self.house+'.semseg.json']
            house_groups = load_semseg_json(archive.read(name), name, self.repairs)['segGroups']
        self.objects = {}
        for p in raw_objects:
            hid, category = int(p[1]), int(p[3])
            class_id = mapping[categories[category]] if category >= 0 else 41
            axes = np.array(p[7:13], dtype=float).reshape(2, 3)
            axes = np.vstack([axes, np.cross(axes[0], axes[1])])
            axis_norm = np.linalg.norm(axes, axis=1, keepdims=True)
            axes_usable = np.isfinite(axes).all() and np.all(axis_norm > 1e-10)
            if axes_usable:
                axes = axes/axis_norm
            center, radii = np.array(p[4:7], dtype=float), np.array(p[13:16], dtype=float)
            valid = (axes_usable and np.isfinite(center).all() and np.isfinite(radii).all()
                     and np.all(radii > 0) and np.allclose(axes @ axes.T, np.eye(3), atol=1e-3))
            if not valid:
                self.issues.append(dict(instance_id=hid+1, reason='invalid_source_OBB_not_a_geometry_target'))
            if hid+1 in self.objects:
                raise ValueError('Duplicate source object ID')
            self.objects[hid+1] = dict(category_id=class_id, center_world=center,
                                      radii=radii, axes_world=axes, geometry_valid=bool(valid))
        by_segments = {}
        for group in house_groups:
            key = frozenset(group['segments'])
            if key in by_segments:
                raise ValueError('Ambiguous house instance segment set')
            by_segments[key] = int(group['id'])
        with ZipFile(self.scan/'undistorted_camera_parameters.zip') as archive:
            names = member_index(archive)
            _, observations = parse_conf(archive.read(names[self.house+'.conf']).decode())
        self.observations = {}
        for observation in observations:
            uuid, sensor, yaw = Path(observation['color']).stem.rsplit('_', 2)
            observation.update(camera_index=int(sensor[1:]), yaw_index=int(yaw))
            self.observations.setdefault(uuid, []).append(observation)
        for group in self.observations.values():
            group.sort(key=lambda r: (r['camera_index'], r['yaw_index']))
        import open3d as o3d
        self.o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=threads)
        self.labels = {}
        with ZipFile(self.scan/'region_segmentations.zip') as archive:
            names = member_index(archive)
            meshes = sorted((n for n in names if n.startswith('region') and n.endswith('.ply')),
                            key=lambda n: int(n[6:-4]))
            if not meshes:
                raise ValueError('No region meshes')
            for name in meshes:
                rid, stem = int(name[6:-4]), name[:-4]
                arrays = read_region_ply(archive.read(names[name]))
                semseg = names[stem+'.semseg.json']
                groups = load_semseg_json(archive.read(semseg), semseg, self.repairs)['segGroups']
                ids = np.full(len(arrays['triangles']), -1, dtype=np.int32)
                for group in groups:
                    local_segments = group['segments']
                    if not local_segments or min(local_segments) < 0 or max(local_segments) >= 1_000_000:
                        raise ValueError(f'Uninspected region segment convention: {rid}')
                    key = frozenset(rid*1_000_000+s for s in local_segments)
                    hid = by_segments.get(key)
                    if hid is None:
                        self.issues.append(dict(region_id=rid, local_instance=group['objectId'],
                                                reason='no_exact_house_segment_link'))
                        continue
                    if hid+1 not in self.objects:
                        raise ValueError('Mesh label references absent house object')
                    ids[arrays['face_instance_local'] == group['objectId']] = hid+1
                xyz = np.ascontiguousarray(arrays['vertices_world'], dtype=np.float32)
                triangles = np.ascontiguousarray(arrays['triangles'], dtype=np.int64)
                if (not np.isfinite(xyz).all() or triangles.min() < 0 or triangles.max() >= len(xyz)):
                    raise ValueError('Invalid region mesh vertices/triangles')
                gid = self.scene.add_triangles(
                    o3d.core.Tensor(xyz, device=o3d.core.Device('CPU:0')),
                    o3d.core.Tensor(triangles.astype(np.uint32), device=o3d.core.Device('CPU:0')))
                self.labels[gid] = ids

    def cast(self, origin, basis, rays):
        directions = rays @ basis.T
        packed = np.concatenate([np.broadcast_to(origin, directions.shape), directions], axis=-1).astype(np.float32)
        hit = self.scene.cast_rays(self.o3d.core.Tensor(packed, device=self.o3d.core.Device('CPU:0')),
                                   nthreads=self.threads)
        depth = hit['t_hit'].numpy()
        gids, tids = hit['geometry_ids'].numpy(), hit['primitive_ids'].numpy()
        hit_valid = np.isfinite(depth) & (depth > 0)
        instance = np.full(depth.shape, -1, dtype=np.int32)
        for gid, labels in self.labels.items():
            selected = hit_valid & (gids == gid)
            instance[selected] = labels[tids[selected].astype(np.int64)]
        return instance, hit_valid

    def register(self, row, features=3600, trials=3000, fit_parities=(0,)):
        """Fit on even source yaws, independently check odd source yaws.

        Four *calibration* faces improve rotation fit; the training input remains
        one face2. This is not four-direction training or extra model input.
        """
        uuid = row['source_view_id']
        if uuid not in self.origins or uuid not in self.observations:
            raise ValueError('Panorama missing from house position/camera table')
        sift = cv2.SIFT_create(nfeatures=features)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        virtual = []
        with ZipFile(self.scan/'matterport_skybox_images.zip') as archive:
            names = member_index(archive)
            for face, yaw in ((1, -90), (2, 0), (3, 90), (4, 180)):
                rgb = image_from_zip(archive, names[f'{uuid}_skybox{face}_sami.jpg'])
                view = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
                kp, desc = sift.detectAndCompute(cv2.cvtColor(view, cv2.COLOR_RGB2GRAY), None)
                if desc is not None:
                    virtual.append((yaw, kp, desc))
        aa, bb, groups, observations_report = [], [], [], []
        with ZipFile(self.scan/'undistorted_color_images.zip') as archive:
            names = member_index(archive)
            for index, obs in enumerate(self.observations[uuid]):
                rgb = image_from_zip(archive, names[Path(obs['color']).name])
                oh, ow = rgb.shape[:2]
                rw, rh = 640, round(oh*640/ow)
                view = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
                kp, desc = sift.detectAndCompute(cv2.cvtColor(view, cv2.COLOR_RGB2GRAY), None)
                best = {}
                if desc is not None:
                    for yaw, vkp, vd in virtual:
                        if len(vd) < 2:
                            continue
                        for pair in matcher.knnMatch(desc, vd, k=2):
                            if len(pair) != 2:
                                continue
                            m, n = pair
                            if m.distance < .70*n.distance:
                                ratio = m.distance/n.distance
                                if m.queryIdx not in best or ratio < best[m.queryIdx][0]:
                                    best[m.queryIdx] = (ratio, yaw, vkp[m.trainIdx].pt)
                k = np.array(obs['intrinsics'], dtype=float).copy()
                pose = np.array(obs['camera_to_world'], dtype=float)
                if (k.shape != (3, 3) or pose.shape != (4, 4)
                        or not np.isfinite(k).all() or not np.isfinite(pose).all()
                        or k[0, 0] <= 0 or k[1, 1] <= 0
                        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3)):
                    raise ValueError('Invalid source camera intrinsics/pose')
                k[1, 2] = oh-1-k[1, 2]
                source_basis = pose[:3, :3] @ np.diag([1., -1., -1.])
                for qi, (_, yaw, point) in best.items():
                    pixel = (np.array(kp[qi].pt)+.5)*[ow/rw, oh/rh]-.5
                    ray = np.array([(pixel[0]-k[0, 2])/k[0, 0], (pixel[1]-k[1, 2])/k[1, 1], 1.])
                    aa.append(view_rays([point], 512, yaw)[0])
                    bb.append(unit(ray) @ source_basis.T)
                    groups.append(index)
                observations_report.append(dict(camera_index=obs['camera_index'], yaw_index=obs['yaw_index'],
                                                matches=len(best), fit=obs['yaw_index'] % 2 == 0))
        a, b, groups = np.asarray(aa), np.asarray(bb), np.asarray(groups, dtype=int)
        if len(a) < 12:
            raise ValueError(f'Insufficient registration matches: {len(a)}')
        attempts = []
        for parity in fit_parities:
            try:
                basis, metrics = fit_registration_partition(a,b,groups,observations_report,trials,parity)
                break
            except ValueError as exc:
                attempts.append(dict(fit_yaw_parity=parity, error=str(exc)))
        else:
            if len(attempts) == 1:
                raise ValueError(attempts[0]['error'])
            raise ValueError('All registration partitions failed: '+str(attempts))
        for observation in observations_report:
            observation['fit'] = observation['yaw_index'] % 2 == parity
        report = dict(method=('four_raw_calibration_faces_to_known_perspective_cameras_sift_v1'
                              if tuple(fit_parities)==(0,) else 'four_raw_calibration_faces_disjoint_yaw_partitions_sift_v2'),
                      training_Local='single_face2', observations=observations_report,
                      **metrics, requested_fit_parities=list(fit_parities), failed_partition_attempts=attempts,
                      sift_features=features, ransac_trials=trials,
                      basis_local_to_world=basis.tolist(), origin_world=self.origins[uuid].tolist(),
                      determinant=float(np.linalg.det(basis)),
                      limitations='Rotation fit with inherited house P origin; not full RGB/mesh SE3 or parallax correction')
        return self.origins[uuid], basis, report
