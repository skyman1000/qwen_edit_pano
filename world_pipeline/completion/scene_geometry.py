"""Camera-frame geometry with explicit gravity/floor priors; no GT dependency."""
import math
import numpy as np

from ..common import FRAME, finite_vector


def camera_geometry(up_local=None, camera_height=1.5):
    if up_local is not None and (not finite_vector(up_local, 3) or np.linalg.norm(up_local) < 1e-8):
        raise ValueError('up_local must be a finite nonzero vector in the camera frame')
    if not math.isfinite(camera_height) or camera_height <= 0:
        raise ValueError('Positive camera-height prior required')
    up = np.array([0., 1., 0.] if up_local is None else up_local, dtype=float)
    up /= np.linalg.norm(up)
    return dict(frame=FRAME, up_local=up.tolist(),
                up_source='assumed_image_up' if up_local is None else 'user_supplied_direction',
                gravity_verified=False, levelled=False,
                floor_plane_local=[*up.tolist(), camera_height],
                floor_source='camera_height_prior_not_measured', floor_verified=False,
                note='Camera axes are retained; image up is not measured gravity. No GT is read.')


def horizontal_axes(geometry):
    up = np.array(geometry['up_local'])
    right = np.array([1., 0., 0.])-up*up[0]
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0., 0., 1.])-up*up[2]
    right /= np.linalg.norm(right)
    forward = np.array([0., 0., 1.])-up*up[2]
    forward -= right*np.dot(forward, right)
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError('Gravity direction leaves no usable camera-forward ground direction')
    forward /= np.linalg.norm(forward)
    return right, up, forward


def half_extent(size, direction):
    """AABB support radius along an arbitrary unit direction, not OBB height."""
    return float(np.dot(np.abs(direction), np.asarray(size))/2)


def ground_center(size, bottom, radius, angle, geometry):
    right, up, forward = horizontal_axes(geometry)
    # Plane is n.p+d=0. Distances and box support radius use the same n.
    d = geometry['floor_plane_local'][3]
    a = math.radians(angle)
    return (radius*(right*math.sin(a)+forward*math.cos(a))
            +up*(bottom+half_extent(size, up)-d)).tolist()


def projection_measurement(center, size, target, hfov, image_size):
    from .box_layout import projected_box
    tan_h = math.tan(math.radians(hfov)/2)
    tan_v = tan_h*image_size[1]/image_size[0]
    projected = projected_box(np.array(center), np.array(size), tan_h, tan_v)
    if projected is None:
        return dict(projected_box=None, visible_bbox_coverage=None,
                    note='AABB crosses camera near plane; simple corner projection is not used')
    target = np.array(target)
    intersect = float(np.prod(np.maximum(0, np.minimum(projected[2:], target[2:])
                                             -np.maximum(projected[:2], target[:2]))))
    target_area = float(np.prod(target[2:]-target[:2]))
    proj_area = float(np.prod(projected[2:]-projected[:2]))
    return dict(projected_box=projected.tolist(), visible_bbox_coverage=intersect/target_area,
                visible_bbox_iou=intersect/(target_area+proj_area-intersect),
                projected_center_uv=[.5+center[0]/center[2]/(2*tan_h),
                                     .5-center[1]/center[2]/(2*tan_v)],
                note='Visible mask fragment and full 3D AABB projection need not have equal boxes; '
                     'containment tests internal consistency, not metric truth')


def observation_report(state, hfov, image_size, geometry):
    records = []
    n = np.array(geometry['up_local']); d = geometry['floor_plane_local'][3]
    for obj in state['objects']:
        center = obj['center_local_m']['value']; size = obj['size_aabb_local_m']['value']
        bbox = obj['bbox2d_visible_xyxy_norm']['value']
        if center is None or size is None or bbox is None:
            continue
        rec = dict(id=obj['track_id'], category_id=obj['category_id']['value'],
                   **projection_measurement(center, size, bbox, hfov, image_size),
                   min_box_signed_distance_to_floor_prior_m=float(np.dot(n, center)+d-half_extent(size, n)),
                   floor_collision_verified=False)
        records.append(rec)
    return dict(protocol='camera_projection_and_explicit_prior_v1', camera_geometry=geometry,
                objects=records, observation_geometry_verified=False,
                note='No positions or sizes were modified. Negative distance to an assumed floor '
                     'is not proof of intersection with a real floor. No level camera is inferred.')
