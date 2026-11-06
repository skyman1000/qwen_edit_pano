"""Small declarative layout -> camera xyz/full AABB. No model-generated code.

Qwen chooses contents, functional groups and camera regions. This module
constructs the declared edge-to-edge relationships and places whole groups
outside the given FOV. It does not repair arbitrary predicted xyz, infer walls,
level the camera, move observations, or certify mesh collisions/support.
"""
import json
import math

import numpy as np

from ..common import finite_vector
from .box_layout import frustum_margin
from .geometry import CANONICAL_CATEGORIES

PROTOCOL = 'camera_region_group_observed_y_anchor_v3'
DIRECTIONS = dict(left=(0,-1), right=(0,1), below=(1,-1), above=(1,1),
                  front=(2,-1), behind=(2,1))
Y_EDGES = dict(bottom=-.5, center=0., top=.5)


def number(value, label, positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'{label} must be finite'+(' and positive' if positive else ''))
    return float(value)


def aligned_y(reference, size, alignment):
    """Align camera-y AABB edges/centers, without inferring a floor or gravity."""
    if not isinstance(alignment,str) or alignment not in Y_EDGES:
        raise ValueError('Camera-y alignment must be bottom, center, or top')
    return reference['center'][1]+Y_EDGES[alignment]*(reference['size'][1]-size[1])


def prompt(observed, observed_layout, hfov, capacity):
    by_id={o['id']:o for o in observed_layout['objects']}
    fixed=[dict(id=o['id'],category=o['category'],description=o['description'],
                center=[round(x,2)for x in by_id[o['id']]['center']],
                size=[round(x,2)for x in by_id[o['id']]['size']],
                camera_y_min=round(by_id[o['id']]['center'][1]-by_id[o['id']]['size'][1]/2,2),
                camera_y_max=round(by_id[o['id']]['center'][1]+by_id[o['id']]['size'][1]/2,2))for o in observed]
    example=dict(room_type='bedroom',scene_hypothesis='A storage area in the unobserved left part of a bedroom.',
        groups=[dict(id='storage',region='left',along_m=1.2,clearance_m=.4,
            objects=[dict(id='new_cabinet',category='cabinet',size=[1.2,1.8,.6],
                          height_anchor_id=observed[0]['id'],height_alignment='bottom',height_offset_m=0.,
                          description='storage cabinet',reason='Provides bedroom storage in the unobserved area.'),
                     dict(id='new_plant',category='plant',size=[.4,.7,.4],anchor_id='new_cabinet',
                          direction='right',gap_m=.3,vertical_alignment='bottom',
                          description='potted plant beside cabinet',reason='Decorates the storage area.'),
                     dict(id='new_art',category='picture',size=[.5,.6,.15],anchor_id='new_cabinet',direction='above',gap_m=.2,
                          description='wall art above cabinet',reason='Decorates the new storage area.')])])
    return (
        'Design ONE plausible continuation of the current room beyond this partial photograph. '
        'The photograph and FIXED OBSERVED boxes describe visible evidence, not the entire room. '
        'Keep those instances unchanged. Propose only useful NEW objects, grouped by function. '
        'Do not place new chairs/tables into the photographed bed, add another copy of visible wall art, '
        'or call a new cabinet near a visible window without a coherent spatial explanation. '
        'Use a small DECLARATIVE layout; do NOT calculate final xyz coordinates. '
        'A deterministic compiler handles projection arithmetic and edge-to-edge spacing. '
        'All axes remain CAMERA axes: x right, y IMAGE-UP, z forward, in metres. '
        'No measured gravity, floor, room boundary or support surfaces are available. '
        f'Photo horizontal FOV is {hfov:g} degrees. Choose each group region as left, right, or back '
        'relative to CAMERA, not relative to the bed or window. '
        'For left/right, along_m>=0 is the group forward z; clearance_m>0 places its WHOLE AABB '
        'beyond that side of the viewing cone. Side regions fan away from the photograph: at '
        '90 degrees and z=4m, the right region starts beyond x=4m even BEFORE group width. '
        'For back, along_m is the group signed x and its WHOLE AABB is behind the camera. '
        'Choose continuous along_m/clearance_m to keep a plausible compact room and separate groups; '
        'do not place everything far away or pack unrelated furniture in one area. '
        'A group is a functional furniture cluster, e.g. a reading area or storage area. '
        'First object: id, category, FULL camera-axis AABB size[3], description, reason, '
        'height_anchor_id naming a FIXED OBSERVED object, height_alignment (bottom/center/top), '
        'and height_offset_m (signed CAMERA-y displacement after alignment). '
        'The compiler aligns the SAME camera-y edge/center of the new root and its observed reference, '
        'then adds height_offset_m. Do not predict an independent center_y_m. '
        'Choose a reference with a relevant height: for an ordinary floor-standing furniture cluster, '
        'the lower edge of visible floor-standing furniture is a useful COARSE prior; do not use '
        'wall art or a ceiling lamp as its lower-edge reference. Use zero offset unless another '
        'relative height is intended. This is camera-y alignment, NOT a measured shared floor: '
        'camera tilt, AABB inflation, and visible-geometry errors remain limitations. '
        'Later objects: id/category/size/description/reason plus anchor_id naming an EARLIER object '
        'IN THIS GROUP, direction, gap_m>=0. Directions: left/right/front/behind/above/below. '
        'The compiler puts their BOX EDGES gap_m apart along that camera axis. '
        'For left/right/front/behind also specify vertical_alignment (bottom/center/top) relative '
        'to the anchor; choose bottom for side-by-side floor-standing furniture of different heights, '
        'not center by default. The remaining horizontal center coordinate is aligned. '
        'For above/below, the y position follows gap_m and both horizontal centers are aligned. '
        'vertical_alignment applies ONLY to horizontal directions; it is not used by above/below. '
        'No independent center_y_m or height_anchor_id for children. This is relative placement, '
        'not physical support: above does not mean on. Do not invent on-top/support claims. '
        'The whole group is translated together into the chosen unobserved region, so its internal '
        'arrangement is preserved. This does not certify walls, physical collisions or room semantics. '
        'Return ONLY JSON with room_type, scene_hypothesis, groups, following this syntax example. '
        'Choose contents for the CURRENT photo; the example is not a required furniture list. '
        f'At most {capacity} NEW objects total, not a target count. No observed objects in groups. '
        'No center xyz, rotations, executable code or extra relation graph. '
        'Allowed categories: '+', '.join(CANONICAL_CATEGORIES.values())
        +'\nSYNTAX EXAMPLE: '+json.dumps(example,separators=(',',':'))
        +'\nFIXED OBSERVED BOXES: '+json.dumps(fixed,separators=(',',':'),ensure_ascii=False))


def overlap_report(additions, observed_layout):
    """Conservative box risks, NOT mesh collisions; observed/observed ignored."""
    boxes=[dict(o,role='observed')for o in observed_layout['objects']]
    boxes += [dict(o,role='new')for o in additions]
    pairs=[]
    for i,a in enumerate(boxes):
        for b in boxes[i+1:]:
            if a['role']==b['role']=='observed':continue
            ca,cb=np.array(a['center']),np.array(b['center'])
            sa,sb=np.array(a['size']),np.array(b['size'])
            length=np.minimum(ca+sa/2,cb+sb/2)-np.maximum(ca-sa/2,cb-sb/2)
            if min(length)>1e-6:
                pairs.append(dict(a=a['id'],b=b['id'],
                    fraction_of_smaller_aabb=float(np.prod(length)/min(np.prod(sa),np.prod(sb)))))
    return dict(pairs=pairs, note='Read-only camera-AABB overlap; not mesh collision or support certification')


def compile_program(answer, observed, observed_layout, hfov, capacity):
    if (not isinstance(answer,dict) or any(not isinstance(answer.get(k),str) or not answer[k].strip()
            for k in ('room_type','scene_hypothesis')) or not isinstance(answer.get('groups'),list)):
        raise ValueError('Placement program needs room_type, scene_hypothesis and groups')
    tangent=math.tan(math.radians(hfov)/2)
    ids={o['id'] for o in observed}; group_ids=set(); additions=[]; diagnostics=[]; heights=[]; unused=[]
    references={o['id']:o for o in observed_layout['objects'] if o['id'] in ids}
    for group in answer['groups']:
        if (not isinstance(group,dict) or not isinstance(group.get('id'),str) or not group['id'].strip()
                or group['id'] in group_ids or group.get('region') not in ('left','right','back')
                or not isinstance(group.get('objects'),list) or not group['objects']):
            raise ValueError('Each group needs a distinct ID, camera region and nonempty objects')
        group_ids.add(group['id'])
        along=number(group.get('along_m'),'along_m')
        clearance=number(group.get('clearance_m'),'clearance_m',positive=True)
        if group['region']!='back' and along<0:
            raise ValueError('Side group along_m is nonnegative forward z; use back for negative z')
        placed={}; local=[]
        for i,item in enumerate(group['objects']):
            if (not isinstance(item,dict) or not isinstance(item.get('id'),str) or not item['id'].strip()
                    or item['id'] in ids or item.get('category') not in CANONICAL_CATEGORIES.values()
                    or not finite_vector(item.get('size'),3) or min(item['size'])<=0
                    or any(not isinstance(item.get(k),str) or not item[k].strip() for k in ('description','reason'))):
                raise ValueError('Each item needs a unique ID, allowed category, positive full size, description and reason')
            if 'center' in item:
                raise ValueError('Program items specify placements, not a second independent center xyz')
            if len(additions)+len(local)>=capacity:
                raise ValueError('Program exceeds addition capacity; no silent truncation')
            size=np.array(item['size'])
            if 'center_y_m' in item:
                raise ValueError('Use an observed height reference for roots or a relative placement for children; no independent center_y_m')
            if i==0:
                if 'anchor_id' in item:
                    raise ValueError('First group object is its local root, not an anchored child')
                ref_id=item.get('height_anchor_id')
                if not isinstance(ref_id,str) or ref_id not in references:
                    raise ValueError('Root height_anchor_id must reference a fixed observed object')
                alignment=item.get('height_alignment')
                offset=number(item.get('height_offset_m'),'height_offset_m')
                center=np.array([0.,aligned_y(references[ref_id],size,alignment)+offset,0.])
                heights.append(dict(id=item['id'],reference_id=ref_id,alignment=alignment,offset_m=offset,
                                    center_y_m=float(center[1]),source='observed_camera_y_aabb_prior'))
            else:
                if (not isinstance(item.get('anchor_id'),str) or item['anchor_id'] not in placed or
                        not isinstance(item.get('direction'),str) or item['direction'] not in DIRECTIONS):
                    raise ValueError('Child must reference an earlier object in its group and a supported direction')
                if 'height_anchor_id' in item:
                    raise ValueError('Child height follows its existing group anchor, not a second height anchor')
                parent=placed[item['anchor_id']]
                axis,sign=DIRECTIONS[item['direction']]
                gap=number(item.get('gap_m'),'gap_m')
                if gap<0:raise ValueError('gap_m is edge-to-edge and must be nonnegative')
                center=np.array(parent['center']).copy()
                center[axis]+=sign*((size[axis]+parent['size'][axis])/2+gap)
                if axis!=1:
                    center[1]=aligned_y(parent,size,item.get('vertical_alignment'))
                elif 'vertical_alignment' in item:
                    value=item['vertical_alignment']
                    if not isinstance(value,str) or value not in Y_EDGES:
                        raise ValueError('vertical_alignment, when supplied, must be bottom/center/top')
                    # Conditional DSL parameter: direction/gap already determine
                    # y. An unused valid alignment is not a second constraint.
                    unused.append(dict(id=item['id'],field='vertical_alignment',value=value,
                        reason='Not applicable to above/below; declared direction and gap preserved'))
            box=dict(id=item['id'],category=item['category'],description=item['description'],
                     reason=item['reason'],unseen_reason='outside_fov',size=size.tolist(),center=center.tolist())
            ids.add(item['id']);placed[item['id']]=box;local.append(box)
        centers=np.array([o['center']for o in local]);sizes=np.array([o['size']for o in local])
        lo=(centers-sizes/2).min(0);hi=(centers+sizes/2).max(0)
        midpoint=(lo+hi)/2; half=(hi-lo)/2
        if group['region']=='back':
            # Existing read-only frustum checker conservatively extends the
            # camera plane 5cm backwards. Include that same guard explicitly.
            target=np.array([along,midpoint[1],-half[2]-.05-clearance])
        else:
            x=tangent*along+half[0]+tangent*half[2]+clearance*math.sqrt(1+tangent*tangent)
            target=np.array([-x if group['region']=='left' else x,midpoint[1],along])
        translation=target-midpoint
        for box in local:
            box['center']=(np.array(box['center'])+translation).tolist()
            if frustum_margin(np.array(box['center']),np.array(box['size']),tangent,tangent)>1e-8:
                raise ValueError('Compiled group did not satisfy its offscreen construction')
        additions.extend(local)
        diagnostics.append(dict(id=group['id'],region=group['region'],translation=translation.tolist(),
            compiled_group_center=target.tolist(),full_group_size=(hi-lo).tolist(),
            member_ids=[o['id']for o in local],clearance_m=clearance))
    result=dict(room_type=answer['room_type'],scene_hypothesis=answer['scene_hypothesis'],additions=additions)
    return result,dict(protocol=PROTOCOL,legacy_region_ids=[],groups=diagnostics,
        position_source='compiled_group_regions_and_edge_relations',floor_snapping=False,
        height_references=heights,height_source='observed_camera_y_aabb_prior_not_measured_floor',
        unused_parameters=unused,
        candidate_grid_used=False,optimizer_enabled=False,back_camera_plane_guard_m=.05,
        overlap_report=overlap_report(additions,observed_layout),
        note='Construction enforces selected offscreen half-space and declared pair separation only; '
             'contents, room extent, gravity, floor, support and semantics remain unverified hypotheses')


def complete(ask, observed, observed_layout, hfov, capacity):
    answer=ask(prompt(observed,observed_layout,hfov,capacity),'placement_program',4096)
    return compile_program(answer,observed,observed_layout,hfov,capacity)
