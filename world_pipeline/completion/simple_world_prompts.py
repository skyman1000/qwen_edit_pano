"""Short grounding query and explicit hypothetical layout planning; no answer coordinates."""
import json

from .qwen_upstream_core import VOCAB, compact_observations


def native_3d_prompt(instance, camera, marked=False):
    target = (f'Locate only the {instance["category"]} indicated by the red rectangle. '
              'The rectangle is an annotation, not part of the scene. Use the original whole-image camera. '
              if marked else f'Locate the {instance["category"]} in the image region '
              f'{json.dumps(instance["bbox_2d"])} (xyxy normalized to 0..1000). ')
    return (target+
            'Return its 3D bounding box as a JSON list containing one object with keys bbox_3d and label. '
            'bbox_3d contains center xyz, full box sizes xyz, then rotations about x/y/z divided by pi. '
            'Use metres in the original image camera: x right, y down, z forward. '+camera)


def semantic_plan_prompt(observed, description, scene_text, capacity, camera=''):
    # No GT room, source IDs, hidden labels or sample identifiers enter this prompt.
    context = compact_observations(observed)
    return (
        'Infer one plausible existing off-camera continuation for a panorama from the SAME camera. '
        'This is a hypothesis about unseen content, not a renovation or decoration request. '
        'For this off-camera experiment, propose objects outside the input camera field of view. '
        'Preserve the entire visible image, including bare walls and empty space, and keep observed objects fixed. '
        'Name a few plausible missing objects, their directions relative to the camera or observed '
        'objects, and why each lies outside the photographed view. Use no metric coordinates or JSON. '
        'Uncertainty about the real room is expected: label the arrangement as a hypothesis. '
        f'At most {capacity} additions; you need not use the full budget. '
        'If the image is mostly a blank wall, a sparse continuation is appropriate.\n'
        'Camera-local axes: x right, y up, z forward; negative z is behind the CAMERA, not behind an object. '
        'Choose only these category names: '+VOCAB+'. Do not propose floors, rugs or ceilings. '
        'Camera: '+camera+'\nObserved state: '+json.dumps(context)+'\n'
        'Visible description: '+description+'\nOptional scene requirements: '+scene_text)


def plan_geometry_prompt(observed, plan, capacity, camera=''):
    return (
        'Convert the proposed hypothetical arrangement below into camera-local object boxes. '
        'Keep observed objects and visible empty space fixed; output only proposed OFF-CAMERA objects. '
        'Place them outside the supplied camera field of view, not on currently visible surfaces. '
        'Use metres: origin at the camera, x=image-right, y=image-up, z=forward; '
        'negative z is behind the camera. Positive y is UP, not down; a higher object has larger y. '
        'Camera axes need not be gravity aligned. Choose plausible approximate positions and full camera-axis '
        'AABB sizes consistent with the observed state. An unknown observed value remains unknown. '
        'Return a JSON object with the key hidden_objects containing a list. Each list entry has '
        'category (one allowed name), center_local_m (three numbers), size_aabb_local_m '
        '(three positive full edge lengths). Do not output observed objects or rotation. '
        f'At most {capacity} new objects. Allowed names: '+VOCAB+'.\n'
        'CAMERA='+camera+'\nOBSERVED='+json.dumps(compact_observations(observed))+'\n'
        'PROPOSED HYPOTHETICAL ARRANGEMENT:\n'+plan)
