"""One bounded, image-grounded revision of offscreen candidates. No box pushing."""
from copy import deepcopy
import json
import math

import numpy as np

from .box_layout import directional_layout, frustum_margin, projected_box


def conflicts(answer, hfov, image_size):
    tan_h = math.tan(math.radians(hfov)/2)
    tan_v = tan_h*image_size[1]/image_size[0]
    issues = []
    for obj in answer['additions']:
        center, size = np.array(obj['center']), np.array(obj['size'])
        margin = float(frustum_margin(center, size, tan_h, tan_v))
        if margin > 0:
            box = projected_box(center, size, tan_h, tan_v)
            issues.append(dict(id=obj['id'], issue='whole_box_not_certified_outside_photo',
                               projected_bbox_xyxy_norm=None if box is None else box.tolist(),
                               note='Center outside or on image edge is insufficient; object has width. '
                                    'This test does not establish occlusion by walls/furniture.'))
    return issues


def revise_once(answer, base_prompt, ask, validate, hfov, image_size, camera_height):
    """Freeze valid candidates. Revise or explicitly withdraw only flagged IDs.

    Invalid revision leaves the original candidate available with unresolved
    diagnostics; it does not crash or silently delete objects. No second retry.
    """
    before = conflicts(answer, hfov, image_size)
    report = dict(protocol='one_offscreen_feedback_v1', used=bool(before),
                  before=before, after=before, withdrawn=[], revision_error=None)
    if not before:
        return answer, report
    failed = {o['id'] for o in before}
    prompt = (base_prompt + '\nREVISION OF YOUR PREVIOUS CANDIDATES: '
              + json.dumps(answer, ensure_ascii=False)
              + '\nCOMPUTED GEOMETRY FEEDBACK: ' + json.dumps(before)
              + '\nRevise ONLY the flagged IDs using the original photo and fixed observed state. '
              'Keep a coherent room, and make descriptions agree with the proposed positions. '
              'Do not invent a fixture in a visible area merely because this room could use one. '
              'Do not simply move an above-bed or near-window object behind the camera and keep its old description. '
              'You may withdraw an unsupported candidate instead of forcing it elsewhere. '
              'Do not output observed objects or unflagged additions. No new IDs or categories. '
              'Return JSON {"additions":[revised objects in the same parameter format], '
              '"withdrawn":[{"id":"flagged ID","reason":"why this hypothesis is withdrawn"}]}. '
              'Account for EVERY flagged ID exactly once, either revised or withdrawn. '
              'Only one revision is available; do not output relations or prose.')
    try:
        reply = ask(prompt, 'completion_revision', 4096)
        if not isinstance(reply, dict) or not isinstance(reply.get('additions'), list) or not isinstance(reply.get('withdrawn'), list):
            raise ValueError('Revision needs additions and withdrawn lists')
        withdrawn = reply['withdrawn']
        if any(not isinstance(o, dict) or not isinstance(o.get('reason'), str) or not o['reason'].strip() for o in withdrawn):
            raise ValueError('Withdrawal requires an ID and reason')
        revised, _ = directional_layout(dict(additions=reply['additions']), camera_height)
        ids = [o.get('id') for o in revised['additions'] + withdrawn]
        if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids) or set(ids) != failed:
            raise ValueError('Revision must account for exactly the flagged IDs once')
        original = {o['id']: o for o in answer['additions']}
        if any(o.get('category') != original[o['id']]['category'] for o in revised['additions']):
            raise ValueError('Revision cannot change categories')
        replacements = {o['id']: o for o in revised['additions']}
        candidate = deepcopy(answer)
        candidate['additions'] = [deepcopy(replacements[o['id']] if o['id'] in replacements else o)
                                  for o in answer['additions'] if o['id'] not in failed or o['id'] in replacements]
        validate(candidate)
        report.update(withdrawn=withdrawn, after=conflicts(candidate, hfov, image_size))
        return candidate, report
    except (ValueError, KeyError, TypeError) as exc:
        report['revision_error'] = str(exc)
        return answer, report
