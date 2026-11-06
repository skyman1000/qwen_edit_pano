"""Specific object semantics are distinct from the downstream Adapter taxonomy.

No invented categories, coordinates or instances. A coarse model category can be
replaced only by its own explicit specific name, with the original field retained.
"""
import json
import re

from .fresh_observer import PROMPT, answer, state_from_answer

COARSE = {'furniture', 'appliance', 'appliances', 'everyday belonging', 'everyday belongings',
          'decor', 'decoration', 'object', 'objects', 'misc', 'thing', 'things', 'item', 'items',
          'unknown', 'household item', 'lighting'}


def normalized_text(value):
    return re.sub(r'\s+', ' ', value.replace('_', ' ').replace('-', ' ').strip())


def inventory_answer(value):
    clean = answer(value)
    status = value.get('review_status')
    if status not in ('complete', 'empty_scene'):
        raise ValueError('Final visible inventory unresolved; use complete or explicitly confirmed empty_scene')
    if bool(clean['objects']) != (status == 'complete'):
        raise ValueError('Objects contradict review_status')
    for obj in clean['objects']:
        reported = obj['category'];specific = normalized_text(reported)
        if specific.casefold() in COARSE:
            specific = normalized_text(obj['name'])
            if specific.casefold() in COARSE:
                raise ValueError('A specific visible noun or name is required; a coarse group is insufficient')
            source = 'model_explicit_name_used_instead_of_coarse_category'
        else:source = 'model_specific_category'
        obj.update(category=specific, reported_category=reported, semantic_label_source=source)
    return dict(clean, review_status=status)


def export_inventory(clean, config):
    state, mapping = state_from_answer(clean, config, 'qwen')
    state['split'] = config.get('split', 'prediction_input')
    for obj, row in zip(state['objects'], clean['objects']):
        obj.update(reported_category=row['reported_category'], semantic_label_source=row['semantic_label_source'])
    return state, mapping


def review_prompt(draft_context):
    return (PROMPT+'\nPerform an INDEPENDENT WHOLE-IMAGE visible inventory. A draft is optional, '
        'may have failed, and is not a prerequisite or a continuation instruction. '
        'Use the original image even when the draft is empty. Include identifiable objects through glass; '
        'avoid duplicate reflections, subparts, and multiple boxes for the same physical instance. '
        'category must be a SPECIFIC noun, e.g. the object type, never furniture/appliance/decor/everyday belonging. '
        'name retains the specific visible appearance. Do not force an Adapter category. '
        'on_floor is true only for direct floor support, not merely because the object is low in the image. '
        'Add review_status: complete / empty_scene / unresolved. Empty objects require empty_scene, '
        'confirming no identifiable objects; unresolved means recognition failed. '
        'Your evidence_description and object list must agree. No hidden objects or target count. '
        '\nDRAFT_CONTEXT: '+json.dumps(draft_context, ensure_ascii=False))


def quality_audit(clean):
    warnings = []
    objects = clean['objects']
    for i, a in enumerate(objects):
        for j, b in enumerate(objects[:i]):
            if a['category'].casefold() != b['category'].casefold():continue
            aa, bb = a['bbox_1000'], b['bbox_1000']
            intersection = max(0, min(aa[2],bb[2])-max(aa[0],bb[0]))*max(0,min(aa[3],bb[3])-max(aa[1],bb[1]))
            areas = [(r[2]-r[0])*(r[3]-r[1]) for r in (aa,bb)]
            if intersection/min(areas) >= .95:
                warnings.append(dict(issue='same_category_nested_boxes_need_review', indices=[j,i],
                    containment=intersection/min(areas), action='preserved_not_automatically_merged'))
    return dict(instances=len(objects), semantic_normalizations=sum(
        o['semantic_label_source'] != 'model_specific_category' for o in objects),
        warnings=warnings, quality_certified=False, new_instances_fabricated=False)
