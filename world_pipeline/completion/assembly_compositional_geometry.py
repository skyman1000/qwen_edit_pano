"""Fixed positive geometry for matched text-only or RGB-assisted layout."""
import json
from . import assembly_holistic_geometry as base

PARAMETERIZATION='complete_programs_protected_regions_positive_geometry_v12'
layout_answer=base.layout_answer
annotate_state=base.annotate_state


def layout_prompt(case,observed,world,plane):
    text=base.layout_prompt(case,observed,world,plane)
    text=text.replace('The attached observation board shows the original photographed view, not a layout canvas for the added objects.',
        'The fixed positive observation packet describes the original photographed camera view.')
    packet=[dict(id=o['track_id'],name=o['description'],predicted_center_m=o['center_local_m']['value'],
        predicted_full_camera_AABB_m=o['size_aabb_local_m']['value']) for o in observed['objects']]
    return text+'\nV12 FIXED POSITIVE GEOMETRY: '+json.dumps(packet,ensure_ascii=False)+(
        '\nThese are frozen predicted observations, not additional objects or measured walls. '
        'Use their positions for relationships, never paste absolute XYZ into relative offset_m. '
        'An image may additionally be supplied in the matched RGB option; all required observed coordinates are already in text. '
        'Estimate thickness and full dimensions for the actual concrete variant. Numeric proxy success is not a semantic certificate.')


def compile_layout(case,observed,world,layout,plane):
    solved=base.compile_layout(case,observed,world,layout,plane)
    solved['review'].update(geometry_parameterization=PARAMETERIZATION,observed_absolute_positions_sent_to_metric_LM=True)
    return solved
