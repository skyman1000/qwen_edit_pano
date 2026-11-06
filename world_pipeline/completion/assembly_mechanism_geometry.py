"""v7 strict geometry with program-owned direction domains, no LM region field."""
from copy import deepcopy
import json

from . import assembly_realization_geometry as base

PARAMETERIZATION='program_owned_directions_reference_mechanism_v8'


def layout_prompt(case,observed,world,plane):
    # Drop v7's trailing region-hint explanation as well as the template field;
    # retaining it would contradict program-owned directions in actual calls.
    prompt=base.layout_prompt(case,observed,world,plane).split('\nV7: ',1)[0]
    start=prompt.index('TEMPLATE: ')+len('TEMPLATE: ');end=prompt.index('\nFIXED SEMANTIC WORLD:',start)
    template=json.loads(prompt[start:end])
    for group in template['groups']:group.pop('region',None)
    prefix=prompt[:start]
    text=('For each group choose region from left,right,rear relative to the original camera; '
        'use unspecified when the evidence/placement description does not determine an off-view direction. '
        'The compiler will consider multiple hidden directions for unspecified; it never treats a central visible area as hidden. ')
    if text not in prefix:raise ValueError('Metric prompt changed; review program-owned direction policy')
    prefix=prefix.replace(text,'Directions are PROGRAM-OWNED. Do not output region or azimuth_deg. '
        'The solver considers left/right/rear hidden domains from fixed observations and typed relations. '
        'A furniture-relative front does not specify the original camera sector. ')
    return prefix+json.dumps(template)+prompt[end:]+(
        '\nV8: Supply numeric attributes only; never add region/azimuth fields. All final full boxes must remain outside '
        'the ORIGINAL input frustum. farther_than/nearer_than refer to original camera Z, not wall distance. '
        'No observed boxes, identities, dimensions, or attachment ownership may be edited.')


def layout_answer(value,world,hfov):
    if isinstance(value,dict):
        for group in value.get('groups',[]) if isinstance(value.get('groups'),list) else []:
            if isinstance(group,dict) and ('region' in group or 'azimuth_deg' in group):
                raise ValueError('Direction is program-owned: omit region and azimuth_deg fields entirely; preserve every object and numeric attribute. No front-to-rear conversion.')
    # Missing region explicitly denotes all hidden domains in the existing
    # compiler. An explicit conflicting response is never silently rewritten.
    return base.layout_answer(value,world,hfov)


def compile_layout(case,observed,world,layout,plane):
    result=base.compile_layout(case,observed,world,layout,plane)
    result['review'].update(geometry_parameterization=PARAMETERIZATION,direction_policy='program_owned_all_hidden_domains',
        explicit_front_silently_reinterpreted=False)
    return result
