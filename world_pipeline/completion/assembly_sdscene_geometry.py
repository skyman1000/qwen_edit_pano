"""Same geometry compiler, no numeric values masquerading as model predictions."""
import json
from . import assembly_parametric as base

PARAMETERIZATION='parent_relative_null_metric_template_v4'
layout_answer=base.layout_answer
compile_layout=base.compile_layout


def layout_prompt(case,observed,world,plane):
    prompt=base.layout_prompt(case,observed,world,plane)
    prefix,rest=prompt.split('\nTEMPLATE: ',1)
    raw,suffix=rest.split('\nFIXED SEMANTIC WORLD: ',1);schema=json.loads(raw)
    for group in schema['groups']:
        group['range_m']=None
        for obj in group['objects']:
            obj.update(yaw_deg=None,offset_m=[None,None,None],surface_fraction=None,hanging_drop_m=None)
    prefix=prefix.replace('The displayed .5/0 are unverified initialization priors.',
        'Replace each required null with a concrete numeric hypothesis; no numeric template value is a prediction.')
    return prefix+'\nTEMPLATE: '+json.dumps(schema)+'\nFIXED SEMANTIC WORLD: '+suffix
