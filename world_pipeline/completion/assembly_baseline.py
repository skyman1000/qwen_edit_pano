"""Prompt baseline with identical hypothesis/catalog capacities and layout backend.

This compares complete pipelines, not identified causal effects. The finite F
selector is bookkeeping for alternatives, without an explicit need mechanism.
"""
from . import assembly_delta as delta

PROTOCOL = delta.PROTOCOL
INTERFACE = delta.INTERFACE
MODEL_VARIANT = 'ordinary_joint_completion_baseline_v1'
hypotheses_answer = delta.hypotheses_answer
furniture_answer = delta.furniture_answer
active_world = delta.active_world
sample_worlds = delta.sample_worlds
semantic_diagnostics = delta.semantic_diagnostics
intervention_diagnostics = delta.intervention_diagnostics


def hypotheses_prompt(case, grounding):
    import json
    return ('Describe 1..2 plausible complete surrounding indoor scenes compatible with this local view. '
        'Return {"hypotheses":[{"description":"scene description","functions":[],"assumptions":["uncertainty"]}]}. '
        'No object list or coordinates.\nOBSERVED FACTS: ' + json.dumps(grounding, ensure_ascii=False) +
        '\nFIXED INSTANCE INVENTORY: ' + json.dumps(case['context'], ensure_ascii=False))


def furniture_prompt(case, hypothesis):
    return ('Predict the plausible missing objects JOINTLY from the partial photograph and observed inventory. '
        'Use a direct scene completion prediction. No explicit requirement/realization mechanism is requested. '
        'Grouping is an output organization field.\n' + delta.furniture_prompt(case, hypothesis))


def build_model(*args, **kwargs):
    model = delta.build_model(*args, **kwargs)
    model['model_variant'] = MODEL_VARIANT
    model['comparison_scope'] = 'direct_prompt_pipeline_baseline_not_isolated_causal_effect'
    return model
