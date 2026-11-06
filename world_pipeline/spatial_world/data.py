"""Deterministic train-only mechanism panel; no outcome-based selection."""
import random
from pathlib import Path
from ..build_polished_world import ROOT,read

DEFAULT_INITIAL=ROOT/'qwen_edit_pano/outputs/object_world_id_A_dual_full_v1/run-1792820440973994739/checkpoint-step010000'


def select_train(records,limit,seed,smoke_ids=None):
    train=[r for r in records if r['split']=='train']
    if smoke_ids is not None:return [r for r in train if r['id'] in smoke_ids]
    if limit<0 or limit>len(train):raise ValueError('Invalid training limit')
    if not limit:return train
    panel=read(ROOT/'research/audits/id_condition_diagnostic_2026-10-25/PANEL.json')
    anchors=[c['id'] for c in panel['cases'] if c['split']=='train']
    byid={r['id']:r for r in train}
    if not set(anchors)<=set(byid):raise ValueError('Diagnostic anchor is not training data')
    selected=[byid[sid] for sid in anchors[:limit]]
    other=[r for r in train if r['id'] not in anchors]
    random.Random(seed).shuffle(other)
    return selected+other[:limit-len(selected)]
