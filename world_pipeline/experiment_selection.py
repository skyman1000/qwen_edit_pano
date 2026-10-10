"""Explicit, opt-in acceptance of small-object pixel-threshold flags only."""
import re
from .common import sha

VERSION='local_pixel_threshold_spotcheck_v1'


def eligible_flags(flags):
    return all(isinstance(f,str) and re.fullmatch(r'local_object_below_erp_threshold:\d+',f) for f in flags)


def validate_selection(policy,gt,records):
    if (policy['version']!=VERSION or policy['gt_root']!=str(gt.resolve()) or
            policy['manifest_sha256']!=sha(gt/'gt_manifest.jsonl') or
            policy.get('authorization')!='explicit_user_instruction_2026-10-10'):
        raise ValueError('Experiment selection identity/authorization mismatch')
    ids=policy['selected_ids']
    expected={r['id'] for r in records if eligible_flags(r['review_flags'])}
    if len(ids)!=len(set(ids)) or set(ids)!=expected:
        raise ValueError('Selection must contain exactly clean and threshold-only samples')
    # In particular, mixed flags and multiview disagreement remain excluded.
