"""Exact raw-label -> existing Adapter vocabulary bridge, using official MP TSV.

No guessed catch-all, label dropping, new category IDs or GT scene annotations.
"""
import csv
from functools import lru_cache
import re

from ..common import ROOT, sha, read
from .geometry import CANONICAL_CATEGORIES

MAPPING = ROOT/'benchmark_assets/Matterport3D_metadata/official/category_mapping.tsv'


def key(value):
    return re.sub(r'[\s_-]+', ' ', value.strip().casefold())


@lru_cache(maxsize=1)
def official_labels():
    digest=sha(MAPPING)
    vocabulary=read(ROOT/'qwen_edit_pano/data/gt_world_supplement_v1/vocabulary.json')
    if vocabulary['category_mapping_sha256']!=digest:
        raise ValueError('Official mapping differs from the existing Adapter vocabulary binding')
    mapping = {}
    with MAPPING.open(newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            label = key(row['raw_category'])
            mapping.setdefault(label, set()).add(int(row['mpcat40index']))
    return mapping, digest


def category(raw):
    if not isinstance(raw,str) or not raw.strip():
        raise ValueError('Visible object needs a category string')
    if raw in CANONICAL_CATEGORIES.values():
        return raw, None
    canonical = {key(name):name for name in CANONICAL_CATEGORIES.values()}
    if key(raw) in canonical:
        return canonical[key(raw)], dict(method='canonical_label_format_normalization')
    labels, digest = official_labels()
    ids = labels.get(key(raw), set())
    if len(ids)!=1 or next(iter(ids),None) not in CANONICAL_CATEGORIES:
        raise ValueError(f'No unambiguous supported official mpcat40 mapping for category {raw!r}')
    cid = next(iter(ids))
    return CANONICAL_CATEGORIES[cid], dict(method='official_matterport_raw_category_to_mpcat40',
        source=str(MAPPING),source_sha256=digest,category_id=cid,
        note='Adapter receives the coarse category; the original raw label is metadata only')
