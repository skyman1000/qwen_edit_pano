"""Pinned project object-label table; preserve raw nouns and review every mapping."""
from copy import deepcopy
from functools import lru_cache

from ..common import ROOT, read, sha
from .category_bridge import key, MAPPING
from .semantic_categories import resolve_noun

VOCABULARY = ROOT/'qwen_edit_pano/data/gt_world_supplement_v1/vocabulary.json'


@lru_cache(maxsize=1)
def vocabulary():
    value = read(VOCABULARY)
    if value['category_mapping_sha256'] != sha(MAPPING):
        raise ValueError('Dataset taxonomy no longer matches the pinned official MP mapping')
    return value


def review(noun):
    vocab = vocabulary(); categories = vocab['categories']; canonical = {key(v):int(k) for k,v in categories.items()}
    raw_member = isinstance(noun,str) and key(noun) in canonical
    resolved = resolve_noun(noun) if isinstance(noun,str) and noun.strip() else dict(category=None,status='missing_raw_label')
    cid = next((int(k) for k,v in categories.items() if v==resolved['category']),None)
    return dict(raw_label=noun, raw_label_in_dataset_table=raw_member,
        dataset_category_id=cid, dataset_category_label=categories.get(str(cid)),
        mapping_status=resolved['status'], mapping_supported=cid is not None,
        coarse_mapping=resolved['category'] in ('furniture','objects','misc'),
        vocabulary_source=str(VOCABULARY), vocabulary_sha256=sha(VOCABULARY),
        official_mapping_sha256=vocab['category_mapping_sha256'], mapping_evidence=resolved)


def annotate(state, semantic=None):
    """New copied/added records only. Immutable observed objects are untouched."""
    result = deepcopy(state); identities = {o['id']:o for o in (semantic or {}).get('objects',[])}
    for obj in result['objects']:
        if obj['local_evidence']=='visible': continue
        noun = identities.get(obj['track_id'],{}).get('category',obj.get('raw_category'))
        audit = review(noun)
        actual = obj['category_id']['value'] if obj['category_id']['valid'] else None
        supported = str(actual) in vocabulary()['categories']
        audit.update(noun_mapping_category_id=audit['dataset_category_id'],
            compiled_mapping_disagrees=actual!=audit['dataset_category_id'],
            dataset_category_id=actual if supported else None,
            dataset_category_label=vocabulary()['categories'].get(str(actual)),mapping_supported=supported)
        obj['taxonomy_review'] = audit
        obj['dataset_category_label'] = audit['dataset_category_label']
        # Never manufacture a new Adapter ID or reinterpret an unknown as misc.
        obj['visual_color'] = '#c3314d' if not supported else '#e88c13' if audit['raw_label_in_dataset_table'] else '#9c43bb'
        obj['visual_role'] = ('generated: unresolved category' if not supported else 'generated: dataset label' if audit['raw_label_in_dataset_table'] else
            'generated: raw name outside label table (mapped)' if audit['mapping_supported'] else 'generated: unresolved category')
    return result
