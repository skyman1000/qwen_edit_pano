"""Versioned semantic vocabulary projection; legacy dataset mapping is untouched.

The MP raw annotation table is not a synonym dictionary. For example its rare
raw 'television' maps to misc, while the NYU name of its common tv row identifies
the existing tv_monitor class. Keep both interpretations and their evidence.
No substring/fuzzy match, new embedding ID, or scene GT is used.
"""
import csv
from functools import lru_cache

from .category_bridge import MAPPING, category, key, official_labels
from .geometry import CANONICAL_CATEGORIES

POLICY = 'semantic_lexicon_v1'
GENERIC = frozenset(('furniture', 'objects', 'misc'))
FIELDS = ('raw_category', 'category', 'nyuClass')


@lru_cache(maxsize=1)
def semantic_aliases():
    _, digest = official_labels()  # Verify the existing vocabulary binding.
    aliases = {}
    with MAPPING.open(newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            cid = int(row['mpcat40index'])
            if cid not in CANONICAL_CATEGORIES or CANONICAL_CATEGORIES[cid] in GENERIC:
                continue
            for field in FIELDS:
                term = key(row[field])
                if term:
                    aliases.setdefault(term, []).append(dict(field=field,
                        raw_category=row['raw_category'], category_id=cid,
                        annotation_count=int(row['count']), wordnet_synset=row['wnsynsetkey']))
    return aliases, digest


def resolve_noun(noun):
    legacy, legacy_method = None, None
    try:
        legacy, legacy_method = category(noun)
    except ValueError:
        pass
    aliases, digest = semantic_aliases()
    evidence = aliases.get(key(noun), [])
    ids = sorted({r['category_id'] for r in evidence})
    row = dict(noun=noun, category=legacy, legacy_category=legacy,
               legacy_method=legacy_method, specific_candidates=ids,
               method=dict(method=POLICY, mapping_sha256=digest), evidence=evidence)
    canonical = {key(name) for name in CANONICAL_CATEGORIES.values()}
    raw_ids = official_labels()[0].get(key(noun), set())
    if len(raw_ids) == 1 and not raw_ids <= set(CANONICAL_CATEGORIES):
        row.update(category=None, status='known_outside_object_vocabulary')
    elif key(noun) in canonical or legacy is not None and legacy not in GENERIC:
        row['status'] = 'coarse_catchall' if legacy in GENERIC else 'exact_supported_label'
    elif len(ids) == 1:
        row.update(category=CANONICAL_CATEGORIES[ids[0]], status='unambiguous_specific_semantic_alias')
    elif len(ids) > 1:
        row.update(category=None, status='ambiguous_semantic_alias')
    else:
        row['status'] = 'coarse_catchall' if legacy is not None else 'unmapped'
    row['legacy_disagreement'] = row['category'] != legacy
    row['semantics_verified'] = False
    return row


def resolve_identity(object_type, name):
    primary = resolve_noun(object_type)
    named = resolve_noun(name)
    p, n = primary['category'], named['category']
    conflict = p is not None and n is not None and p not in GENERIC and n not in GENERIC and p != n
    chosen = primary
    if p is None and n is not None:
        chosen = named
    result = dict(category=None if conflict else chosen['category'],
                  method=dict(method=POLICY, primary=primary, exact_name_check=named),
                  mapping_status='identity_conflict' if conflict else chosen['status'],
                  mapping_input='name' if chosen is named else 'object_type',
                  legacy_category=primary['legacy_category'], semantics_verified=False)
    if primary['status'] == 'known_outside_object_vocabulary':
        result.update(category=None, mapping_status=primary['status'], mapping_input='object_type')
    result['identity_loss_review'] = result['category'] in GENERIC
    return result
