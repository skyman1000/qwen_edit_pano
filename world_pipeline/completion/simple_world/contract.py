"""A global hypothesis followed by a jointly generated surrounding inventory."""
from copy import deepcopy
import json

from ..candidate_repair import contract as typed

hypotheses_answer = typed.hypotheses_answer


def hypotheses_prompt(case, grounding):
    return typed.hypotheses_prompt(case, grounding) + (
        '\nDescribe the organization of the ENTIRE surrounding scene, including how its uses coexist. '
        'Do not give separate single-role activation profiles. This short hypothesis will directly '
        'condition one joint inventory mechanism; it is not an additional planning or scoring stage.')


def furniture_prompt(case, h):
    schema = dict(alternatives=[dict(description='<concrete complete surrounding-world hypothesis>',
        groups=[dict(function='<canonical function>', reason='<why this arrangement may exist>',
            layout_region='unspecified', observed_refs=[], new_objects=[dict(key='<unique key>',
                category='<concrete noun>', variant='<concrete type>', reason='<distinct hidden instance>',
                placement='<location and use>', attachment=dict(kind='floor', host=None), affordances=[])],
            functional_links=[])], relations=[])])
    return ('Generate one or two complete JOINT hidden-world deltas conditioned on the global hypothesis '
        'and partial RGB. Think of the whole inhabited surrounding scene: how activity arrangements, '
        'independent furniture and their ordinary contents coexist around the fixed visible anchors. '
        'Do not answer with only one example corner or only the minimum equipment for one function. '
        'All groups in an alternative coexist, including multiple groups of the same function. '
        'The executor keeps this WHOLE alternative: it will NOT split it by function, add independent '
        'role-presence gates or mix it with groups from another alternative. '
        'Only distinct UNPHOTOGRAPHED instances belong in new_objects. Photographed but undetected '
        'objects remain visible evidence; do not recreate them elsewhere. The supplied observed '
        'inventory is immutable. Same-category hidden instances need a distinct placement/use hypothesis. '
        'No minimum counts, required classes, functions or camera-sector quotas. Empty deltas are legal. '
        'Use at most four groups and 24 new instances per alternative as interface capacity, not a target. '
        'Explain genuine collections and multiplicity briefly instead of repeating filler records. '
        'Walls/floor/ceiling are unknown building boundaries, not new object records. '
        'Return concise JSON using this structure, replacing all placeholders: ' + json.dumps(schema) +
        '\nSUPPORT: encode installation ONLY in attachment={kind,host}. '
        'kind is floor/top/seat/shelf/hang/wall/ceiling/free. top/seat/shelf/hang need the exact new host '
        'key IN THE SAME GROUP or an observed ID; other kinds require host=null. floor means standing '
        'on the floor, wall means physically mounted, never merely next to a wall. Affordances list '
        'surfaces offered to children: top/seat/shelf/hanging. shelf denotes an interior shelf support '
        'proxy; top denotes an upper surface. Neither proves mesh containment. '
        'Do NOT repeat on, supported_by, inside, in or contained_in in relations or functional_links. '
        'A storage hypothesis is expressed by the child attachment and concise placement/reason. '
        '\nSPATIAL: optional relations={source,target,kind,reason}, kind near/left_of/right_of/'
        'farther_than/nearer_than/above/below. Original camera X points right, Z points into the photo; '
        'farther_than means greater Z, nearer_than smaller Z. above/below additionally need '
        'scope="whole_aabb" and constrain whole boxes, never lighting intention. New-new endpoints '
        'stay in the same group. Optional group.layout_region is left/right/rear/unspecified in the '
        'ORIGINAL camera; rear means Z<0. Select regions jointly for this world if there is a '
        'concrete spatial hypothesis; omit/unspecified if uncertain. No sector must be occupied. '
        '\nFUNCTION: optional group.functional_links={source,target,kind,purpose}, '
        'kind serves/illuminates/faces/accessible_from only. Source is new in this group; target is '
        'new in the same group or observed. These are unverified intentions, not hard geometry. '
        'No dimensions, coordinates, probabilities or long explanations. '
        '\nFUNCTION_TAGS: ' + json.dumps(typed.FUNCTIONS) +
        '\nGLOBAL_HYPOTHESIS: ' + json.dumps(h, ensure_ascii=False) +
        '\nOBSERVED_SUPPORT_DOMAINS: ' + json.dumps(case.get('observed_support_domains', {}), ensure_ascii=False) +
        '\nFROZEN_VISIBLE_INSTANCES: ' + json.dumps(case['context'], ensure_ascii=False))


def furniture_answer(value, case, h):
    # Reject contradictory declarations; never guess a new host or downgrade
    # containment to adjacency. Avoid asking for a redundant containment field.
    for alt in value.get('alternatives', []):
        for r in alt.get('relations', []):
            if r.get('kind') in ('in', 'inside', 'contained_in', 'on', 'on_top_of', 'supported_by'):
                raise ValueError('Support/storage is encoded ONLY by attachment on the child. '
                    'Remove the redundant relation, and explicitly choose top or shelf with the intended exact host.')
        for g in alt.get('groups', []):
            for link in g.get('functional_links', []):
                if link.get('kind') not in ('serves', 'illuminates', 'faces', 'accessible_from'):
                    raise ValueError('Functional links allow serves/illuminates/faces/accessible_from only. '
                        'Storage uses the child attachment; do not duplicate it as contained_in.')
    programs = typed.furniture_answer(deepcopy(value), case, h)
    # The legacy role parser intentionally reduces H to a set of tags. Here H
    # is the actual global hypothesis that conditioned the joint LM mechanism.
    # Restore its complete value for layout conditioning and audit, not a new H.
    for program in programs: program['hypothesis'] = deepcopy(h)
    return programs
