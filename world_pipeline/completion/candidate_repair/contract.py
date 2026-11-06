"""One wire contract for geometry, attachment and non-geometric intent.

The original instance/host validator and the original SCM/geometry remain the
executor. No inference from English reasons silently changes a relation.
"""
from collections import Counter
from copy import deepcopy
import json

from .. import assembly_compositional as original
from ..assembly_modular import FUNCTIONS

SPATIAL = ('near', 'left_of', 'right_of', 'above', 'below', 'farther_than', 'nearer_than')
FUNCTIONAL = ('serves', 'illuminates', 'faces', 'accessible_from', 'contained_in')
SURFACES = {'wall', 'walls', 'floor', 'flooring', 'ceiling', 'room', 'baseboard', 'trim', 'ceiling trim', 'wall trim'}


def short(value, name, limit=320):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(name + ' requires concrete concise text, 1..' + str(limit) + ' characters')
    if value.strip().casefold() in {'...', '<fill>', 'specific whole-world configuration',
        'concrete unseen organization', 'uncertain assumption', 'qualitative enclosure assumption',
        'possible hidden arrangement and contents, not a visible caption'}:
        raise ValueError(name + ': fill the field with scene-specific content, not the schema placeholder')
    return value.strip()


def hypotheses_prompt(case, grounding, previous=None):
    focus = ('Independently infer possible uses of unphotographed surrounding space.' if previous is None else
             'Look for a plausible alternative or complementary surrounding organization not already represented below. '
             'There is no obligation to add a use; the first-round catalog is finite and is not negative evidence.')
    return ('Infer one or two possible functional organizations of the existing UNSEEN surroundings of this partial RGB. '
        'These are uncertain hypotheses, not photographed facts or requirements. ' + focus +
        ' Return one JSON object with hypotheses: an array of objects. Each object has description (one short sentence), '
        'functions (an array using only the canonical tags below, possibly empty), and assumptions '
        '(zero to four short sentences). No inventory, quantities, metric room bounds or probabilities. '
        'Use concrete text, never copy a field description. Missing visible detections are not hidden new objects. '
        '\nCANONICAL_FUNCTIONS: ' + json.dumps(FUNCTIONS) +
        '\nOBSERVED_REPORTS: ' + json.dumps(grounding, ensure_ascii=False) +
        '\nFIXED_OBSERVED_INSTANCES: ' + json.dumps(case['context'], ensure_ascii=False) +
        ('\nFIRST_ROUND_COVERAGE: ' + json.dumps(previous, ensure_ascii=False) if previous is not None else ''))


def hypotheses_answer(value):
    rows = original.hypotheses_answer(value)
    for row in rows:
        short(row['description'], 'hypothesis.description')
        if len(row['assumptions']) > 4:
            raise ValueError('hypothesis.assumptions: at most four concise assumptions, not an essay')
        for s in row['assumptions']: short(s, 'hypothesis.assumption')
    return rows


def furniture_prompt(case, h, channel, previous=None):
    schema = dict(alternatives=[dict(description='<fill>', groups=[dict(function='<canonical tag>', reason='<fill>',
        observed_refs=[], new_objects=[dict(key='<new unique identifier>', category='<concrete noun>', variant='<fill>',
        reason='<separate hidden instance hypothesis>', placement='<qualitative location>',
        attachment=dict(kind='floor', host=None), affordances=[])], functional_links=[])], relations=[])])
    focus = ('Surrounding activity arrangements with ordinary supported contents.' if channel == 'surrounding' else
             'Ordinary possessions, storage/display and secondary arrangements beyond the visible activity.')
    return ('Propose one or two COMPLETE possible HIDDEN-WORLD DELTAS around the fixed observed instances in this '
        'numbered observation board. ' + focus +
        ' The photograph shows only one view of an existing inhabited world. Supply concrete functional arrangements, '
        'including independent hosts and their contents jointly. All groups in an alternative coexist; several groups '
        'of the SAME function can coexist. Each group is self-contained: all new host descendants and new-new relations '
        'stay in that group. Exact observed anchors may be referenced without regenerating them. '
        'Only DISTINCT unphotographed instances go into new_objects. Photographed but undetected furniture is still '
        'visible evidence, not permission to move it offscreen. Same-category unseen instances are possible when distinct. '
        'Walls, floor and ceiling are unknown enclosure assumptions, not ordinary object instances. '
        'No mandatory category, function, object count or camera-sector coverage. An empty delta uses groups=[] and relations=[]. '
        'Interface capacity is at most four groups and 24 new instances per alternative; it is NOT a target. '
        'For repeated items, explain the collection/use and multiplicity in group.reason; do not enumerate identical '
        'glasses/books merely to fill capacity. Each reason/placement should be one short phrase, no long repeated prose. '
        'Return exactly the following structure, replacing every <...> with concrete content: ' + json.dumps(schema) +
        '\nONE TYPED RELATION CONTRACT:\n'
        '1. attachment.kind is floor/top/seat/shelf/hang/wall/ceiling/free. top/seat/shelf/hang need the exact host '
        'key in THIS group or a frozen obs ID. Other kinds require host=null. Hypothesis IDs are not hosts. '
        'A floor-standing item next to a wall uses floor; wall means physically mounted to the building. '
        'affordances is a subset of top,seat,shelf,hanging and describes surfaces offered to children. '
        'Follow OBSERVED_SUPPORT_DOMAINS: an empty candidate list disallows a wholly outside-view attachment there. '
        '2. relations contains ONLY {source,target,kind,reason} with kind from ' + json.dumps(SPATIAL) + '. '
        'All directions use the ORIGINAL CAMERA: left/right are X; farther_than is greater Z, nearer_than smaller Z. '
        'These specify geometric ordering, not furniture-relative language. above/below constrain WHOLE AABB intervals, '
        'so they additionally require scope="whole_aabb". A floor lamp illuminating a table is NOT whole-box above it. '
        'Do not duplicate support in relations. Do not use in/inside as a spatial ordering. '
        '3. group.functional_links contains {source,target,kind,purpose} with kind from ' + json.dumps(FUNCTIONAL) + '. '
        'These are non-geometric semantic hypotheses, never automatic hard spatial constraints. Use illuminates for '
        'lighting, faces for orientation intent, serves for use. A contained_in link must agree with attachment.kind=shelf '
        'and the exact same host; it records an interior-storage hypothesis, not verified mesh containment. '
        'Do not reinterpret top attachment as interior storage. Relations and links use exact new keys or observed IDs. '
        'All endpoints must exist; links between new objects stay in the same group. '
        'No metric coordinates, dimensions, floor estimate, confidence or probabilities. '
        'Optional group.layout_region is left/right/rear/unspecified in original camera axes; omit if uncertain. '
        'rear means original camera Z<0, not deeper in the photo or behind furniture. '
        '\nFUNCTION_TAGS: ' + json.dumps(FUNCTIONS) + '\nHYPOTHESIS: ' + json.dumps(h, ensure_ascii=False) +
        '\nOBSERVED_SUPPORT_DOMAINS: ' + json.dumps(case.get('observed_support_domains', {}), ensure_ascii=False) +
        '\nFROZEN_VISIBLE_INSTANCES: ' + json.dumps(case['context'], ensure_ascii=False) +
        ('\nSECOND FIXED-BUDGET ROUND: The following is the FIRST_ROUND_COVERAGE, not a complete world. '
         'Consider missing independent hosts, different plausible functional uses, and their ordinary contents. '
         'Do not repeat the same catalog merely with renamed keys. Return COMPLETE programs, including hosts, '
         'not an incremental patch: the original executor selects one whole program per role. '
         'It is legal to find no distinct additional proposal.\n' + json.dumps(previous, ensure_ascii=False)
         if previous is not None else ''))


def furniture_answer(value, case, h):
    raw = deepcopy(value)
    alternatives = raw.get('alternatives')
    if not isinstance(alternatives, list): raise ValueError('alternatives must be an array')
    for ai, alt in enumerate(alternatives):
        if not isinstance(alt, dict): raise ValueError('Each alternative must be an object')
        short(alt.get('description'), f'alternatives[{ai}].description')
        groups = alt.get('groups')
        if not isinstance(groups, list): raise ValueError('groups must be an array')
        objects = {}; membership = {}; bindings = []
        for gi, g in enumerate(groups):
            if not isinstance(g, dict): raise ValueError('Each group must be an object')
            short(g.get('reason'), 'group.reason')
            if not isinstance(g.get('new_objects'), list): raise ValueError('new_objects must be an array')
            for o in g['new_objects']:
                if not isinstance(o, dict) or not isinstance(o.get('key'), str):
                    raise ValueError('Each new object needs a string key')
                if o['key'] in objects: raise ValueError('New instance keys must be unique within an alternative')
                for k in ('reason', 'placement', 'variant'): short(o.get(k), 'object.' + k)
                if not isinstance(o.get('category'), str): raise ValueError('category must be a concrete noun')
                if o['category'].lower().replace('_', ' ').strip() in SURFACES:
                    raise ValueError('Building surfaces are enclosure assumptions, not hidden object instances')
                objects[o.get('key')] = o; membership[o.get('key')] = gi
            g.setdefault('functional_links', [])
        relations = []
        if not isinstance(alt.get('relations', []), list): raise ValueError('relations must be an array')
        for ri, r in enumerate(alt.get('relations', [])):
            if not isinstance(r, dict): raise ValueError('Each relation must be an object')
            k = r.get('kind'); a = r.get('source'); b = r.get('target')
            if not isinstance(a, str) or not isinstance(b, str): raise ValueError('Relation endpoints must be exact string IDs')
            if k in ('in', 'inside'):
                attachment = objects.get(a, {}).get('attachment', {})
                if attachment != dict(kind='shelf', host=b):
                    raise ValueError(f'{a}: {k} is an interior-storage hypothesis, not top or adjacency. '
                        f'Actual attachment={attachment}. Use functional_links.kind=contained_in only with '
                        'a compatible shelf attachment to the SAME host; otherwise correct the complete proposal.')
                g = groups[membership[a]]
                g['functional_links'].append(dict(source=a, target=b, kind='contained_in', purpose=r.get('reason')))
                bindings.append(dict(original_relation=deepcopy(r), action='preserved_as_typed_containment_hypothesis',
                                     geometry_verified=False))
                continue
            if k in ('on', 'on_top_of', 'supported_by'):
                # The original validator already checks that this redundant
                # assertion agrees exactly with the encoded child attachment.
                relations.append(r)
                continue
            if k not in SPATIAL:
                raise ValueError('Spatial kind ' + repr(k) + ' is not in ' + str(SPATIAL) +
                    '. Use explicit ORIGINAL CAMERA depth, or group.functional_links for functional intent; no guessed alias.')
            if k in ('above', 'below') and r.get('scope') != 'whole_aabb':
                raise ValueError(f'{a}->{b}: {k} needs explicit scope=whole_aabb. If the intent is illumination, '
                    'facing or service, return a functional_link instead; do not impose whole-box ordering.')
            relations.append(r)
        alt['relations'] = relations
        alt['_typed_bindings'] = bindings
        observed = {o['id'] for o in case['context']}
        for gi, g in enumerate(groups):
            links = g['functional_links']
            if not isinstance(links, list): raise ValueError('functional_links must be an array')
            keys = {k for k, group in membership.items() if group == gi}
            for link in links:
                if not isinstance(link, dict) or link.get('kind') not in FUNCTIONAL:
                    raise ValueError('functional_links.kind must use ' + str(FUNCTIONAL))
                a = link.get('source'); b = link.get('target')
                if not isinstance(a, str) or not isinstance(b, str) or a not in keys or b not in keys | observed or a == b:
                    raise ValueError('Functional link endpoints must be new source in this group and distinct same-group/observed target')
                short(link.get('purpose'), 'functional_link.purpose')
                if link['kind'] == 'contained_in' and objects[a].get('attachment') != dict(kind='shelf', host=b):
                    raise ValueError('contained_in requires matching shelf host; no silent conversion from top')
    programs = original.furniture_answer(raw, case, h)
    for alt, program in zip(raw['alternatives'], programs):
        for g in program['groups']:
            source = alt['groups'][int(g['assembly_key'][1:])]
            own = {o['key'] for o in g['objects']}
            g.update(layout_region=source.get('layout_region', 'unspecified'),
                     spatial_reason=source['reason'], functional_links=deepcopy([l for l in source['functional_links'] if l['source'] in own]))
            if g['layout_region'] not in ('left', 'right', 'rear', 'unspecified'):
                raise ValueError('group.layout_region must be left/right/rear/unspecified')
        # The old revalidator preserves these group fields and object identities.
        clean = original.revalidate_program(program, case)
        clean['catalog_provenance'] = dict(interface='typed_candidate_contract_v1',
            relation_bindings=alt['_typed_bindings'] + program.get('relation_bindings', []), semantic_links_geometry_verified=False)
        original.spatial_choices({'candidate': clean}, case)
        program.clear(); program.update(clean)
    return programs


def coverage(programs):
    roles = {}; repeated = []; summaries = []
    for fid, p in programs.items():
        entries = []
        for g in p['groups']:
            role = roles.setdefault(g['function'], dict(programs=0, groups=0, root_categories=set(), categories=set(), maximum_program_objects=0))
            role['groups'] += 1
            cats = Counter(o['category'] for o in g['objects'])
            role['categories'].update(cats)
            roots = [o['category'] for o in g['objects'] if o['attachment']['host'] is None]
            role['root_categories'].update(roots)
            entries.append(dict(function=g['function'], root_categories=roots, categories=dict(cats),
                                objects=len(g['objects']), intent=g['reason']))
            buckets = Counter((o['category'], o['variant'], o['attachment']['host'], o['distinct_instance_reason']) for o in g['objects'])
            for (category, variant, host, reason), count in buckets.items():
                if count > 1:
                    repeated.append(dict(program=fid, category=category, variant=variant, host=host, count=count,
                                         reason=reason, interpretation='review_collection_hypothesis_not_automatic_deletion'))
        for function in {g['function'] for g in p['groups']}:
            role = roles[function]; role['programs'] += 1
            role['maximum_program_objects'] = max(role['maximum_program_objects'],
                sum(len(g['objects']) for g in p['groups'] if g['function'] == function))
        summaries.append(dict(description=p['description'], groups=entries))
    return dict(roles={r: dict(v, root_categories=sorted(v['root_categories']), categories=sorted(v['categories'])) for r, v in roles.items()},
                maximum_reachable_delta=sum(v['maximum_program_objects'] for v in roles.values()),
                repeated_instances_for_review=repeated, complete_program_summaries=summaries)
