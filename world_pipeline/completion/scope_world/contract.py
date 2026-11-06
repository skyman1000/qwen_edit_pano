"""Bounded room/area commitments, then host-complete local instantiation."""
from copy import deepcopy
import json
import re

from ...common import read
from ..simple_world import contract as base
from ..candidate_repair.contract import short as typed_short, FUNCTIONS


def short(value, name, limit=600):
    text=typed_short(value,name,limit)
    if re.search(r'<[^<>]{1,100}>',text):
        raise ValueError(name+': replace schema placeholders with a concrete hypothesis')
    return text


def context(case):
    observed = read(case['state'])
    return dict(visible_instances=case['context'], predicted_geometry=[dict(id=o['track_id'],
        center_m=o['center_local_m']['value'], full_camera_AABB_m=o['size_aabb_local_m']['value'])
        for o in observed['objects']], geometry_is_predicted=True,
        observed_support_domains=case['observed_support_domains'])


def hypotheses_prompt(case):
    schema = dict(hypotheses=[dict(description='<concrete surrounding configuration>',
        room_kind='<room type hypothesis>', functions=['<canonical use>'], assumptions=['<uncertainty>'],
        visible_unmodelled=['<photographed item missing from the fixed graph, if any>'],
        hidden_areas=[dict(id='a0', function='<canonical use>', region='unspecified',
            configuration='<concrete unseen host arrangement and use>',
            why_plausible='<room-level compatibility and uncertainty>', anchor_refs=[])],
        unmodelled_space='<what remains unknown beyond this proposal>')])
    return ('Infer one or two alternative surrounding configurations from ONE PARTIAL indoor RGB. '
        'The output is a hypothesis about the existing room AROUND the camera, not a caption of the photographed corner. '
        'First reason about the room type and its plausible unseen organization. Describe coexisting hidden areas '
        'as concrete host arrangements and uses. A function label alone, or adding a second copy of a visible item, '
        'does not describe the surrounding configuration. Keep the proposed areas mutually compatible in ONE room. '
        'Use ordinary room-level functional causes, not a resident biography. Decorative cues alone do not establish '
        'a new activity; a sparsely furnished crop does not establish empty surroundings. '
        'Every hidden_area in ONE hypothesis COEXISTS and will be instantiated; they are not alternatives or independent '
        'role switches. Different hypotheses may propose different organizations. No required categories, number of areas '
        'or sector coverage. hidden_areas may be empty when the organization is unresolved; this means UNKNOWN, not empty room. '
        'Capacity is zero to four hidden areas per hypothesis, not a target. Up to four short assumptions and six '
        'visible_unmodelled nouns; concise concrete fields, no inventory counts, coordinates, dimensions or probabilities. '
        'Photographed objects missing from the graph go ONLY in visible_unmodelled. Do not move them offscreen. '
        'anchor_refs are optional exact fixed IDs for context, not authorization to duplicate or relocate them. '
        'region is left/right/rear/unspecified in ORIGINAL camera axes; rear means Z<0. Unknown room walls and extent '
        'remain assumptions. If a direction is uncertain use unspecified rather than claiming a measured layout. '
        'Return JSON and replace all placeholders: '+json.dumps(schema)+
        '\nFUNCTIONS: '+json.dumps(FUNCTIONS)+'\nFIXED_EVIDENCE: '+json.dumps(context(case),ensure_ascii=False))


def hypotheses_answer(value, case):
    hs = base.hypotheses_answer(value); raw = value['hypotheses']; observed = {o['id'] for o in case['context']}
    for h, row in zip(hs,raw):
        h['description']=short(h['description'],'hypothesis.description')
        h['assumptions']=[short(v,'hypothesis.assumptions') for v in h['assumptions']]
        h['room_kind'] = short(row.get('room_kind'),'room_kind')
        h['unmodelled_space'] = short(row.get('unmodelled_space'),'unmodelled_space')
        visible = row.get('visible_unmodelled',[])
        if not isinstance(visible,list) or len(visible)>6: raise ValueError('visible_unmodelled: at most six concise nouns')
        h['visible_unmodelled'] = [short(v,'visible_unmodelled',120) for v in visible]
        areas = row.get('hidden_areas')
        if not isinstance(areas,list) or len(areas)>4: raise ValueError('hidden_areas must be an array of zero to four coexisting areas')
        seen = set(); clean = []
        for a in areas:
            if not isinstance(a,dict): raise ValueError('Each hidden area must be an object')
            key = a.get('id')
            if not isinstance(key,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,23}',key) or key in seen:
                raise ValueError('Area IDs must be distinct short identifiers')
            seen.add(key)
            if a.get('function') not in FUNCTIONS: raise ValueError('Area function must be canonical')
            if a.get('region') not in ('left','right','rear','unspecified'): raise ValueError('Area region must use original camera axes')
            refs = a.get('anchor_refs',[])
            if not isinstance(refs,list) or any(not isinstance(v,str) or v not in observed for v in refs):
                raise ValueError('Area anchor_refs must be exact observed IDs; no exhaustive partition is required')
            clean.append(dict(id=key,function=a['function'],region=a['region'],anchor_refs=list(dict.fromkeys(refs)),
                configuration=short(a.get('configuration'),'area.configuration'),
                why_plausible=short(a.get('why_plausible'),'area.why_plausible')))
        h['hidden_areas'] = clean
    return hs


def area_prompt(case,h,area,remaining,prior):
    schema = dict(status='instantiated',reason='<instantiation or explicit uncertainty>',
        new_objects=[dict(key='<unique local ID>',category='<concrete noun>',variant='<type>',
            reason='<distinct hidden instance>',placement='<location relative to hosts and this area>',
            attachment=dict(kind='floor',host=None),affordances=[])],relations=[],functional_links=[])
    return ('Instantiate ONLY the requested hidden area of the supplied WHOLE surrounding hypothesis. '
        'All its other areas coexist; do not replace the world by this one corner. Supply the concrete independent '
        'host furniture/fixtures and their ordinary supported contents together, at appropriate scales. '
        'Preserve the requested room use, area function and original-camera region. No mandatory class or object count. '
        'Complete this particular area, not a list of tiny additions to already photographed furniture. '
        'Same-category hidden instances need a distinct area/use; visible_unmodelled items are photographed and cannot '
        'be regenerated elsewhere. Avoid broad categories such as storage, belongings or furniture; use actual nouns. '
        'status=instantiated means a nonempty concrete inventory. status=assumed_empty means an explicit empty-area '
        'hypothesis; status=unresolved means insufficient specification, never evidence of absence. The last two use '
        'new_objects=[], relations=[], functional_links=[] and a concrete reason. '
        'Return concise JSON: '+json.dumps(schema)+
        '\nSupport is ONLY attachment kind floor/top/seat/shelf/hang/wall/ceiling/free. Hosted kinds '
        'top/seat/shelf/hang need an exact local new host or observed ID; other kinds require host=null. '
        'Use floor for standing furniture, wall only for physically mounted items. A nearby observed object is an '
        'anchor, not automatically a physical host. New-new hosts stay inside THIS area. Affordances are offered '
        'surfaces: top/seat/shelf/hanging. Consult fixed support domains: a nonempty domain is only a necessary '
        'condition, not a guarantee that a whole additional object can stay outside the input view. '
        'Optional relations use near/left_of/right_of/farther_than/nearer_than/above/below with exact endpoints '
        'and reason. Greater Z is farther_than, smaller Z nearer_than. above/below require scope=whole_aabb. '
        'Functional_links allow serves/illuminates/faces/accessible_from with source,target,kind,purpose. '
        'Do not duplicate support as on/inside/contained_in. Do not invent geometry or return probabilities. '
        'There are '+str(remaining)+' records of remaining SHARED world capacity, not a quantity to fill. '
        'During a schema correction preserve the exact proposed instance keys and categories; correct the faulty '
        'fields rather than clearing the inventory, renaming instances, or changing their categories. '
        '\nWHOLE_CONFIGURATION: '+json.dumps(h,ensure_ascii=False)+
        '\nREQUESTED_AREA: '+json.dumps(area,ensure_ascii=False)+
        '\nALREADY_INSTANTIATED_AREAS_READ_ONLY: '+json.dumps(prior,ensure_ascii=False)+
        '\nFIXED_EVIDENCE: '+json.dumps(context(case),ensure_ascii=False))


def area_answer(value,case,h,area,remaining):
    status = value.get('status'); reason = short(value.get('reason'),'area_result.reason')
    if status not in ('instantiated','assumed_empty','unresolved'): raise ValueError('Use instantiated/assumed_empty/unresolved')
    rows = value.get('new_objects'); relations = value.get('relations',[]); links = value.get('functional_links',[])
    if not isinstance(rows,list) or len(rows)>remaining: raise ValueError('new_objects exceeds shared remaining capacity')
    if status != 'instantiated':
        if rows or relations or links: raise ValueError('Empty or unresolved area cannot contain asserted objects/relations')
        return dict(status=status,reason=reason,group=None,relations=[])
    if not rows: raise ValueError('An unresolved inventory uses status=unresolved; an explicit empty area uses assumed_empty')
    prefix = area['id']+'_'; keys = [o.get('key') if isinstance(o,dict) else None for o in rows]
    if any(not isinstance(k,str) for k in keys) or len(set(keys))!=len(rows):
        raise ValueError('Every area object needs a distinct local string key')
    ids = set(keys)
    raw = deepcopy(value)
    def mapped(v): return prefix+v if v in ids else v
    for o in raw['new_objects']:
        o['key'] = mapped(o['key'])
        if isinstance(o.get('attachment'),dict): o['attachment']['host'] = mapped(o['attachment'].get('host'))
    for r in raw.get('relations',[])+raw.get('functional_links',[]):
        r['source'] = mapped(r.get('source')); r['target'] = mapped(r.get('target'))
    g = dict(function=area['function'],reason=reason,layout_region=area['region'],observed_refs=area['anchor_refs'],
        new_objects=raw['new_objects'],functional_links=raw.get('functional_links',[]))
    # Validate locally using the unchanged typed support/identity contracts.
    wire = dict(alternatives=[dict(description=h['description'],groups=[g],relations=raw.get('relations',[]))])
    base.furniture_answer(wire,case,h)
    return dict(status=status,reason=reason,group=g,relations=raw.get('relations',[]))


class AreaValidator:
    """A schema correction cannot become a new inventory draw or empty world."""
    def __init__(self,case,h,area,remaining):
        self.case,self.h,self.area,self.remaining=case,h,area,remaining
        self.commitment=None

    def __call__(self,value):
        rows=value.get('new_objects')
        valid=isinstance(rows,list) and all(isinstance(o,dict) and isinstance(o.get('key'),str)
            and isinstance(o.get('category'),str) for o in rows)
        identity={o['key']:o['category'] for o in rows} if valid else None
        if valid and len(identity)!=len(rows): identity=None
        if self.commitment is None and identity:
            self.commitment=identity
        if self.commitment is not None and identity!=self.commitment:
            raise ValueError('Schema correction changed the proposed inventory. Preserve exact keys/categories: '+
                json.dumps(self.commitment)+'; correct fields only. A failed correction remains unresolved, not absent.')
        try:
            return area_answer(value,self.case,self.h,self.area,self.remaining)
        except ValueError as exc:
            if self.commitment:
                # Requests intentionally omits the previous full reply from
                # correction prompts. Carry the identity contract explicitly.
                raise ValueError('Preserve these exact proposed keys/categories: '+json.dumps(self.commitment)+
                    '; correct fields only. Field error: '+str(exc)) from exc
            raise


def assemble(case,h,answers):
    groups=[]; relations=[]; ledger=[]
    for area in h['hidden_areas']:
        a = answers[area['id']]
        ledger.append(dict(area=area['id'],configuration=area['configuration'],region=area['region'],
            status=a['status'],reason=a['reason'],uninstantiated_inventory=a.get('uninstantiated_inventory',{})))
        if a['group'] is not None: groups.append(a['group']); relations += a['relations']
    raw=dict(alternatives=[dict(description=h['description'],groups=groups,relations=relations)])
    p=base.furniture_answer(raw,case,h)[0]
    unresolved=[v['area'] for v in ledger if v['status']=='unresolved']
    p['scope_ledger']=ledger
    p['scope_complete_within_hypothesis']=bool(h['hidden_areas']) and not unresolved
    p['unresolved_areas']=unresolved
    p['unmodelled_space']=h['unmodelled_space']
    return p
