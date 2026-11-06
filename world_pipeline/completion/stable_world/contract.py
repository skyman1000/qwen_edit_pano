"""Lossless narrative validation and explicit scalar/list field repairs.

The typed validator's 320-character rule is a presentation constraint, not a
reason to redraw an inventory. Validate a bounded narrative view, then restore
every original narrative before the original semantic/geometric revalidation.
Object identities, membership, attachments and relations are never shortened.
"""
from copy import deepcopy
import json
import re

from ..candidate_repair import contract as typed
from .. import assembly_compositional as original


def prose(value,name):
    if not isinstance(value,str) or not value.strip():raise ValueError(name+' requires nonempty text')
    value=value.strip()
    if re.search(r'<[^<>]{1,100}>',value):raise ValueError(name+': replace schema placeholder')
    typed.short(value[:320],name)
    return value


def hypotheses_answer(value):
    hs=original.hypotheses_answer(value)
    for h in hs:
        prose(h['description'],'hypothesis.description')
        if len(h['assumptions'])>4:raise ValueError('At most four assumptions')
        if any(v not in typed.FUNCTIONS for v in h['functions']):raise ValueError('Use canonical function tags')
        for v in h['assumptions']:prose(v,'hypothesis.assumptions')
    return hs


def furniture_answer(value,case,h):
    if not isinstance(value,dict) or not isinstance(value.get('alternatives'),list):raise ValueError('Need alternatives array')
    raw=deepcopy(value); view=deepcopy(raw)
    def bounded(row,key):row[key]=prose(row.get(key),key)[:320]
    for a in view.get('alternatives',[]):
        if not isinstance(a,dict) or not isinstance(a.get('groups'),list):raise ValueError('Need group objects in groups array')
        bounded(a,'description')
        for g in a.get('groups',[]):
            if not isinstance(g,dict) or not isinstance(g.get('new_objects'),list):raise ValueError('Need new_objects array in each group')
            bounded(g,'reason')
            for o in g.get('new_objects',[]):
                if not isinstance(o,dict):raise ValueError('Each object must be a JSON object')
                for k in ('reason','placement','variant'):bounded(o,k)
            if not isinstance(g.get('functional_links',[]),list):raise ValueError('functional_links must be an array')
            for r in g.get('functional_links',[]):
                if not isinstance(r,dict):raise ValueError('Each functional link must be an object')
                bounded(r,'purpose')
        if not isinstance(a.get('relations',[]),list):raise ValueError('relations must be an array')
        for r in a.get('relations',[]):
            if not isinstance(r,dict):raise ValueError('Each relation must be an object')
            bounded(r,'reason')
    programs=typed.furniture_answer(view,case,h)
    for a,p in zip(raw['alternatives'],programs):
        p['description']=a['description'];p['hypothesis']=deepcopy(h)
        objects={o['key']:o for g in a['groups'] for o in g['new_objects']}
        for g in p['groups']:
            source=a['groups'][int(g['assembly_key'][1:])]
            g.update(reason=source['reason'],spatial_reason=source['reason'])
            for link in g.get('functional_links',[]):
                match=next((r for r in source.get('functional_links',[]) if
                    all(r[k]==link[k] for k in ('source','target','kind'))),None)
                if match:link['purpose']=match['purpose']
                elif link['kind']=='contained_in':
                    match=next((r for r in a.get('relations',[]) if r.get('kind') in ('in','inside')
                        and all(r[k]==link[k] for k in ('source','target'))),None)
                    if match:link['purpose']=match['reason']
            for o in g['objects']:
                old=objects[o['key']]
                o.update(variant=old['variant'],distinct_instance_reason=old['reason'],placement_hint=old['placement'])
        for r in p['relations']:
            match=next((old for old in a.get('relations',[]) if old['source']==r['source'] and old['target']==r['target']
                and {'farther_than':'in_front_of','nearer_than':'behind'}.get(old['kind'],old['kind'])==r['kind']),None)
            if match:r['reason']=match['reason']
        clean=original.revalidate_program(p,case)
        clean['hypothesis']=deepcopy(h)
        original.spatial_choices({'candidate':clean},case)
        p.clear();p.update(clean)
    return programs


def repair_paths(value):
    """Explicit editable fields; no object/category/group/alternative changes."""
    paths=set()
    alternatives=value.get('alternatives')
    if not isinstance(alternatives,list):return paths
    for ai,a in enumerate(alternatives):
        if not isinstance(a,dict) or not isinstance(a.get('groups'),list):continue
        for gi,g in enumerate(a['groups']):
            if not isinstance(g,dict) or not isinstance(g.get('new_objects'),list):continue
            base=f'/alternatives/{ai}/groups/{gi}'
            for oi,o in enumerate(g.get('new_objects',[])):
                if not isinstance(o,dict):continue
                prefix=base+f'/new_objects/{oi}'
                if isinstance(o.get('attachment'),dict):
                    paths.update(prefix+'/attachment/'+k for k in ('kind','host'))
                paths.add(prefix+'/affordances')
            links=g.get('functional_links',[])
            for ri,r in enumerate(links if isinstance(links,list) else []):
                paths.update(base+f'/functional_links/{ri}/'+k for k in ('source','target','kind'))
        relations=a.get('relations',[])
        for ri,r in enumerate(relations if isinstance(relations,list) else []):
            paths.update(f'/alternatives/{ai}/relations/{ri}/'+k for k in ('source','target','kind','scope'))
    return paths


def apply_patch(value,response):
    if not isinstance(response,dict) or set(response)!={'edits'} or not isinstance(response['edits'],list):
        raise ValueError('Return exactly edits:[{path,value,reason}]; never a replacement inventory')
    result=deepcopy(value);allowed=repair_paths(value);seen=set()
    for edit in response['edits']:
        if not isinstance(edit,dict) or set(edit)!={'path','value','reason'}:raise ValueError('Each edit needs path,value,reason')
        path=edit['path']
        if not isinstance(path,str) or path not in allowed or path in seen:raise ValueError('Non-editable or repeated path: '+str(path))
        seen.add(path);prose(edit['reason'],'edit.reason')
        tokens=path.strip('/').split('/');node=result
        for t in tokens[:-1]:node=node[int(t)] if isinstance(node,list) else node[t]
        node[tokens[-1]]=deepcopy(edit['value'])
    return result


def furniture_prompt(case,h,channel,coverage):
    # Keep the productive original interface. No caption/area planner gate.
    return (typed.furniture_prompt(case,h,channel,coverage)+
        '\nTASK INTERPRETATION: Sample a PLAUSIBLE existing hidden-world hypothesis, not a list of '
        'visually proved objects and not a minimal room redesign. Unobserved means uncertain, not absent. '
        'A floor-standing hidden cabinet does not need an observed supporting object. Missing measured walls '
        'do not forbid hypothesizing floor furniture. Keep assumptions explicit; do not claim entailment. '
        'H is suggestive and NON-EXHAUSTIVE, not permission to generate only its named functions. '
        'Describe compatible surrounding functional arrangements with their independent hosts and ordinary '
        'contents; do not turn observed-host attachment into the default for everything. '
        'Do not duplicate photographed but undetected items elsewhere. rear is CAMERA Z<0, not behind a bed. '
        'Building-mounted fixtures use host=null; anchors and physical hosts are different concepts. '
        'Supply concise reasons. Long explanation alone will not invalidate or redraw a correct inventory. '
        'The saved local catalog is an existing hypothesis catalog, not observed fact or a required checklist. '
        'Propose a distinct compatible alternative or complementary use if plausible; no minimum counts or new role requirement.')
