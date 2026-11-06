"""v13 interface correction: closed labels and fixed-inventory installation fields.

No changes to H/A/V, counts, sampling or the geometry solver. Never interpret
missing attachment as free placement. Keep v1 sources/runs intact.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

from PIL import Image

from ..common import ROOT, read, write, sha
from . import v13_category_contract as prior
from . import assembly_sdscene as parser
from .v13_workflow import verify

KIND = 'v13_closed_category_installation_fields_v2'
META = prior.META
GENERIC = {'object', 'objects', 'furniture', 'misc', 'lighting', 'decor',
           'decoration', 'fixture', 'everyday belonging'}


def sources():
    return dict(prior.sources(), **{str(p): sha(p) for p in (
        Path(__file__).resolve(), ROOT/'qwen_edit_pano/scripts/v13_contract_fix.sh')})


def furniture_prompt(case, hypothesis, channel):
    text = prior.furniture_prompt(case, hypothesis, channel)
    text = text.replace(
        'attachment is optional. Omit it when support is unknown; this means independent placement, not verified support. ',
        'Predict a plausible installation for each instance. Missing installation is unknown, never implicitly free. ')
    text = text.replace('"placement":"qualitative surrounding placement"}',
        '"placement":"qualitative surrounding placement",'
        '"attachment":{"kind":"floor","host":null},"affordances":[]}')
    return text + ('\nINSTALLATION FIELDS: The floor example is syntax, not a default. '
        'Choose floor for ordinary floor-standing furniture, wall/ceiling for physical mounting, '
        'top/seat/shelf/hang with an exact compatible host when applicable. '
        'free is an explicit independent-placement hypothesis whose camera Y must later be predicted; '
        'do not use it merely because the field was omitted. No general relation graph is required. '
        'Keep concrete types in variant, not bare objects/furniture/lighting/misc. '
        'A dataset coarse label is allowed as category_id even when its variant is more specific. '
        'Do not turn observed objects or described visible fixtures into hidden new instances. '
        'Consider coherent assemblies and ordinary possessions, not only minimum unmet needs.')


@contextmanager
def closed_labels():
    # The old parser predates the closed table and rejects legal labels lighting
    # and furniture. Permit ONLY exact existing table labels; other broad terms
    # remain rejected. Concrete variants are checked separately below.
    with patch.object(parser, 'BROAD', parser.BROAD-set(prior.categories().values())):
        yield


def objects(value):
    for ai, alt in enumerate(value['alternatives']):
        for group in alt['groups']:
            for obj in group['new_objects']:
                yield ai, obj


def generic_variant(obj):
    return isinstance(obj.get('variant'),str) and obj['variant'].strip().casefold() in GENERIC


def raw_validator():
    """Bound a format correction to the inventory; defer installation to fields."""
    initial = None
    def accept(value):
        nonlocal initial
        signature = prior.identity_signature(value)
        if initial is None: initial = signature
        elif signature is None or len(initial) != len(signature):
            raise ValueError('Preserve every original alternative and object identity')
        else:
            for before, after in zip(initial, signature):
                if [(k,v) for k,v,_ in before] != [(k,v) for k,v,_ in after]:
                    raise ValueError('Do not add/drop/rename objects or rewrite their variants')
                if any(a is not None and a != b for (_,_,a),(_,_,b) in zip(before,after)):
                    raise ValueError('Do not change a valid original category_id')
        # Check the numeric/category interface without accepting synthesized free
        # attachments. The normalized copy is discarded at this stage.
        try: prior.normalize(value)
        except (KeyError,TypeError,AttributeError) as exc:
            raise ValueError('Malformed instance record: '+str(exc)) from exc
        if not 1 <= len(value['alternatives']) <= 2:
            raise ValueError('Use 1..2 alternatives')
        for alt in value['alternatives']:
            if len(alt['groups']) > 4: raise ValueError('At most 4 groups')
            rows = [o for g in alt['groups'] for o in g['new_objects']]
            if len(rows)>24 or len({o['key'] for o in rows}) != len(rows):
                raise ValueError('Use at most 24 distinct instance keys per alternative')
            if any(g.get('function') not in prior.FUNCTIONS for g in alt['groups']):
                raise ValueError('Use an allowed function tag')
        return deepcopy(value)
    return accept


def parse(value, case, hypothesis):
    for _, obj in objects(value):
        if 'attachment' not in obj:
            raise ValueError(obj['key']+': installation missing; never default it to free')
        if generic_variant(obj):
            raise ValueError(obj['key']+': refine the coarse variant to a concrete type without changing category_id')
    normalized, audit = prior.normalize(value)
    with closed_labels():
        programs = prior.original.furniture_answer(normalized, case, hypothesis)
        for p in programs: prior.original.revalidate_program(p, case)
    return programs, normalized, audit


def field_prompt(raw, case, error):
    template = [dict(alternative=ai, key=o['key'], attachment=dict(kind=None,host=None),
        affordances=o.get('affordances',[]), **({'variant': None} if generic_variant(o) else {}))
        for ai,o in objects(raw)]
    return ('Complete ONLY installation/affordance fields for the FIXED hidden inventory. '
        'No adding, removing, merging or renaming objects; category IDs are frozen. '
        'Return exactly {"objects": TEMPLATE}. Include each (alternative,key) once. '
        'Only rows whose template includes variant may refine that generic noun into a concrete type '
        'consistent with the SAME category_id and original reason. All other variants are frozen. '
        'Predict plausible support, not observed truth. floor/wall/ceiling/free require host=null; '
        'top/seat/shelf/hang require an exact observed ID or new key in the SAME semantic group. '
        'Choose meaningful floor-standing or mounting hypotheses. free is explicit independent placement, '
        'not a synonym for missing support. Do not add a host: use a plausible existing host or root placement. '
        'For new hosts, set offered affordances among top/seat/shelf/hanging as needed. '
        'Do not force every instance onto a floor, or force a semantic relationship graph. '
        'Observed hosts must have a nonempty eligible support domain; visible anchors are immutable. '
        'No coordinates, sizes, probabilities, category fields or extra explanations. '
        '\nTEMPLATE: '+json.dumps(template)+
        '\nFIXED_INVENTORY: '+json.dumps(raw,ensure_ascii=False)+
        '\nOBSERVATIONS: '+json.dumps(case['context'],ensure_ascii=False)+
        '\nOBSERVED_SUPPORT_DOMAINS: '+json.dumps(case.get('observed_support_domains',{}),ensure_ascii=False)+
        '\nINTERFACE_ERROR: '+str(error))


def field_validator(raw, case, hypothesis):
    expected = {(ai,o['key']):o for ai,o in objects(raw)}
    def accept(value):
        rows=value.get('objects') if isinstance(value,dict) else None
        if not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows):
            raise ValueError('Return objects array')
        keys=[]
        for row in rows:
            if type(row.get('alternative')) is not int or not isinstance(row.get('key'),str):
                raise ValueError('Use exact integer alternative and string key')
            keys.append((row['alternative'],row['key']))
        if len(keys)!=len(set(keys)) or set(keys)!=set(expected):
            raise ValueError('Field completion must preserve every fixed instance exactly once')
        completed=deepcopy(raw);target={(ai,o['key']):o for ai,o in objects(completed)}
        for key,row in zip(keys,rows):
            allowed={'alternative','key','attachment','affordances'}
            if generic_variant(expected[key]):allowed.add('variant')
            if set(row)!=allowed:
                raise ValueError('Return only exact fields for '+str(key)+': '+str(sorted(allowed)))
            target[key]['attachment']=deepcopy(row['attachment'])
            target[key]['affordances']=deepcopy(row['affordances'])
            if 'variant' in allowed:target[key]['variant']=row['variant']
        parse(completed,case,hypothesis)
        return completed
    return accept


def finalize(raw, case, hypothesis, client, run, board, name, out):
    raw_sha=None;used=False
    try:
        programs, normalized, audit = parse(raw,case,hypothesis)
    except ValueError as exc:
        used=True
        path=out/'calls'/(name+'_installation.txt')
        completed=prior.requests._ask(client,run,name+'_installation',board,field_prompt(raw,case,exc),
            path,field_validator(raw,case,hypothesis),run['stage_token_budgets']['proposal'])
        programs, normalized, audit = parse(completed,case,hypothesis)
        raw_sha=sha(path.with_suffix('.json')) if path.with_suffix('.json').exists() else None
    write(out/'calls'/(name+'.contract.json'),dict(kind=KIND,normalized_reply=normalized,
        advisory=audit,category_table_sha256=sha(prior.VOCABULARY),
        field_completion_used=used,field_reply_sha256=raw_sha,
        inventory_additions_deletions_allowed=False,default_free_used=False,
        support_is_predicted_not_verified=True,relations_are_hard_constraints=False))
    return programs


def propose_scene(root, client):
    run=read(root/'run.json');case=read(root/'shared/case.json')
    grounding=read(root/'context/qwen/observer2d/grounding.json')
    out=root/'proposals/qwen';out.mkdir(parents=True,exist_ok=False);(out/'calls').mkdir()
    programs={};failures=[];hypotheses=[]
    with Image.open(root/'shared/observation_board.png') as image:board=image.convert('RGB')
    try:
        hypotheses=prior.requests._ask(client,run,'hypotheses',board,prior.original.hypotheses_prompt(case,grounding),
            out/'calls/hypotheses.txt',prior.original.hypotheses_answer,run['stage_token_budgets']['proposal'])
        write(out/'hypotheses.json',hypotheses)
        for h in hypotheses:
            for channel in ('surrounding','belongings'):
                name='furniture_'+h['id']+'_'+channel
                try:
                    raw=prior.requests._ask(client,run,name,board,furniture_prompt(case,h,channel),
                        out/'calls'/(name+'.txt'),raw_validator(),run['stage_token_budgets']['proposal'])
                    values=finalize(raw,case,h,client,run,board,name,out)
                    for value in values:
                        if not any(g['objects'] for g in value['groups']):continue
                        value['proposal_source']=dict(backend='qwen',channel=channel,interface=KIND)
                        with closed_labels():programs['f'+str(len(programs))]=prior.original.revalidate_program(value,case)
                except Exception as exc:failures.append(dict(stage=name,error=type(exc).__name__+': '+str(exc)))
                write(out/'programs.json',programs)
    except Exception as exc:failures.append(dict(stage='hypotheses',error=type(exc).__name__+': '+str(exc)))
    write(out/'hypotheses.json',hypotheses);write(out/'programs.json',programs)
    result=dict(status='partial' if failures and programs else 'failed' if failures else 'complete',
        backend='qwen',model_id=run['local_model'],programs=len(programs),
        candidate_instances=sum(len(g['objects']) for p in programs.values() for g in p['groups']),
        failed_queries=failures,scope=KIND,category_ids_from_dataset=True,
        hard_auxiliary_relations=False,default_free_used=False,old_candidates_reused=False,
        installation_field_queries=len(list((out/'calls').glob('*_installation.json'))))
    write(out/'summary.json',result);return result


def prepare(root, observations, worlds=1, draws=4):
    verify();prior.categories();result=prior.local.prepare(root,observations,worlds,draws)
    write(root/META,dict(kind=KIND,source_sha256=sources(),original_model_unchanged=True,
        category_source=str(prior.VOCABULARY),relations_policy='advisory_only',
        category_parser_policy='allow_exact_closed_labels_with_concrete_variant',
        absent_attachment='bounded_fixed_inventory_field_completion_never_default_free',
        original_candidates_reused=False,API_used=False))
    check(root);return result


def check(root):
    verify();meta=read(root/META)
    if meta.get('kind')!=KIND or meta.get('source_sha256')!=sources():
        raise ValueError('v2 contract source changed; preserve this run and use a new directory')
    return prior.local.check_run(root)


@contextmanager
def runtime():
    # Scope compatibility to this explicit entry. No change to core source or
    # to the default/v1 behavior; restore every patched attribute on exit.
    with closed_labels(), patch.multiple(prior,KIND=KIND,prepare=prepare,check=check,propose_scene=propose_scene):
        yield


def main():
    with runtime():return prior.main()


if __name__=='__main__':sys.exit(main())
