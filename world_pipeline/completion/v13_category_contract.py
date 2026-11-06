"""Opt-in v13 proposal interface: dataset IDs and advisory object relations.

The pinned H/A/V estimator, CPT compiler, sampler and geometry are unchanged.
This changes the finite candidate domain, so it is not a causal ablation.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

from PIL import Image

from ..common import ROOT, read, write, sha
from . import organized_local_world as local
from . import assembly_compositional as original
from . import assembly_sdscene_experiment as requests
from .assembly_experiment import check as context_check
from .assembly_modular import FUNCTIONS
from .assembly_taxonomy import vocabulary, VOCABULARY
from .category_bridge import CANONICAL_CATEGORIES
from .semantic_categories import resolve_identity
from .fresh_pipeline import provider
from .v13_workflow import verify

KIND = 'v13_closed_category_optional_relations_v1'
META = 'v13_category_contract.json'


def sources():
    paths = (Path(__file__).resolve(), VOCABULARY,
             ROOT/'qwen_edit_pano/scripts/v13_category_contract.sh')
    return {str(p): sha(p) for p in paths}


def categories():
    cats = vocabulary()['categories']
    if any(CANONICAL_CATEGORIES.get(int(k)) != v for k, v in cats.items()):
        raise ValueError('Vocabulary differs from the state compiler category table')
    return cats


def furniture_prompt(case, hypothesis, channel):
    return (
        'Propose 1..2 concrete plausible COMPLETE hidden-world deltas around this partial indoor RGB. '
        'The attached observation board and FIXED_OBSERVATIONS are the only evidence. '
        'Preserve visible instances; do not regenerate them or their integral parts. '
        'A plausible unseen object need not have direct visual proof. H is a soft, non-exhaustive use hypothesis, '
        'not a resident biography and not a list of unmet needs. '
        'Each alternative describes ONE coherent surrounding world, with its coexisting functional assemblies '
        'and ordinary belongings. Think about the whole room, not only surfaces next to visible objects. '
        'Several groups of one function inside an alternative coexist. Different alternatives are mutually exclusive. '
        'An empty alternative is legal. No class, quantity, direction or extra activity is mandatory. '
        'Repeated objects may be real collections, but do not fill the response capacity with numbered copies. '
        'Use category_id as an INTEGER from DATASET_CATEGORIES; it is a coarse dataset label. '
        'Keep the specific noun/type in variant. Select objects/furniture/misc only when that coarse meaning is appropriate; '
        'never change the noun merely to obtain a valid ID. No new IDs, null IDs, or guessed label names. '
        'Optional relations are advisory, not causal edges or strict acceptance constraints; [] is sufficient. '
        'Put useful arrangement intent in placement and group.reason. Do not equate facing with camera depth. '
        'attachment is optional. Omit it when support is unknown; this means independent placement, not verified support. '
        'When confident, declare floor/wall/ceiling/free with host=null, or top/seat/shelf/hang with an exact host key '
        'in the SAME group or an exact observed ID. Physical mounting is different from being next to a wall. '
        'Include a new host in this alternative if choosing a supported object. '
        'affordances are optional offered support surfaces: top/seat/shelf/hanging. '
        'Return only {"alternatives":[{"description":"specific uncertain world",'
        '"groups":[{"function":"one allowed function", "reason":"why this assembly fits the world",'
        '"observed_refs":[],"new_objects":[{"key":"unique_id","category_id":5,'
        '"variant":"specific concrete type","reason":"why this separate hidden instance is plausible",'
        '"placement":"qualitative surrounding placement"}]}],"relations":[]}]}. '
        'The example ID 5 is syntax only, not a requested object. Capacity per alternative: 4 groups, 24 objects; '
        'capacity is not a target. No metric sizes, coordinates, probabilities or prose outside JSON.\n'
        'CHANNEL: '+channel+'\nHYPOTHESIS: '+json.dumps(hypothesis, ensure_ascii=False)+
        '\nALLOWED_FUNCTIONS: '+json.dumps(FUNCTIONS)+
        '\nDATASET_CATEGORIES: '+json.dumps(categories(), ensure_ascii=False)+
        '\nFIXED_OBSERVATIONS: '+json.dumps(case['context'], ensure_ascii=False)+
        '\nOBSERVED_SUPPORT_DOMAINS: '+json.dumps(case.get('observed_support_domains', {}), ensure_ascii=False))


def identity_signature(value):
    """Freeze parseable instance identities across the bounded format correction."""
    try:
        result = []
        for alt in value['alternatives']:
            rows = []
            for group in alt['groups']:
                for obj in group['new_objects']:
                    if not isinstance(obj['key'], str) or not isinstance(obj['variant'], str):
                        return None
                    cid = obj.get('category_id')
                    locked_id = (cid if type(cid) is int and str(cid) in categories()
                        and resolve_identity(categories()[str(cid)], obj['variant'])['category'] == categories()[str(cid)] else None)
                    rows.append((obj['key'], obj['variant'], locked_id))
            result.append(sorted(rows))
        return result
    except (KeyError, TypeError, AttributeError):
        return None


def normalize(value):
    """Convert declared IDs, retain advisory information; never guess a category."""
    cats = categories(); clean = deepcopy(value); audit = []
    if not isinstance(clean, dict) or not isinstance(clean.get('alternatives'), list):
        raise ValueError('Return an object with alternatives array')
    for ai, alt in enumerate(clean['alternatives']):
        if not isinstance(alt, dict) or not isinstance(alt.get('groups'), list):
            raise ValueError('Each alternative needs groups')
        advisory = dict(alternative=ai, relations=deepcopy(alt.get('relations', [])), groups=[])
        alt['relations'] = []
        for gi, group in enumerate(alt['groups']):
            if not isinstance(group, dict) or not isinstance(group.get('new_objects'), list):
                raise ValueError('Each group needs new_objects')
            advisory['groups'].append(dict(group=gi, relations=group.pop('relations', []),
                functional_links=group.pop('functional_links', []), unknown_support=[]))
            for obj in group['new_objects']:
                if not isinstance(obj, dict): raise ValueError('Each new object must be a record')
                cid = obj.get('category_id')
                if type(cid) is not int or str(cid) not in cats:
                    raise ValueError(str(obj.get('key'))+': category_id must be an integer from DATASET_CATEGORIES')
                label = cats[str(cid)]
                if 'category' in obj and obj['category'] != label:
                    raise ValueError(str(obj.get('key'))+': category and category_id disagree; exact label is '+label)
                obj['category'] = label
                if not isinstance(obj.get('variant'), str) or not obj['variant'].strip():
                    raise ValueError('Keep a specific noun/type in variant')
                resolved = resolve_identity(label, obj['variant'])
                if resolved['category'] != label:
                    raise ValueError(str(obj.get('key'))+': declared category conflicts with the specific variant')
                if 'attachment' not in obj:
                    obj['attachment'] = dict(kind='free', host=None)
                    advisory['groups'][-1]['unknown_support'].append(obj['key'])
                obj.setdefault('affordances', [])
        audit.append(advisory)
    return clean, audit


def validator(case, hypothesis, audit_path):
    first = None
    def accept(value):
        nonlocal first
        signature = identity_signature(value)
        if first is None: first = signature
        elif signature is None or len(signature) != len(first):
            raise ValueError('Format correction must preserve every original alternative and instance')
        else:
            for old, new in zip(first, signature):
                if [(k, v) for k, v, _ in old] != [(k, v) for k, v, _ in new]:
                    raise ValueError('Format correction cannot add/drop/rename instances or rewrite variants')
                if any(cid is not None and cid != actual for (_, _, cid), (_, _, actual) in zip(old, new)):
                    raise ValueError('Format correction cannot change a valid original category_id')
        clean, advisory = normalize(value)
        programs = original.furniture_answer(clean, case, hypothesis)
        for program in programs: original.revalidate_program(program, case)
        write(audit_path, dict(category_table_sha256=sha(VOCABULARY),
            identity_signature=signature, advisory=advisory,
            relations_are_hard_constraints=False, unknown_support_is_not_verified=True,
            normalized_reply=clean))
        return programs
    return accept


def propose_scene(root, client):
    run = read(root/'run.json'); case = read(root/'shared/case.json')
    grounding = read(root/'context/qwen/observer2d/grounding.json')
    out = root/'proposals/qwen'; out.mkdir(parents=True, exist_ok=False); (out/'calls').mkdir()
    programs = {}; failures = []; hypotheses = []
    with Image.open(root/'shared/observation_board.png') as image:
        board = image.convert('RGB')
    try:
        hypotheses = requests._ask(client, run, 'hypotheses', board, original.hypotheses_prompt(case, grounding),
            out/'calls/hypotheses.txt', original.hypotheses_answer, run['stage_token_budgets']['proposal'])
        write(out/'hypotheses.json', hypotheses)
        for h in hypotheses:
            for channel in ('surrounding', 'belongings'):
                name = 'furniture_'+h['id']+'_'+channel
                try:
                    values = requests._ask(client, run, name, board, furniture_prompt(case, h, channel),
                        out/'calls'/(name+'.txt'), validator(case, h, out/'calls'/(name+'.contract.json')),
                        run['stage_token_budgets']['proposal'])
                    for value in values:
                        if not any(g['objects'] for g in value['groups']): continue
                        value['proposal_source'] = dict(backend='qwen', channel=channel, interface=KIND)
                        programs['f'+str(len(programs))] = original.revalidate_program(value, case)
                except Exception as exc:
                    failures.append(dict(stage=name, error=type(exc).__name__+': '+str(exc)))
                write(out/'programs.json', programs)
    except Exception as exc:
        failures.append(dict(stage='hypotheses', error=type(exc).__name__+': '+str(exc)))
    # Explicit empty files permit downstream gates to report an empty domain honestly.
    write(out/'hypotheses.json', hypotheses); write(out/'programs.json', programs)
    result = dict(status='partial' if failures and programs else 'failed' if failures else 'complete',
        backend='qwen', model_id=run['local_model'], programs=len(programs),
        candidate_instances=sum(len(g['objects']) for p in programs.values() for g in p['groups']),
        failed_queries=failures, scope=KIND, category_ids_from_dataset=True,
        hard_auxiliary_relations=False, old_candidates_reused=False)
    write(out/'summary.json', result)
    return result


def prepare(root, observations, worlds=1, draws=4):
    verify(); categories(); result = local.prepare(root, observations, worlds, draws)
    write(root/META, dict(kind=KIND, source_sha256=sources(), original_model_unchanged=True,
        category_source=str(VOCABULARY), relations_policy='advisory_only',
        explicit_installation_used_for_geometry=True, absent_attachment='free_independent_placement_unknown_support',
        original_candidates_reused=False, API_used=False))
    check(root); return result


def check(root):
    verify(); meta = read(root/META)
    if meta.get('kind') != KIND or meta.get('source_sha256') != sources():
        raise ValueError('Proposal contract source changed; preserve this run and start a new directory')
    return local.check_run(root)


def propose(root):
    suite = check(root); client = None; results = []
    for scene in suite['scenes']:
        marker = root/'phase_records'/(scene['id']+'_propose_qwen.json')
        if marker.exists(): results.append(read(marker)); continue
        comp = root/'scenes'/scene['id']/'completion'
        try:
            config = context_check(comp/'context')
            if client is None: client = provider(config, 'qwen')
            result = propose_scene(comp, client)
            status = 'complete' if result['status'] == 'complete' else 'partial' if result['programs'] else 'failed'
            record = dict(scene=scene['id'], phase='propose', backend='qwen', status=status, result=result)
        except Exception as exc:
            record = dict(scene=scene['id'], phase='propose', backend='qwen', status='failed', error=type(exc).__name__+': '+str(exc))
        write(marker, record); results.append(record); print(json.dumps(record, ensure_ascii=False), flush=True)
    check(root)
    return dict(status='partial_or_failed' if any(r['status'] != 'complete' for r in results) else 'complete')


def report(root):
    suite = check(root); cats = categories()
    for scene in suite['scenes']:
        comp = root/'scenes'/scene['id']/'completion/end_to_end/qwen/completion'
        for path in comp.glob('world_*/G_full.json'):
            for obj in read(path)['objects']:
                if obj['local_evidence'] == 'visible': continue
                cid = obj['category_id']
                if not cid['valid'] or type(cid['value']) is not int or str(cid['value']) not in cats:
                    raise ValueError('Generated state contains a non-dataset category: '+str(path)+' '+obj['track_id'])
    from . import assembly_presentation_visual as visual
    original_labels = visual.label_rows
    def labels(state):
        rows = original_labels(state)
        for row, obj in zip(rows, state['objects']):
            if row['mapped']:
                row['label'] = row['code']+' '+str(obj['category_id']['value'])+':'+row['label'].split(' ', 1)[1]
        return rows
    with patch.object(visual, 'label_rows', labels): result = local.report(root)
    write(root/'category_contract_report.json', dict(kind=KIND, generated_category_ids_checked=True,
        observations_unchanged=True, candidate_interface_changed=True, original_H_A_V_and_geometry=True,
        quality_verified=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'report')+local.PHASES)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--observations-from', type=Path)
    parser.add_argument('--worlds', type=int, default=1)
    parser.add_argument('--draws', type=int, default=4)
    args = parser.parse_args(); root = args.root.resolve()
    try:
        if args.action == 'prepare':
            if args.observations_from is None: parser.error('prepare needs --observations-from')
            result = prepare(root, args.observations_from, args.worlds, args.draws)
        else:
            check(root)
            if args.action == 'check': result = dict(status='verified_no_model_calls')
            elif args.action == 'propose': result = propose(root)
            elif args.action == 'report': result = report(root)
            else: result = local.phase(root, args.action)
        print(json.dumps({k:v for k,v in result.items() if k != 'records'}, ensure_ascii=False), flush=True)
        return 1 if result['status'] in ('failed', 'partial', 'partial_or_failed') else 0
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, str(exc)+'\n')


if __name__ == '__main__': sys.exit(main())
