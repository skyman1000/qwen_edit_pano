"""Qualitative hidden-world planning. No room floor, direct XYZ, or new SCM.

G_obs membership is never a language-model decision. Identity review is a
fallible evidence check, not a certificate of semantic correctness.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from PIL import Image

from ...common import read, write, sha
from .. import assembly_compositional as original
from ..assembly_modular import FUNCTIONS
from ..assembly_experiment import SeededProvider, stage_seed
from ..world_scm_runner import ask_valid


class RecordedFailure(ValueError):
    pass


class Requests:
    """Fixed two-attempt schema budget, bound cache, original failures retained."""
    def __init__(self, folder, seed, board, client):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.seed, self.board, self.client = seed, Path(board), client

    def ask(self, name, prompt, validator, budget=8192):
        path = self.folder / (name + '.txt')
        accepted = path.with_suffix('.accepted.json')
        failed = path.with_suffix('.failed.json')
        binding = dict(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                       image_sha256=sha(self.board), seed=stage_seed(self.seed, name), max_tokens=budget)
        for saved in (accepted, failed):
            if not saved.exists():
                continue
            record = read(saved)
            if record['binding'] != binding:
                raise ValueError('Request binding changed: ' + str(saved))
            for raw, digest in record['raw_sha256'].items():
                if sha(raw) != digest:
                    raise ValueError('Saved response changed: ' + raw)
            if saved == failed:
                raise RecordedFailure(record['error'])
            if hashlib.sha256(json.dumps(record['value'], sort_keys=True).encode()).hexdigest() != record['value_sha256']:
                raise ValueError('Accepted response changed: ' + str(saved))
            return record['value']
        # A killed process may have consumed an unknown number of samples.
        # Never silently overwrite it and pretend a fresh sample was original.
        if list(self.folder.glob(name + '*.request.json')) or path.exists():
            raise RecordedFailure('Interrupted request without final receipt: ' + str(path) + '; preserve it and use a new run')
        try:
            with Image.open(self.board) as im:
                value = ask_valid(SeededProvider(self.client(), binding['seed']), im.convert('RGB'),
                                  prompt, path, validator, budget)
        except Exception as exc:
            write(failed, dict(binding=binding, error=type(exc).__name__ + ': ' + str(exc),
                               raw_sha256=self.raw_hashes(path)))
            raise
        write(accepted, dict(binding=binding, value=value,
              value_sha256=hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(),
              raw_sha256=self.raw_hashes(path)))
        return value

    @staticmethod
    def raw_hashes(path):
        return {str(p): sha(p) for p in path.parent.glob(path.stem + '*')
                if p.is_file() and not p.name.endswith(('.accepted.json', '.failed.json'))}


def plan_prompt(case, count):
    return ("Propose " + str(count) + " alternative qualitative organizations of the UNSEEN surrounding indoor world, "
        "conditioned on this partial RGB and frozen visible instances. These are hypotheses, not recovered truth. "
        "First separate photographed evidence from unseen assumptions. Planning must extend outside the original camera, "
        "not just describe the regions in the photograph. A small or open continuation is legal; no furniture count, "
        "category or camera-sector quota. Describe possible coherent functional arrangements, their ordinary contents, "
        "and what would make each hypothesis wrong. Same-function groups may coexist. Use uncertainty when scale or room extent is unknown. "
        "Do not output metric bounds, a floor height, object IDs for new furniture, probabilities, or boxes. "
        "rear means behind the ORIGINAL CAMERA, Z<0; deeper in the photograph is NOT rear. "
        "visible_unmodeled records photographed objects missed by the detector: these must NOT become hidden new objects. "
        "architecture_assumptions describes walls/floor/ceiling separately; these are not furniture instances. "
        "observed_refs are optional nonexclusive annotations. Omitted IDs remain in G_obs; do not partition the inventory. "
        "Return exactly {\"plans\":[{\"description\":\"concrete unseen organization\","
        "\"functions\":[\"canonical tag\"],\"assumptions\":[\"uncertain assumption\"],"
        "\"visible_unmodeled\":[\"visible item/location or empty list\"],"
        "\"architecture_assumptions\":[\"qualitative enclosure assumption\"],"
        "\"areas\":[{\"function\":\"canonical tag\",\"camera_region\":\"unspecified\","
        "\"hidden_intent\":\"possible hidden arrangement and contents, not a visible caption\","
        "\"observed_refs\":[]}],\"failure_conditions\":[\"what would contradict this hypothesis\"]}]}. "
        "camera_region is left/right/rear/unspecified; unspecified is preferable to invented certainty. "
        "Canonical functions: " + json.dumps(FUNCTIONS) +
        "\nFROZEN_VISIBLE_INSTANCES: " + json.dumps(case['context'], ensure_ascii=False))


def plan_answer(value, case, count):
    plans = value.get('plans')
    if not isinstance(plans, list) or len(plans) != count:
        raise ValueError('Return exactly ' + str(count) + ' alternative plans; no object-count requirement')
    observed = {o['id'] for o in case['context']}
    result = []
    for i, item in enumerate(plans):
        if not isinstance(item, dict) or not isinstance(item.get('description'), str) or not item['description'].strip():
            raise ValueError('Each plan requires a nonempty description of an unseen hypothesis')
        p = dict(id=f'plan_{i+1:02d}', description=item['description'])
        for key in ('functions', 'assumptions', 'visible_unmodeled', 'architecture_assumptions', 'failure_conditions'):
            if not isinstance(item.get(key), list) or any(not isinstance(s, str) for s in item[key]):
                raise ValueError(key + ' must be a string array (possibly empty)')
            p[key] = deepcopy(item[key])
        original.profile('h0', p['functions'])
        if not isinstance(item.get('areas'), list):
            raise ValueError('areas must be an array, possibly empty')
        p['areas'] = []
        for j, area in enumerate(item['areas']):
            if not isinstance(area, dict) or area.get('function') not in FUNCTIONS:
                raise ValueError('area.function must be a canonical tag')
            if area.get('camera_region') not in ('left', 'right', 'rear', 'unspecified'):
                raise ValueError('camera_region must be left/right/rear/unspecified in original camera frame')
            if not isinstance(area.get('hidden_intent'), str) or not area['hidden_intent'].strip():
                raise ValueError('area requires hidden_intent')
            refs = area.get('observed_refs', [])
            if not isinstance(refs, list) or any(not isinstance(r, str) for r in refs):
                raise ValueError('observed_refs must be a string array')
            unknown = sorted(set(refs) - observed)
            if unknown:
                raise ValueError('Unknown observed_refs: ' + str(unknown) + '; allowed: ' + str(sorted(observed)))
            # Repeated/missing references are NOT lost/duplicated object instances.
            p['areas'].append(dict(id=f'area_{j+1:02d}', function=area['function'],
                camera_region=area['camera_region'], hidden_intent=area['hidden_intent'], observed_refs=refs))
        result.append(p)
    return result


def candidate_prompt(case, h, channel, plan):
    return original.furniture_prompt(case, h, channel) + (
        '\nPLAN-CONDITIONED HIDDEN DELTA CONTRACT: This branch represents ONE possible unseen world organization. '
        'Use its hidden_intent, not its photographed inventory, to construct complete candidate programs. '
        'Other global plans are executed in separate branches and must not be mixed here. '
        'All new_objects are DISTINCT HIDDEN instances. Photographed but undetected objects belong to visible_unmodeled, '
        'not this hidden delta; never move or rename a photographed object to make it hidden. '
        'Same category is permitted for a distinct unseen instance with an explicit reason. '
        'Walls, floors, ceilings and architectural trim are enclosure assumptions, not new_objects. '
        'Hypothesis and area identifiers (h0, plan_XX, area_XX) are NOT object hosts. '
        'Host must be null for floor/wall/ceiling/free, or an exact observed object ID or new object key in the SAME group. '
        'Keep all host descendants and dependencies together. Do not silently replace missing hosts. '
        'Use one or two complete alternatives, including ordinary contents, not a minimum number of objects. '
        'For each group optionally add camera_region left/right/rear/unspecified and spatial_reason. '
        'Use unspecified for uncertain direction or groups attached to observed hosts. rear is original-camera Z<0. '
        'No XYZ, room bounds, floor height or probabilities. The original executor compiles dimensions and placements later. '
        '\nGLOBAL_PLAN: ' + json.dumps(plan, ensure_ascii=False))


ARCHITECTURAL_SURFACES = {'wall', 'walls', 'floor', 'ceiling', 'room', 'trim', 'baseboard',
                          'ceiling trim', 'wall trim', 'flooring', 'carpet flooring'}


def candidate_answer(value, case, h):
    # Validate the unchanged mature instance/attachment grammar first.
    programs = original.furniture_answer(value, case, h)
    for raw, program in zip(value['alternatives'], programs):
        for group in program['groups']:
            source = raw['groups'][int(group['assembly_key'][1:])]
            region = source.get('camera_region', 'unspecified')
            if region not in ('left', 'right', 'rear', 'unspecified'):
                raise ValueError('group.camera_region must be left/right/rear/unspecified')
            group.update(layout_region=region, spatial_reason=source.get('spatial_reason', source['reason']))
            for obj in group['objects']:
                if obj['category'].lower().replace('_', ' ').strip() in ARCHITECTURAL_SURFACES:
                    raise ValueError(obj['key'] + ': architectural surface belongs to architecture_assumptions, not hidden object inventory')
        # Uses original coordinate-region semantics, NOT metric floor guesses.
        original.spatial_choices({'candidate': program}, case)
    return programs


def identity_prompt(program, case, plan):
    objects = [o for g in program['groups'] for o in g['objects']]
    return ('Check INSTANCE IDENTITY against the photograph and numbered visible instances, not whether this story sounds plausible. '
        'For EACH candidate key decide hidden_hypothesis, uncertain, same_visible_instance or visible_part. '
        'A shared category alone is NOT identity evidence. An extra unseen chair is not the photographed chair. '
        'Positive visible matches require a specific frozen observed_id OR a bbox_1000 [x1,y1,x2,y2] in the ORIGINAL RGB '
        '(exclude annotation margins), plus a reason identifying the same instance. Use uncertain if you cannot substantiate it. '
        'Look especially for renaming photographed appliances, furniture, and their parts as unseen objects. '
        'Do not invent missing evidence, move objects, suggest alternative furniture, or rate overall plausibility. '
        'Return {"checks":[{"key":"candidate key","verdict":"uncertain","observed_id":null,'
        '"bbox_1000":null,"reason":"instance-level evidence or uncertainty"}]}. '
        '\nFROZEN_VISIBLE_INSTANCES: ' + json.dumps(case['context'], ensure_ascii=False) +
        '\nVISIBLE_UNMODELED: ' + json.dumps(plan['visible_unmodeled'], ensure_ascii=False) +
        '\nCANDIDATES: ' + json.dumps(objects, ensure_ascii=False))


def identity_answer(value, program, case):
    expected = {o['key'] for g in program['groups'] for o in g['objects']}
    observed = {o['id'] for o in case['context']}
    checks = value.get('checks')
    if not isinstance(checks, list) or any(not isinstance(c, dict) or not isinstance(c.get('key'), str) for c in checks):
        raise ValueError('checks must identify each candidate key')
    actual = [c['key'] for c in checks]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Identity checks must cover candidate keys exactly: ' + str(sorted(expected)))
    for c in checks:
        if c.get('verdict') not in ('hidden_hypothesis', 'uncertain', 'same_visible_instance', 'visible_part'):
            raise ValueError('Invalid identity verdict')
        if not isinstance(c.get('reason'), str) or not c['reason'].strip():
            raise ValueError('Identity check requires reason')
        if c.get('observed_id') is not None and c['observed_id'] not in observed:
            raise ValueError('Identity evidence refers to unknown observed ID')
        bbox = c.get('bbox_1000')
        if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4 or
                any(type(x) not in (int, float) or not 0 <= x <= 1000 for x in bbox) or
                bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
            raise ValueError('bbox_1000 must be a nonempty normalized original-image rectangle')
        if c['verdict'] in ('same_visible_instance', 'visible_part') and c.get('observed_id') is None and bbox is None:
            raise ValueError('A positive visible match requires observed_id or bbox_1000 evidence; otherwise use uncertain')
    return checks


def propose(comp, plan, client):
    """Two original channels, one identity check per alternative. No deletion.

    A rejected alternative remains on disk with evidence. It is excluded as a
    WHOLE program, never repaired by moving photographed objects offscreen.
    """
    out = comp / 'proposals/qwen'
    if (out / 'summary.json').exists():
        return read(out / 'summary.json')
    run = read(comp / 'run.json'); case = read(comp / 'shared/case.json')
    requests = Requests(out / 'calls', run['seed'], comp / 'shared/observation_board.png', client)
    h = original.profile('h0', plan['functions'])
    write(out / 'hypotheses.json', [h]); write(out / 'global_plan.json', plan)
    programs = {}; rejected = []; errors = []; empty = []
    for channel in ('surrounding', 'belongings'):
        try:
            candidates = requests.ask('furniture_h0_' + channel, candidate_prompt(case, h, channel, plan),
                                      lambda v: candidate_answer(v, case, h))
        except Exception as exc:
            errors.append(dict(stage=channel, error=type(exc).__name__ + ': ' + str(exc)))
            continue
        for i, candidate in enumerate(candidates):
            name = channel + '_' + str(i)
            if not any(g['objects'] for g in candidate['groups']):
                empty.append(dict(candidate=name, program=candidate)); continue
            try:
                checks = requests.ask('identity_' + name, identity_prompt(candidate, case, plan),
                                      lambda v: identity_answer(v, candidate, case))
            except Exception as exc:
                errors.append(dict(stage='identity_' + name, error=type(exc).__name__ + ': ' + str(exc)))
                continue
            conflicts = [c for c in checks if c['verdict'] in ('same_visible_instance', 'visible_part')]
            if conflicts:
                rejected.append(dict(candidate=name, program=candidate, evidence=checks,
                                     reason='instance_identity_conflict; whole alternative retained for review'))
                continue
            clean = original.revalidate_program(candidate, case)
            clean['proposal_source'] = dict(backend='qwen', channel=channel, plan=plan['id'],
                                            identity_checks=checks, semantic_correctness_proven=False)
            programs['f' + str(len(programs))] = clean
    write(out / 'programs.json', programs)
    write(out / 'rejected_programs.json', rejected); write(out / 'empty_programs.json', empty)
    summary = dict(status='partial' if errors else 'complete', backend='qwen', programs=len(programs),
        candidate_instances=sum(len(g['objects']) for p in programs.values() for g in p['groups']),
        rejected_alternatives=len(rejected), empty_alternatives=len(empty), failed_queries=errors,
        model_id=run['local_model'], API_used=False, plan=plan['id'],
        original_v13_executor=True, semantic_correctness_proven=False)
    write(out / 'summary.json', summary)
    return summary
