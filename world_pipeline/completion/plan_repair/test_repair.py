"""CPU contracts + real saved-response regressions; no quality claim about Qwen."""
from contextlib import ExitStack
from copy import deepcopy
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from ...common import ROOT, read, write, sha
from .. import assembly_multiscene as pipeline, assembly_compositional as original
from .. import v13_workflow as pinned
from ..fresh_observer import load_case
from ..test_compositional import CPUProgramProvider
from ..test_multiscene import CPUProvider
from . import proposals as p, workflow as w

SOURCE = ROOT / 'qwen_edit_pano/outputs/local_observation_fixed_002'
OLD = ROOT / 'qwen_edit_pano/outputs/v13_local_world_001'
FAILED = ROOT / 'qwen_edit_pano/outputs/panorama_plan_001'
AUDIT = ROOT / 'qwen_edit_pano/reports/panorama_plan_001_root_cause_20261105/diagnosis.json'


def plan(description='Possible hidden work area', refs=()):
    return dict(description=description, functions=['work'], assumptions=['Unseen extent is unknown'],
        visible_unmodeled=[], architecture_assumptions=['Enclosure extent is unknown'],
        areas=[dict(function='work', camera_region='unspecified',
                    hidden_intent='Possible desk, chair and ordinary work contents outside the camera', observed_refs=list(refs))],
        failure_conditions=['No surrounding space exists'])


def wire(program):
    return dict(alternatives=[dict(description=program['description'], groups=[
        dict(function=g['function'], reason=g['reason'], observed_refs=g.get('observed_refs', []),
             camera_region=g.get('layout_region', 'unspecified'),
             new_objects=[dict(key=o['key'], category=o['category'], variant=o['variant'],
                 reason=o['distinct_instance_reason'], placement=o['placement_hint'],
                 attachment=o['attachment'], affordances=o['affordances']) for o in g['objects']])
        for g in program['groups']], relations=[dict(r, kind={'in_front_of': 'farther_than',
            'behind': 'nearer_than'}.get(r['kind'], r['kind'])) for r in program['relations']])])


class PlanningCPUProvider(CPUProgramProvider):
    """Explicit synthetic semantic replies; downstream executor is real v13."""
    def ask(self, image, prompt, path, **kwargs):
        path = Path(path)
        if path.stem == 'global_plans':
            count = int(prompt.split('Propose ')[1].split(' ')[0])
            value = dict(plans=[plan('CPU synthetic plan ' + str(i)) for i in range(count)])
        elif path.stem.startswith('identity_'):
            objects = json.loads(prompt.split('\nCANDIDATES: ')[1])
            value = dict(checks=[dict(key=o['key'], verdict='uncertain', observed_id=None,
                                     bbox_1000=None, reason='Synthetic fixture, no semantic evidence') for o in objects])
        else:
            return super().ask(image, prompt, path, **kwargs)
        self.requests.append(dict(path=str(path), prompt=prompt, **kwargs))
        path.write_text(json.dumps(value)); write(path.with_suffix('.json'), value)
        write(path.with_suffix('.request.json'), dict(prompt=prompt, image_delivered=image is not None, **kwargs))
        return value


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'research/audits', prefix='plan_repair_CPU_')
        self.addCleanup(self.tmp.cleanup); self.base = Path(self.tmp.name)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, CAUPANO_PROGRAM_CATALOG='', CAUPANO_LAYOUT_RGB='0'))
        self.stack.enter_context(patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight', return_value={}))

    def prepare(self, plans=2, worlds=2):
        source = self.base / 'source'; source.mkdir()
        suite = read(SOURCE / 'suite.json'); suite['scenes'] = suite['scenes'][:1]
        write(source / 'suite.json', suite)
        sid = suite['scenes'][0]['id']
        shutil.copytree(SOURCE / 'scenes' / sid / 'observation', source / 'scenes' / sid / 'observation')
        root = self.base / 'run'
        w.prepare(root, source, plans=plans, worlds=worlds, draws=4)
        return root, sid

    def test_real_five_partition_failures_become_optional_annotations_only(self):
        failures = [f for f in read(AUDIT)['failures'] if f['stage'] == 'blueprint']
        self.assertEqual(len(failures), 5)
        for failure in failures:
            case = load_case(FAILED / 'scenes' / failure['scene'] / 'observation', 'qwen')
            before = sha(case['state'])
            raw = read(failure['response'])
            # Replay the actual omitted/repeated reference sets, not ideal IDs.
            refs = [r for z in raw['zones'] for r in z['observed_ids']]
            value = plan(refs=refs)
            parsed = p.plan_answer(dict(plans=[value]), case, 1)[0]
            self.assertEqual(parsed['areas'][0]['observed_refs'], refs)
            self.assertEqual(sha(case['state']), before)
            self.assertNotIn('room_envelope', parsed)
            self.assertNotIn('floor_y_m', parsed)

    def test_unknown_refs_still_rejected_but_unassigned_observations_preserved(self):
        case = load_case(SOURCE / 'scenes' / read(SOURCE / 'suite.json')['scenes'][0]['id'] / 'observation', 'qwen')
        self.assertTrue(p.plan_answer(dict(plans=[plan()]), case, 1))
        with self.assertRaisesRegex(ValueError, 'Unknown observed_refs'):
            p.plan_answer(dict(plans=[plan(refs=['obs_999'])]), case, 1)

    def test_all_real_v13_valid_programs_survive_original_grammar_and_region_metadata(self):
        count = 0
        for path in sorted(OLD.glob('scenes/*/completion/proposals/qwen/programs.json')):
            case = read(path.parents[2] / 'shared/case.json')
            for saved in read(path).values():
                clean = p.candidate_answer(wire(saved), case, saved['hypothesis'])[0]
                again = original.revalidate_program(clean, case)
                self.assertEqual({o['key'] for g in saved['groups'] for o in g['objects']},
                                 {o['key'] for g in again['groups'] for o in g['objects']})
                self.assertEqual([g['layout_region'] for g in clean['groups']], [g['layout_region'] for g in again['groups']])
                count += 1
        self.assertEqual(count, 17)

    def test_actual_kitchen_hypothesis_host_is_not_accepted_as_an_object(self):
        f = next(f for f in read(AUDIT)['failures'] if f['stage'].startswith('inventory'))
        case = load_case(FAILED / 'scenes' / f['scene'] / 'observation', 'qwen')
        raw = read(f['response']); obj = next(o for o in raw['objects'] if o['support']['kind'] == 'top')
        self.assertEqual(obj['support']['host'], 'h0')
        candidate = dict(alternatives=[dict(description='Replay actual invalid support reference', relations=[],
            groups=[dict(function='cooking', reason='Actual failed proposal', observed_refs=[], new_objects=[dict(
                key=obj['id'], category=obj['category'], variant=obj['description'], reason=obj['reason'],
                placement=obj['placement'], attachment=obj['support'], affordances=[])])])])
        with self.assertRaisesRegex(ValueError, 'h0'):
            p.candidate_answer(candidate, case, original.profile('h0', ['cooking']))

    def test_architecture_not_added_as_furniture(self):
        case = load_case(SOURCE / 'scenes' / read(SOURCE / 'suite.json')['scenes'][0]['id'] / 'observation', 'qwen')
        candidate = dict(alternatives=[dict(description='Wrong layer', relations=[], groups=[dict(function='background',
            reason='Wrong layer', observed_refs=[], new_objects=[dict(key='new_wall', category='wall', variant='wall',
            reason='Guessed enclosure', placement='side', attachment=dict(kind='wall', host=None), affordances=[])])])])
        with self.assertRaisesRegex(ValueError, 'architectural surface'):
            p.candidate_answer(candidate, case, original.profile('h0', []))

    def test_identity_requires_evidence_same_category_and_uncertainty_are_not_vetoes(self):
        case = dict(context=[dict(id='obs_000')]); program = dict(groups=[dict(objects=[dict(key='new_chair')])])
        check = dict(key='new_chair', verdict='same_visible_instance', reason='same category', observed_id=None, bbox_1000=None)
        with self.assertRaisesRegex(ValueError, 'requires observed_id or bbox_1000'):
            p.identity_answer(dict(checks=[check]), program, case)
        check['verdict'] = 'uncertain'
        self.assertEqual(p.identity_answer(dict(checks=[check]), program, case)[0]['verdict'], 'uncertain')
        check.update(verdict='same_visible_instance', observed_id='obs_000', reason='Same photographed chair at marked location')
        self.assertEqual(p.identity_answer(dict(checks=[check]), program, case)[0]['observed_id'], 'obs_000')

    def test_bound_cache_and_failure_resume_do_not_call_provider(self):
        image = self.base / 'rgb.png'; Image.new('RGB', (10, 10)).save(image)
        class Provider:
            def __init__(self): self.calls = 0
            def ask(self, image, prompt, path, **kw):
                self.calls += 1; path.write_text('{}'); write(path.with_suffix('.json'), {})
                return {}
        provider = Provider(); request = p.Requests(self.base / 'calls', 9, image, lambda: provider)
        request.ask('valid', 'valid request', lambda v: ['normalized'])
        self.assertEqual(provider.calls, 1)
        self.assertEqual(request.ask('valid', 'valid request', lambda v: self.fail('validator should not rerun model')), ['normalized'])
        def reject(v): raise ValueError('specific instance h0 is not a host')
        with self.assertRaisesRegex(ValueError, 'specific instance h0'):
            request.ask('invalid', 'invalid request', reject)
        self.assertEqual(provider.calls, 3)
        with self.assertRaisesRegex(p.RecordedFailure, 'specific instance h0'):
            request.ask('invalid', 'invalid request', reject)
        self.assertEqual(provider.calls, 3)
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            request.ask('valid', 'modified request', lambda v: v)

    def test_end_to_end_original_executor_with_separate_plans_and_repeatable_report(self):
        root, sid = self.prepare()
        proposal = PlanningCPUProvider(); executor = CPUProvider(all_present=True)
        def factory(config, backend):
            self.assertEqual(backend, 'qwen')
            return executor
        with patch.object(w, 'provider', return_value=proposal), patch.object(pipeline, 'provider', side_effect=factory):
            for name in w.PHASES:
                w.phase(root, name)
        for b in ('plan_01', 'plan_02'):
            branch = root / 'branches' / b
            comp = branch / 'scenes' / sid / 'completion'
            domain = read(comp / 'shared/domain.json')
            self.assertEqual(set(domain['proposal_provenance']), {'qwen'})
            self.assertEqual(read(comp / 'proposals/qwen/global_plan.json')['id'], b)
            self.assertEqual({pr['proposal_source']['plan'] for fid, pr in domain['programs'].items() if fid != 'f_null'}, {b})
            observed = read(branch / 'scenes' / sid / 'observation/qwen/observer3d/G_obs.json')['objects']
            for folder in (comp / 'end_to_end/qwen/completion').glob('world_*'):
                full = read(folder / 'G_full.json')
                self.assertEqual(full['objects'][:len(observed)], observed)
                self.assertGreaterEqual(len(full['objects']), len(observed))
                self.assertTrue(read(folder / 'numeric_review.json')['geometry_contract_passed'])
            self.assertTrue((comp / 'end_to_end/qwen/completion/causal_model.json').exists())
            self.assertTrue((comp / 'end_to_end/qwen/completion/world_01/host_interventions.json').exists())
        result = w.report(root)
        self.assertEqual(result['status'], 'completed_needs_review', result)
        # Original smoothing + fixed IID noise can yield a null draw even with
        # all-positive fixture votes. Keep it; never resample to satisfy a test.
        self.assertEqual(result['counts'], {'exported_with_new_objects': 3, 'observed_only_sample': 1})
        self.assertFalse(result['execution_errors'])
        with patch.object(w, 'provider', side_effect=AssertionError('resume must not load model')), \
             patch.object(pipeline, 'provider', side_effect=AssertionError('resume must not load model')):
            for name in w.PHASES: w.phase(root, name)
            again = w.report(root)
        self.assertEqual(again['counts'], result['counts'])
        self.assertEqual(again['status'], result['status'])
        self.assertTrue((root / 'branches/plan_01/report' / sid / 'world_01/views/scene_3d.html').exists())
        class Links(HTMLParser):
            def handle_starttag(inner, tag, attrs):
                for name, value in attrs:
                    if name in ('href', 'src') and value and not value.startswith(('http:', 'https:', 'data:', '#', 'javascript:')):
                        self.assertTrue((page.parent / value.split('#')[0]).exists(), (page, value))
        for page in [root / 'index.html'] + list((root / 'branches').glob('*/report/**/*.html')):
            Links().feed(page.read_text())
        self.assertEqual(pinned.verify()['source_files'], 118)

    def test_all_rejected_candidate_supply_is_not_sampled_empty_world(self):
        root, sid = self.prepare(plans=1, worlds=1)
        class VisibleOnly(PlanningCPUProvider):
            def ask(self, image, prompt, path, **kw):
                if Path(path).stem.startswith('identity_'):
                    value = dict(checks=[dict(key=o['key'], verdict='same_visible_instance', observed_id='obs_000',
                        bbox_1000=None, reason='CPU synthetic identity-conflict fixture') for o in json.loads(prompt.split('\nCANDIDATES: ')[1])])
                    path.write_text(json.dumps(value)); write(path.with_suffix('.json'), value); return value
                return super().ask(image, prompt, path, **kw)
        with patch.object(w, 'provider', return_value=VisibleOnly()):
            for name in w.PHASES: w.phase(root, name)
        result = w.report(root)
        self.assertEqual(result['status'], 'completion_incomplete')
        self.assertEqual(result['counts'], {'empty_candidate_domain': 1})
        self.assertFalse(result['execution_errors'])
        comp = root / 'branches/plan_01/scenes' / sid / 'completion'
        self.assertFalse((comp / 'shared/domain.json').exists())
        rejected = read(comp / 'proposals/qwen/rejected_programs.json')
        self.assertTrue(rejected)
        self.assertTrue(rejected[0]['program']['groups'][0]['objects'])

    def test_real_numeric_rejection_does_not_masquerade_as_request_error(self):
        matched = 0
        for path in OLD.glob('scenes/*/completion/end_to_end/qwen/completion/summary.json'):
            outcome = read(path)
            if outcome.get('failures'):
                self.assertEqual(w.execution_failures(outcome, path.parent), [])
                modified = deepcopy(outcome); modified['failures'][0]['error'] = 'ValueError: model returned invalid JSON'
                self.assertTrue(w.execution_failures(modified, path.parent)); matched += 1
        self.assertEqual(matched, 1)


if __name__ == '__main__':
    unittest.main()
