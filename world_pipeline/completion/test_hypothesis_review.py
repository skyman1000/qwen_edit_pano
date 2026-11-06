"""CPU checks of assessment accounting and evidence preservation, not model quality."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from ..common import read, sha, write
from .hypothesis_review import prompt, run, validate_assessment
from .semantic_probe import state_context
from .test_content_review import plan
from .test_semantic_probe import state


def assessment(count=3):
    return dict(reviews=[dict(id=f'new_{i}', existence='plausible',
        existence_reason='An optional functional object; not a claim that it is observed.',
        placement='underdetermined', placement_reason='The proposed wall is outside the view.',
        evidence_ids=[]) for i in range(count)])


class HypothesisReviewTests(unittest.TestCase):
    def test_unknown_or_conflicting_placement_does_not_remove_existence_hypothesis(self):
        original = plan(3); case = dict(context=[]); answer = assessment()
        answer['reviews'][1]['placement'] = 'conflict'
        answer['reviews'][1]['placement_reason'] = 'The specified surface is visibly occupied.'
        before = deepcopy((original, case, answer))
        result = validate_assessment(answer, original, case)
        self.assertEqual((original, case, answer), before)
        self.assertEqual(len(result['reviews']), 3)
        self.assertEqual([r['existence'] for r in result['reviews']], ['plausible']*3)
        result['reviews'][0]['existence'] = 'unresolved'
        self.assertEqual(answer, before[2])

    def test_rejects_observation_targets_missing_duplicates_and_foreign_ids(self):
        for failure in ('observations', 'missing', 'duplicate', 'foreign'):
            answer = assessment(); observed = [dict(id=f'observed_{i:04d}') for i in range(6)]
            if failure == 'observations':
                answer = assessment(6)
                for item, obj in zip(answer['reviews'], observed):item['id'] = obj['id']
            elif failure == 'missing':answer['reviews'].pop()
            elif failure == 'duplicate':answer['reviews'][1]['id'] = 'new_0'
            else:answer['reviews'][0]['id'] = 'foreign'
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                validate_assessment(answer, plan(3), dict(context=observed))

    def test_rejects_identity_changes_geometry_and_mutating_decisions(self):
        for field, value in [('name','replacement'), ('center',[0,0,-2]), ('size',[1,1,1]),
                             ('replacement_placement','Behind the camera.'), ('decision','withdraw')]:
            answer = assessment();answer['reviews'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_assessment(answer, plan(3), dict(context=[]))

    def test_invalid_judgments_missing_reasons_and_bad_evidence_are_rejected(self):
        for field, value in [('existence','retain'), ('placement','relocate'), ('existence_reason',' '),
                             ('placement_reason',None), ('evidence_ids',['new_0']),
                             ('evidence_ids',['observed_0000','observed_0000'])]:
            answer = assessment();answer['reviews'][0][field] = value
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                validate_assessment(answer, plan(3), dict(context=[dict(id='observed_0000')]))

    def test_no_forced_positive_assessment_or_minimum_accepted_count(self):
        answer = assessment()
        for item, judgment in zip(answer['reviews'], ('incompatible','outside_scope','unresolved')):
            item['existence'] = judgment
        self.assertEqual(validate_assessment(answer, plan(3), dict(context=[])), answer)

    def test_run_preserves_sources_candidates_and_raw_response_without_world_export(self):
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); image = root/'rgb.png'; Image.new('RGB',(64,64)).save(image)
                graph_path = root/'observed.json'; observed = state();write(graph_path, observed)
                case = dict(id='case', image=str(image), state=str(graph_path), context=state_context(observed))
                entry = dict(case=case, plan=plan(3), source_sha256={str(image):sha(image),str(graph_path):sha(graph_path)})
                before = deepcopy(entry); answer = assessment()
                if not valid:answer['reviews'][0]['id'] = case['context'][0]['id']
                class Model:
                    calls = 0
                    def ask(self, image, question, path, **settings):
                        self.calls += 1
                        self.settings = settings
                        path.write_text(json.dumps(answer));write(path.with_suffix('.json'), answer)
                        return answer
                model = Model(); output = root/'output';output.mkdir()
                self.assertEqual(run(model, entry, output), valid)
                self.assertEqual(model.calls, 1)
                self.assertEqual(model.settings, dict(seed=3407, sample=True, profile='official',
                                                     max_tokens=4096, guard_repetition=True))
                self.assertEqual(entry, before)
                self.assertTrue(all(sha(p)==h for p,h in entry['source_sha256'].items()))
                folder = output/'case'
                self.assertEqual(read(folder/'original_plan.json'), entry['plan'])
                self.assertEqual(read(folder/'G_obs.json'), observed)
                self.assertEqual(read(folder/'assessment.json'), answer)
                self.assertEqual((folder/'assessment_validated.json').exists(), valid)
                summary = read(output/'summary.json')
                self.assertFalse(summary['semantics_verified'])
                self.assertFalse(summary['geometry_generated'])
                self.assertFalse(summary['plan_modified'])
                self.assertFalse((folder/'semantic_plan.json').exists())
                self.assertFalse((folder/'G_full.json').exists())

    def test_prompt_has_explicit_targets_and_no_case_specific_answers(self):
        question = prompt(dict(context=[]), plan(3))
        self.assertIn('REFERENCE_ONLY_OBSERVATIONS', question)
        self.assertIn('OUTPUT_TARGET_IDS_ONLY: ["new_0", "new_1", "new_2"]', question)
        self.assertNotIn('sauna', question)
        self.assertNotIn('addition_0001', question)


if __name__ == '__main__':
    unittest.main()
