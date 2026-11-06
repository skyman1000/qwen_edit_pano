"""Finite visibility math and actual support-collapse regression; no inference."""
from pathlib import Path
import unittest

from .assembly_support_audit import empty_probability,hidden_count_posterior,inspect
from . import test_assembly_functional as functional_tests


class CountTests(unittest.TestCase):
    def test_partial_observation_keeps_positive_hidden_probability(self):
        result=hidden_count_posterior({0:.25,1:.45,2:.25,3:.05},1,.3)
        p=result['hidden_count_probabilities']
        self.assertAlmostEqual(sum(p.values()),1.)
        self.assertAlmostEqual(p[0],.135/(.135+.105+.02205))
        self.assertGreater(1-p[0],.4)
        full=hidden_count_posterior({0:.25,1:.45,2:.25,3:.05},1,1.)
        self.assertEqual(full['hidden_count_probabilities'][0],1.)

    def test_no_report_is_not_absence_when_detection_is_zero(self):
        result=hidden_count_posterior({0:.2,1:.8},0,0.)
        self.assertEqual(result['hidden_count_probabilities'],{0:.2,1:.8})
        with self.assertRaisesRegex(ValueError,'zero probability'):
            hidden_count_posterior({0:1.},1,.3)

    def test_saved_zero_support_is_deterministic_not_a_bad_random_seed(self):
        root=Path(__file__).resolve().parents[2]/'outputs/functional_completion_20261025_092354_2337884/posterior'
        if not root.is_dir():self.skipTest('Actual run unavailable')
        row=inspect(root)
        self.assertTrue(row['candidate_support_degenerate'])
        self.assertEqual(row['proposed_counts'],[0]*6)
        self.assertAlmostEqual(row['posterior_empty_probability'],1.)
        self.assertEqual(row['layout_requests'],0)


class EmptyProbabilityTests(unittest.TestCase):
    setUp=functional_tests.FunctionalTests.setUp
    initialize=functional_tests.FunctionalTests.initialize
    model=functional_tests.FunctionalTests.model

    def test_empty_probability_matches_finite_conditional_bundle_formula(self):
        model=self.model();expected=0.
        from .assembly_functional import role_posterior
        for row in role_posterior(model)['rows']:
            h=row['roles'];reading=.12+.66*int(h['reading'])
            storage=.12+.66*int(h['storage']);background=.12+.66*int(h['background'])
            expected+=row['probability']*(1-reading/2)*(1-storage)*(1-background/2)
        self.assertAlmostEqual(empty_probability(model),expected)


if __name__=='__main__':unittest.main()
