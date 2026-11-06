"""Mathematical inference tests only; no invented scene probabilities in production."""
import unittest

from .causal_probability import CausalGraph,fit_cpts


def node(name,parents,probabilities):
    import itertools
    return dict(id=name,states=['no','yes'],parents=parents,cpt=[
        dict(when=list(values),probabilities={'no':1-p,'yes':p})
        for values,p in zip(itertools.product(['no','yes'],repeat=len(parents)),probabilities)])


class CausalProbabilityTests(unittest.TestCase):
    def test_conditioning_and_intervening_are_different_under_common_cause(self):
        # Z -> X and Z -> Y; no X -> Y. All parameters are synthetic test values.
        model=CausalGraph(dict(parameter_source='synthetic_test_only',nodes=[
            node('context',[],[.5]),node('visible_object',['context'],[.1,.9]),
            node('hidden_object',['context'],[.1,.9])]))
        obs=model.query('hidden_object',{'visible_object':'yes'})
        effect=model.query('hidden_object',interventions={'visible_object':'yes'})
        self.assertAlmostEqual(obs['probabilities']['yes'],.82)
        self.assertAlmostEqual(effect['probabilities']['yes'],.5)
        self.assertAlmostEqual(model.query('context',{'visible_object':'yes'})['probabilities']['yes'],.9)

    def test_lamp_presence_is_not_lamp_operating_and_does_not_force_power(self):
        model=CausalGraph(dict(parameter_source='synthetic_test_only',nodes=[
            node('lamp_present',[],[.8]),node('power_available',[],[.6]),
            node('switch_on',[],[.5]),
            node('lamp_lit',['lamp_present','power_available','switch_on'],[0,0,0,0,0,0,0,1])]))
        self.assertAlmostEqual(model.query('power_available',{'lamp_present':'yes'})['probabilities']['yes'],.6)
        self.assertEqual(model.query('power_available',{'lamp_lit':'yes'})['probabilities']['yes'],1.)
        self.assertEqual(model.query('lamp_lit',interventions={'power_available':'no'})['probabilities']['yes'],0.)
        self.assertAlmostEqual(model.query('lamp_present',interventions={'power_available':'no'})['probabilities']['yes'],.8)

    def test_do_cuts_incoming_mechanism(self):
        model=CausalGraph(dict(nodes=[node('parent',[],[0.]),node('child',['parent'],[0.,1.])]))
        self.assertEqual(model.query('child',interventions={'child':'yes'})['probabilities']['yes'],1.)

    def test_impossible_evidence_and_unknown_nodes_rejected(self):
        model=CausalGraph(dict(nodes=[node('x',[],[0.])]))
        for kwargs in [dict(evidence={'x':'yes'}),dict(evidence={'missing':'yes'}),
                       dict(evidence={'x':'no'},interventions={'x':'yes'})]:
            with self.assertRaises(ValueError):model.query('x',**kwargs)

    def test_cpt_alone_is_not_an_individual_counterfactual_model(self):
        with self.assertRaisesRegex(NotImplementedError,'noise coupling'):
            CausalGraph(dict(nodes=[node('x',[],[.5])])).counterfactual()

    def test_parameters_can_be_fitted_without_claiming_graph_identification(self):
        graph=[dict(id='context',states=['no','yes'],parents=[]),
               dict(id='object',states=['no','yes'],parents=['context'])]
        spec=fit_cpts(graph,[{'context':'no','object':'no'},{'context':'yes','object':'yes'}])
        self.assertFalse(spec['identified_from_data'])
        self.assertAlmostEqual(CausalGraph(spec).query('object',{'context':'yes'})['probabilities']['yes'],2/3)
        with self.assertRaisesRegex(ValueError,'Missing training annotation'):
            fit_cpts(graph,[{'object':'yes'}])

    def test_cycle_incomplete_table_and_unnormalized_probabilities_rejected(self):
        for spec in [dict(nodes=[node('x',['x'],[.1,.2])]),
                     dict(nodes=[node('x',[],[.5]),node('y',['x'],[.1])]),
                     dict(nodes=[dict(id='x',states=['no','yes'],parents=[],
                                      cpt=[dict(when=[],probabilities={'no':.8,'yes':.8})])])]:
            with self.assertRaises(ValueError):CausalGraph(spec)


if __name__=='__main__':unittest.main()
