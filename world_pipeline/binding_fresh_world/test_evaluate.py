"""CPU: read-only richer sampling probe and independent probability check."""
import math
import unittest
import torch
from .. import oracle_test
from ..id_binding_world.runtime import WorldBranch
from .evaluate import SamplingProbe,attention_stats


class EvaluationTests(unittest.TestCase):
    def test_instance_category_and_global_statistics_match_direct_probabilities(self):
        q=torch.tensor([[[[1.,0.],[0.,1.]]]])
        k=torch.tensor([[[[0.,0.],[5.,0.],[0.,5.],[4.,0.]]]])
        mask=torch.tensor([[True,True,True,False]])
        labels=(torch.tensor([5,9]),torch.tensor([0,1,0,1]),torch.tensor([1,2,2,1]),
            torch.tensor([.5,.5,0.,0.]),torch.tensor([0.,0.,.5,.5]))
        categories=torch.tensor([[0,3,3,22]])
        actual=attention_stats(q,k,mask,labels,categories)
        a=(q[0,0]@k[0,0].T/math.sqrt(2)).masked_fill(~mask[0],-torch.inf).softmax(-1)
        self.assertAlmostEqual(actual['positive_target_attention_mean'],float((a[0,1]+a[1,2])/2),places=7)
        self.assertAlmostEqual(actual['positive_global_attention_mean'],float(a[:,0].mean()),places=7)
        self.assertAlmostEqual(actual['negative_target_attention_mean'],float((a[0,2]+a[1,1])/2),places=7)
        self.assertEqual(actual['positive_correct_instance_token_top1'],1.)
        # Wrong same-class slots: category-token agreement is NOT instance agreement.
        switched=(*labels[:2],torch.tensor([2,1,1,2]),*labels[3:])
        other=attention_stats(q,k,mask,switched,categories)
        self.assertEqual(other['positive_correct_instance_token_top1'],0.)
        self.assertEqual(other['positive_correct_category_token_top1'],1.)

    def test_actual_tiny_edit_probe_does_not_change_generated_prediction(self):
        torch.set_num_threads(2);model=oracle_test.model(False)
        torch.manual_seed(61);world=WorldBranch(32,(0,2),16,binding_block=2)
        for branch in world.branches.values():branch.gate.data.fill_(.1)
        world.attach(model);state=oracle_test.tensorize(oracle_test.graph(),3);kwargs=oracle_test.inputs()
        labels=(torch.tensor([1,2,11,18]),torch.tensor([0,1,2,3,0,1,2,3]),torch.tensor([1,1,2,2,2,2,1,1]),
            torch.tensor([.25]*4+[0.]*4),torch.tensor([0.]*4+[.25]*4))
        with torch.inference_mode():
            world.set_world(state);expected=model(**kwargs)[0]
            probe=SamplingProbe(world);probe.reset(labels,state[0]);world.clear_context();world.set_world(state)
            observed=model(**kwargs)[0]
            torch.testing.assert_close(expected,observed,rtol=0,atol=0)
            self.assertEqual(probe.calls,1);self.assertEqual(len(probe.records),1)
            row=probe.records[0]
            for k in ('positive_target_attention_mean','positive_global_attention_mean',
                      'positive_correct_instance_token_top1','positive_correct_category_token_top1',
                      'negative_target_attention_mean'):
                self.assertTrue(0<=row[k]<=1)
            self.assertAlmostEqual(sum(row[k] for k in ('positive_target_attention_mean',
                'positive_global_attention_mean','positive_other_object_attention_mean')),1.,places=6)
            probe.reset(None);world.clear_context();world.set_world(state)
            torch.testing.assert_close(expected,model(**kwargs)[0],rtol=0,atol=0)
            self.assertFalse(probe.records);probe.close()


if __name__=='__main__':unittest.main()
