"""CPU actual-Q/K observer must not alter tiny Edit inference."""
import unittest
import torch
from .. import oracle_test
from ..id_binding_world.runtime import WorldBranch
from .evaluate import SamplingProbe
from .supervision import binding_terms


class EvaluationTests(unittest.TestCase):
    def test_sampling_probe_matches_actual_qk_and_never_changes_output(self):
        torch.set_num_threads(2);model=oracle_test.model(False)
        torch.manual_seed(61);world=WorldBranch(32,(0,2),16,binding_block=2)
        for b in world.branches.values():b.gate.data.fill_(.1)
        world.attach(model);graph=oracle_test.tensorize(oracle_test.graph(),3);kwargs=oracle_test.inputs()
        labels=(torch.tensor([1,2,11,18]),torch.tensor([0,1,2,3,0,1,2,3]),torch.tensor([1,1,2,2,2,2,1,1]),
            torch.tensor([.25]*4+[0.]*4),torch.tensor([0.]*4+[.25]*4))
        with torch.inference_mode():
            world.set_world(graph);expected=model(**kwargs)[0]
            probe=SamplingProbe(world);probe.reset(labels)
            saved={}
            def collect(name):
                def hook(module,args,output):saved[name]=output.detach().float()
                return hook
            hooks=[getattr(world.branches['2'],k).register_forward_hook(collect(k)) for k in ('q','k')]
            world.clear_context();world.set_world(graph);observed=model(**kwargs)[0]
            torch.testing.assert_close(expected,observed,rtol=0,atol=0)
            self.assertEqual(probe.calls,1);self.assertEqual(len(probe.records),1)
            def heads(x):return x.reshape(1,-1,8,2).transpose(1,2)
            coverage,concentration=binding_terms(heads(saved['q']),heads(saved['k']),graph[2],labels)
            self.assertEqual(float(coverage),probe.records[0]['coverage'])
            self.assertEqual(float(concentration),probe.records[0]['concentration'])
            for h in hooks:h.remove()
            probe.reset(None);world.clear_context();world.set_world(graph);plain=model(**kwargs)[0]
            torch.testing.assert_close(expected,plain,rtol=0,atol=0);self.assertEqual(probe.calls,0)
            probe.close()


if __name__=='__main__':unittest.main()
