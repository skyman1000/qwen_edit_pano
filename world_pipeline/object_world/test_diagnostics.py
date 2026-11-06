"""CPU checks for condition interventions, passive hooks and checkpoint parity."""
import unittest
import json
import tempfile
from pathlib import Path
import torch

from .. import oracle_test
from .runtime import WorldBranch
from .diagnostic_features import condition_variant,set_condition,AttentionProbe,compare_gradients,MODES
from .diagnose import backward_check,resume_records


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def tensors(self):return oracle_test.tensorize(oracle_test.graph(),3)

    def world(self):
        torch.manual_seed(72)
        world=WorldBranch(32,torch.randn(42,12),(0,2),16)
        for branch in world.branches.values():branch.gate.data.fill_(.1)
        return world

    def test_interventions_preserve_inputs_masks_units_and_validity(self):
        original=self.tensors();saved=tuple(t.clone() for t in original)
        for mode in MODES:
            out=condition_variant(original,mode,range(3,41))
            if out is not None:
                self.assertTrue(out[2].any())
                self.assertEqual(out[0].shape,original[0].shape)
        for x,y in zip(original,saved):torch.testing.assert_close(x,y,rtol=0,atol=0)
        c,g,m=condition_variant(original,'yaw90',range(3,41))
        torch.testing.assert_close(g[0,2,:6],torch.tensor([3.,2.,-1.,3.,2.,1.]))
        torch.testing.assert_close(g[...,6:],original[1][...,6:])
        torch.testing.assert_close(c,original[0]);torch.testing.assert_close(m,original[2])
        changed=condition_variant(original,'class_cycle',range(3,41))
        torch.testing.assert_close(changed[1],original[1]);torch.testing.assert_close(changed[2],original[2])
        self.assertTrue(torch.all(changed[0][original[2]][1:]!=original[0][original[2]][1:]))

    def test_passive_attention_probe_does_not_change_output_or_gradients(self):
        model=oracle_test.model(True);world=self.world();world.attach(model)
        inputs=oracle_test.inputs();tensors=self.tensors()
        set_condition(world,tensors,'correct',range(3,41));first=model(**inputs)[0]
        first.square().mean().backward()
        grads={k:p.grad.clone() for k,p in world.named_parameters() if p.grad is not None}
        world.zero_grad(set_to_none=True)
        probe=AttentionProbe(world);probe.active=True
        set_condition(world,tensors,'correct',range(3,41));second=model(**inputs)[0]
        world.set_world(None);second.square().mean().backward()
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        for k,p in world.named_parameters():
            if p.grad is not None:torch.testing.assert_close(grads[k],p.grad,rtol=0,atol=0)
        self.assertEqual(set(probe.records),{'0','2'})
        for r in probe.records.values():
            self.assertGreaterEqual(r['global_attention_mass'],0.)
            self.assertLessEqual(r['global_attention_mass'],1.)

    def test_order_is_neutral_and_mean_and_global_differ(self):
        model=oracle_test.model(False);world=self.world();world.attach(model)
        with torch.no_grad():
            set_condition(world,self.tensors(),'correct',range(3,41));first=model(**oracle_test.inputs())[0]
            set_condition(world,self.tensors(),'order_permute',range(3,41));second=model(**oracle_test.inputs())[0]
        torch.testing.assert_close(first,second,rtol=1e-5,atol=1e-6)
        set_condition(world,self.tensors(),'mean_objects',range(3,41));tokens,mask=world.current
        torch.testing.assert_close(tokens[0,1],tokens[0,2],rtol=0,atol=0)
        set_condition(world,self.tensors(),'global_only',range(3,41));self.assertEqual(int(world.current[1].sum()),1)

    def test_full_loss_checkpoint_gradient_comparison(self):
        model=oracle_test.model(True);world=self.world();world.attach(model)
        inputs=oracle_test.inputs();target=torch.randn(1,4,1,8,16)
        noise=torch.randn_like(target);clean=noise-target
        config=dict(pack_plan=dict(height=64,names={'3':'chair','4':'door'}),region_weight=.25)
        wrapper=model._gradient_checkpointing_func
        result=backward_check(model,world,inputs,self.tensors(),target,torch.ones(8,16),config,42,noise,clean,mode='parity')
        self.assertIs(model._gradient_checkpointing_func,wrapper)
        self.assertEqual(result['relative_l2_difference'],0.)
        self.assertEqual(result['loss_without_checkpoint'],result['loss_with_checkpoint'])
        self.assertTrue(result['base_frozen'])

    def test_48gb_default_never_disables_checkpoint(self):
        model=oracle_test.model(True);world=self.world();world.attach(model)
        target=torch.randn(1,4,1,8,16);noise=torch.randn_like(target)
        config=dict(pack_plan=dict(height=64,names={'3':'chair','4':'door'}),region_weight=.25)
        events=[]
        result=backward_check(model,world,oracle_test.inputs(),self.tensors(),target,torch.ones(8,16),config,42,
            noise,noise-target,memory_callback=lambda event,**kwargs:events.append((event,kwargs)))
        starts=[value['checkpointed'] for event,value in events if event=='backward_start']
        self.assertEqual(starts,[True])
        self.assertTrue(result['encoder_gradient_nonzero']);self.assertTrue(result['semantic_gradient_nonzero'])
        self.assertFalse(result['checkpoint_parity_tested'])
        self.assertNotIn('loss_without_checkpoint',result)

    def test_resume_accepts_old_memory_policy_without_overwriting_records(self):
        budget=dict(samples=2,sigmas=[.2,.5,.8],checkpoint_steps=[250,500],modes=list(MODES),
            cases=[dict(id='a',seen_by_step500=True),dict(id='b',seen_by_step500=False)],scope='mechanism test')
        previous=dict(budget,backward_comparison_passes=2)
        budget.update(backward_mode='checkpointed',backward_passes=1)
        row=dict(step=250,id='a',scheduler_index=800,mode='correct',seen_by_step500=True,sigma=.2,
            flow=.1,object_region=.2,flow_plus_region=.15,relative_prediction_difference_from_correct=0.)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder);(output/'PLAN.json').write_text(json.dumps(previous))
            path=output/'results.jsonl';path.write_text(json.dumps(row)+'\n');original=path.read_bytes()
            saved=resume_records(output,budget)
            self.assertEqual(len(saved),1);self.assertEqual(path.read_bytes(),original)
            with self.assertRaises(ValueError):resume_records(output,dict(budget,sigmas=[.1,.5,.8]))
            path.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaises(ValueError):resume_records(output,budget)


if __name__=='__main__':unittest.main()
