"""CPU regressions for the actual tiny Edit path, labels, RNG and checkpoint aux."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from .. import oracle_test
from ..oracle_model import TargetAttention
from ..fourier_world.runtime import WorldBranch as FourierBranch
from ..oracle_memory import ActivationStorage,training_autocast
from ..scaled_world.runtime import seed_step
from ..object_world.data import tensorize
from .runtime import WorldBranch,BindingAttention,object_region_loss
from .supervision import sample_binding,attention_coverage_loss,POLICY


class ABTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def world(self):
        torch.manual_seed(72)
        w=WorldBranch(32,(0,2),16,binding_block=2)
        for branch in w.branches.values():branch.gate.data.fill_(.1)
        return w

    def labels(self):
        return (torch.tensor([1,2,11,18]),torch.arange(4),torch.tensor([1,1,2,2]),torch.full((4,),.25))

    def test_aux_reads_real_qk_and_never_changes_forward(self):
        torch.manual_seed(4);old=TargetAttention(32,16);new=BindingAttention(32,16)
        new.load_state_dict(old.state_dict());new.gate.data.fill_(.1);old.gate.data.fill_(.1)
        h=torch.randn(1,44,32);tokens=torch.randn(1,3,16);mask=torch.ones(1,3,dtype=torch.bool)
        with training_autocast('cpu'):
            expected=old(h,tokens,mask,40)
            without,_=new(h,tokens,mask,40)
            with_loss,loss=new(h,tokens,mask,40,self.labels())
        torch.testing.assert_close(expected,without,rtol=0,atol=0)
        torch.testing.assert_close(without,with_loss,rtol=0,atol=0)
        self.assertGreater(float(loss),0.)
        loss.backward()
        self.assertGreater(float(new.q.weight.grad.abs().sum()),0.)
        self.assertGreater(float(new.k.weight.grad.abs().sum()),0.)
        self.assertIsNone(new.v.weight.grad)

    def test_correct_attention_has_lower_loss_and_permutation_is_neutral(self):
        q=torch.tensor([[[[1.,0.],[0.,1.]]]])
        k=torch.tensor([[[[0.,0.],[8.,0.],[0.,8.]]]],requires_grad=True)
        mask=torch.ones(1,3,dtype=torch.bool)
        s=(torch.arange(2),torch.arange(2),torch.tensor([1,2]),torch.full((2,),.5))
        right=attention_coverage_loss(q,k,mask,s)
        wrong=attention_coverage_loss(q,k,mask,(*s[:2],torch.tensor([2,1]),s[-1]))
        self.assertLess(float(right),float(wrong))
        perm=torch.tensor([0,2,1])
        same=attention_coverage_loss(q,k[:,:,perm],mask[:,perm],(*s[:2],torch.tensor([2,1]),s[-1]))
        torch.testing.assert_close(right,same,rtol=0,atol=0)
        right.backward();self.assertTrue(torch.isfinite(k.grad).all())

    def test_empty_support_is_finite_zero_and_global_is_not_a_label(self):
        q=torch.randn(1,2,3,4,requires_grad=True);k=torch.randn(1,2,3,4,requires_grad=True)
        empty=(torch.empty(0,dtype=torch.long),)*3+(torch.empty(0),)
        loss=attention_coverage_loss(q,k,torch.ones(1,3,dtype=torch.bool),empty)
        self.assertEqual(float(loss),0.);loss.backward();self.assertTrue(torch.isfinite(q.grad).all())
        with self.assertRaises(ValueError):
            attention_coverage_loss(q,k,torch.ones(1,3,dtype=torch.bool),
                (torch.tensor([0]),torch.tensor([0]),torch.tensor([0]),torch.ones(1)))

    def test_labels_keep_same_class_instances_and_do_not_consume_global_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mask.npz'
            ids=np.ones((8,16),dtype=np.int32);ids[:,8:]=2
            np.savez(path,instance=ids,instance_valid=np.ones_like(ids,dtype=bool),object_loss_valid=np.ones_like(ids,dtype=bool))
            row=dict(objects=[dict(instance_id='1',category_id=3),dict(instance_id='2',category_id=3)],
                object_region_supervision=dict(path=str(path)))
            seed_step(7,2);torch_state=torch.get_rng_state().clone();np.random.seed(12);numpy_state=np.random.get_state()
            labels,meta=sample_binding(row,64,1,117)
            torch.testing.assert_close(torch_state,torch.get_rng_state(),rtol=0,atol=0)
            self.assertTrue(np.array_equal(numpy_state[1],np.random.get_state()[1]))
            self.assertEqual(meta['selected_instances'],2);self.assertEqual(set(labels[2].tolist()),{1,2})
            xs=(labels[0][labels[1]]%10)-1
            self.assertTrue(torch.all(xs[labels[2]==1]<4));self.assertTrue(torch.all(xs[labels[2]==2]>=4))
            reversed_row=dict(row,objects=row['objects'][::-1]);other,_=sample_binding(reversed_row,64,1,117)
            for a,b in zip(labels[:2],other[:2]):torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(3-labels[2],other[2],rtol=0,atol=0)

    def test_fractional_subcell_support_is_sampled_without_binarization(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mask.npz';ids=np.zeros((32,64),dtype=np.int32)
            ids[::4,:]=1  # 25% coverage in each affected query cell; total >64 px.
            valid=ids>0
            np.savez(path,instance=ids,instance_valid=valid,object_loss_valid=valid)
            row=dict(objects=[dict(instance_id='1',category_id=22)],object_region_supervision=dict(path=str(path)))
            labels,meta=sample_binding(row,64,1,0)
            self.assertEqual(meta['selected_instances'],1);self.assertEqual(len(labels[2]),8)

    def test_full_objective_checkpoint_offload_and_cleared_runtime_match(self):
        from qwen_edit_pano.losses import panorama_loss
        from qwen_pano.profiles import REPO_PANORAMA
        from diffusers import QwenImageEditPlusPipeline as Pipeline
        kwargs=oracle_test.inputs();graph=oracle_test.tensorize(oracle_test.graph(),3)
        records=[]
        for checkpointed,offload in ((False,False),(True,False),(True,True)):
            model=oracle_test.model(checkpointed);w=self.world();w.attach(model)
            storage=ActivationStorage(enabled=offload,device_type='cpu')
            seed_step(21,0)
            with storage.context():
                w.set_supervision(self.labels());w.set_world(graph)
                out=model(**kwargs)[0][:,:32]
                aux=w.take_binding_loss()
                pred=Pipeline._unpack_latents(out,64,128,8)
                target=torch.randn_like(pred);noise=torch.randn_like(pred)
                args=SimpleNamespace(**REPO_PANORAMA)
                global_loss,_=panorama_loss(pred,target,torch.ones(1,1,8,16,dtype=torch.bool),
                    torch.ones(1,dtype=torch.bool),torch.ones(1),args.geometry_start_epoch,args,noise=noise,clean=noise-target)
                loss=global_loss+.25*object_region_loss(pred,target,torch.ones(8,16))+.01*aux
                # A deliberately different runtime state before recomputation.
                w.set_world(oracle_test.tensorize(oracle_test.graph(),3,'constant'))
                w.set_supervision(tuple(t.clone()*0 for t in self.labels()));w.clear_context()
                loss.backward()
            self.assertFalse(w._losses)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertGreater(float(w.encoder.category.weight.grad.abs().sum()),0.)
            records.append((out.detach(),float(loss),{k:p.grad.clone() for k,p in w.named_parameters() if p.grad is not None}))
        for out,loss,grads in records[1:]:
            torch.testing.assert_close(out,records[0][0],rtol=0,atol=0)
            self.assertEqual(loss,records[0][1]);self.assertEqual(grads.keys(),records[0][2].keys())
            for k in grads:torch.testing.assert_close(grads[k],records[0][2][k],rtol=0,atol=0)

    def test_disabled_aux_matches_original_fourier_forward_and_gradients(self):
        results=[];state=self.world().state_dict()
        for cls in (FourierBranch,WorldBranch):
            model=oracle_test.model(True)
            w=cls(32,(0,2),16,**({'binding_block':2} if cls is WorldBranch else {}));w.load_state_dict(state);w.attach(model)
            w.set_world(oracle_test.tensorize(oracle_test.graph(),3))
            out=model(**oracle_test.inputs())[0]
            w.set_world(None);out.square().mean().backward()
            results.append((out.detach(),{k:p.grad.clone() for k,p in w.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(results[0][0],results[1][0],rtol=0,atol=0)
        self.assertEqual(results[0][1].keys(),results[1][1].keys())
        for k in results[0][1]:torch.testing.assert_close(results[0][1][k],results[1][1][k],rtol=0,atol=0)

    def test_a_b_initialization_and_noise_stream_are_identical(self):
        a=self.world();b=self.world()
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        # Binding itself is deterministic and contains no dropout/random operations.
        q=torch.randn(1,8,40,2);k=torch.randn(1,8,3,2);mask=torch.ones(1,3,dtype=torch.bool)
        before=torch.get_rng_state().clone();attention_coverage_loss(q,k,mask,self.labels())
        torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
