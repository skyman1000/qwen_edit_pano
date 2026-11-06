"""CPU tests using real tiny Edit blocks, LoRA, circular padding and checkpointing."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from .. import oracle_test
from ..oracle_memory import training_autocast, ActivationStorage
from ..grounded_world.continue_train import predict
from ..binding_refine_world.supervision import binding_terms
from .runtime import WorldBranch
from . import data as d
from .train import save_checkpoint


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setup(self, checkpointed=False):
        model = oracle_test.model(checkpointed).bfloat16()
        generator = torch.Generator().manual_seed(87)
        features = torch.randn(42, 32, generator=generator)
        torch.manual_seed(18)
        world = WorldBranch.from_model(model, features, blocks=(0,2), binding_block=2,
                                       rank=4, connector_rank=4, geometry_width=16)
        world.attach(model)
        kwargs = oracle_test.inputs()
        for key in ('hidden_states','encoder_hidden_states'):
            kwargs[key] = kwargs[key].bfloat16()
        state = (torch.tensor([[0,3,22]]),torch.tensor([[[0.]*8,[1.,2.,3.,1.,1.,1.,1.,1.],
                                                      [-1.,0.,-2.,1.,2.,1.,1.,1.]]]),
                 torch.ones(1,3,dtype=torch.bool))
        labels = (torch.tensor([1,2,11,18]),torch.tensor([0,1,2,3]),torch.tensor([1,2,2,1]),
                  torch.tensor([.5,.5,0.,0.]),torch.tensor([0.,0.,.5,.5]))
        return model,world,kwargs,state,labels

    def test_native_anchor_and_base_registration(self):
        model, world, _, state, _ = self.setup()
        features = torch.randn(42,32,generator=torch.Generator().manual_seed(87))
        with torch.no_grad(),training_autocast('cpu'):
            expected = model.txt_in(model.txt_norm(features.bfloat16())).float()
        torch.testing.assert_close(expected,world.encoder.semantic_base,rtol=0,atol=0)
        with training_autocast('cpu'):
            actual,_ = world.encoder(*state)
        torch.testing.assert_close(actual[:,1:],expected[state[0]][:,1:],rtol=0,atol=0)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertTrue(all(p.requires_grad and p.dtype==torch.float32 for p in world.parameters()))
        self.assertFalse(any('transformer_blocks' in key for key in world.state_dict()))
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in oracle_test.model(False).parameters()))

    def test_zero_gate_baseline_labels_and_two_step_gradients(self):
        model,world,kwargs,state,labels = self.setup(True)
        opt=torch.optim.AdamW(world.parameters(),lr=1e-3,eps=1e-6)
        with torch.no_grad(),training_autocast('cpu'):
            world.set_world(None);off=model(**kwargs)[0].clone()
            world.set_world(state);plain=model(**kwargs)[0].clone()
            world.set_supervision(labels);labelled=model(**kwargs)[0].clone()
            aux=world.take_binding_loss()
        torch.testing.assert_close(off,plain,rtol=0,atol=0)
        torch.testing.assert_close(plain,labelled,rtol=0,atol=0)
        self.assertTrue(bool((aux>0).all()))
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            with ActivationStorage(True,'cpu').context():
                with training_autocast('cpu'):
                    pred=predict(model,world,state,kwargs,64,labels)
                    terms=world.take_binding_loss()
                    loss=pred.float().square().mean()+.01*terms[0]+.001*terms[1]
                world.clear_context()
                loss.backward()
            for p in (world.encoder.semantic.up.weight,world.encoder.geometry[-1].weight,
                      world.branches['2'].key_delta.up.weight):
                self.assertIsNotNone(p.grad)
                self.assertGreater(float(p.grad.abs().sum()),0)
            if step==1:
                self.assertGreater(float(world.branches['2'].value_delta.up.weight.grad.abs().sum()),0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in world.parameters()))
            opt.step();self.assertEqual(d.optimizer_step(opt),step+1)
        with torch.no_grad(),training_autocast('cpu'):
            a=predict(model,world,state,kwargs,64,labels);world.take_binding_loss()
            b=predict(model,world,state,kwargs,64)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_checkpoint_replay_clear_context_and_cpu_storage_exact(self):
        results=[]
        for checkpointed,offload in ((False,False),(True,False),(True,True)):
            model,world,kwargs,state,labels=self.setup(checkpointed)
            for b in world.branches.values():b.gate.data.fill_(.12)
            with ActivationStorage(offload,'cpu').context():
                with training_autocast('cpu'):
                    pred=predict(model,world,state,kwargs,64,labels);aux=world.take_binding_loss()
                    loss=pred.float().square().mean()+.01*aux[0]+.001*aux[1]
                world.clear_context()
                loss.backward()
            results.append((pred.detach(),{n:p.grad.clone() for n,p in world.named_parameters()}))
            self.assertEqual(world._losses,[])
        for value, grads in results[1:]:
            torch.testing.assert_close(value,results[0][0],rtol=0,atol=0)
            for key,g in grads.items():
                torch.testing.assert_close(g,results[0][1][key],rtol=0,atol=0)

    def test_native_projections_auxiliary_and_reference_text_direct_outputs(self):
        model,world,_,_,labels=self.setup()
        block=model.transformer_blocks[2];attn=block.attn;processor=attn.processor
        image=torch.randn(1,44,32).bfloat16();text=torch.randn(1,5,32).bfloat16()
        objects=torch.randn(1,3,32).bfloat16();mask=torch.ones(1,3,dtype=torch.bool)
        world.branches['2'].gate.data.fill_(.2)
        with torch.no_grad(),training_autocast('cpu'):
            expected_image,expected_text=processor.original(attn,image,text)
            collected=[]
            actual_image,actual_text=processor(attn,image,text,
                native_layout_context=(objects,mask,40,labels),native_layout_losses=collected)
            def heads(x):return x.unflatten(-1,(attn.heads,-1))
            q=attn.norm_q(heads(attn.to_q(image[:,:40]))).transpose(1,2)
            k=attn.norm_added_k(heads(attn.add_k_proj(objects))).transpose(1,2)
            v=heads(attn.add_v_proj(objects)).transpose(1,2)
            attended=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:]).transpose(1,2).flatten(2)
            projected=attn.to_out[0](attended)-attn.to_out[0].bias.to(attended.dtype)
            expected=(expected_image[:,:40].float()+torch.tanh(world.branches['2'].gate)*projected.float()).to(image.dtype)
            terms=torch.stack(binding_terms(q,k,mask,labels))
        torch.testing.assert_close(actual_image[:,:40],expected,rtol=0,atol=0)
        torch.testing.assert_close(actual_image[:,40:],expected_image[:,40:],rtol=0,atol=0)
        torch.testing.assert_close(actual_text,expected_text,rtol=0,atol=0)
        torch.testing.assert_close(collected[0],terms,rtol=0,atol=0)

    def test_permutation_and_missing_geometry(self):
        model,world,kwargs,state,_=self.setup()
        for b in world.branches.values():b.gate.data.fill_(.1)
        perm=torch.tensor([0,2,1])
        other=tuple(t[:,perm] for t in state)
        with torch.no_grad(),training_autocast('cpu'):
            a=predict(model,world,state,kwargs,64)
            b=predict(model,world,other,kwargs,64)
        torch.testing.assert_close(a,b,rtol=.02,atol=.01)
        geometry=state[1].clone();geometry[:,:,6:]=0
        expected=world.encoder.geometry[0](geometry)
        geometry[:,:,:6]=999
        torch.testing.assert_close(world.encoder.geometry[0](geometry),expected,rtol=0,atol=0)

    def test_own_checkpoint_exact_adam_restore_and_foreign_rejection(self):
        model,world,kwargs,state,labels=self.setup()
        config=dict(version=d.VERSION,smoke=False)
        opt=torch.optim.AdamW(world.parameters(),lr=1e-3,eps=1e-6)
        def step(w,o,m):
            o.zero_grad(set_to_none=True)
            with training_autocast('cpu'):
                pred=predict(m,w,state,kwargs,64,labels);aux=w.take_binding_loss()
                loss=pred.float().square().mean()+.01*aux[0]+.001*aux[1]
            w.clear_context();loss.backward();o.step()
        step(world,opt,model)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run-1';out.mkdir()
            path=save_checkpoint(out,world,opt,config,1)
            model2,world2,_,_,_=self.setup()
            opt2=torch.optim.AdamW(world2.parameters(),lr=1e-3,eps=1e-6)
            self.assertEqual(d.resume_step(path,config),1)
            d.restore(world2,opt2,path)
            step(world,opt,model);step(world2,opt2,model2)
            for key,value in world.state_dict().items():
                torch.testing.assert_close(value,world2.state_dict()[key],rtol=0,atol=0)
            for v1,v2 in zip(opt.state.values(),opt2.state.values()):
                for key in v1:torch.testing.assert_close(v1[key],v2[key],rtol=0,atol=0)
            with self.assertRaises(ValueError):d.resume_step(path,dict(config,version='old_A'))

    def test_two_rank_local_means_equal_global_batch_gradients(self):
        bundles=[self.setup() for _ in range(3)]
        for _,w,_,_,_ in bundles:
            for b in w.branches.values():b.gate.data.fill_(.1)
        for bundle,indices,divisor in ((bundles[0],range(4),4),
                                        (bundles[1],range(2),2),(bundles[2],range(2,4),2)):
            m,w,kw,state,labels=bundle
            for i in indices:
                inputs=dict(kw,hidden_states=kw['hidden_states']+i*.01)
                with training_autocast('cpu'):
                    pred=predict(m,w,state,inputs,64,labels);aux=w.take_binding_loss()
                    loss=(pred.float().square().mean()+.01*aux[0]+.001*aux[1])/divisor
                w.clear_context();loss.backward()
        for p,a,b in zip(bundles[0][1].parameters(),bundles[1][1].parameters(),bundles[2][1].parameters()):
            torch.testing.assert_close(p.grad,(a.grad+b.grad)/2,rtol=2e-5,atol=2e-7)


if __name__=='__main__':unittest.main()
