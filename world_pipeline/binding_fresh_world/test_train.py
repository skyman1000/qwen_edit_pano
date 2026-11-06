"""CPU regressions for fresh initialization, two-rank math, and exact restart."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from ..binding_refine_world.runtime import WorldBranch
from ..id_binding_world.runtime import WorldBranch as OriginalWorld
from ..id_binding_world.parallel import micro_indices, seed_microbatch
from ..scaled_world.runtime import EpochSampler
from . import data as d
from .train import schedule, save_checkpoint


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def world(self):
        torch.manual_seed(31)
        return WorldBranch(32,(0,2),16,binding_block=2)

    def optimizer(self, world):
        return torch.optim.AdamW(world.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-6,weight_decay=1e-5)

    def loss(self, world, index):
        seed_microbatch(0,index)
        category=torch.tensor([[0,3,22]])
        geometry=torch.randn(1,3,8);geometry[:,:,6:]=1
        mask=torch.ones(1,3,dtype=torch.bool)
        tokens,_=world.encoder(category,geometry,mask)
        hidden=torch.randn(1,9,32)
        labels=(torch.tensor([0,1,2,3]),torch.tensor([0,1,2,3]),torch.tensor([1,2,2,1]),
            torch.tensor([.5,.5,0.,0.]),torch.tensor([0.,0.,.5,.5]))
        value=hidden.new_zeros(())
        for branch in world.branches.values():
            hidden,aux=branch(hidden,tokens,mask,8,labels)
            value=value+.01*aux[0]+.001*aux[1]
        return hidden.float().square().mean()+value

    def test_fresh_uses_actual_A_initial_weights_not_refined_constructor_rng(self):
        torch.manual_seed(0);original=OriginalWorld(32,(0,2),16,binding_block=2)
        torch.manual_seed(0);fresh=WorldBranch(32,(0,2),16,binding_block=2)
        self.assertTrue(any(not torch.equal(v,fresh.state_dict()[k]) for k,v in original.state_dict().items()))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'initial.safetensors';save_file(original.state_dict(),str(path))
            with patch.object(d,'INITIAL',path):d.load_initial(fresh)
        for k,v in original.state_dict().items():
            torch.testing.assert_close(v,fresh.state_dict()[k],rtol=0,atol=0)
        self.assertFalse(self.optimizer(fresh).state)
        self.assertTrue(all(float(b.gate)==0 for b in fresh.branches.values()))

    def test_two_rank_local_means_equal_global_four_microbatch_update(self):
        source=self.world();single=copy.deepcopy(source);ranks=[copy.deepcopy(source) for _ in range(2)]
        for index in range(4):(self.loss(single,index)/4).backward()
        for rank,world in enumerate(ranks):
            for index in micro_indices(0,4,rank,2):(self.loss(world,index)/2).backward()
        for p,a,b in zip(single.parameters(),ranks[0].parameters(),ranks[1].parameters()):
            self.assertEqual(p.grad is None,a.grad is None);self.assertEqual(a.grad is None,b.grad is None)
            if p.grad is not None:
                mean=(a.grad+b.grad)/2
                torch.testing.assert_close(p.grad,mean,rtol=3e-5,atol=2e-7)
                a.grad=mean.clone();b.grad=mean.clone()
        for w in (single,*ranks):
            torch.nn.utils.clip_grad_norm_(w.parameters(),1.)
            self.optimizer(w).step()
        for world in ranks:
            for a,b in zip(single.parameters(),world.parameters()):
                torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-7)
        self.assertGreater(float(single.encoder.category.weight.grad.abs().sum()),0.)

    def test_checkpoint_adam_rng_sampler_and_schedule_resume(self):
        config=dict(version=d.VERSION,smoke=False,lr=1e-4,warmup_steps=100,
            binding_ramp_steps=200,coverage_weight=.01,concentration_weight=.001)
        for actual,expected in zip(schedule(config,0),(1e-6,.00005,.000005)):
            self.assertAlmostEqual(actual,expected,places=15)
        self.assertEqual(schedule(config,199),(1e-4,.01,.001))
        world=self.world();opt=self.optimizer(world)
        self.loss(world,0).backward();opt.step();opt.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'run-1';out.mkdir()
            checkpoint=save_checkpoint(out,world,opt,config,1)
            restored=self.world();other=self.optimizer(restored)
            self.assertEqual(d.resume_step(checkpoint,config),1)
            self.assertEqual(d.restore(restored,other,checkpoint),1)
            for w,o in ((world,opt),(restored,other)):
                o.zero_grad(set_to_none=True);self.loss(w,4).backward();o.step()
            for k,v in world.state_dict().items():
                torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
            for a,b in zip(opt.state.values(),other.state.values()):
                for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
            self.assertEqual(d.latest(Path(directory),config),checkpoint)
            with self.assertRaises(ValueError):d.resume_step(checkpoint,dict(config,smoke=True))
            marker=json.loads((checkpoint/'COMPLETE.json').read_text());marker['version']='old_A10000'
            (checkpoint/'COMPLETE.json').write_text(json.dumps(marker))
            with self.assertRaises(ValueError):d.resume_step(checkpoint,config)
            next(iter(opt.state.values()))['exp_avg'].fill_(float('inf'))
            with self.assertRaises(FloatingPointError):save_checkpoint(out,world,opt,config,2)
            self.assertFalse((out/'checkpoint-step000002').exists())
        sampler=EpochSampler(7,0);prefix=[sampler.index(i) for i in range(16)]
        restored_sampler=EpochSampler(7,0)
        self.assertEqual(prefix[8:],[restored_sampler.index(i) for i in range(8,16)])

    def test_fresh_zero_gate_real_edit_two_steps_and_forward_only_inference(self):
        from types import SimpleNamespace
        from .. import oracle_test
        from ..oracle_memory import ActivationStorage,training_autocast
        from ..grounded_world.continue_train import predict
        from ..binding_refine_world.train import objective
        from qwen_pano.profiles import REPO_PANORAMA
        model=oracle_test.model(True);world=self.world();world.attach(model);opt=self.optimizer(world)
        state=oracle_test.tensorize(oracle_test.graph(),3);kwargs=oracle_test.inputs()
        labels=(torch.tensor([1,2,11,18]),torch.tensor([0,1,2,3]),torch.tensor([1,2,2,1]),
            torch.tensor([.5,.5,0.,0.]),torch.tensor([0.,0.,.5,.5]))
        with torch.no_grad(),training_autocast('cpu'):
            world.set_world(None);off=model(**kwargs)[0]
            plain=predict(model,world,state,kwargs,64)
            labelled=predict(model,world,state,kwargs,64,labels);world.take_binding_loss()
            from diffusers import QwenImageEditPlusPipeline as Pipeline
            off=Pipeline._unpack_latents(off[:,:32],64,128,8)
            torch.testing.assert_close(plain,off,rtol=0,atol=0)
            torch.testing.assert_close(plain,labelled,rtol=0,atol=0)
        config=dict(lr=1e-4,warmup_steps=100,binding_ramp_steps=200,coverage_weight=.01,concentration_weight=.001)
        for step in range(2):
            opt.zero_grad(set_to_none=True);lr,wc,wf=schedule(config,step)
            for group in opt.param_groups:group['lr']=lr
            with ActivationStorage(enabled=True,device_type='cpu').context():
                with training_autocast('cpu'):
                    pred=predict(model,world,state,kwargs,64,labels);aux=world.take_binding_loss()
                    noise=torch.randn_like(pred);z=torch.randn_like(pred)
                    loss,_=objective(pred,noise-z,torch.ones(8,16),noise,z,SimpleNamespace(**REPO_PANORAMA),aux,wc,wf)
                world.clear_context();loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for param in (world.branches['2'].q.weight,world.branches['2'].k.weight,world.encoder.category.weight):
                self.assertGreater(float(param.grad.abs().sum()),0.)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            opt.step();self.assertEqual(d.optimizer_step(opt),step+1)
        # Labels are train-only: removing them changes no generated prediction.
        with torch.no_grad(),training_autocast('cpu'):
            labelled=predict(model,world,state,kwargs,64,labels);world.take_binding_loss()
            plain=predict(model,world,state,kwargs,64)
            torch.testing.assert_close(plain,labelled,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
