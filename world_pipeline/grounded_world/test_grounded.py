"""CPU checks on real tiny Edit, with checkpointing and immutable GT labels."""
import copy
import math
import unittest
import torch
from .. import oracle_test
from ..oracle_memory import training_autocast,ActivationStorage
from ..spatial_world.test_spatial import state
from .runtime import WorldBranch,ObjectAttention,probabilities_loss


class GroundedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def model(self,checkpointed=False):
        model=oracle_test.model(checkpointed).bfloat16();torch.manual_seed(9)
        features=torch.randn(42,24);world=WorldBranch(32,features,(0,2),16,64,1,2)
        world.attach(model);kw=oracle_test.inputs()
        for key in ('hidden_states','encoder_hidden_states'):kw[key]=kw[key].bfloat16()
        return model,world,kw

    def labels(self,world):
        chosen=(world.current[2][0,:,1:]).nonzero()[0];query=int(chosen[0]);slot=int(chosen[1])+1
        return (torch.tensor([query]),torch.tensor([0]),torch.tensor([slot]),torch.tensor([1.]))

    def test_zero_gate_is_panorama_exact_and_labels_never_change_prediction(self):
        model,world,kw=self.model()
        with torch.no_grad(),training_autocast('cpu'):
            world.set_world(None);base=model(**kw)[0]
            world.set_world(state());plain=model(**kw)[0]
            labels=self.labels(world);world.set_supervision(labels);supervised=model(**kw)[0];loss=world.take_binding_loss()
        torch.testing.assert_close(base,plain,rtol=0,atol=0)
        torch.testing.assert_close(plain,supervised,rtol=0,atol=0);self.assertGreater(float(loss),0)
        self.assertNotIn('empty',dict(world.encoder.named_parameters()))
        self.assertFalse(world.encoder.class_features.requires_grad)

    def test_binding_has_real_qk_and_semantic_gradient_at_zero_gate(self):
        model,world,kw=self.model();world.set_world(state());world.set_supervision(self.labels(world))
        with training_autocast('cpu'):output=model(**kw)[0];loss=world.take_binding_loss()
        loss.backward()
        for p in (world.branches['2'].q.weight,world.branches['2'].k.weight,world.encoder.semantic[0].weight):
            self.assertIsNotNone(p.grad);self.assertGreater(float(p.grad.abs().sum()),0);self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(world.branches['2'].v.weight.grad is None or not world.branches['2'].v.weight.grad.count_nonzero())

    def test_labels_are_readout_only_with_nonzero_gates(self):
        model,world,kw=self.model()
        for branch in world.branches.values():branch.gate.data.fill_(.1)
        with torch.no_grad():
            world.set_world(state());a=model(**kw)[0];labels=self.labels(world)
            world.set_supervision(labels);b=model(**kw)[0];self.assertGreater(float(world.take_binding_loss()),0)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_no_candidate_is_exact_identity_despite_norm_bias_and_reference_unchanged(self):
        branch=ObjectAttention(32,16);branch.gate.data.fill_(.1);branch.world_norm.bias.data.fill_(.5)
        hidden=torch.randn(1,8,32).bfloat16();tokens=torch.randn(1,4,16);mask=torch.ones(1,4,dtype=torch.bool)
        candidates=torch.zeros(1,6,4,dtype=torch.bool);candidates[:,2:4,1]=True
        with torch.no_grad():output,_=branch(hidden,tokens,mask,6,candidates)
        torch.testing.assert_close(output[:,[0,1,4,5,6,7]],hidden[:,[0,1,4,5,6,7]],rtol=0,atol=0)
        self.assertTrue(torch.isfinite(output).all());self.assertGreater(float((output[:,2:4]-hidden[:,2:4]).abs().sum()),0)

    def test_coverage_matches_explicit_probabilities_and_rejects_unreachable_targets(self):
        torch.manual_seed(14);q=torch.randn(1,2,5,4,requires_grad=True);k=torch.randn(1,2,4,4,requires_grad=True)
        allowed=torch.zeros(1,5,4,dtype=torch.bool);allowed[:,:,0]=True;allowed[:,1,2]=True;allowed[:,3,1:]=True
        labels=(torch.tensor([1,3]),torch.tensor([0,1]),torch.tensor([2,1]),torch.tensor([.5,.5]))
        logits=(q[0]@k[0].transpose(-1,-2))/2;logits=logits.masked_fill(~allowed[0][None],-torch.inf)
        probability=logits.softmax(-1).mean(0)
        expected=-(probability[1,2].log()+probability[3,1].log())/2
        actual=probabilities_loss(q,k,allowed,labels);torch.testing.assert_close(actual,expected)
        actual.backward();self.assertTrue(torch.isfinite(q.grad).all());self.assertTrue(torch.isfinite(k.grad).all())
        bad=(labels[0],labels[1],torch.tensor([1,1]),labels[3])
        with self.assertRaises(ValueError):probabilities_loss(q,k,allowed,bad)
        empty=tuple(t[:0] for t in labels);self.assertEqual(float(probabilities_loss(q,k,allowed,empty)),0)

    def test_checkpoint_and_cpu_offload_backward_capture_sample_and_labels(self):
        outputs=[];gradients=[]
        for checkpointed,offload in ((False,False),(True,False),(True,True)):
            model,world,kw=self.model(checkpointed)
            for branch in world.branches.values():branch.gate.data.fill_(.03)
            with ActivationStorage(enabled=offload,device_type='cpu').context():
                world.set_world(state());world.set_supervision(self.labels(world))
                with training_autocast('cpu'):output=model(**kw)[0];loss=output.float().square().mean()+.01*world.take_binding_loss()
                world.clear_context();loss.backward()
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertGreater(float(world.encoder.semantic[0].weight.grad.abs().sum()),0)
            outputs.append(output.detach());gradients.append({k:p.grad.clone() for k,p in world.named_parameters() if p.grad is not None})
        for value,grads in zip(outputs[1:],gradients[1:]):
            torch.testing.assert_close(value,outputs[0],rtol=0,atol=0);self.assertEqual(grads.keys(),gradients[0].keys())
            for name in grads:torch.testing.assert_close(grads[name],gradients[0][name],rtol=0,atol=0)

    def test_token_permutation_invariance_and_persistent_inference_grid(self):
        model,world,kw=self.model()
        for b in world.branches.values():b.gate.data.fill_(.05)
        c,g,m=state();order=torch.tensor([0,3,1,2])
        with torch.no_grad():
            world.set_world((c,g,m));a=model(**kw)[0]
            world.set_world((c[:,order],g[:,order],m[:,order]));b=model(**kw)[0]
        torch.testing.assert_close(a,b,rtol=.02,atol=.002)
        other=oracle_test.model(False,mode='persistent').bfloat16()
        v=WorldBranch(32,world.encoder.class_features,(0,2),16,64,1,2);v.load_state_dict(world.state_dict(),strict=True);v.attach(other)
        kw2=oracle_test.inputs('persistent')
        for key in ('hidden_states','encoder_hidden_states'):kw2[key]=kw2[key].bfloat16()
        with torch.no_grad():v.set_world(state());out=other(**kw2)[0]
        from qwen_edit_pano.circular import crop_columns
        restored=torch.cat((crop_columns(out[:,:40],4,8,1),out[:,40:]),1)
        torch.testing.assert_close(restored,a,rtol=0,atol=0)

    def test_optimizer_checkpoint_resume_is_exact(self):
        model,world,kw=self.model();opt=torch.optim.AdamW(world.parameters(),lr=.001,eps=1e-8)
        def step(m,w,o,k):
            o.zero_grad(set_to_none=True);w.set_world(state());w.set_supervision(self.labels(w))
            value=m(**k)[0].float().square().mean()+.01*w.take_binding_loss();w.clear_context();value.backward();o.step()
        step(model,world,opt,kw);saved=copy.deepcopy(world.state_dict());os=copy.deepcopy(opt.state_dict());step(model,world,opt,kw)
        m,w,k=self.model();w.load_state_dict(saved,strict=True);o=torch.optim.AdamW(w.parameters(),lr=.001,eps=1e-8);o.load_state_dict(os);step(m,w,o,k)
        for name,value in world.state_dict().items():torch.testing.assert_close(value,w.state_dict()[name],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
