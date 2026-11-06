"""CPU regression of real tiny Edit, sampled labels, stable losses and resume."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

from .. import oracle_test
from ..oracle_memory import ActivationStorage
from ..scaled_world.runtime import seed_step
from ..id_binding_world.runtime import WorldBranch as OriginalWorld
from ..object_world.data import tensorize
from .runtime import WorldBranch
from .supervision import POLICY,sample_binding,binding_terms,guard
from .train import objective,resume_step
from . import data as d
from .run import latest


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def labels(self):
        return (torch.tensor([1,2,11,18]),torch.tensor([0,1,2,3,0,1,2,3]),
            torch.tensor([1,1,2,2,2,2,1,1]),torch.tensor([.25]*4+[0.]*4),torch.tensor([0.]*4+[.25]*4))

    def world(self,cls=WorldBranch):
        torch.manual_seed(61);w=cls(32,(0,2),16,binding_block=2)
        for branch in w.branches.values():branch.gate.data.fill_(.1)
        return w

    def test_losses_equal_independent_probability_calculation_and_reject_wrong_object(self):
        q=torch.tensor([[[[1.,0.],[0.,1.]],[[.8,0.],[0.,.8]]]],requires_grad=True)
        k=torch.tensor([[[[0.,0.],[8.,0.],[0.,8.]],[[0.,0.],[8.,0.],[0.,8.]]]],requires_grad=True)
        mask=torch.ones(1,3,dtype=torch.bool)
        labels=(torch.arange(2),torch.tensor([0,1,0,1]),torch.tensor([1,2,2,1]),
            torch.tensor([.5,.5,0.,0.]),torch.tensor([0.,0.,.5,.5]))
        cov,focus=binding_terms(q,k,mask,labels)
        a=(q@k.transpose(-1,-2)/2**.5).softmax(-1).mean(1)[0]
        expected_cov=-(a[0,1].log()+a[1,2].log())/2
        expected_focus=-((1-a[0,2]).log()+(1-a[1,1]).log())/2
        torch.testing.assert_close(cov,expected_cov);torch.testing.assert_close(focus,expected_focus)
        wrong=binding_terms(q,k,mask,(*labels[:2],torch.tensor([2,1,1,2]),*labels[3:]))
        self.assertLess(float(cov+focus),float(sum(wrong)))
        (cov+focus).backward();self.assertTrue(torch.isfinite(q.grad).all());self.assertTrue(torch.isfinite(k.grad).all())

    def test_padding_extreme_logits_empty_and_permutation(self):
        q=torch.tensor([[[[1000.,0.],[0.,1000.]]]],requires_grad=True)
        k=torch.tensor([[[[0.,0.],[1000.,0.],[0.,1000.],[7.,7.]]]],requires_grad=True)
        mask=torch.tensor([[True,True,True,False]])
        labels=(torch.arange(2),torch.tensor([0,1,0,1]),torch.tensor([1,2,1,2]),
            torch.tensor([.5,.5,0.,0.]),torch.tensor([0.,0.,.5,.5]))
        values=binding_terms(q,k,mask,labels);self.assertTrue(all(torch.isfinite(v) for v in values))
        sum(values).backward();self.assertTrue(torch.isfinite(q.grad).all());self.assertTrue(torch.isfinite(k.grad).all())
        perm=torch.tensor([0,2,1,3]);other=binding_terms(q,k[:,:,perm],mask[:,perm],(*labels[:2],3-labels[2],*labels[3:]))
        for a,b in zip(values,other):torch.testing.assert_close(a,b,rtol=0,atol=0)
        empty=(torch.empty(0,dtype=torch.long),)*3+(torch.empty(0),torch.empty(0))
        self.assertEqual(float(sum(binding_terms(q,k,mask,empty))),0.)

    def test_label_regions_guard_rng_and_same_class_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            ids=np.zeros((32,64),dtype=np.int32);ids[4:28,4:20]=1;ids[4:28,44:60]=2
            ids[:4,24:40]=3  # generic object cannot become a negative.
            valid=ids>0;path=Path(directory)/'support.npz'
            np.savez(path,instance=ids,instance_valid=valid,object_loss_valid=valid)
            row=dict(objects=[dict(instance_id=str(i),category_id=c) for i,c in ((1,3),(2,3),(3,39))],
                object_region_supervision=dict(path=str(path)))
            before=torch.get_rng_state().clone();np.random.seed(9);before_np=np.random.get_state()
            labels,meta=sample_binding(row,128,1,51)
            self.assertEqual(meta['selected_instances'],2);self.assertEqual(meta['negative_instances'],2)
            torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
            self.assertTrue(np.array_equal(before_np[1],np.random.get_state()[1]))
            queries,pairs,slots,pos,neg=labels;y=(queries[pairs]//18).numpy();x=(queries[pairs]%18-1).numpy()
            self.assertTrue(((x>=0)&(x<16)).all());self.assertTrue(torch.isclose(pos.sum(),torch.tensor(1.)))
            self.assertTrue(torch.isclose(neg.sum(),torch.tensor(1.)))
            pooled=ids.reshape(8,4,16,4)
            for j,slot in enumerate(slots.tolist()):
                patch=pooled[y[j],:,x[j],:]
                if pos[j]>0:self.assertTrue((patch==slot).any())
                if neg[j]>0:
                    self.assertTrue((patch==(3-slot)).any())
                    mass=(ids==slot).reshape(8,4,16,4).sum((1,3))
                    self.assertFalse(guard(mass>0)[y[j],x[j]])
            reversed_row=dict(row,objects=row['objects'][1::-1]+row['objects'][2:])
            other,_=sample_binding(reversed_row,128,1,51)
            for j in (0,1,3,4):torch.testing.assert_close(labels[j],other[j],rtol=0,atol=0)
            torch.testing.assert_close(3-labels[2],other[2],rtol=0,atol=0)

    def test_guard_is_periodic_and_unknown_only_has_no_negative(self):
        support=np.zeros((4,8),dtype=bool);support[0,0]=True
        self.assertTrue(guard(support)[0,-1]);self.assertFalse(guard(support)[-1,0])
        with tempfile.TemporaryDirectory() as directory:
            ids=np.zeros((32,64),dtype=np.int32);ids[::4]=1
            path=Path(directory)/'one.npz';np.savez(path,instance=ids,instance_valid=ids>0,object_loss_valid=ids>0)
            row=dict(objects=[dict(instance_id='1',category_id=22)],object_region_supervision=dict(path=str(path)))
            labels,meta=sample_binding(row,64,1,7)
            self.assertEqual(meta['positive_pairs'],8);self.assertEqual(meta['negative_pairs'],0)
            self.assertTrue((labels[3]>0).all());self.assertEqual(float(labels[4].sum()),0.)

    def test_real_edit_disabled_forward_and_gradients_equal_original_A(self):
        source=self.world(OriginalWorld).state_dict();records=[]
        for cls in (OriginalWorld,WorldBranch):
            model=oracle_test.model(True);world=self.world(cls);world.load_state_dict(source);world.attach(model)
            world.set_world(oracle_test.tensorize(oracle_test.graph(),3))
            out=model(**oracle_test.inputs())[0];world.clear_context();out.square().mean().backward()
            records.append((out.detach(),{k:p.grad.clone() for k,p in world.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(records[0][0],records[1][0],rtol=0,atol=0)
        self.assertEqual(records[0][1].keys(),records[1][1].keys())
        for key in records[0][1]:torch.testing.assert_close(records[0][1][key],records[1][1][key],rtol=0,atol=0)

    def test_real_edit_aux_checkpoint_cpu_storage_and_mutated_context(self):
        from diffusers import QwenImageEditPlusPipeline as Pipeline
        from qwen_pano.profiles import REPO_PANORAMA
        records=[]
        for checkpointed,offload in ((False,False),(True,False),(True,True)):
            model=oracle_test.model(checkpointed);world=self.world();world.attach(model);kwargs=oracle_test.inputs()
            graph=oracle_test.tensorize(oracle_test.graph(),3)
            with torch.no_grad():
                world.set_world(graph);plain=model(**kwargs)[0]
                world.set_supervision(self.labels());world.set_world(graph);labelled=model(**kwargs)[0];world.take_binding_loss()
                torch.testing.assert_close(plain,labelled,rtol=0,atol=0)
            world.clear_context();storage=ActivationStorage(enabled=offload,device_type='cpu')
            with storage.context():
                world.set_supervision(self.labels());world.set_world(graph);out=model(**kwargs)[0];aux=world.take_binding_loss()
                prediction=Pipeline._unpack_latents(out[:,:32],64,128,8)
                seed_step(32,0);target=torch.randn_like(prediction);noise=torch.randn_like(prediction)
                loss,_=objective(prediction,target,torch.ones(8,16),noise,noise-target,
                    SimpleNamespace(**REPO_PANORAMA),aux,.01,.001)
                world.set_world(oracle_test.tensorize(oracle_test.graph(),3,'constant'));world.clear_context();loss.backward()
            self.assertFalse(world._losses);self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertGreater(float(world.encoder.category.weight.grad.abs().sum()),0.)
            records.append((out.detach(),float(loss),{k:p.grad.clone() for k,p in world.named_parameters() if p.grad is not None}))
        for out,loss,grads in records[1:]:
            torch.testing.assert_close(out,records[0][0],rtol=0,atol=0);self.assertEqual(loss,records[0][1])
            self.assertEqual(grads.keys(),records[0][2].keys())
            for key in grads:torch.testing.assert_close(grads[key],records[0][2][key],rtol=0,atol=0)

    def test_exact_world_adam_restore_and_cross_arm_rejection(self):
        from ..object_world.data import atomic_tensors
        world=self.world();opt=torch.optim.AdamW(world.parameters(),lr=1e-4,eps=1e-6)
        def update(w,o):
            o.zero_grad(set_to_none=True);sum(p.square().sum() for p in w.parameters()).backward();o.step()
        update(world,opt)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'run-1/checkpoint-step000001';checkpoint.mkdir(parents=True)
            config=dict(version=d.VERSION,group='binding')
            atomic_tensors(checkpoint/'world.safetensors',world.state_dict());torch.save(opt.state_dict(),checkpoint/'optimizer.pt')
            (checkpoint/'config.json').write_text(json.dumps(config))
            (checkpoint/'COMPLETE.json').write_text(json.dumps(dict(version=d.VERSION,step=1,optimizer_step=1)))
            restored=self.world();other=torch.optim.AdamW(restored.parameters(),lr=1e-4,eps=1e-6)
            self.assertEqual(d.restore(restored,other,checkpoint),1)
            update(world,opt);update(restored,other)
            for k,v in world.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
            for a,b in zip(opt.state.values(),other.state.values()):
                for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
            self.assertEqual(resume_step(checkpoint,config),1)
            with self.assertRaises(ValueError):resume_step(checkpoint,dict(config,group='control'))
            self.assertEqual(latest(root,config),checkpoint)
            partial=root/'run-2/checkpoint-step010100';partial.mkdir(parents=True)
            (partial/'config.json').write_text(json.dumps(config));(partial/'COMPLETE.json').write_text(
                json.dumps(dict(version=d.VERSION,step=10100,optimizer_step=10100)))
            self.assertEqual(latest(root,config),checkpoint)


if __name__=='__main__':unittest.main()
