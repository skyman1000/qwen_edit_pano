"""CPU tests for geometry, real Edit integration, and checkpoint recomputation."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file,load_file
from .. import oracle_test
from ..id_binding_world.runtime import WorldBranch as Legacy,VERSION as OLD
from ..oracle_memory import training_autocast,ActivationStorage
from .runtime import WorldBranch,LocalResidual,load_world,VERSION
from .projection import erp_support


def state():
    return (torch.tensor([[0,3,22,7]]),torch.tensor([[[0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,3.,1.,2.,1.,1.,1.],[-3.,.2,-2.,1.,1.,.4,1.,1.],
        [3.,0.,0.,1.,1.,1.,1.,1.]]]),torch.ones(1,4,dtype=torch.bool))


class SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_cardinal_axes_seam_poles_full_sizes_and_padding(self):
        g=torch.zeros(1,7,8);g[:,:,3:6]=1;g[:,:,6:]=1
        g[0,1,:3]=torch.tensor([0,0,3]);g[0,2,:3]=torch.tensor([3,0,0])
        g[0,3,:3]=torch.tensor([-3,0,0]);g[0,4,:3]=torch.tensor([0,0,-3])
        g[0,5,:3]=torch.tensor([0,3,0]);g[0,6,:3]=torch.tensor([0,-3,0])
        w,meta=erp_support(g,torch.ones(1,7,dtype=torch.bool),32,64,2)
        a=w[0].reshape(32,68,7)
        self.assertEqual(float(a[...,0].sum()),0)
        self.assertGreater(float(a[16,34,1]),0);self.assertGreater(float(a[16,50,2]),0)
        self.assertGreater(float(a[16,18,3]),0)
        self.assertGreater(float(a[16,2,4]),0);self.assertGreater(float(a[16,65,4]),0)
        self.assertGreater(float(a[0,34,5]),0);self.assertGreater(float(a[-1,34,6]),0)
        torch.testing.assert_close(a[:,:2],a[:,64:66],rtol=0,atol=0)
        torch.testing.assert_close(a[:,-2:],a[:,2:4],rtol=0,atol=0)
        self.assertLessEqual(meta['maximum_overlap_weight'],1.000001)
        g=state()[1][:,:2].clone();g[0,1,3:6]=2
        weights,_=erp_support(g,torch.ones(1,2,dtype=torch.bool),32,64)
        a=weights[0,:,1].reshape(32,64)
        self.assertGreater(float(a[16,35]),0) # ~20 degrees, inside full-size box
        self.assertEqual(float(a[16,40]),0) # ~48 degrees, outside including blur

    def test_missing_inside_and_empty_do_not_create_all_sky_object(self):
        cat,g,mask=state();g=g.clone();g[0,1,:3]=0;g[0,2,6]=0;mask[0,3]=False
        w,meta=erp_support(g,mask,8,16)
        self.assertEqual(float(w.sum()),0);self.assertEqual(meta['routed_boxes'],0)
        w,_=erp_support(torch.zeros(1,1,8),torch.ones(1,1,dtype=torch.bool),4,8)
        self.assertEqual(float(w.sum()),0)

    def test_matches_independent_numpy_slab_intersection(self):
        # At supersample=1, compare exact intersections + declared smoothing.
        g=state()[1];h,w=16,32
        projected,_=erp_support(g,state()[2],h,w,supersample=1)
        from qwen_pano.caupano.world_view import erp_rays
        rays=erp_rays(h,w);masks=[]
        for slot in range(g.shape[1]):
            if slot==0:masks.append(np.zeros((h,w)));continue
            center=g[0,slot,:3].numpy();half=g[0,slot,3:6].numpy()/2
            with np.errstate(divide='ignore'):
                aa=(center-half)/rays;bb=(center+half)/rays
            near=np.minimum(aa,bb).max(-1);far=np.maximum(aa,bb).min(-1)
            a=((far>=np.maximum(near,0))&(far>0)).astype(float)
            a=(np.roll(a,1,1)+2*a+np.roll(a,-1,1))/4
            a=(np.concatenate((a[:1],a[:-1]),0)+2*a+np.concatenate((a[1:],a[-1:]),0))/4
            masks.append(a)
        a=np.stack(masks);a/=np.maximum(a.sum(0,keepdims=True),1)
        np.testing.assert_allclose(projected[0].numpy(),a.reshape(4,-1).T,atol=2e-7)

    def test_token_permutation_moves_support_with_token(self):
        c,g,m=state();order=torch.tensor([0,3,1,2])
        a,_=erp_support(g,m,8,16);b,_=erp_support(g[:,order],m[:,order],8,16)
        torch.testing.assert_close(a[:,:,order],b,rtol=0,atol=0)

    def test_local_branch_has_no_global_or_unrouted_token_gradient(self):
        module=LocalResidual(32,16);module.gate.data.fill_(.1)
        tokens=torch.randn(1,4,16,requires_grad=True);target=torch.randn(1,5,32,requires_grad=True)
        route=torch.zeros(1,5,4);route[:,1:3,1]=1
        delta=module(target,tokens,route)
        self.assertEqual(float(delta[:,[0,3,4]].abs().sum()),0)
        delta.square().sum().backward()
        self.assertGreater(float(tokens.grad[:,1].abs().sum()),0)
        self.assertEqual(float(tokens.grad[:,[0,2,3]].abs().sum()),0)

    def test_warm_start_matches_legacy_exact_and_strict_resume(self):
        oldmodel=oracle_test.model(False).bfloat16();old=Legacy(32,(0,2),16,binding_block=2)
        for branch in old.branches.values():branch.gate.data.fill_(.1)
        old.attach(oldmodel)
        model=oracle_test.model(False).bfloat16();world=WorldBranch(32,(0,2),16,64,1)
        load_world(world,old.state_dict(),dict(version=OLD,group='A'));world.attach(model)
        kw=oracle_test.inputs()
        for key in ('hidden_states','encoder_hidden_states'):kw[key]=kw[key].bfloat16()
        with torch.no_grad(),training_autocast('cpu'):
            old.set_world(state());expected=oldmodel(**kw)[0]
            world.set_world(state());actual=model(**kw)[0]
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        bad=dict(old.state_dict());bad.pop('encoder.category.weight')
        with self.assertRaises(ValueError):load_world(world,bad,dict(version=OLD,group='A'))
        with self.assertRaises(RuntimeError):load_world(world,old.state_dict(),dict(version=VERSION))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'world.safetensors';save_file(world.state_dict(),str(p))
            other=WorldBranch(32,(0,2),16,64,1);load_world(other,load_file(str(p)),dict(version=VERSION))
            for key,value in other.state_dict().items():torch.testing.assert_close(value,world.state_dict()[key],rtol=0,atol=0)

    def test_checkpoint_backward_captures_routing_and_cpu_offload(self):
        outputs=[];gradients=[]
        for checkpointed,offload in ((False,False),(True,False),(True,True)):
            model=oracle_test.model(checkpointed).bfloat16();torch.manual_seed(9)
            world=WorldBranch(32,(0,2),16,64,1)
            for branch in world.branches.values():branch.gate.data.fill_(.1);branch.local.gate.data.fill_(.05)
            world.attach(model);kw=oracle_test.inputs()
            for key in ('hidden_states','encoder_hidden_states'):kw[key]=kw[key].bfloat16()
            with ActivationStorage(enabled=offload,device_type='cpu').context():
                with training_autocast('cpu'):
                    world.set_world(state());out=model(**kw)[0]
                world.clear_context();world.enabled=False;world.set_world(None)
                out.float().square().mean().backward()
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for k,p in world.named_parameters() if '.local.' in k and not k.endswith('gate')))
            outputs.append(out.detach());gradients.append({k:p.grad.clone() for k,p in world.named_parameters() if p.grad is not None})
        for out,grads in zip(outputs[1:],gradients[1:]):
            torch.testing.assert_close(out,outputs[0],rtol=0,atol=0)
            self.assertEqual(grads.keys(),gradients[0].keys())
            for key in grads:torch.testing.assert_close(grads[key],gradients[0][key],rtol=0,atol=0)

    def test_zero_gate_opens_then_local_body_learns(self):
        torch.manual_seed(8);module=LocalResidual(32,16)
        optimizer=torch.optim.AdamW(module.parameters(),lr=.001)
        h=torch.randn(1,8,32);tokens=torch.randn(1,4,16)
        route=torch.zeros(1,8,4);route[:,:4,1]=1;route[:,4:,2]=1
        target=torch.randn_like(h)
        for step in range(2):
            optimizer.zero_grad();loss=(module(h,tokens,route)-target).square().mean();loss.backward()
            if step==0:self.assertGreater(float(module.gate.grad.abs()),0)
            else:self.assertGreater(float(module.value.weight.grad.abs().sum()),0)
            optimizer.step()

    def test_persistent_generation_grid_matches_temporary_training_grid(self):
        from qwen_edit_pano.circular import crop_columns
        first=oracle_test.model(False);torch.manual_seed(18)
        w=WorldBranch(32,(0,2),16,64,1)
        for b in w.branches.values():b.gate.data.fill_(.1);b.local.gate.data.fill_(.03)
        w.attach(first);w.set_world(state());a=first(**oracle_test.inputs())[0]
        second=oracle_test.model(False,mode='persistent');v=WorldBranch(32,(0,2),16,64,1)
        v.load_state_dict(w.state_dict(),strict=True);v.attach(second);v.set_world(state())
        b=second(**oracle_test.inputs('persistent'))[0]
        restored=torch.cat((crop_columns(b[:,:40],4,8,1),b[:,40:]),1)
        torch.testing.assert_close(a,restored,rtol=0,atol=0)

    def test_new_controls_leave_local_intersecting_objects_unchanged(self):
        from .controls import condition_variant,fully_outside
        c,g,m=state();c=torch.cat((c,torch.tensor([[11,14]])),1)
        g=torch.cat((g,torch.tensor([[[3.,0.,-4.,1.,1.,1.,1.,1.],[-2.,1.,-5.,1.,1.,1.,1.,1.]]])),1)
        m=torch.ones_like(c,dtype=torch.bool);original=(c,g,m)
        local=~fully_outside(g)[0];local[0]=True
        for mode in ('wrong_category','wrong_position'):
            altered,meta=condition_variant(original,mode)
            for a,b in zip(altered,original):torch.testing.assert_close(a[:,local],b[:,local],rtol=0,atol=0)
            changed=meta['changed_slots'];self.assertTrue(fully_outside(altered[1][0,changed]).all())
            self.assertEqual(sorted(c[0].tolist()),sorted(altered[0][0].tolist()))
            self.assertEqual(sorted(map(tuple,g[0,:,:3].tolist())),sorted(map(tuple,altered[1][0,:,:3].tolist())))

    def test_restart_ignores_incomplete_checkpoint_and_panel_excludes_val(self):
        from .run import latest
        from .data import select_train
        from ..object_world.data import load_pack,DEFAULT_PACK
        plan,rows=load_pack(DEFAULT_PACK);selected=select_train(rows,32,0)
        self.assertEqual(len(selected),32);self.assertTrue(all(r['split']=='train' for r in selected))
        self.assertEqual([r['id'] for r in selected],[r['id'] for r in select_train(rows,32,0)])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'run-one/checkpoint-step000050';p.mkdir(parents=True)
            (p/'COMPLETE.json').write_text('{"version":"'+VERSION+'","step":50}')
            self.assertIsNone(latest(root)[1])
            for name in ('world.safetensors','optimizer.pt'):(p/name).touch()
            (p/'config.json').write_text('{"version":"'+VERSION+'"}')
            self.assertEqual(latest(root),(50,p))


if __name__=='__main__':unittest.main()
