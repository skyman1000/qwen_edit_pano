"""CPU checks of mathematical supervision and frozen-base gradient integration."""
import unittest
import torch

from .runtime import region_weights,object_region_loss,WorldBranch,SemanticEncoder
from .data import tensorize
from .. import oracle_test


class SupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_object_balance_fractional_cells_and_gradients(self):
        # One one-pixel object and one seven-pixel object contribute equally.
        instance=torch.tensor([[1,2,2,2],[2,2,2,2]])
        weights=region_weights(instance,torch.ones_like(instance,dtype=torch.bool),[1,2],(2,4))
        prediction=torch.ones(1,1,1,2,4,requires_grad=True)
        loss=object_region_loss(prediction,torch.zeros_like(prediction),weights)
        loss.backward()
        self.assertAlmostEqual(float(loss),1.,places=6)
        grad=prediction.grad[0,0,0]
        self.assertAlmostEqual(float(grad[instance==1].sum()),float(grad[instance==2].sum()),places=6)
        down=region_weights(instance,torch.ones_like(instance,dtype=torch.bool),[1,2],(1,2))
        self.assertAlmostEqual(float(down.mean()),1.,places=6)
        self.assertGreater(float(down[0,0]),float(down[0,1]))

    def test_invalid_unrepresented_regions_and_empty_support(self):
        ids=torch.tensor([[1,2,3,3],[1,2,3,3]])
        valid=torch.tensor([[True,False,True,True],[True,False,True,True]])
        weights=region_weights(ids,valid,[1,2],(2,4))
        prediction=torch.ones(1,2,1,2,4,requires_grad=True)
        object_region_loss(prediction,torch.zeros_like(prediction),weights).backward()
        self.assertTrue(torch.all(prediction.grad[0,:,0,:,1:]==0))
        empty=region_weights(ids,torch.zeros_like(valid),[1,2],(2,4))
        zero=object_region_loss(prediction,torch.zeros_like(prediction),empty)
        self.assertEqual(float(zero),0.)

    def test_capacity_and_missing_geometry(self):
        obj=dict(category_id=22,center_valid=False,center_local_m=None,size_valid=True,size_aabb_local_m=[1,2,3])
        row=dict(objects=[dict(obj) for _ in range(215)])
        category,geometry,mask=tensorize(row,256)
        self.assertEqual(category.shape,(1,216));self.assertTrue(mask.all())
        self.assertEqual(float(geometry[0,1,:3].sum()),0.)
        self.assertEqual(float(geometry[0,1,6]),0.);self.assertEqual(float(geometry[0,1,7]),1.)
        with self.assertRaises(ValueError):tensorize(row,160)

    def test_frozen_semantic_buffer_and_permutation(self):
        features=torch.randn(42,12)
        encoder=SemanticEncoder(features,16)
        category=torch.tensor([[0,3,22]])
        geometry=torch.randn(1,3,8);geometry[...,6:]=1
        mask=torch.ones(1,3,dtype=torch.bool)
        first,_=encoder(category,geometry,mask)
        perm=torch.tensor([0,2,1])
        second,_=encoder(category[:,perm],geometry[:,perm],mask[:,perm])
        torch.testing.assert_close(first[:,perm],second)
        first.square().mean().backward()
        self.assertFalse(encoder.class_features.requires_grad)
        self.assertIsNone(encoder.class_features.grad)
        self.assertGreater(float(encoder.semantic.weight.grad.abs().sum()),0.)

    def test_checkpointed_frozen_base_receives_region_gradient(self):
        torch.manual_seed(5)
        features=torch.randn(42,12)
        first=WorldBranch(32,features,(0,2),16)
        for branch in first.branches.values():branch.gate.data.fill_(.1)
        state=first.state_dict();gradients=[]
        graph=oracle_test.graph()
        inputs=oracle_test.tensorize(graph,3)
        for checkpointed in (False,True):
            world=WorldBranch(32,features,(0,2),16);world.load_state_dict(state)
            model=oracle_test.model(checkpointed);world.attach(model);world.set_world(inputs)
            if checkpointed:
                from ..inference_diagnostics import RelativeResidualRecorder
                recorder=RelativeResidualRecorder(world)
            result=model(**oracle_test.inputs())[0][:,:4,:]
            world.set_world(None)
            prediction=result.transpose(1,2).reshape(1,result.shape[-1],1,2,2)
            loss=object_region_loss(prediction,torch.zeros_like(prediction),torch.tensor([[4.,0.],[0.,0.]]))
            loss.backward()
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertGreater(float(world.encoder.semantic.weight.grad.abs().sum()),0.)
            if checkpointed:
                self.assertEqual(set(recorder.summary()),{'0','2'})
                self.assertGreater(recorder.summary()['2']['last']['target_rms'],0.)
            gradients.append({key:p.grad.clone() for key,p in world.named_parameters() if p.grad is not None})
        self.assertEqual(gradients[0].keys(),gradients[1].keys())
        for key in gradients[0]:torch.testing.assert_close(gradients[0][key],gradients[1][key],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
