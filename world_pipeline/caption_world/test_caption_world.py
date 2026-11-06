"""CPU checks of fresh initialization, caption provenance, no-binding gradients."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from safetensors.torch import save_file

from ..build_polished_world import atomic_json
from .. import oracle_test
from ..oracle_memory import training_autocast,ActivationStorage
from ..spatial_world.test_spatial import state
from ..id_binding_world.parallel import micro_indices,seed_microbatch
from ..grounded_world.continue_train import predict
from ..grounded_world.train import norms
from . import data as d
from .cache import shard
from .train import fresh_world,objective,resume_step
from .run import latest


class CaptionWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_cache_shards_and_training_indices_have_no_inherited_offset(self):
        rows=list(range(7));a,b=shard(rows,0,2),shard(rows,1,2)
        self.assertFalse(set(a)&set(b));self.assertEqual(sorted(a+b),rows)
        for step in (0,1,2500):
            single=list(micro_indices(step,4));dual=[i for rank in (0,1) for i in micro_indices(step,4,rank,2)]
            self.assertEqual(single,dual);self.assertEqual(single[0],4*step)

    def test_scene_caption_removes_unaligned_directions_without_inventing_objects(self):
        text='A sauna on the left, a shower in the center, and a bed on the right.'
        self.assertEqual(d.scene_caption(text),'A sauna, a shower, and a bed.')
        text='A mirror on the right wall and a table in the center of the room.'
        self.assertEqual(d.scene_caption(text),'A mirror on the wall and a table in the center of the room.')
        self.assertEqual(d.scene_caption('Doors on the left and right sides.'),'Doors on both sides.')
        self.assertEqual(d.scene_caption('A hallway extending to the right.'),'A hallway extending.')
        self.assertEqual(d.scene_caption('A room with a bed and a window.'),'A room with a bed and a window.')

    def test_new_prompt_cache_is_used_instead_of_old_text_reference_is_reused(self):
        with TemporaryDirectory() as directory:
            folder=Path(directory);old=folder/'old.safetensors';new=folder/'new.safetensors'
            save_file(dict(embeddings=torch.full((2,3584),-7.),mask=torch.ones(2,dtype=torch.bool),
                reference=torch.full((4,64),9.),ref_shape=torch.tensor([1,2,2])),str(old))
            save_file(dict(embeddings=torch.full((3,3584),5.),mask=torch.ones(3,dtype=torch.bool)),str(new))
            actual=d.load_condition(dict(condition_cache=str(old),caption_cache=str(new)),'cpu')
            self.assertEqual(actual['embeddings'].shape,(3,3584));self.assertTrue((actual['embeddings']==5).all())
            self.assertTrue((actual['reference']==9).all());self.assertEqual(actual['ref_shape'].tolist(),[1,2,2])

    def test_caption_metadata_rejects_wrong_prompt_even_with_correct_dimensions(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'cache.safetensors'
            row=dict(id='train1',caption_prompt='Expand.\nScene description: a bedroom',caption_cache=str(path))
            entry=dict(source=dict(id='train1',split='train'),reference_cache=dict(path='existing reference'))
            prepared=dict(base_snapshot='existing local Qwen')
            values=dict(embeddings=torch.ones(3,3584,dtype=torch.bfloat16),mask=torch.ones(3,dtype=torch.bool))
            save_file(values,str(path),metadata={'contract':json.dumps(d.cache_contract(row,entry,prepared),sort_keys=True)})
            d.check_caption_cache(row,entry,prepared)
            row['caption_prompt']='Expand.\nScene description: a kitchen'
            with self.assertRaises(ValueError):d.check_caption_cache(row,entry,prepared)

    def test_resume_rejects_old_C_and_incomplete_checkpoints(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);path=root/'run-1/checkpoint-step000100';path.mkdir(parents=True)
            config=dict(version=d.VERSION,smoke=False)
            atomic_json(path/'config.json',config);atomic_json(path/'COMPLETE.json',dict(version=d.VERSION,step=100,optimizer_step=100))
            self.assertEqual(latest(root,config),(0,None))
            with self.assertRaises(ValueError):resume_step(path,config)
            for name in ('world.safetensors','optimizer.pt'):(path/name).write_bytes(b'CPU fixture, not weights')
            self.assertEqual(latest(root,config),(100,path));self.assertEqual(resume_step(path,config),100)
            atomic_json(path/'COMPLETE.json',dict(version='grounded_full_distribution_C_v1',step=100,optimizer_step=700))
            with self.assertRaises(ValueError):latest(root,config)
            with self.assertRaises(ValueError):resume_step(path,config)

    def test_actual_tiny_frozen_transformer_two_steps_without_binding_and_exact_resume(self):
        from qwen_pano.profiles import REPO_PANORAMA
        plan=dict(height=64,padding_columns=1)
        config=dict(seed=0,blocks=[0,2],width=16,binding_block=2,lr=1e-3,betas=[.9,.999],adam_eps=1e-8,weight_decay=1e-5)
        torch.manual_seed(31);features=torch.randn(42,24)
        model=oracle_test.model(True).bfloat16().eval().requires_grad_(False)
        world,opt=fresh_world(32,features,plan,config,'cpu');self.assertFalse(opt.state)
        initial={k:v.clone() for k,v in world.state_dict().items()}
        other,other_opt=fresh_world(32,features,plan,config,'cpu')
        self.assertTrue(all(torch.equal(t,other.state_dict()[k]) for k,t in initial.items()));self.assertFalse(other_opt.state)
        world.attach(model);kwargs=oracle_test.inputs()
        for key in ('hidden_states','encoder_hidden_states'):kwargs[key]=kwargs[key].bfloat16()
        with torch.no_grad(),training_autocast('cpu'):
            base=predict(model,world,None,kwargs,64);zero=predict(model,world,state(),kwargs,64)
            torch.testing.assert_close(base,zero,rtol=0,atol=0)
        world.clear_context();lossargs=SimpleNamespace(**REPO_PANORAMA)
        generator=torch.Generator().manual_seed(42)
        clean=torch.randn((1,4,1,8,16),generator=generator);noise=torch.randn(clean.shape,generator=generator);velocity=noise-clean
        def step(m,w,o,index):
            seed_microbatch(0,index);o.zero_grad(set_to_none=True)
            with ActivationStorage(enabled=True,device_type='cpu').context():
                with training_autocast('cpu'):
                    prediction=predict(m,w,state(),kwargs,64)
                    loss,parts=objective(prediction,velocity,torch.ones(8,16),lossargs,noise,clean)
                self.assertEqual(set(parts),{'flow','flow_panorama','cube','yaw_resampled','object_region'})
                self.assertIsNone(w.supervision);self.assertEqual(w._losses,[])
                w.clear_context();loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is None for p in m.parameters()))
            gradient=norms(w);o.step();return gradient
        step(model,world,opt,0);saved=copy.deepcopy(world.state_dict());optimizer=copy.deepcopy(opt.state_dict())
        gradient=step(model,world,opt,1)
        self.assertGreater(gradient['semantic'],0);self.assertGreater(gradient['value_output'],0)
        resumed=oracle_test.model(True).bfloat16().eval().requires_grad_(False)
        other.load_state_dict(saved,strict=True);other_opt.load_state_dict(optimizer);other.attach(resumed)
        step(resumed,other,other_opt,1)
        for name,value in world.state_dict().items():torch.testing.assert_close(value,other.state_dict()[name],rtol=0,atol=0)
        torch.testing.assert_close(world.encoder.class_features,initial['encoder.class_features'],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
