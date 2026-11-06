"""CPU regressions for staged training, tensor layout and supervision boundaries."""
import tempfile
from pathlib import Path
import unittest
import torch

from .runtime import WorldBranch,object_region_loss
from .data import atomic_tensors
from .. import oracle_test
from ..scaled_world.runtime import EpochSampler,seed_step,learning_rate


class DeepReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_pack_unpack_preserves_latent_pixel_and_channel_order(self):
        from diffusers import QwenImageEditPlusPipeline as Pipeline
        z=torch.arange(16*8*16).reshape(1,16,1,8,16).float()
        packed=Pipeline._pack_latents(z,1,16,8,16)
        restored=Pipeline._unpack_latents(packed,64,128,8)
        torch.testing.assert_close(z,restored,rtol=0,atol=0)

    def test_region_mask_cannot_silently_broadcast_across_samples(self):
        with self.assertRaisesRegex(ValueError,'one sample'):
            object_region_loss(torch.ones(2,16,1,4,8),torch.zeros(2,16,1,4,8),torch.ones(4,8))

    def test_sampler_has_full_coverage_across_epoch_and_restart(self):
        sampler=EpochSampler(7,11)
        ids=[sampler.index(i) for i in range(21)]
        for begin in (0,7,14):self.assertEqual(set(ids[begin:begin+7]),set(range(7)))
        restored=EpochSampler(7,11)
        self.assertEqual([restored.index(i) for i in range(9,21)],ids[9:])

    def test_saved_adam_and_step_rng_match_uninterrupted_training(self):
        torch.manual_seed(8);features=torch.randn(42,12)
        initial=WorldBranch(32,features,(0,2),16).state_dict()
        def fresh():
            model=oracle_test.model(True)
            world=WorldBranch(32,features,(0,2),16);world.load_state_dict(initial);world.attach(model)
            return model,world,torch.optim.AdamW(world.parameters(),lr=1e-4,eps=1e-6)
        kwargs=oracle_test.inputs()
        graph=oracle_test.tensorize(oracle_test.graph(),3)
        def steps(model,world,optimizer,begin,end):
            sampler=EpochSampler(3,21)
            for step in range(begin,end):
                seed_step(21,step)
                for group in optimizer.param_groups:group['lr']=learning_rate(1e-4,2,step)
                optimizer.zero_grad(set_to_none=True)
                for micro in range(2):
                    sid=sampler.index(step*2+micro)
                    local=dict(kwargs);local['hidden_states']=torch.randn_like(kwargs['hidden_states'])
                    g=tuple(t.clone() for t in graph);g[1][...,0]+=sid
                    world.set_world(g)
                    out=model(**local)[0][:,:32].transpose(1,2).reshape(1,16,1,4,8)
                    world.set_world(None)
                    target=torch.randn_like(out)
                    loss=object_region_loss(out,target,torch.ones(4,8))
                    (loss/2).backward()
                torch.nn.utils.clip_grad_norm_(world.parameters(),1.)
                optimizer.step()
        m1,w1,o1=fresh();steps(m1,w1,o1,0,3)
        m2,w2,o2=fresh();steps(m2,w2,o2,0,1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            atomic_tensors(root/'world.safetensors',w2.state_dict());torch.save(o2.state_dict(),root/'optimizer.pt')
            from safetensors.torch import load_file
            m3,w3,o3=fresh();w3.load_state_dict(load_file(str(root/'world.safetensors')))
            o3.load_state_dict(torch.load(root/'optimizer.pt',weights_only=True))
            steps(m3,w3,o3,1,3)
        for key,value in w1.state_dict().items():torch.testing.assert_close(value,w3.state_dict()[key],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in m1.parameters()))


if __name__=='__main__':unittest.main()
