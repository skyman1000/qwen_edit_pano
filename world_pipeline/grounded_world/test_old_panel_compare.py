"""CPU comparison checks: preserve legacy/current forwards, no mixed hooks."""
import copy
import unittest

import torch
from diffusers import QwenImageTransformer2DModel

from .old_panel_compare import make_world
from .runtime import WorldBranch as CurrentWorld
from ..id_binding_world.runtime import WorldBranch as OldWorld
from ..oracle_memory import training_autocast
from ..spatial_world.test_spatial import state
from ..oracle_test import inputs
from qwen_edit_pano.paired_circular import install_circular_padding
from .cfg_policy import world_cfg_policy


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_factory_keeps_two_checkpoint_architectures_distinct(self):
        features=torch.randn(42,24);plan=dict(height=64,padding_columns=1)
        config=dict(blocks=[20,40,59],width=256,binding_block=40)
        self.assertIsNone(make_world('pano_only',32,features,plan,config))
        self.assertIsInstance(make_world('A10000',32,features,plan,config),OldWorld)
        self.assertIsInstance(make_world('C6000',32,features,plan,config),CurrentWorld)
        with self.assertRaises(ValueError):make_world('unknown',32,features,plan,config)

    def test_tiny_native_forward_matches_each_original_implementation(self):
        torch.manual_seed(31)
        base=QwenImageTransformer2DModel(patch_size=2,in_channels=16,out_channels=4,num_layers=60,
            attention_head_dim=16,num_attention_heads=2,joint_attention_dim=32,
            axes_dims_rope=(4,6,6),zero_cond_t=True).bfloat16().eval().requires_grad_(False)
        features=torch.randn(42,24);plan=dict(height=64,padding_columns=1)
        config=dict(blocks=[20,40,59],width=256,binding_block=40)
        for arm in ('A10000','C6000'):
            with self.subTest(arm=arm):
                actual=make_world(arm,32,features,plan,config).eval().requires_grad_(False)
                expected=(OldWorld(32,config['blocks'],256,40) if arm=='A10000' else
                    CurrentWorld(32,features,height=64,padding=1)).eval().requires_grad_(False)
                for branch in actual.branches.values():branch.gate.data.fill_(.05)
                expected.load_state_dict(actual.state_dict(),strict=True)
                ma,me,plain=copy.deepcopy(base),copy.deepcopy(base),copy.deepcopy(base)
                for model in (ma,me,plain):install_circular_padding(model,1,'temporary')
                actual.attach(ma);expected.attach(me)
                kw=inputs()
                for key in ('hidden_states','encoder_hidden_states'):kw[key]=kw[key].bfloat16()
                with torch.inference_mode(),training_autocast('cpu'):
                    actual.set_world(state());expected.set_world(state())
                    reference=me(**kw)[0]
                    with world_cfg_policy(ma,actual,'both') as calls:
                        with ma.cache_context('cond'):positive=ma(**kw)[0]
                        with ma.cache_context('uncond'):negative=ma(**kw)[0]
                    torch.testing.assert_close(positive,reference,rtol=0,atol=0)
                    torch.testing.assert_close(negative,reference,rtol=0,atol=0)
                    self.assertEqual(calls,dict(policy='both',cond=1,uncond=1,disabled_uncond=0))
                    actual.clear_context();expected.clear_context()
                    torch.testing.assert_close(ma(**kw)[0],me(**kw)[0],rtol=0,atol=0)
                    torch.testing.assert_close(ma(**kw)[0],plain(**kw)[0],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
