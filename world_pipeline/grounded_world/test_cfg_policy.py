"""CPU checks: CFG scoping, restoration, and actual tiny Qwen World bypass."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest

from .cfg_policy import world_cfg_policy


class FakeTransformer:
    @contextmanager
    def cache_context(self,name):yield


class PolicyTests(unittest.TestCase):
    def test_positive_only_and_restore_on_exception(self):
        model=FakeTransformer();token=object();world=SimpleNamespace(current=token)
        with world_cfg_policy(model,world,'positive_only') as calls:
            with model.cache_context('cond'):self.assertIs(world.current,token)
            with self.assertRaisesRegex(RuntimeError,'test'):
                with model.cache_context('uncond'):
                    self.assertIsNone(world.current);raise RuntimeError('test')
            self.assertIs(world.current,token)
        self.assertNotIn('cache_context',model.__dict__)
        self.assertEqual(calls,dict(policy='positive_only',cond=1,uncond=1,disabled_uncond=1))

    def test_both_preserves_world_for_both_branches(self):
        model=FakeTransformer();token=object();world=SimpleNamespace(current=token)
        with world_cfg_policy(model,world,'both') as calls:
            for name in ('cond','uncond'):
                with model.cache_context(name):self.assertIs(world.current,token)
        self.assertEqual(calls['disabled_uncond'],0)

    def test_tiny_native_Qwen_negative_equals_world_off(self):
        import torch
        from .test_category_contrast import tiny
        from ..spatial_world.test_spatial import state
        from ..oracle_memory import training_autocast
        torch.set_num_threads(2);model,world,kw=tiny(False);model.eval();world.eval()
        with torch.inference_mode(),training_autocast('cpu'):
            world.set_world(state())
            with world_cfg_policy(model,world,'positive_only'):
                with model.cache_context('cond'):positive=model(**kw)[0]
                with model.cache_context('uncond'):negative=model(**kw)[0]
            self.assertIsNotNone(world.current)
            world.clear_context();off=model(**kw)[0]
        torch.testing.assert_close(negative,off,rtol=0,atol=0)
        self.assertGreater(int((positive!=off).count_nonzero()),0)


if __name__=='__main__':unittest.main()
