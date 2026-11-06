"""CPU contract and integration tests; no production weights or GPU execution."""
import unittest
from unittest.mock import patch
import torch
from .. import oracle_test
from ..scaled_world import test_runtime as baseline_tests
from ..scaled_world.runtime import WorldBranch as BaselineWorldBranch
from .runtime import WorldBranch, GeometryFeatures, ENCODING
from .binding import verify_binding


class FourierOracleTests(oracle_test.OracleTests):
    def setUp(self):
        super().setUp()
        self.patcher = patch.object(oracle_test, 'WorldBranch', WorldBranch)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_checkpoint_backward_explicit_world_and_freeze(self):
        # Do not deepcopy modules with scoped-autocast bound forward closures:
        # Python function copies keep the original branch binding. Production
        # restores state into a fresh module, which this fixture must also do.
        first=WorldBranch(32,(0,2),16)
        for branch in first.branches.values():branch.gate.data.fill_(.1)
        second=WorldBranch(32,(0,2),16)
        second.load_state_dict(first.state_dict())
        gradients=[]
        for checkpointed,w in ((False,first),(True,second)):
            m=oracle_test.model(checkpointed)
            w.attach(m);w.set_world(oracle_test.tensorize(oracle_test.graph(),3))
            y=m(**oracle_test.inputs())[0]
            w.set_world(None)
            y.square().mean().backward()
            self.assertTrue(all(p.grad is None for p in m.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in w.encoder.parameters()))
            gradients.append({k:p.grad.clone() for k,p in w.named_parameters() if p.grad is not None})
        self.assertEqual(gradients[0].keys(),gradients[1].keys())
        for key in gradients[0]:
            torch.testing.assert_close(gradients[0][key],gradients[1][key],rtol=0,atol=0)


class FourierRuntimeTests(baseline_tests.RuntimeTests):
    def setUp(self):
        super().setUp()
        self.patcher = patch.object(baseline_tests, 'WorldBranch', WorldBranch)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)


class EncodingTests(unittest.TestCase):
    def test_formula_masks_and_known_zero(self):
        f=GeometryFeatures()
        g=torch.tensor([[[0., -1., 2., 1., 2., 3., 1., 1.]]])
        y=f(g)
        self.assertEqual(y.shape[-1],392)
        pe=y[...,8:].reshape(1,1,6,64)
        self.assertEqual(float(pe[0,0,0,32]),1.)
        self.assertAlmostEqual(float(pe[0,0,1,0]),float(torch.sin(torch.tensor(-1.))))
        self.assertAlmostEqual(float(pe[0,0,2,31]),float(torch.sin(torch.tensor(256.))),places=5)
        g[...,6]=0;g[...,:3]=float('nan')
        missing=f(g)
        self.assertTrue(torch.isfinite(missing).all())
        self.assertEqual(int(torch.count_nonzero(missing[...,8:8+3*64])),0)
        self.assertEqual(int(torch.count_nonzero(missing[...,:3])),0)

    def test_trig_stays_fp32_and_direction_is_distinguished(self):
        f=GeometryFeatures()
        g=torch.tensor([[[1.2345,0.,3.4567,1.,2.,3.,1.,1.]]])
        a=f(g)
        with torch.autocast('cpu',dtype=torch.bfloat16):b=f(g)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        back=g.clone();back[...,2]*=-1
        self.assertFalse(torch.equal(a,f(back)))
        # Cartesian encoding is continuous across the ERP back-direction seam.
        left=g.clone();left[...,:3]=torch.tensor([1e-6,0.,-3.])
        right=left.clone();right[...,0]=-1e-6
        self.assertLess(float((f(left)-f(right)).abs().max()),.0003)

    def test_unchanged_adapter_initialization_and_rng(self):
        torch.manual_seed(42);old=BaselineWorldBranch(32,(0,2),16);a=torch.rand(3)
        torch.manual_seed(42);new=WorldBranch(32,(0,2),16);b=torch.rand(3)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for key,value in old.state_dict().items():
            if not key.startswith('encoder.geometry.'):
                torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
        with self.assertRaises(RuntimeError):new.load_state_dict(old.state_dict(),strict=True)

    def test_reject_wrong_checkpoint_contract(self):
        with self.assertRaises(ValueError):
            verify_binding({'experiment_version':'frozen_pano_scale_world_v2'}, {})
        with self.assertRaises(ValueError):
            verify_binding({'experiment_version':'frozen_pano_scale_fourier_world_v1',
                            'geometry_encoding':dict(ENCODING,coordinate_scale_m=10.)}, {})


if __name__=='__main__':unittest.main()
