"""CPU numerical tests for the audited v2 training/inference contract."""
import copy
import random
import unittest
import torch
from ..oracle_test import graph, model, inputs
from ..oracle_model import tensorize
from ..oracle_memory import training_autocast
from .runtime import WorldBranch, EpochSampler, seed_step, learning_rate
from .binding import apply_scale


class RuntimeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_full_coverage_and_restart(self):
        s=EpochSampler(1262,0)
        a=[s.index(i) for i in range(4000)]
        for offset in (0,1262,2524):
            self.assertEqual(sorted(a[offset:offset+1262]),list(range(1262)))
        resumed=EpochSampler(1262,0)
        self.assertEqual(a[2000:],[resumed.index(i) for i in range(2000,4000)])
        counts=torch.bincount(torch.tensor(a),minlength=1262)
        self.assertEqual((int(counts.min()),int(counts.max())),(3,4))

    def test_warmup_absolute_step(self):
        self.assertAlmostEqual(learning_rate(1e-4,100,0),1e-6)
        self.assertAlmostEqual(learning_rate(1e-4,100,49),5e-5)
        self.assertEqual(learning_rate(1e-4,100,99),1e-4)
        self.assertEqual(learning_rate(1e-4,100,1000),1e-4)

    def test_precision_match_and_reference_unchanged(self):
        torch.manual_seed(0)
        w=WorldBranch(32,(0,),16).eval()
        w.branches['0'].gate.data.fill_(.1)
        hidden=torch.randn(1,12,32).bfloat16()
        features=tensorize(graph(),3)
        with torch.no_grad():
            with training_autocast('cpu'):
                w.set_world(features);training=w.branches['0'](hidden,*w.current,8)
            w.set_world(features);inference=w.branches['0'](hidden,*w.current,8)
        torch.testing.assert_close(training,inference,rtol=0,atol=0)
        torch.testing.assert_close(inference[:,8:],hidden[:,8:],rtol=0,atol=0)

    def test_checkpointed_scoped_precision_backward(self):
        m=model(checkpointed=True).bfloat16()
        apply_scale(m,.5,'default')
        w=WorldBranch(32,(0,2),16)
        for branch in w.branches.values():branch.gate.data.fill_(.1)
        w.attach(m)
        kw=inputs()
        for key in ('hidden_states','encoder_hidden_states'):kw[key]=kw[key].bfloat16()
        with training_autocast('cpu'):
            w.set_world(tensorize(graph(),3));y=m(**kw)[0]
        w.set_world(None)
        y.float().square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in w.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_optimizer_resume_includes_python_yaw_rng(self):
        # Exercise the same absolute-step RNG, sampler, schedule and optimizer state.
        def make():
            p=torch.nn.Parameter(torch.tensor([.25]))
            return p,torch.optim.AdamW([p],lr=1e-4)
        def advance(p,opt,begin,end):
            sampler=EpochSampler(7,0);trace=[]
            for step in range(begin,end):
                seed_step(0,step);opt.zero_grad()
                opt.param_groups[0]['lr']=learning_rate(1e-4,2,step)
                for micro in range(4):
                    idx=sampler.index(step*4+micro)
                    yaw=random.choice((60,180,300));noise=torch.randn(1)
                    (((p-noise-idx/10).square()*yaw/360)/4).sum().backward()
                    trace.append((idx,yaw,float(noise)))
                opt.step()
            return trace
        full,opt=make();trace=advance(full,opt,0,4)
        part,opt2=make();prefix=advance(part,opt2,0,2)
        state=copy.deepcopy(opt2.state_dict());weights=part.detach().clone()
        restored,opt3=make();restored.data.copy_(weights);opt3.load_state_dict(state)
        random.seed(935);torch.manual_seed(928)
        suffix=advance(restored,opt3,2,4)
        self.assertEqual(trace,prefix+suffix)
        torch.testing.assert_close(full,restored,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
