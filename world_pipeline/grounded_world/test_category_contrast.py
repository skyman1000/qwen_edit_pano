"""Meaningful CPU checks of paired gradients, frozen source, data and recovery."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import torch
from safetensors.torch import load_file,save_file
from .. import oracle_test
from ..oracle_memory import training_autocast,ActivationStorage
from ..spatial_world.test_spatial import state
from .runtime import WorldBranch
from .data import inputs
from . import category_contrast as cc
from .continue_train import configuration,predict,error_forward
from . import continue_run as run


def tiny(checkpointed=True):
    model=oracle_test.model(checkpointed).bfloat16();torch.manual_seed(9)
    world=WorldBranch(32,torch.randn(42,24),(0,2),16,64,1,2)
    for b in world.branches.values():b.gate.data.fill_(.05)
    world.attach(model);kw=oracle_test.inputs()
    for key in ('hidden_states','encoder_hidden_states'):kw[key]=kw[key].bfloat16()
    return model,world,kw


def pair_gradients(serial,checkpointed=True):
    model,world,kw=tiny(checkpointed);s=state();entry=dict(slot=1,category_id=int(s[0][0,1]),wrong_category_id=14)
    neg=cc.negative_state(s,entry);target=torch.randn(1,4,1,8,16).bfloat16();weights=torch.rand(8,16);weights/=weights.sum()
    store=ActivationStorage(device_type='cpu')
    if serial:
        n0=error_forward(model,world,neg,kw,target,weights,64);world.clear_context()
        with store.context():
            with training_autocast('cpu'):
                output=predict(model,world,s,kw,64);positive=cc.regional_error(output,target,weights)
                base=(output.float()-target.float()).square().mean();loss,cp,cn=cc.coefficients(positive.detach(),n0)
                backward=base+.01*cp*positive
            world.clear_context();backward.backward()
        del output,positive,base,backward
        with store.context():
            negative=error_forward(model,world,neg,kw,target,weights,64,True)
            torch.testing.assert_close(negative.detach(),n0,rtol=0,atol=0)
            world.clear_context();(.01*cn*negative).backward()
    else:
        with store.context():
            with training_autocast('cpu'):
                output=predict(model,world,s,kw,64);positive=cc.regional_error(output,target,weights)
                base=(output.float()-target.float()).square().mean();negative=error_forward(model,world,neg,kw,target,weights,64,True)
                loss=cc.objective(positive,negative);backward=base+.01*loss
            world.clear_context();backward.backward()
    assert all(p.grad is None for p in model.parameters())
    return {n:p.grad.clone() for n,p in world.named_parameters() if p.grad is not None}


class MathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_bounded_objective_and_chain_coefficients_both_branches(self):
        for a,b in ((.1,.1),(.2,.3),(10.,.1),(0.,1000.)):
            pos=torch.tensor(a,requires_grad=True);neg=torch.tensor(b,requires_grad=True)
            loss=cc.objective(pos,neg);loss.backward();value,cp,cn=cc.coefficients(pos.detach(),neg.detach())
            torch.testing.assert_close(loss,value,rtol=0,atol=0);torch.testing.assert_close(pos.grad,cp);torch.testing.assert_close(neg.grad,cn)
            self.assertTrue(torch.isfinite(value));self.assertGreaterEqual(float(cp),0);self.assertLessEqual(float(cn),0)
        self.assertLess(cc.objective(torch.tensor(.1),torch.tensor(.3)),cc.objective(torch.tensor(.3),torch.tensor(.1)))
        self.assertEqual(cc.weight(200),.0002);self.assertEqual(cc.weight(249),.01);self.assertEqual(cc.weight(399),.01)

    def test_serial_backward_matches_joint_real_tiny_edit_and_checkpoint_offload(self):
        joint=pair_gradients(False,True);serial=pair_gradients(True,True);plain=pair_gradients(True,False)
        self.assertEqual(joint.keys(),serial.keys())
        for key in joint:
            torch.testing.assert_close(serial[key],joint[key],rtol=.03,atol=2e-5,msg=key)
            torch.testing.assert_close(serial[key],plain[key],rtol=0,atol=0,msg=key)
        self.assertTrue(any(bool(g.count_nonzero()) for key,g in serial.items() if '.v.weight' in key))

    def test_positive_unpack_excludes_reference_and_control_prediction_matches_original(self):
        from diffusers import QwenImageEditPlusPipeline as Pipeline
        model,world,kw=tiny();world.set_world(state())
        with torch.no_grad(),training_autocast('cpu'):
            original=Pipeline._unpack_latents(model(**kw)[0][:,:32],64,128,8)
            actual=predict(model,world,state(),kw,64)
        self.assertEqual(actual.shape,(1,4,1,8,16));torch.testing.assert_close(actual,original,rtol=0,atol=0)

    def test_optimizer_resume_serial_updates_exact(self):
        model,world,kw=tiny();opt=torch.optim.AdamW(world.parameters(),lr=1e-4,eps=1e-8)
        target=torch.zeros(1,4,1,8,16);weights=torch.ones(8,16)/128;s=state()
        def step(w,m,o,k):
            o.zero_grad(set_to_none=True)
            with training_autocast('cpu'):
                output=predict(m,w,s,k,64);error=cc.regional_error(output,target,weights)
            w.clear_context();error.backward();torch.nn.utils.clip_grad_norm_(w.parameters(),1.);o.step()
        step(world,model,opt,kw);saved=copy.deepcopy(world.state_dict());moments=copy.deepcopy(opt.state_dict());step(world,model,opt,kw)
        m,w,k=tiny();w.load_state_dict(saved);o=torch.optim.AdamW(w.parameters(),lr=1e-4,eps=1e-8);o.load_state_dict(moments);step(w,m,o,k)
        for key,value in world.state_dict().items():torch.testing.assert_close(value,w.state_dict()[key],rtol=0,atol=0)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);cls.prepared=cc.prepare();cls.plan,cls.allrows,cls.rows,cls.features=inputs()

    def test_real_targets_one_category_only_gt_support_normalized_no_global_rng_consumption(self):
        count=0;before=torch.random.get_rng_state().clone()
        for row in self.rows:
            chosen=cc.select_target(self.prepared['targets'][row['id']],self.plan['names'],2026)
            if chosen is None:continue
            s=tensorize(row);n=cc.negative_state(s,chosen);difference=(n[0]!=s[0]).nonzero()
            self.assertEqual(difference.tolist(),[[0,chosen['slot']]])
            torch.testing.assert_close(s[1],n[1],rtol=0,atol=0);torch.testing.assert_close(s[2],n[2],rtol=0,atol=0)
            weights=cc.instance_weights(row,chosen,self.plan['height']);torch.testing.assert_close(weights.sum(),torch.tensor(1.))
            self.assertGreaterEqual(chosen['pixels'],256);self.assertGreaterEqual(chosen['packed_cells'],4);count+=1
        self.assertEqual(count,self.prepared['eligible_images']);torch.testing.assert_close(before,torch.random.get_rng_state(),rtol=0,atol=0)

    def test_source_world_adam_exact_and_configs_only_change_category_loss(self):
        a=configuration(self.plan,self.rows,self.prepared,'C');b=configuration(self.plan,self.rows,self.prepared,'D')
        strip=lambda c:{k:v for k,v in c.items() if k not in ('group','category_contrast_weight')}
        self.assertEqual(strip(a),strip(b));self.assertEqual(a['binding_weight'],.01);self.assertEqual(a['region_weight'],.25)
        source=load_file(str(cc.INITIAL/'world.safetensors'));arch=json.loads((Path(self.plan['base_snapshot'])/'transformer/config.json').read_text())
        world=WorldBranch(arch['num_attention_heads']*arch['attention_head_dim'],self.features);world.load_state_dict(source,strict=True)
        opt=torch.optim.AdamW(world.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=1e-5,eps=1e-8)
        original=torch.load(cc.INITIAL/'optimizer.pt',map_location='cpu',weights_only=True);opt.load_state_dict(original)
        self.assertEqual(max(float(v['step']) for v in opt.state.values()),200.)
        for entry in opt.state.values():
            self.assertTrue(all(torch.isfinite(v).all() for v in entry.values() if isinstance(v,torch.Tensor)))
            self.assertEqual(entry['exp_avg'].dtype,torch.float32);self.assertEqual(entry['exp_avg_sq'].dtype,torch.float32)

    def test_launcher_smoke_requirement_correct_arm_recovery_no_auto_eval(self):
        from ..build_polished_world import atomic_json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with mock.patch('sys.argv',['run','train','--group','D','--root',d]),mock.patch.object(run.subprocess,'run') as execute:
                with self.assertRaises(ValueError):run.main()
                execute.assert_not_called()
            for group in ('C','D'):
                path=root/'smoke'/group/'run-fixture/checkpoint-step000202';path.mkdir(parents=True)
                atomic_json(path/'config.json',configuration(self.plan,self.rows,self.prepared,group,True));atomic_json(path/'COMPLETE.json',dict(version=cc.EXPERIMENT,step=202))
                atomic_json(path/'SMOKE_COMPLETE.json',dict(base_trainable=0,actual_value_output_gradient_verified=group=='D'))
                for name in ('world.safetensors','optimizer.pt'):(path/name).touch()
            path=root/'train/D/run-fixture/checkpoint-step000350';path.mkdir(parents=True)
            atomic_json(path/'config.json',configuration(self.plan,self.rows,self.prepared,'D'));atomic_json(path/'COMPLETE.json',dict(version=cc.EXPERIMENT,step=350))
            for name in ('world.safetensors','optimizer.pt'):(path/name).touch()
            with mock.patch('sys.argv',['run','train','--group','D','--root',d]),mock.patch.object(run.subprocess,'run') as execute:
                run.main();command=execute.call_args.args[0]
                self.assertEqual(command[command.index('--resume')+1],str(path));self.assertEqual(execute.call_count,1)

    def test_cpu_evaluation_protocol_single_object_both_arms_and_old_pipeline_compatible(self):
        from .continue_evaluate import protocol,single_variant
        from . import evaluate as ev
        from ..build_polished_world import atomic_json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);paths={};state0=load_file(str(cc.INITIAL/'world.safetensors'))
            for group in ('C','D'):
                path=root/group/'run-explicit-untrained-CPU-fixture/checkpoint-step000600';path.mkdir(parents=True);paths[group]=path
                save_file(state0,str(path/'world.safetensors'));save_file(state0,str(path.parent/'initial_world.safetensors'))
                atomic_json(path/'config.json',configuration(self.plan,self.rows,self.prepared,group));atomic_json(path/'COMPLETE.json',dict(version=cc.EXPERIMENT,step=600))
            plan,byid,features,p,out=protocol(paths,root,self.prepared)
            self.assertEqual(len(p['single_object_render']['ids']),4)
            for sid,target in p['single_object_render']['targets'].items():
                changed,meta=single_variant(byid[sid],target);self.assertEqual(len(meta['changed_slots']),1)
                self.assertGreaterEqual(target['pixels'],256)
            ev.diagnose(plan,byid,features,p,out/'diagnose',paths,True);ev.render(plan,byid,features,p,out/'render',paths,True)
            self.assertFalse(out.exists())


from ..object_world.data import tensorize
if __name__=='__main__':unittest.main()
