"""CPU-only full-distribution/recovery/replicated-gradient checks."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import load_file,save_file

from ..build_polished_world import atomic_json,read
from ..object_world.data import tensorize
from ..oracle_memory import training_autocast,ActivationStorage
from ..scaled_world.runtime import EpochSampler
from ..id_binding_world.parallel import seed_microbatch
from ..spatial_world.test_spatial import state
from . import full_data as fd
from . import full_run
from .full_train import sample_indices,resume_step
from .data import ensure_support,load_support,check_rows
from .supervision import sample
from .test_category_contrast import tiny
from .continue_train import predict
from .runtime import WorldBranch


def four_micro_gradients(ranks):
    from qwen_edit_pano.losses import panorama_loss
    from qwen_edit_pano.world_pipeline.object_world.runtime import object_region_loss
    from qwen_pano.profiles import REPO_PANORAMA
    from types import SimpleNamespace
    args=SimpleNamespace(**REPO_PANORAMA);outputs=[]
    torch.manual_seed(42);clean=[torch.randn(1,4,1,8,16).bfloat16() for _ in range(4)]
    noise=[torch.randn_like(x) for x in clean]
    for rank in range(ranks):
        model,world,kw=tiny(True)
        for data_index,noise_index in sample_indices(0,ranks,rank):
            seed_microbatch(0,noise_index)
            s=state();world.set_world(s);chosen=world.current[2][0,:,1:].nonzero()[0]
            labels=(torch.tensor([int(chosen[0])]),torch.tensor([0]),torch.tensor([int(chosen[1])+1]),torch.ones(1))
            with ActivationStorage(device_type='cpu').context():
                with training_autocast('cpu'):
                    pred=predict(model,world,s,kw,64,labels);target=noise[data_index]-clean[data_index]
                    base,_=panorama_loss(pred,target,torch.ones(1,1,8,16,dtype=torch.bool),torch.ones(1,dtype=torch.bool),
                        torch.ones(1),args.geometry_start_epoch,args,noise=noise[data_index],clean=clean[data_index])
                    loss=base+.25*object_region_loss(pred,target,torch.ones(8,16))+.01*world.take_binding_loss()
                (loss/(4//ranks)).backward()
            world.clear_context()
        assert all(p.grad is None for p in model.parameters())
        outputs.append({n:p.grad.clone() for n,p in world.named_parameters() if p.grad is not None})
    return {key:sum(r[key] for r in outputs)/ranks for key in outputs[0]}


class FullTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);cls.plan,cls.allrows,cls.rows,cls.features=fd.full_inputs()
        cls.config=fd.configuration(cls.plan,cls.rows)

    def test_real_full_split_all_caches_and_max_capacity(self):
        self.assertEqual(len(self.rows),7390);self.assertEqual(len(self.allrows)-len(self.rows),923)
        self.assertEqual(len({r['scene_id'] for r in self.rows}),61)
        train={r['scene_id'] for r in self.rows};val={r['scene_id'] for r in self.allrows if r['split']=='val'}
        self.assertFalse(train&val);check_rows(self.rows,self.plan)
        largest=max(self.rows,key=lambda r:len(r['objects']));self.assertEqual(len(largest['objects']),215)
        tensors=tensorize(largest);self.assertEqual(tensors[0].shape,(1,216))
        with tempfile.TemporaryDirectory() as d:
            summary=ensure_support(largest,self.plan,Path(d));labels,_=sample(load_support(Path(d),largest['id']),7)
            self.assertGreater(summary['eligible_instances'],0);self.assertGreater(len(labels[2]),0)
            from ..spatial_world.projection import erp_support
            route,_=erp_support(tensors[1],tensors[2],64,128,1)
            self.assertTrue((route[0,labels[0][labels[1]],labels[2]]>0).all())

    def test_source_C600_world_and_adam_no_reinitialization(self):
        arch=read(Path(self.plan['base_snapshot'])/'transformer/config.json')
        world=WorldBranch(arch['num_attention_heads']*arch['attention_head_dim'],self.features)
        saved=load_file(str(fd.SOURCE/'world.safetensors'));world.load_state_dict(saved,strict=True)
        for k,v in world.state_dict().items():torch.testing.assert_close(v,saved[k],rtol=0,atol=0)
        original=torch.load(fd.SOURCE/'optimizer.pt',map_location='cpu',weights_only=True)
        opt=torch.optim.AdamW(world.parameters(),lr=self.config['lr'],betas=tuple(self.config['betas']),eps=self.config['adam_eps'])
        opt.load_state_dict(original)
        self.assertEqual(max(float(v['step']) for v in opt.state.values()),600.)
        for v in opt.state.values():
            self.assertEqual(v['exp_avg'].dtype,torch.float32);self.assertEqual(v['exp_avg_sq'].dtype,torch.float32)
            self.assertTrue(all(torch.isfinite(t).all() for t in v.values() if isinstance(t,torch.Tensor)))
        self.assertEqual(self.config['category_contrast_weight'],0.);self.assertEqual(self.config['region_weight'],.25)
        self.assertEqual(self.config['binding_weight'],.01);self.assertEqual(self.config['pano_lora_scale'],.5)

    def test_full_epoch_no_skip_or_val_and_same_one_two_rank_noise_stream(self):
        sampler=EpochSampler(7390,0);first=[sampler.index(i) for i in range(7390)]
        self.assertEqual(set(first),set(range(7390)))
        for step in (0,1,250,1999):
            one=sample_indices(step);two=sample_indices(step,2,0)+sample_indices(step,2,1)
            self.assertEqual(one,two)
            self.assertEqual([x[0] for x in one],list(range(step*4,step*4+4)))
            self.assertEqual([x[1] for x in one],list(range(2400+step*4,2400+step*4+4)))

    def test_actual_C_objective_replica_average_equals_single_global_batch(self):
        one=four_micro_gradients(1);two=four_micro_gradients(2);self.assertEqual(one.keys(),two.keys())
        for k in one:torch.testing.assert_close(one[k],two[k],rtol=5e-6,atol=2e-8,msg=k)
        self.assertTrue(any(bool(v.count_nonzero()) for k,v in one.items() if '.v.weight' in k))
        self.assertTrue(bool(one['encoder.semantic.0.weight'].count_nonzero()))

    def test_no_eligible_binding_labels_keeps_main_loss_and_finite_gradients(self):
        model,world,kw=tiny(True)
        empty=(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),
               torch.empty(0,dtype=torch.long),torch.empty(0))
        with ActivationStorage(device_type='cpu').context():
            with training_autocast('cpu'):
                pred=predict(model,world,state(),kw,64,empty);binding=world.take_binding_loss()
                loss=pred.float().square().mean()+.01*binding
            self.assertEqual(float(binding),0.);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in world.parameters()))
        self.assertTrue(bool(world.encoder.semantic[0].weight.grad.count_nonzero()))

    def test_launcher_requires_correct_gpu_smoke_and_can_extend_same_training(self):
        with tempfile.TemporaryDirectory() as d,mock.patch.object(fd,'require_prepared'):
            root=Path(d)
            with mock.patch('sys.argv',['full','train','--gpus','2','--root',d]),mock.patch.object(full_run.subprocess,'run') as execute:
                with self.assertRaises(ValueError):full_run.main()
                execute.assert_not_called()
            smoke=root/'smoke/2/run-explicit-CPU-fixture/checkpoint-step000002';smoke.mkdir(parents=True)
            atomic_json(smoke/'COMPLETE.json',dict(version=fd.VERSION,step=2));atomic_json(smoke/'config.json',fd.configuration(self.plan,self.rows,True))
            atomic_json(smoke/'SMOKE_COMPLETE.json',dict(version=fd.VERSION,replicas=2,maximum_objects=215,base_trainable=0,
                semantic_gradient=True,value_output_gradient=True))
            for name in ('world.safetensors','optimizer.pt'):(smoke/name).touch()
            checkpoint=root/'train/run-explicit-CPU-fixture/checkpoint-step000250';checkpoint.mkdir(parents=True)
            atomic_json(checkpoint/'COMPLETE.json',dict(version=fd.VERSION,step=250));atomic_json(checkpoint/'config.json',self.config)
            for name in ('world.safetensors','optimizer.pt'):(checkpoint/name).touch()
            for target in (2000,10000):
                with mock.patch('sys.argv',['full','train','--gpus','2','--steps',str(target),'--root',d]),mock.patch.object(full_run.subprocess,'run') as execute:
                    full_run.main();command=execute.call_args.args[0]
                    self.assertEqual(command[command.index('--resume')+1],str(checkpoint));self.assertEqual(execute.call_count,1)
                    self.assertIn('torch.distributed.run',command);self.assertIn('--nproc_per_node=2',command)
                    self.assertEqual(command[command.index('--steps')+1],str(target))
            start,path=resume_step(checkpoint,self.config);self.assertEqual(start,250)
            self.assertEqual(resume_step(None,self.config),(0,fd.SOURCE))
            with self.assertRaises(ValueError):resume_step(checkpoint,dict(self.config,region_weight=.5))

    def test_cpu_evaluation_of_explicit_untrained_fixture_uses_same_objects(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);checkpoint=root/'train/run-explicit-untrained-CPU-fixture/checkpoint-step002000';checkpoint.mkdir(parents=True)
            atomic_json(checkpoint/'COMPLETE.json',dict(version=fd.VERSION,step=2000));atomic_json(checkpoint/'config.json',self.config)
            state0=load_file(str(fd.SOURCE/'world.safetensors'));save_file(state0,str(checkpoint/'world.safetensors'));(checkpoint/'optimizer.pt').touch()
            for stage in ('diagnose','render'):full_run.evaluation(root,stage,True)
            self.assertFalse((root/'evaluation').exists())


if __name__=='__main__':unittest.main()
