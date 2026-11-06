"""CPU data, recovery, native-loss and evaluation protocol integration checks."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import torch
from safetensors.torch import save_file
from .. import oracle_test
from ..oracle_memory import training_autocast,ActivationStorage
from ..spatial_world.test_spatial import state
from . import run
from .runtime import WorldBranch,VERSION
from .data import inputs,smoke_rows,DEFAULT_PACK,DEFAULT_PANEL,DEFAULT_SUPPORT,load_support
from .train import configuration,draw_latents
from .supervision import sample


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);cls.plan,cls.rows,cls.train,cls.features=inputs()

    def args(self,group):return SimpleNamespace(pack=DEFAULT_PACK,panel=DEFAULT_PANEL,group=group,smoke=False,seed=0,support=DEFAULT_SUPPORT)

    def test_real_panel_and_sparse_labels_match_actual_input_candidates(self):
        from ..object_world.data import tensorize
        from ..spatial_world.projection import erp_support
        self.assertEqual(len(self.train),128);self.assertEqual(len({r['scene_id'] for r in self.train}),61)
        self.assertTrue(all(r['split']=='train' for r in self.train))
        count=0
        for row in self.train:
            items=load_support(DEFAULT_SUPPORT,row['id']);labels,_=sample(items,17)
            _,geometry,mask=tensorize(row);routing,_=erp_support(geometry,mask,64,128,1)
            if len(labels[2]):self.assertTrue((routing[0,labels[0][labels[1]],labels[2]]>0).all())
            count+=len(items)
        self.assertEqual(count,4128)
        self.assertFalse(torch.equal(self.features[22],self.features[21]))

    def test_group_configs_only_change_binding_and_initial_tensors_match(self):
        configs=[configuration(self.plan,self.args(g),self.train) for g in ('A','B')]
        stripped=[{k:v for k,v in c.items() if k not in ('group','binding_weight')} for c in configs]
        self.assertEqual(stripped[0],stripped[1]);states=[]
        for _ in range(2):
            torch.manual_seed(0);world=WorldBranch(32,self.features,(0,2),16,64,1,2);states.append(world.state_dict())
        for k,v in states[0].items():torch.testing.assert_close(v,states[1][k],rtol=0,atol=0)

    def test_latent_inputs_exact_across_arms_and_resume_without_global_rng_side_effect(self):
        from safetensors.torch import load_file
        from ..build_polished_world import read
        vae=read(Path(self.plan['base_snapshot'])/'vae/config.json')
        mean=torch.tensor(vae['latents_mean']).reshape(1,-1,1,1,1);std=torch.tensor(vae['latents_std']).reshape(1,-1,1,1,1)
        posterior=load_file(self.train[0]['posterior_cache'],device='cpu');before=torch.random.get_rng_state().clone()
        a=draw_latents(posterior,mean,std,0,57,1000);b=draw_latents(posterior,mean,std,0,57,1000)
        c=draw_latents(posterior,mean,std,0,58,1000)
        for x,y in zip(a[:2],b[:2]):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertEqual(a[2],b[2]);self.assertTrue(0<=a[2]<1000);self.assertFalse(torch.equal(a[1],c[1]))
        self.assertEqual(a[0].dtype,torch.bfloat16);torch.testing.assert_close(torch.random.get_rng_state(),before,rtol=0,atol=0)

    def test_launcher_requires_smoke_and_recovers_only_correct_arm(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with mock.patch('sys.argv',['run','pilot','--group','A','--root',d]),mock.patch.object(run.subprocess,'run') as execute:
                with self.assertRaises(ValueError):run.main()
                execute.assert_not_called()
            for group in ('A','B'):
                path=root/'smoke'/group/'run-fixture/checkpoint-step000002';path.mkdir(parents=True)
                (path/'COMPLETE.json').write_text(json.dumps(dict(version=VERSION,step=2)))
                setting=self.args(group);setting.smoke=True
                (path/'config.json').write_text(json.dumps(configuration(self.plan,setting,smoke_rows(self.train))))
                (path/'SMOKE_COMPLETE.json').write_text(json.dumps(dict(zero_gate_panorama_exact=True,semantic_gradient=True,value_output_gradient=True)))
                for name in ('world.safetensors','optimizer.pt'):(path/name).touch()
            saved=root/'pilot/A/run-fixture/checkpoint-step000050';saved.mkdir(parents=True)
            (saved/'COMPLETE.json').write_text(json.dumps(dict(version=VERSION,step=50)));(saved/'config.json').write_text(json.dumps(configuration(self.plan,self.args('A'),self.train)))
            for name in ('world.safetensors','optimizer.pt'):(saved/name).touch()
            with mock.patch('sys.argv',['run','pilot','--group','A','--root',d]),mock.patch.object(run.subprocess,'run') as execute:
                run.main();command=execute.call_args.args[0]
            self.assertEqual(command[command.index('--resume')+1],str(saved));self.assertEqual(command[command.index('--steps')+1],'200')
            self.assertEqual(execute.call_count,1)

    def test_actual_panorama_region_objective_two_steps_updates_both_groups(self):
        from qwen_edit_pano.losses import panorama_loss
        from qwen_pano.profiles import REPO_PANORAMA
        from diffusers import QwenImageEditPlusPipeline as Pipeline
        from ..object_world.runtime import object_region_loss
        for beta in (0.,.01):
            model=oracle_test.model(True).bfloat16();torch.manual_seed(9)
            world=WorldBranch(32,torch.randn(42,24),(0,2),16,64,1,2);world.attach(model)
            opt=torch.optim.AdamW(world.parameters(),lr=1e-4,eps=1e-8);kw=oracle_test.inputs()
            for key in ('hidden_states','encoder_hidden_states'):kw[key]=kw[key].bfloat16()
            z=torch.randn(1,4,1,8,16).bfloat16();noise=torch.randn_like(z);args=SimpleNamespace(**REPO_PANORAMA)
            for step in range(2):
                opt.zero_grad(set_to_none=True);world.set_world(state())
                chosen=world.current[2][0,:,1:].nonzero()[0];labels=(torch.tensor([int(chosen[0])]),torch.tensor([0]),torch.tensor([int(chosen[1])+1]),torch.ones(1))
                if beta:world.set_supervision(labels)
                with ActivationStorage(device_type='cpu').context():
                    with training_autocast('cpu'):
                        packed=model(**kw)[0][:,:32];prediction=Pipeline._unpack_latents(packed,64,128,8)
                        value,_=panorama_loss(prediction,noise-z,torch.ones(1,1,8,16,dtype=torch.bool),torch.ones(1,dtype=torch.bool),
                            torch.ones(1),args.geometry_start_epoch,args,noise=noise,clean=z)
                        loss=value+.25*object_region_loss(prediction,noise-z,torch.ones(8,16))
                        if beta:loss=loss+beta*world.take_binding_loss()
                    world.clear_context();loss.backward()
                self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is None for p in model.parameters()))
                opt.step()
            self.assertGreater(float(world.encoder.semantic[0].weight.grad.abs().sum()),0)
            self.assertTrue(any(p.grad is not None and bool(p.grad.count_nonzero()) for n,p in world.named_parameters() if '.v.weight' in n or '.out.weight' in n))

    def test_cpu_evaluation_preflight_on_explicit_untrained_fixture(self):
        from . import evaluate
        from ..build_polished_world import read
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);arch=read(Path(self.plan['base_snapshot'])/'transformer/config.json')
            torch.manual_seed(0);world=WorldBranch(arch['num_attention_heads']*arch['attention_head_dim'],self.features)
            paths={}
            for group in ('A','B'):
                path=base/group/'run-cpu-untrained-fixture/checkpoint-step000200';path.mkdir(parents=True);paths[group]=path
                save_file(world.state_dict(),str(path/'world.safetensors'));save_file(world.state_dict(),str(path.parent/'initial_world.safetensors'))
                (path/'config.json').write_text(json.dumps(configuration(self.plan,self.args(group),self.train)))
                (path/'COMPLETE.json').write_text(json.dumps(dict(version=VERSION,step=200)))
            plan,byid,features,protocol,output=evaluate.load_protocol(paths,base)
            evaluate.diagnose(plan,byid,features,protocol,output/'diagnose',paths,True)
            evaluate.render(plan,byid,features,protocol,output/'render',paths,True)
            self.assertFalse(output.exists())
            chosen=evaluate.render_scenes(protocol,byid)
            self.assertEqual([byid[s]['split'] for s in chosen],['train','train','val','val'])
            for sid in chosen:
                for mode in ('wrong_category','wrong_position'):
                    altered,meta=evaluate.render_variant(byid[sid],mode);self.assertTrue(meta['changed_slots'])


if __name__=='__main__':unittest.main()
