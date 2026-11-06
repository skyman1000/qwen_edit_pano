"""CPU checks of local-only optimization on the real tiny Edit transformer."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch
from .. import oracle_test
from ..oracle_memory import training_autocast,ActivationStorage
from ..id_binding_world.parallel import micro_indices
from .runtime import WorldBranch,VERSION,LEGACY_VERSION
from .test_spatial import state
from . import calibration


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def model(self,checkpointed=True):
        model=oracle_test.model(checkpointed).bfloat16();torch.manual_seed(9)
        world=WorldBranch(32,(0,2),16,64,1)
        for branch in world.branches.values():branch.gate.data.fill_(.1);branch.local.gate.data.fill_(.005)
        world.attach(model)
        kw=oracle_test.inputs()
        for key in ('hidden_states','encoder_hidden_states'):kw[key]=kw[key].bfloat16()
        return model,world,kw

    def test_initialization_distinguishes_warm_start_and_joint_resume(self):
        config=dict(version=VERSION,group='A',spatial_mode='spatial',binding_weight=0.,initial_step=10000,smoke=False)
        complete=dict(version=VERSION,step=200)
        self.assertEqual(calibration.initialization(config,complete,True),(10200,200))
        with self.assertRaises(ValueError):calibration.initialization(config,complete,False)
        for altered in (dict(config,spatial_mode='control'),dict(config,binding_weight=.01),
                dict(config,optimization_policy=calibration.POLICY),dict(config,smoke=True)):
            with self.assertRaises(ValueError):calibration.initialization(altered,complete,True)
        self.assertEqual(calibration.initialization(dict(version=LEGACY_VERSION,group='A'),dict(version=LEGACY_VERSION,step=10000)),(10000,0))

    def test_freezing_does_not_change_forward_and_detects_frozen_mutation(self):
        model,world,kw=self.model();reference=calibration.frozen_snapshot(world)
        with torch.inference_mode(),training_autocast('cpu'):
            world.set_world(state());before=model(**kw)[0]
            policy=calibration.configure(world);world.set_world(state());after=model(**kw)[0]
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        self.assertTrue(policy['trainable_names']);self.assertTrue(all('.local.' in key for key in policy['trainable_names']))
        self.assertFalse(any(p.requires_grad for p in world.encoder.parameters()))
        self.assertTrue(calibration.check_frozen(world,reference)['all_frozen_tensors_exact'])
        with torch.no_grad():world.encoder.category.weight[0,0]+=1
        with self.assertRaises(AssertionError):calibration.check_frozen(world,reference)

    def test_local_backward_matches_checkpoint_and_offload_with_no_frozen_gradients(self):
        values=[];grads=[]
        for checkpointed,offload in ((False,False),(True,False),(True,True)):
            model,world,kw=self.model(checkpointed);calibration.configure(world)
            frozen=calibration.frozen_snapshot(world)
            with ActivationStorage(enabled=offload,device_type='cpu').context():
                with training_autocast('cpu'):world.set_world(state());value=model(**kw)[0]
                world.clear_context();world.enabled=False
                value.float().square().mean().backward()
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(all(p.grad is None for name,p in world.named_parameters() if '.local.' not in name))
            active={name:p.grad.clone() for name,p in world.named_parameters() if p.grad is not None}
            self.assertTrue(any('local.value.' in name and grad.abs().sum()>0 for name,grad in active.items()))
            self.assertTrue(calibration.check_frozen(world,frozen)['all_frozen_tensors_exact'])
            values.append(value.detach());grads.append(active)
        for value,gradient in zip(values[1:],grads[1:]):
            torch.testing.assert_close(value,values[0],rtol=0,atol=0)
            self.assertEqual(gradient.keys(),grads[0].keys())
            for key in gradient:torch.testing.assert_close(gradient[key],grads[0][key],rtol=0,atol=0)

    def test_optimizer_resume_and_frozen_state_are_exact(self):
        model,world,kw=self.model();calibration.configure(world)
        frozen=calibration.frozen_snapshot(world)
        initial={name:value.clone() for name,value in world.state_dict().items()}
        opt=torch.optim.AdamW((p for p in world.parameters() if p.requires_grad),lr=1e-4,eps=1e-8,weight_decay=1e-5)
        def step(m,w,o,k):
            o.zero_grad(set_to_none=True)
            with training_autocast('cpu'):w.set_world(state());loss=m(**k)[0].float().square().mean()
            w.clear_context();loss.backward();o.step()
        step(model,world,opt,kw)
        saved=copy.deepcopy(world.state_dict());optimizer=copy.deepcopy(opt.state_dict())
        step(model,world,opt,kw)
        other,restored,k=self.model();calibration.configure(restored);restored.load_state_dict(saved)
        resumed=torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad),lr=1e-4,eps=1e-8,weight_decay=1e-5)
        resumed.load_state_dict(optimizer);step(other,restored,resumed,k)
        for name,value in world.state_dict().items():torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)
        self.assertTrue(calibration.check_frozen(world,frozen)['all_frozen_tensors_exact'])
        self.assertTrue(any('.local.' in name and not name.endswith('.gate') and not torch.equal(value,initial[name]) for name,value in world.state_dict().items()))
        self.assertTrue(all(state['exp_avg'].dtype==torch.float32 for state in opt.state.values()))
        self.assertEqual(list(micro_indices(200,4)),[800,801,802,803])
        self.assertEqual(list(micro_indices(201,4)),[804,805,806,807])

    def test_launcher_recovers_only_calibration_checkpoints_and_never_evaluates(self):
        from . import run
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);saved=root/'calibrate/spatial/run-fixture/checkpoint-step000025';saved.mkdir(parents=True)
            (saved/'COMPLETE.json').write_text(json.dumps(dict(version=VERSION,step=25)))
            (saved/'config.json').write_text(json.dumps(dict(version=VERSION)))
            for name in ('world.safetensors','optimizer.pt'):(saved/name).touch()
            with mock.patch('sys.argv',['run','calibrate','--root',d]),mock.patch.object(run.subprocess,'run') as execute:
                run.main();command=execute.call_args.args[0]
            self.assertEqual(command[command.index('--resume')+1],str(saved))
            self.assertEqual(command[command.index('--steps')+1],'100')
            self.assertEqual(command[command.index('--init-checkpoint')+1],str(run.CALIBRATION_INITIAL))
            self.assertIn('--local-only-calibration',command);self.assertEqual(execute.call_count,1)
            (saved/'COMPLETE.json').write_text(json.dumps(dict(version=VERSION,step=100)))
            with mock.patch('sys.argv',['run','calibrate','--root',d]),mock.patch.object(run.subprocess,'run') as execute:
                run.main();execute.assert_not_called()

    def test_diagnostic_compares_selected_checkpoint_with_its_spatial_parent(self):
        from . import run
        with tempfile.TemporaryDirectory() as d:
            checkpoint=Path(d)/'selected';checkpoint.mkdir()
            source='/temporary-fixture/spatial200'
            (checkpoint/'config.json').write_text(json.dumps(dict(initial_checkpoint=source)))
            with mock.patch('sys.argv',['run','calibrate-diagnose','--checkpoint',str(checkpoint)]),mock.patch.object(run.subprocess,'run') as execute:
                run.main();command=execute.call_args.args[0]
            self.assertEqual(command[-3:],['--checkpoints',source,str(checkpoint)])

    def test_calibration_probe_uses_trained_checkpoint_parent_and_fixed_interventions(self):
        from . import run
        with tempfile.TemporaryDirectory() as d:
            checkpoint=Path(d)/'selected';checkpoint.mkdir();source='/temporary-fixture/spatial200'
            (checkpoint/'config.json').write_text(json.dumps(dict(initial_checkpoint=source)))
            with mock.patch('sys.argv',['run','calibrate-probe','--checkpoint',str(checkpoint)]),mock.patch.object(run.subprocess,'run') as execute:
                run.main();command=execute.call_args.args[0]
            self.assertEqual(command[command.index('--checkpoint')+1],str(checkpoint))
            self.assertEqual(command[command.index('--reference-checkpoint')+1],source)
            cases=command[command.index('--cases')+1:]
            self.assertEqual(cases,['baseline','correct','move_tv','mirror_tv','remove_tv','hidden_chair_to_plant','add_floor_plant'])
            self.assertEqual(execute.call_count,1)


if __name__=='__main__':unittest.main()
