"""No CUDA/Slurm: driver fresh/smoke isolation, partial saves and own resume."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import run as driver
from . import data as d


class DriverTests(unittest.TestCase):
    def test_isolated_dual_smoke_then_fresh_and_formal_only_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);root=base/'qwen_edit_pano/outputs/fresh';calls=[]
            def configuration(plan,smoke=False):
                return dict(version=d.VERSION,smoke=smoke,source_step=0)
            def fake_run(command,check):
                self.assertTrue(check);self.assertIn('--nproc-per-node=2',command)
                self.assertIn('qwen_edit_pano.world_pipeline.native_world.train',command)
                calls.append(command);out=Path(command[command.index('--output')+1])
                smoke='--smoke' in command
                target=2 if smoke else int(command[command.index('--steps')+1])
                checkpoint=out/f'checkpoint-step{target:06d}';checkpoint.mkdir(parents=True)
                config=configuration(None,smoke)
                (checkpoint/'config.json').write_text(json.dumps(config))
                (checkpoint/'world.safetensors').touch();(checkpoint/'optimizer.pt').touch()
                (checkpoint/'COMPLETE.json').write_text(json.dumps(dict(version=d.VERSION,step=target,optimizer_step=target)))
                if smoke:
                    fields=('both_ranks_passed','zero_gate_baseline_exact','labels_do_not_change_forward',
                        'zero_gate_connector_key_gradients','encoder_gradient','base_frozen',
                        'replica_parameters_equal','fresh_initialization','both_binding_terms','value_generation_gradient')
                    (checkpoint/'SMOKE_COMPLETE.json').write_text(json.dumps(dict(version=d.VERSION,config=config,
                        optimizer_step=2,**{key:True for key in fields})))
            with patch.object(driver,'ROOT',base),patch.object(d,'inputs',return_value=({},[])),\
                    patch.object(d,'check',return_value={}),patch.object(d,'configuration',side_effect=configuration),\
                    patch.object(driver.subprocess,'run',side_effect=fake_run):
                def invoke(steps):
                    with patch('sys.argv',['run','--root',str(root),'--steps',str(steps)]):driver.main()
                invoke(500)
                self.assertEqual(len(calls),2)
                self.assertIn('--smoke',calls[0]);self.assertNotIn('--resume',calls[0])
                self.assertNotIn('--smoke',calls[1]);self.assertNotIn('--resume',calls[1])
                # An incomplete newer write has no COMPLETE marker and is ignored.
                partial=root/'train/run-999/checkpoint-step000800';partial.mkdir(parents=True)
                (partial/'world.safetensors').touch()
                invoke(10000)
                self.assertEqual(len(calls),3)
                resume=calls[-1][calls[-1].index('--resume')+1]
                self.assertIn('/train/',resume);self.assertTrue(resume.endswith('checkpoint-step000500'))
                invoke(10000);self.assertEqual(len(calls),3)
            finished=json.loads((root/'FINISHED.json').read_text())
            self.assertEqual(finished['target'],10000);self.assertFalse(finished['automatic_evaluation'])


if __name__=='__main__':unittest.main()
