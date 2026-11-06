"""No Slurm/CUDA: verify smoke isolation, finite budgets and formal-only resume."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import run as driver
from . import data as d


class RunTests(unittest.TestCase):
    def test_smoke_is_not_warm_start_and_repeated_command_resumes_formal_only(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'qwen_edit_pano/outputs/test';calls=[]
            def configuration(plan,old,group,smoke=False):
                return dict(version=d.VERSION,group=group,smoke=smoke,source_step=10000)
            def fake_run(command,check):
                self.assertTrue(check);calls.append(command)
                out=Path(command[command.index('--output')+1]);smoke='--smoke' in command
                group=command[command.index('--group')+1]
                target=10002 if smoke else 10000+int(command[command.index('--additional-steps')+1])
                checkpoint=out/f'checkpoint-step{target:06d}';checkpoint.mkdir(parents=True)
                config=configuration(None,None,group,smoke)
                (checkpoint/'config.json').write_text(json.dumps(config))
                (checkpoint/'world.safetensors').touch();(checkpoint/'optimizer.pt').touch()
                (checkpoint/'COMPLETE.json').write_text(json.dumps(dict(version=d.VERSION,step=target,optimizer_step=target)))
                if smoke:(checkpoint/'SMOKE_COMPLETE.json').write_text(json.dumps(dict(config=config,base_frozen=True,
                    forward_unchanged_with_labels=True,both_binding_terms=group=='binding')))
            with patch.object(driver,'ROOT',base),patch.object(driver.d,'check',return_value={}),\
                    patch.object(driver.d,'inputs',return_value=({},[],{})),\
                    patch.object(driver.d,'configuration',side_effect=configuration),\
                    patch.object(driver.subprocess,'run',side_effect=fake_run):
                def invoke(steps):
                    with patch('sys.argv',['run','binding','--root',str(root),'--additional-steps',str(steps)]):driver.main()
                invoke(500)
                self.assertEqual(len(calls),2);self.assertIn('--smoke',calls[0]);self.assertNotIn('--smoke',calls[1])
                self.assertNotIn('--resume',calls[1])
                invoke(1000)
                self.assertEqual(len(calls),3)
                source=calls[-1][calls[-1].index('--resume')+1]
                self.assertIn('/train/',source);self.assertTrue(source.endswith('checkpoint-step010500'))
                invoke(1000);self.assertEqual(len(calls),3)
            finished=json.loads((root/'binding/FINISHED.json').read_text());self.assertEqual(finished['target'],11000)


if __name__=='__main__':unittest.main()
