"""Execute a copied runner with fake Python, never Slurm or a GPU."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from ..build_polished_world import ROOT


class RunnerTests(unittest.TestCase):
    def fixture(self,directory,fail_smoke=False):
        root=Path(directory)
        for path in ('research/audits/object_world_deep_review_2026-10-18/diagnostic_panel6.json',
                     'qwen_edit_pano/outputs/object_world_layer_pilot_legacy_v1_val6/COMPLETE.json'):
            p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
        fake=root/'fake-python'
        fake.write_text('#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n'
            'args=sys.argv[1:]\n'
            "with Path('commands.jsonl').open('a') as f:f.write(json.dumps(args)+'\\n')\n"
            f"if '--smoke' in args and {fail_smoke!r}:sys.exit(9)\n"
            "if '--smoke' in args:\n"
            " p=Path(args[args.index('--output')+1])/'checkpoint-step000002/SMOKE_COMPLETE.json'\n"
            " p.parent.mkdir(parents=True);p.write_text('{}')\n")
        fake.chmod(0o700)
        source=(ROOT/'qwen_edit_pano/world_pipeline/submit_object_world_id_ab.sh').read_text()
        source=source.replace('cd /data-nfs/gpu1-2/u13529658780/DIT360',f'cd {root}')
        source=source.replace('ab_python=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python',f'ab_python={fake}')
        script=root/'runner.sh';script.write_text(source)
        return root,script

    def test_two_arms_each_smoke_then_fresh_train_then_fixed_six_eval(self):
        runs=[]
        for group in ('A','B'):
            with tempfile.TemporaryDirectory() as directory:
                root,script=self.fixture(directory)
                result=subprocess.run(['bash',str(script),group],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                commands=[json.loads(l) for l in (root/'commands.jsonl').read_text().splitlines()]
                self.assertEqual(len(commands),3)
                self.assertIn('--smoke',commands[0]);self.assertNotIn('--resume',commands[1])
                self.assertEqual(commands[1][commands[1].index('--steps')+1],'2000')
                self.assertEqual(commands[1][commands[1].index('--warmup-steps')+1],'100')
                self.assertEqual(commands[2][commands[2].index('--limit')+1],'6')
                self.assertIn('--baseline-reference',commands[2])
                runs.append(commands)
        # Except arm name/output directory, both commands are identical.
        normalized=lambda commands:[['GROUP' if s in ('A','B') else s.replace('_A_','_GROUP_').replace('_B_','_GROUP_') for s in cmd] for cmd in commands]
        self.assertEqual(normalized(runs[0]),normalized(runs[1]))

    def test_smoke_failure_stops_before_long_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root,script=self.fixture(directory,True)
            result=subprocess.run(['bash',str(script),'B'],capture_output=True,text=True)
            self.assertEqual(result.returncode,9)
            self.assertEqual(len((root/'commands.jsonl').read_text().splitlines()),1)

    def test_output_collision_and_invalid_group_do_not_run_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root,script=self.fixture(directory)
            (root/'qwen_edit_pano/outputs/object_world_id_ab_A_v1').mkdir()
            for args in (['A'],['C'],['B','../unsafe']):
                result=subprocess.run(['bash',str(script),*args],capture_output=True,text=True)
                self.assertEqual(result.returncode,2)
            self.assertFalse((root/'commands.jsonl').exists())


if __name__=='__main__':unittest.main()
