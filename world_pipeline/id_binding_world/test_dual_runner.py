"""User-run shell launcher checks with fake Python/scontrol; never submit jobs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from ..common import ROOT


class DualRunnerTests(unittest.TestCase):
    def fixture(self, root, mode):
        fake = root / 'fake-python'
        fake.write_text(f'#!{sys.executable}\n'
            'import json,sys,time\nfrom pathlib import Path\n'
            'args=sys.argv[1:]\n'
            'with Path("commands.jsonl").open("a") as f:f.write(json.dumps(args)+"\\n")\n'
            'if "qwen_edit_pano.world_pipeline.id_binding_world.restart" in args:\n'
            ' print(Path("qwen_edit_pano/outputs/object_world_id_A_dual_full_v1/run-fixture").resolve())\n'
            ' print("checkpoint-step001000");print(1000);sys.exit(0)\n'
            'out=Path(args[args.index("--output")+1]);out.mkdir(parents=True)\n'
            f'if {mode!r}=="failure":sys.exit(9)\n'
            f'if {mode!r}=="stop":\n'
            ' stop=Path(args[args.index("--stop-file")+1])\n'
            ' for i in range(300):\n'
            '  if stop.exists():break\n'
            '  time.sleep(.02)\n'
            ' else:sys.exit(8)\n'
            ' (out/"STOPPED.json").write_text("{}")\n'
            'else:(out/"COMPLETE.json").write_text("{}")\n')
        fake.chmod(0o700)
        binary = root / 'bin'
        binary.mkdir()
        scontrol = binary / 'scontrol'
        scontrol.write_text(f'#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n'
                           'Path("requeue.json").write_text(json.dumps(sys.argv[1:]))\n')
        scontrol.chmod(0o700)
        source = (ROOT / 'qwen_edit_pano/world_pipeline/submit_object_world_a_dual.sh').read_text()
        source = source.replace(f'cd {ROOT}', f'cd {root}')
        source = source.replace('dual_python=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python', f'dual_python={fake}')
        script = root / 'runner.sh'
        script.write_text(source)
        env = {**os.environ, 'SLURM_JOB_ID': '999_TEST', 'PATH': str(binary)+os.pathsep+os.environ['PATH']}
        return script, env

    def test_training_only_launch_and_error_propagation(self):
        for mode in ('complete', 'failure'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                script, env = self.fixture(root, mode)
                result = subprocess.run(['bash', str(script)], cwd=root, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0 if mode == 'complete' else 9, result.stderr)
                commands = [json.loads(s) for s in (root / 'commands.jsonl').read_text().splitlines()]
                self.assertEqual(len(commands), 2)
                train = commands[1]
                self.assertIn('--nproc-per-node=2', train)
                self.assertEqual(train[train.index('--steps')+1], '10000')
                self.assertEqual(train[train.index('--group')+1], 'A')
                self.assertEqual(train[train.index('--accumulation')+1], '4')
                self.assertFalse(any('inference' in s for c in commands for s in c))
                self.assertFalse((root / 'requeue.json').exists())

    def test_time_signal_waits_for_saved_checkpoint_before_requeue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, env = self.fixture(root, 'stop')
            process = subprocess.Popen(['bash', str(script)], cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = root / 'qwen_edit_pano/outputs/object_world_id_A_dual_full_v1/run-fixture'
            deadline = time.monotonic()+10
            while not output.exists() and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertTrue(output.exists())
            os.kill(process.pid, signal.SIGUSR1)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, (stdout, stderr))
            self.assertTrue((output / 'STOPPED.json').exists())
            self.assertEqual(json.loads((root / 'requeue.json').read_text()), ['requeue', '999_TEST'])


if __name__ == '__main__':
    unittest.main()
