"""No Slurm contact: fake submissions verify dependency and partial-failure behavior."""
import subprocess
from pathlib import Path
import unittest

from ..submit_object_world_chain import submit_jobs


class SubmissionTests(unittest.TestCase):
    def test_three_stages_resume_previous_exact_final_step(self):
        commands=[];saved=[];answers=iter(('101;cluster','102','103'))
        def submit(command):
            commands.append(command);return next(answers)
        outputs=[Path('/tmp/test-object-part'+str(i)) for i in range(3)]
        result=submit_jobs(outputs,submit,saved.append)
        self.assertEqual([r['job_id'] for r in result],['101','102','103'])
        self.assertEqual([r['target_step'] for r in result],[3500,7000,10000])
        self.assertFalse(any('--dependency' in s for s in commands[0]))
        self.assertIn('--dependency=afterok:101',commands[1])
        self.assertIn('--dependency=afterok:102',commands[2])
        self.assertEqual(commands[1][-1],str(outputs[0]/'checkpoint-step003500'))
        self.assertEqual(commands[2][-1],str(outputs[1]/'checkpoint-step007000'))
        self.assertEqual([len(r) for r in saved],[1,2,3])

    def test_partial_submission_is_recorded_and_stops(self):
        commands=[];saved=[]
        def submit(command):
            commands.append(command)
            if len(commands)==2:raise subprocess.CalledProcessError(1,command)
            return '101'
        with self.assertRaises(subprocess.CalledProcessError):
            submit_jobs([Path('/tmp/a'),Path('/tmp/b'),Path('/tmp/c')],submit,saved.append)
        self.assertEqual(len(commands),2)
        self.assertEqual(len(saved),1)
        self.assertEqual(saved[0][0]['job_id'],'101')


if __name__=='__main__':unittest.main()
