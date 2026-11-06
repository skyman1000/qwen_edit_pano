"""CPU orchestration tests with synthetic stage outputs; no GPU/model inference."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from ..common import ROOT,read,write,write_rows,sha
from .batch_world import run,command,inputs,QWEN_PY,BOX_PY
from .placement_program import PROTOCOL
from .semantic_world import make_graph
from .test_semantic_world import fixture


def manifest(root):
    cases=[]
    for i in range(2):
        image=root/f'image{i}.png';Image.new('RGB',(64,64),(30+i,80,120)).save(image)
        cases.append(dict(id=f'val_case_{i}',split='val',local_rgb=str(image),hfov_deg=90.,
                          image_sha256=sha(image),display_label='REVIEW_LABEL_NOT_MODEL_INPUT'))
    path=root/'inputs.jsonl';write_rows(path,cases)
    return path,cases


def synthetic_launcher(fail_first_observer=False):
    calls=[]; failed=False
    def launch(cmd,log):
        nonlocal failed
        image=Path(cmd[cmd.index('--image')+1]);dest=Path(cmd[cmd.index('--output')+1])
        stage='inventory' if '--inventory-only' in cmd else 'observer' if 'box_observer' in cmd[3] else 'completion'
        calls.append((image.name,stage));log.write_text('Synthetic CPU stage, no model\n')
        if fail_first_observer and not failed and stage=='observer':
            failed=True
            return 1
        dest.mkdir();Image.open(image).save(dest/'local.png')
        observed,plan,layout=fixture()
        config=dict(image_sha256=sha(image))
        if stage=='inventory':
            graph=make_graph(observed,None,'test','not_estimated','observed')
            summary=dict(status='completed',phase='inventory_only')
        elif stage=='observer':
            source=Path(cmd[cmd.index('--source')+1])
            graph=make_graph(observed,layout,'test','wilddet3d_camera_aabb_center','observed')
            config['source_sha256']=sha(source)
            summary=dict(status='completed',numeric_boxes=2)
        else:
            source=Path(cmd[cmd.index('--reuse-state')+1]);graph=read(source)
            full=make_graph(plan['objects'],layout,'test','synthetic_completion')
            full['objects'][:2]=deepcopy(graph['objects'])
            config['protocol']=PROTOCOL
            summary=dict(status='completed',observed=2,hypothesized=1,outside_fov_failed=1,
                         suspected_aabb_overlap_pairs=1,quality_status='needs_completion_revision')
            write(dest/'G_full.json',full);write(dest/'placement_program.json',{})
            write(dest/'completion_conversion.json',{})
        write(dest/'G_obs.json',graph);write(dest/'config.json',config);write(dest/'summary.json',summary)
        return 0
    return launch,calls


class BatchWorldTests(unittest.TestCase):
    def test_commands_use_correct_env_and_only_same_image_observation(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'research/audits') as tmp:
            root=Path(tmp);_,cases=manifest(root);case=cases[0]
            previous=dict(inventory=root/'inventory',observer=root/'observer')
            first=command('inventory',case,root/'out1',previous)
            second=command('observer',case,root/'out2',previous)
            third=command('completion',case,root/'out3',previous)
            self.assertEqual([first[0],second[0],third[0]],[str(QWEN_PY),str(BOX_PY),str(QWEN_PY)])
            self.assertIn('--inventory-only',first)
            self.assertIn(str(previous['inventory']/'G_obs.json'),second)
            self.assertIn(str(previous['observer']/'G_obs.json'),third)
            for cmd in (first,second,third):
                self.assertIn(case['local_rgb'],cmd)
                self.assertNotIn(case['display_label'],cmd)

    def test_failure_kept_other_cases_continue_and_resume_reuses_success(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'research/audits') as tmp:
            root=Path(tmp);path,_=manifest(root);output=root/'batch'
            launch,calls=synthetic_launcher(fail_first_observer=True)
            first=run(path,output,launch)
            self.assertEqual((first['completed_cases'],first['failed_cases']),(1,1))
            self.assertIn('subprocess exit code 1',first['cases_detail'][0]['error'])
            self.assertFalse(first['quality_certified'])
            second=run(path,output,launch)
            self.assertEqual((second['completed_cases'],second['failed_cases']),(2,0))
            self.assertEqual(sum(stage=='inventory' for _,stage in calls),2)
            self.assertTrue((output/'val_case_0/observer_001.log').is_file())
            self.assertTrue((output/'val_case_0/observer_002/G_obs.json').is_file())
            self.assertIn('needs_completion_revision',(output/'index.html').read_text())
            self.assertTrue((output/'val_case_1/scene_3d.html').is_file())
            self.assertNotIn('<script src=',(output/'val_case_1/scene_3d.html').read_text())

    def test_changed_rgb_rejected_before_any_model_execution(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'research/audits') as tmp:
            root=Path(tmp);path,cases=manifest(root)
            Image.new('RGB',(64,64),'red').save(cases[0]['local_rgb'])
            with self.assertRaisesRegex(ValueError,'RGB changed'):inputs(path)

    def test_changed_saved_observation_rejected_on_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'research/audits') as tmp:
            root=Path(tmp);path,_=manifest(root);output=root/'batch'
            launch,_=synthetic_launcher();run(path,output,launch)
            graph_path=output/'val_case_0/observer_001/G_obs.json'
            graph=read(graph_path);graph['objects'][0]['center_local_m']['value'][0]+=.4
            write(graph_path,graph)
            with self.assertRaisesRegex(ValueError,'Saved stage changed'):run(path,output,launch)

    def test_cpu_prepared_pending_cases_are_not_reported_as_gpu_failures(self):
        from .batch_world_review import review
        with tempfile.TemporaryDirectory(dir=ROOT/'research/audits') as tmp:
            root=Path(tmp);path,_=manifest(root);output=root/'batch'
            launch,_=synthetic_launcher();run(path,output,launch)
            progress=read(output/'progress.json')
            for stage in ('observer','completion'):
                progress['val_case_0'][stage]=dict(status='pending',error='Awaiting manual GPU stage')
            write(output/'progress.json',progress)
            summary=review(path,output)
            self.assertEqual((summary['completed_cases'],summary['failed_cases'],summary['pending_cases']),(1,0,1))
            self.assertEqual(summary['cases_detail'][0]['observed'],2)


if __name__=='__main__':
    unittest.main()
