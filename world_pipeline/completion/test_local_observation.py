"""Observer-only boundaries: no hidden graph, no invented geometry or API."""
from copy import deepcopy
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from ..common import ROOT,read,write
from . import local_observation as local

SAVED=ROOT/'qwen_edit_pano/outputs/kitchen_observer_20261028_144049/scenes'/local.KITCHEN/'observation'


class LocalObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='local_observation_test_',dir=ROOT/'research/audits')
        self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)

    def fixture(self,name='source',three=True):
        root=self.base/name;obs=root/'scenes'/local.KITCHEN/'observation';obs.mkdir(parents=True)
        write(root/'suite.json',dict(scenes=[dict(id=local.KITCHEN,hfov=90)]));shutil.copyfile(SAVED/'input.png',obs/'input.png')
        for folder in ['observer2d']+(['observer3d'] if three else []):
            target=obs/'qwen'/folder;target.mkdir(parents=True)
            for f in (SAVED/'qwen'/folder).glob('*.json'):
                if f.name in ['G_2d.json','G_obs.json','grounding.json','summary.json']:shutil.copyfile(f,target/f.name)
        # A report must not parse or use these files, even when present.
        poison=root/'scenes'/local.KITCHEN/'completion/end_to_end/qwen/completion/world_01';poison.mkdir(parents=True)
        (poison/'G_full.json').write_text('DO NOT READ ANY COMPLETED WORLD')
        return root,obs

    @staticmethod
    def renderer(state,image,hfov,out):
        out.mkdir();assert all(o['local_evidence']=='visible' for o in state['objects'])
        for name in ['overview','local_projection','layout_xz','direction_projections','scene_3d','erp_overlay','erp_state_map']:
            Image.new('RGB',(80,40),'white').save(out/(name+'.png'))
        (out/'index.html').write_text('<p>CPU test fixture</p>');(out/'scene_3d.html').write_text('<p>CPU test fixture</p>')
        return dict(labels=len(state['objects']))

    def test_rejects_full_hidden_GT_and_missing_metric_without_filtering(self):
        state=read(SAVED/'qwen/observer3d/G_obs.json')
        for change in [dict(scope='full'),dict(stage='GT'),dict(scene_gt_used=True)]:
            with self.assertRaises(ValueError):local.visible_state(dict(state,**change),metric=True)
        bad=deepcopy(state);bad['objects'][0]['local_evidence']='unobserved'
        with self.assertRaisesRegex(ValueError,'non-visible'):local.visible_state(bad,metric=True)
        two=read(SAVED/'qwen/observer2d/G_2d.json')
        with self.assertRaisesRegex(ValueError,'missing geometry'):local.visible_state(two,metric=True)

    def test_2d_3d_binding_rejects_dropped_or_changed_instances(self):
        two=read(SAVED/'qwen/observer2d/G_2d.json');three=read(SAVED/'qwen/observer3d/G_obs.json')
        local.bind_observations(two,three)
        bad=deepcopy(three);bad['objects'].pop()
        with self.assertRaisesRegex(ValueError,'instance mismatch'):local.bind_observations(two,bad)
        bad=deepcopy(three);bad['objects'][0]['raw_category']='chair'
        with self.assertRaisesRegex(ValueError,'identity'):local.bind_observations(two,bad)

    def test_saved_report_uses_observation_only_and_preserves_exact_source(self):
        root,obs=self.fixture();out=self.base/'review';before=(obs/'qwen/observer3d/G_obs.json').read_bytes()
        local.report(root,out,renderer=self.renderer)
        self.assertEqual((out/'scenes'/local.KITCHEN/'G_obs.json').read_bytes(),before)
        self.assertEqual((obs/'qwen/observer3d/G_obs.json').read_bytes(),before)
        result=read(out/'results.json');self.assertEqual(result['records'][0]['observed_3d'],3)
        self.assertEqual(result['records'][0]['generated_instances'],0)
        self.assertTrue(all(not r['generated'] for r in read(out/'instances.json')))
        check=read(out/'validation.json');self.assertFalse(check['completion_files_read']);self.assertFalse(check['broken_links'])
        self.assertTrue(check['source_files_unchanged'])
        self.assertEqual(read(out/'scenes'/local.KITCHEN/'views/visual_validation.json')['palette'],[local.visual.OBSERVED])

    def test_missing_3d_remains_unknown_with_only_real_2d_boxes(self):
        root,_=self.fixture(three=False);out=self.base/'review'
        def forbidden(*args):raise AssertionError('Cannot render missing 3D')
        local.report(root,out,renderer=forbidden)
        self.assertEqual(read(out/'results.json')['records'][0]['status'],'2d_only')
        self.assertFalse((out/'scenes'/local.KITCHEN/'G_obs.json').exists())
        self.assertFalse((out/'scenes'/local.KITCHEN/'views').exists())
        self.assertTrue(all(r['center_m'] is None for r in read(out/'instances.json')))

    def test_replacement_must_match_original_pixels_and_camera(self):
        root,_=self.fixture();other,obs=self.fixture('replacement')
        suite,selected=local.sources(root,other);self.assertEqual(selected[local.KITCHEN],other)
        Image.new('RGB',(1024,1024),'black').save(obs/'input.png')
        with self.assertRaisesRegex(ValueError,'RGB/HFOV'):local.sources(root,other)

    def test_report_distinguishes_2d_failure_3d_failure_and_blocked_dependency(self):
        for phase in ('observer2d','observer3d','blocked'):
            root,obs=self.fixture('failure_'+phase,three=False)
            if phase=='observer2d':
                (obs/'qwen/observer2d/G_2d.json').unlink()
                write(obs/'qwen/observer2d/summary.json',dict(status='failed',error='fixture 2D schema failure'))
                expected='observer2d_failed';message='fixture 2D schema failure'
            elif phase=='observer3d':
                (obs/'qwen/observer3d').mkdir()
                write(obs/'qwen/observer3d/summary.json',dict(status='failed',error='fixture 3D failure'))
                expected='observer3d_failed';message='fixture 3D failure'
            else:
                (root/'phase_records').mkdir()
                write(root/'phase_records'/(local.KITCHEN+'_observer3d_qwen.json'),dict(status='blocked_dependency',error='fixture upstream failure'))
                expected='blocked_3d_dependency';message='fixture upstream failure'
            out=self.base/('review_'+phase)
            def forbidden(*args):raise AssertionError('No complete 3D exists')
            local.report(root,out,renderer=forbidden)
            row=read(out/'results.json')['records'][0]
            self.assertEqual(row['status'],expected);self.assertEqual(row['error'],message)
            self.assertIn(message,(out/'scenes'/local.KITCHEN/'index.html').read_text())
            self.assertFalse((out/'scenes'/local.KITCHEN/'G_obs.json').exists())

    def test_stale_graph_from_failed_3d_stage_is_not_published_as_completed(self):
        root,obs=self.fixture('stale')
        write(obs/'qwen/observer3d/summary.json',dict(status='failed',error='fixture post-export crash'))
        def forbidden(*args):raise AssertionError('Failed-stage graph cannot be rendered as completed')
        out=self.base/'stale_review';local.report(root,out,renderer=forbidden)
        row=read(out/'results.json')['records'][0]
        self.assertEqual(row['status'],'invalid_observation');self.assertIn('failed observation stage',row['error'])
        self.assertFalse((out/'scenes'/local.KITCHEN/'G_obs.json').exists())

    def test_shell_runs_only_2d_3d_and_report_without_API_key(self):
        fake=self.base/'python';srun=self.base/'srun';log=self.base/'calls.txt'
        fake.write_text('#!'+sys.executable+'\nimport os,sys\nfrom pathlib import Path\na=sys.argv[1:];stage=a[a.index("-m")+2]\nwith open(os.environ["LOCAL_OBSERVER_SMOKE_LOG"],"a") as f:f.write(stage+"\\n")\nif stage=="prepare":Path(a[a.index("--root")+1]).mkdir(parents=True)\nif stage==os.environ.get("LOCAL_OBSERVER_SMOKE_FAILURE"):sys.exit(1)\n')
        srun.write_text('#!'+sys.executable+'\nimport os,sys\np=os.environ["CAUPANO_QWEN_PYTHON"];i=sys.argv.index(p);os.execv(p,sys.argv[i:])\n')
        fake.chmod(0o755);srun.chmod(0o755)
        for mode,failure in [(mode,failure) for mode in ('run','local') for failure in ('','observer2d','observer3d')]:
            env=dict(os.environ,CAUPANO_QWEN_PYTHON=str(fake),CAUPANO_WILDDET_PYTHON=str(fake),LOCAL_OBSERVER_SMOKE_LOG=str(log),LOCAL_OBSERVER_SMOKE_FAILURE=failure,PATH=str(self.base)+os.pathsep+os.environ['PATH'])
            env.pop('SILICONFLOW_API_KEY',None);log.unlink(missing_ok=True)
            args=([mode,'CPU_FIXTURE'] if mode=='run' else [mode])+[str(self.base/(mode+'_'+(failure or 'passed')))]
            p=subprocess.run(['bash',str(ROOT/'qwen_edit_pano/scripts/local_observation.sh')]+args,env=env,text=True,capture_output=True)
            self.assertEqual(p.returncode,1 if failure else 0,(p.stdout,p.stderr))
            phases=log.read_text().splitlines();self.assertEqual(phases,['prepare','observer2d','observer3d','report'])
            self.assertNotIn('mechanisms',phases);self.assertNotIn('worlds',phases)


if __name__=='__main__':unittest.main()
