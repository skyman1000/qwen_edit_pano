"""Actual duplicate inventory replay, old successful detections, and CLI gates."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from ..common import ROOT,read,write
from . import assembly_io_recovery as io,assembly_observer_recovery as recovery

SOURCE=ROOT/'qwen_edit_pano/outputs/organized_multiscene_named_prepared_20261028_001/scenes'
LATEST=ROOT/'qwen_edit_pano/outputs/organized_multiscene_20261028_110616_948850'
KITCHEN='8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029'


class ObserverRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
    def root(self,sid=KITCHEN):
        src=SOURCE/sid/'observation';root=self.base/sid;root.mkdir();(root/'qwen').mkdir()
        for name in ('input.png','experiment.json'):shutil.copyfile(src/name,root/name)
        return root
    def duplicate_reply(self):return read(LATEST/'scenes'/KITCHEN/'observation/qwen/observer2d/chunk_00_0.json')

    def test_actual_duplicates_are_collapsed_exactly_without_inventing_boxes(self):
        raw=self.duplicate_reply();before=deepcopy(raw);clean,duplicates=recovery.unique_inventory(raw)
        self.assertEqual(len(raw['objects']),8);self.assertEqual(len(clean['objects']),4);self.assertEqual(len(duplicates),4)
        self.assertEqual(raw,before);self.assertEqual(recovery.identity_candidates(clean['objects']),[['p002','p003']])

    def test_actual_duplicate_response_routes_to_full_review_then_explicit_identity(self):
        root=self.root();raw=self.duplicate_reply();prompts=[]
        class P:
            def ask(inner,image,prompt,path,**kwargs):
                prompts.append(prompt)
                if path.stem=='chunk_00_0':value=deepcopy(raw)
                elif path.stem=='whole_image_review':value=deepcopy(raw);value.pop('done')
                elif path.stem=='instance_identity_review':value=dict(groups=[dict(members=['p002','p003'],representative_id='p002')])
                else:raise AssertionError('Unexpected incremental retry or model call: '+str(path))
                path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value
        result=io.observe_chunked(P(),root);self.assertEqual(result['objects'],3);self.assertEqual(result['actual_calls'],3)
        self.assertIn('WHOLE-IMAGE INVENTORY REVIEW',prompts[1]);self.assertNotIn('INCREMENTAL VISIBLE INVENTORY',prompts[1])
        d=root/'qwen/observer2d';self.assertTrue((d/'reconciliation_candidates.json').exists());self.assertFalse((d/'chunk_00_1.txt').exists())
        self.assertFalse(result['automatic_overlap_suppression']);self.assertFalse(result['visual_quality_verified'])

    def test_whole_image_review_can_add_missed_types_without_changing_world_model(self):
        root=self.root();raw=self.duplicate_reply();clean,_=recovery.unique_inventory(raw);review=deepcopy(clean)
        review['objects']+= [dict(name='range hood',category='range hood',bbox_1000=[350,380,600,490],on_floor=False)]
        class P:
            def ask(inner,image,prompt,path,**kwargs):
                value=raw if path.stem=='chunk_00_0' else review if path.stem=='whole_image_review' else dict(groups=[dict(members=['p002','p003'],representative_id='p003')])
                path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value
        result=io.observe_chunked(P(),root);self.assertEqual(result['objects'],4)
        self.assertIn('range hood',[o['raw_category'] for o in read(root/'qwen/observer2d/G_2d.json')['objects']])

    def test_overlap_review_may_keep_distinct_objects_and_cannot_omit_or_invent_IDs(self):
        ids=['p002','p003'];value=dict(groups=[dict(members=[i],representative_id=i) for i in ids])
        self.assertEqual(recovery.identity_answer(value,ids)['keep_ids'],ids)
        for bad in (dict(groups=[]),dict(groups=[dict(members=ids,representative_id='invented')]),dict(groups=[dict(members=['p002'],representative_id='p002')])):
            with self.assertRaises(ValueError):recovery.identity_answer(bad,ids)
        ids=['p000','p001','p002','p003'];bad=dict(groups=[dict(members=['p000','p002'],representative_id='p000'),dict(members=['p001','p003'],representative_id='p001')])
        with self.assertRaisesRegex(ValueError,'unrelated'):recovery.identity_answer(bad,ids,[['p000','p001'],['p002','p003']])

    def test_empty_review_preserves_proposals_and_does_not_publish_a_false_zero_graph(self):
        root=self.root();raw=self.duplicate_reply()
        class P:
            def ask(inner,image,prompt,path,**kwargs):
                value=raw if path.stem=='chunk_00_0' else dict(objects=[],evidence_description='No objects accepted',done=False)
                path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value
        with self.assertRaisesRegex(ValueError,'Contradictory empty'):io.observe_chunked(P(),root)
        d=root/'qwen/observer2d';self.assertFalse((d/'G_2d.json').exists());self.assertEqual(len(read(d/'reconciliation_candidates.json')['objects']),4)
        self.assertEqual(read(d/'summary.json')['actual_calls'],3)

    def test_four_previously_successful_visible_inventories_remain_identical(self):
        for s in sorted(SOURCE.iterdir()):
            if s.name==KITCHEN:continue
            root=self.root(s.name);src=s/'observation/qwen/observer2d'
            class P:
                def ask(inner,image,prompt,path,**kwargs):
                    original=src/(path.stem+'.json')
                    if original.exists():value=read(original)
                    elif path.stem=='chunk_00_0' and (src/'answer.json').exists():value=dict(read(src/'answer.json'),done=True)
                    else:raise AssertionError('Successful observer unexpectedly entered a new path: '+str(path))
                    path.write_text(json.dumps(value));return value
            io.observe_chunked(P(),root)
            actual=read(root/'qwen/observer2d/G_2d.json')
            first=src/'chunk_00_0.json'
            if first.exists() and len(read(first)['objects'])>8:
                # A previously rejected but valid first reply is now retained,
                # rather than replaced by an 8-row regeneration.
                from .fresh_observer import answer,state_from_answer
                expected,_=state_from_answer(answer(read(first)),read(root/'experiment.json'),'qwen')
            else:expected=read(src/'G_2d.json')
            self.assertEqual(actual,expected,s.name)

    def test_observation_only_cli_needs_no_API_key_and_stops_downstream_on_failure(self):
        fake=self.base/'python';srun=self.base/'srun';log=self.base/'calls.jsonl'
        fake.write_text('#!'+sys.executable+'\nimport json,os,sys\nfrom pathlib import Path\na=sys.argv[1:];m=a[a.index("-m")+1];stage=a[a.index("-m")+2] if m.endswith("assembly_multiscene") else "preview"\nwith open(os.environ["OBSERVER_SMOKE_LOG"],"a") as f:f.write(json.dumps(stage)+"\\n")\nif stage=="prepare":Path(a[a.index("--root")+1]).mkdir(parents=True)\nif stage=="observer2d" and os.environ.get("OBSERVER_SMOKE_FAIL")=="1":sys.exit(1)\n')
        srun.write_text('#!'+sys.executable+'\nimport os,sys\np=os.environ["CAUPANO_QWEN_PYTHON"];i=sys.argv.index(p);os.execv(p,sys.argv[i:])\n')
        fake.chmod(0o755);srun.chmod(0o755)
        for failed in (False,True):
            env=dict(os.environ,CAUPANO_QWEN_PYTHON=str(fake),OBSERVER_SMOKE_LOG=str(log),OBSERVER_SMOKE_FAIL='1' if failed else '0',PATH=str(self.base)+os.pathsep+os.environ['PATH']);env.pop('SILICONFLOW_API_KEY',None)
            log.unlink(missing_ok=True);root=self.base/('failed' if failed else 'passed')
            p=subprocess.run(['bash',str(ROOT/'qwen_edit_pano/scripts/organized_multiscene.sh'),'observe','CPU_FIXTURE',str(root)],env=env,text=True,capture_output=True)
            self.assertEqual(p.returncode,1 if failed else 0,(p.stdout,p.stderr));self.assertEqual([json.loads(s) for s in log.read_text().splitlines()],['prepare','observer2d','preview'])
            self.assertNotIn('observer3d',p.stdout);self.assertFalse((root/'completion').exists())


if __name__=='__main__':unittest.main()
