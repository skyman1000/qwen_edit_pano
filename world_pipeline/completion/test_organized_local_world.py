"""CPU integration checks; fixture replies are never real model results."""
from contextlib import ExitStack
from copy import deepcopy
from html.parser import HTMLParser
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ..common import ROOT,read,write,sha
from . import organized_local_world as local,assembly_multiscene as pipeline
from .test_compositional import CPUProgramProvider
from .test_multiscene import CPUProvider

SOURCE=ROOT/'qwen_edit_pano/outputs/local_observation_fixed_002'


class LocalWorldTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'research/audits',prefix='v13_local_CPU_')
        self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG='',CAUPANO_LAYOUT_RGB='0'))
        self.stack.enter_context(patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}))

    def source(self):
        source=self.base/'source';source.mkdir();suite=read(SOURCE/'suite.json');suite['scenes']=suite['scenes'][:1]
        write(source/'suite.json',suite)
        sid=suite['scenes'][0]['id'];shutil.copytree(SOURCE/'scenes'/sid/'observation',source/'scenes'/sid/'observation')
        # Poison old completion: preparing a fresh run must never read it.
        folder=source/'scenes'/sid/'completion';folder.mkdir();(folder/'G_full.json').write_text('DO NOT REUSE')
        return source,sid

    def prepared(self,worlds=2):
        source,sid=self.source();root=self.base/'run'
        result=local.prepare(root,source,worlds=worlds)
        self.assertEqual(result['observed'],16)
        return root,source,sid

    def test_preparation_reuses_every_observation_byte_and_schedules_full_completion(self):
        root,source,sid=self.prepared();meta=read(root/'local_world_run.json')
        self.assertTrue(meta['hidden_state_generation']);self.assertFalse(meta['API_used'])
        self.assertEqual(meta['scheduled_phases'],list(local.PHASES));self.assertNotIn('observer2d',local.PHASES)
        self.assertEqual(local.check_run(root)['backends'],['qwen'])
        self.assertFalse((root/'scenes'/sid/'completion').exists())
        for name,digest in meta['observation_bindings'][0]['files'].items():
            self.assertEqual(sha(root/'scenes'/sid/'observation'/name),digest)
            self.assertEqual(sha(source/'scenes'/sid/'observation'/name),digest)

    def test_failed_or_mismatched_observation_is_rejected_before_run_creation(self):
        source,sid=self.source();obs=source/'scenes'/sid/'observation'
        p=obs/'qwen/observer3d/summary.json';summary=read(p);write(p,dict(summary,status='failed'))
        with self.assertRaisesRegex(ValueError,'Require completed'):local.prepare(self.base/'run',source)
        self.assertFalse((self.base/'run').exists())
        write(p,summary);p=obs/'qwen/observer3d/config.json';config=read(p);config['source_sha256']='different';write(p,config)
        with self.assertRaisesRegex(ValueError,'not bound'):local.prepare(self.base/'run',source)
        self.assertFalse((self.base/'run').exists())

    def test_mutation_of_original_or_imported_G_obs_is_blocked(self):
        root,source,sid=self.prepared()
        for base in (source,root):
            p=base/'scenes'/sid/'observation/qwen/observer3d/G_obs.json';before=p.read_bytes();p.write_bytes(before+b' ')
            with self.assertRaisesRegex(ValueError,'Frozen observation changed'):local.check_run(root)
            p.write_bytes(before)

    def test_no_completion_is_not_an_exported_empty_world(self):
        root,_,_=self.prepared();result=local.report(root)
        self.assertEqual(result['status'],'partial_or_failed')
        self.assertTrue(all(r['status']=='not_run' and r['placed_new'] is None for r in result['records']))
        self.assertEqual(result['worlds_with_new_objects'],0)

    def test_full_v13_Qwen_only_pipeline_exports_new_objects_and_visualization(self):
        root,source,sid=self.prepared(worlds=2)
        def provider(config,backend):
            self.assertEqual(backend,'qwen','No API factory may be selected')
            return CPUProgramProvider() if not (root/'scenes'/sid/'completion/shared/domain.json').exists() else CPUProvider(all_present=True)
        with patch.object(pipeline,'provider',side_effect=provider):
            for name in local.PHASES:
                result=local.phase(root,name)
                self.assertIn(result['status'],('complete','complete_with_warnings'),(name,result))
        comp=root/'scenes'/sid/'completion';domain=read(comp/'shared/domain.json')
        self.assertEqual(set(domain['proposal_provenance']),{'qwen'})
        self.assertFalse((comp/'proposals/api').exists());self.assertFalse((comp/'shared/mechanisms/api.json').exists())
        self.assertEqual(read(comp/'run.json')['experiments'],['end_to_end'])
        original=read(source/'scenes'/sid/'observation/qwen/observer3d/G_obs.json')['objects']
        for folder in (comp/'end_to_end/qwen/completion').glob('world_*'):
            graph=read(folder/'G_full.json');self.assertEqual(graph['objects'][:len(original)],original)
            self.assertGreater(len(graph['objects']),len(original))
        # Exercise real renderer for one world, then a lightweight renderer for
        # state-classification tests. These are synthetic replies, not GPU runs.
        result=local.report(root)
        self.assertEqual(result['status'],'complete_needs_review');self.assertEqual(result['worlds_with_new_objects'],2)
        self.assertTrue(all(r['placed_new']>0 and r['observed_preserved'] for r in result['records']))
        self.assertEqual(read(root/'report'/sid/'world_01/views/visual_validation.json')['labels'],len(read(comp/'end_to_end/qwen/completion/world_01/G_full.json')['objects']))
        self.assertTrue((root/'report'/sid/'world_01/views/scene_3d.html').exists())
        self.check_links(root)
        # A rejected numeric candidate remains visible but never a success.
        folder=comp/'end_to_end/qwen/completion/world_01';(folder/'G_full.json').rename(folder/'G_full_candidates.json')
        with patch.object(local,'render_world',side_effect=self.renderer):
            # Renderer fixture creates an output directory; remove previous views.
            for p in (root/'report').glob('*/world_*/views'):shutil.rmtree(p)
            result=local.report(root)
        self.assertEqual(result['records'][0]['status'],'rejected_layout_candidate')
        self.assertEqual(result['status'],'partial_or_failed')

    @staticmethod
    def renderer(state,image,hfov,out):
        from PIL import Image
        out.mkdir();Image.new('RGB',(80,40),'white').save(out/'overview.png')
        (out/'index.html').write_text('<p>CPU synthetic world fixture</p>')
        return dict(labels=len(state['objects']))

    def test_empty_increment_is_recorded_without_retry_or_false_success(self):
        root,_,sid=self.prepared(worlds=1)
        folder=root/'scenes'/sid/'completion/end_to_end/qwen/completion/world_01';folder.mkdir(parents=True)
        state=deepcopy(read(root/'scenes'/sid/'observation/qwen/observer3d/G_obs.json'))
        state.update(scope='full',world_state_ready=True)
        write(folder/'G_full.json',state);write(folder/'numeric_review.json',dict(geometry_contract_passed=True))
        write(folder/'semantic_world.json',dict(objects=[]))
        write(folder.parent/'summary.json',dict(status='worlds_exported_needs_review',worlds=[dict(folder='world_01')],failures=[]))
        with patch.object(local,'render_world',side_effect=self.renderer):result=local.report(root)
        self.assertEqual(result['records'][0]['status'],'observed_only_sample')
        self.assertEqual(result['records'][0]['placed_new'],0)
        self.assertEqual(result['status'],'partial_or_failed')

    def test_single_backend_does_not_satisfy_legacy_paired_dependencies(self):
        root,_,sid=self.prepared();local.phase(root,'prepare-scenes')
        comp=root/'scenes'/sid/'completion'
        pipeline.fresh.propose(comp,'qwen',CPUProgramProvider())
        self.assertFalse(pipeline.missing_inputs(comp.parent,'pool','qwen',('qwen',)))
        self.assertTrue(any('/api/' in p for p in pipeline.missing_inputs(comp.parent,'pool','qwen')))

    def test_shell_schedules_worlds_and_report_without_observer_or_API_stages(self):
        text=(ROOT/'qwen_edit_pano/scripts/organized_local_world.sh').read_text()
        self.assertIn('for ow_phase in prepare-scenes propose pool mechanisms worlds',text)
        self.assertNotIn('--backend api',text);self.assertNotIn('SILICONFLOW_API_KEY',text)
        self.assertNotIn('lo_phase observer2d',text);self.assertIn('report --root',text)

    def check_links(self,root):
        class Links(HTMLParser):
            def handle_starttag(inner,tag,attrs):
                for name,value in attrs:
                    if name in ('src','href') and value and not value.startswith(('http:','https:','data:','#','javascript:')):
                        self.assertTrue((page.parent/value.split('#')[0]).exists(),(page,value))
        for page in (root/'report').rglob('*.html'):
            Links().feed(page.read_text())


if __name__=='__main__':unittest.main()
