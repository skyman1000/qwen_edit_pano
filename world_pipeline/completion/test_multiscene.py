"""CPU fixtures: same v13 mechanisms, fresh domains and projection geometry."""
from copy import deepcopy
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ..common import ROOT,read,write,sha,field
from . import assembly_multiscene as suite,assembly_presentation_visual as visual
from . import assembly_multiscene_report as report,assembly_organized_experiment as organized,assembly_organized as scm
from .test_compositional import CPUProgramProvider,SOURCE
from .test_organized import CPUProvider as OriginalCPUProvider


class CPUProvider(OriginalCPUProvider):
    def ask(self,image,prompt,path,**kwargs):
        if '\nREQUESTED_OBJECT_CONTRACTS: ' in prompt:
            requested=json.loads(prompt.split('\nREQUESTED_OBJECT_CONTRACTS: ')[1].split('\n')[0]);request=json.loads(prompt.split('\nREQUEST: ')[1].split('\n')[0])
            value=dict(group_id=request['group_id'],objects=[]);root=0
            if request['range_required']:value['range_m']=3.
            for obj in requested:
                keys=obj['required_named_fields'];row=dict(id=obj['object_id'],yaw_deg=0.)
                if 'root_offset_m' in keys:row['root_offset_m']=[root*2.,-.5,0.];root+=1
                else:row['host_offset_uv']=[0.,0.]
                for key in ('surface_fraction','hanging_drop_m','host_offset_y_m'):
                    if key in keys:row[key]=.5 if key=='surface_fraction' else .1 if key=='hanging_drop_m' else 0.
                value['objects'].append(row)
            path=Path(path);self.requests.append(dict(path=str(path),prompt=prompt,**kwargs));path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value
        if '\nREQUIRED_FIELDS_BY_ID: ' in prompt:
            fields=json.loads(prompt.split('\nREQUIRED_FIELDS_BY_ID: ')[1].split('\n')[0]);gid=prompt.split('\nGROUP_ID: ')[1].split('\n')[0]
            value=dict(group_id=gid,roots=[],hosted=[]);root=0
            if '\"range_m\"' not in prompt.split('\nFROZEN_GROUP_RANGE: ')[1].split('\n')[0] and 'Do not return range_m' not in prompt:value['range_m']=3.
            for oid,keys in fields.items():
                row=dict(id=oid,yaw_deg=0.)
                if 'root_offset_m' in keys:row['root_offset_m']=[root*2.,-.5,0.];root+=1
                else:row['host_offset_uv']=[0.,0.]
                for key in ('surface_fraction','hanging_drop_m','host_offset_y_m'):
                    if key in keys:row[key]=.5 if key=='surface_fraction' else .1 if key=='hanging_drop_m' else 0.
                value['roots' if 'root_offset_m' in row else 'hosted'].append(row)
            path=Path(path);self.requests.append(dict(path=str(path),prompt=prompt,**kwargs));path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value
        if '\nROW_COLUMNS_BY_ID: ' not in prompt:return super().ask(image,prompt,path,**kwargs)
        rows=json.loads(prompt.split('\nROW_COLUMNS_BY_ID: ')[1].split('\n')[0]);group=prompt.split('\nGROUP_ID: ')[1].split('\n')[0]
        values=[];root=0
        for oid,columns in rows.items():
            row=[oid]
            for c in columns[1:]:row.append(root*2. if c=='root_dx_m' else -.5 if c=='root_dy_m' else .5 if c=='surface_fraction' else .1 if c=='hanging_drop_m' else 0.)
            if 'root_dx_m' in columns:root+=1
            values.append(row)
        value=dict(group_id=group,range_m=3.,placements=values);path=Path(path)
        self.requests.append(dict(path=str(path),prompt=prompt,**kwargs));path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value


def create(path):path.mkdir(parents=True,exist_ok=False);return path.resolve()


def graph_fixture():
    full=read(ROOT/'qwen_edit_pano/outputs/organized_range_repair_20261027_183331_2074997/end_to_end/qwen/completion/world_01/G_full.json')
    observed=deepcopy(full['objects'][0]);observed.update(track_id='obs_000',raw_category='sofa',category_id=field(10,'fixture'),center_local_m=field([0.,0.,3.],'fixture'),size_aabb_local_m=field([1.,1.,1.],'fixture'))
    new=deepcopy(observed);new.update(track_id='hidden_table',raw_category='side table',category_id=field(5,'fixture'),local_evidence='unobserved',visual_color='#9c43bb',center_local_m=field([0.,0.,-3.],'fixture'))
    unknown=deepcopy(new);unknown.update(track_id='unknown',raw_category='unknown fixture',category_id=field(None,reason='unmapped'),center_local_m=field([3.,0.,0.],'fixture'))
    full['objects']=[observed,new,unknown];return full


class MultisceneTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)

    def initialized_suite(self):
        im=self.base/'fixture_input.png';Image.new('RGB',(160,160),'#ccddee').save(im)
        manifest=self.base/'inputs.jsonl';manifest.write_text(json.dumps(dict(id='fixture',local_rgb=str(im),local_sha256=sha(im),hfov_degrees=90,split='CPU_fixture'))+'\n')
        root=self.base/'suite'
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG=''),patch.object(suite,'new_output',side_effect=create),\
            patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=create),\
            patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}):suite.initialize_suite(root,manifest,worlds=2)
        return root

    def fresh_run(self,name='run',single=False):
        scene=self.base/name;scene.mkdir(parents=True);observation=scene/'observation';observation.mkdir()
        for filename in ('experiment.json','input.png'):shutil.copyfile(SOURCE/filename,observation/filename)
        for b in ('qwen','api'):
            (observation/b).mkdir()
            for stage in ('observer2d','observer3d'):shutil.copytree(SOURCE/b/stage,observation/b/stage)
        config=dict(seed=3407,worlds=2,draws=4,concentration=1.,variant_concentration=1.)
        if single:config.update(mode='single',worlds=1,experiments=['end_to_end'])
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG='',CAUPANO_LAYOUT_RGB='0'),\
            patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
            patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=create),\
            patch('qwen_edit_pano.world_pipeline.completion.assembly_sdscene_experiment.new_output',side_effect=create):suite.prepare_scene(scene,config)
        root=scene/'completion'
        for b in ('qwen','api'):self.assertEqual(suite.fresh.propose(root,b,CPUProgramProvider())['status'],'complete')
        suite.pool_domain(root);return root

    def test_default_five_RGB_inputs_exist_are_distinct_and_not_development_image(self):
        from ..common import rows
        files=rows(suite.DEFAULT_INPUTS);self.assertEqual(len(files),5)
        pixels=[]
        for r in files:
            self.assertEqual(sha(Path(r['local_rgb'])),r['local_sha256'])
            with Image.open(r['local_rgb']) as im:pixels.append(im.convert('RGB').tobytes())
        with Image.open(SOURCE/'input.png') as im:dev=im.convert('RGB').tobytes()
        self.assertEqual(len(set(pixels)),5);self.assertNotIn(dev,pixels)

    def test_fresh_pool_promotes_profiles_before_v13_queries_without_catalog_reuse(self):
        root=self.fresh_run();d=read(root/'shared/domain.json');r=read(root/'run.json')
        self.assertEqual(d['version'],13);self.assertTrue(any(h.get('primary_role')=='work' for h in d['profiles']))
        self.assertIsNone(r['catalog_source']);self.assertFalse(d['candidate_proposals_reused'])
        self.assertEqual(set(d['proposal_provenance']),{'qwen','api'})
        self.assertEqual(read(root/'call_budget.json')['initial_interface_probe_calls_per_backend'],0)
        for branch in suite.EXPERIMENTS:self.assertEqual(read(root/branch/'experiment.json')['completion_revision'],'organized_v13')

    def test_new_estimator_matches_original_v13_main_tables_and_actual_CPT(self):
        a=self.fresh_run();b=self.base/'old_estimator';shutil.copytree(a,b)
        # Original estimator's additional historical probe does not alter its main CPT.
        write(b/'shared/domain_v12.json',read(b/'shared/proposal_domain.json'))
        budget=read(b/'call_budget.json');budget['maximum_mechanism_calls_with_one_repair_each']+=128;write(b/'call_budget.json',budget)
        suite.estimate(a,'qwen',CPUProvider());organized.mechanisms(b,'qwen',CPUProvider())
        x=read(a/'shared/mechanisms/qwen.json');y=read(b/'shared/mechanisms/qwen.json')
        for key in ('recognition','presence','variants','original_parent_prompt_plan'):self.assertEqual(x[key],y[key])
        d=read(a/'shared/domain.json');self.assertEqual(scm.compile_model(d,x)['nodes'],scm.compile_model(d,y)['nodes'])

    def test_matched_backends_pool_raw_counts_once_and_preserve_Hcut_law(self):
        root=self.fresh_run()
        for b in ('qwen','api'):suite.estimate(root,b,CPUProvider())
        organized.pool_mechanisms(root);d=read(root/'shared/domain.json');t=read(root/'shared/mechanisms/pooled.json')
        functional=scm.compile_model(d,t);cut=scm.compile_model(d,t,'function_free')
        for key in ('object_presence','hidden_count_probabilities','any_declared_region_presence'):
            for k,v in scm.marginals(functional)[key].items():self.assertAlmostEqual(v,scm.marginals(cut)[key][k])
        for branch in suite.EXPERIMENTS:
            for b in ('qwen','api'):
                result=organized.world(root,branch,b,CPUProvider())
                self.assertEqual(len(result['worlds'])+len(result['blocked_worlds']),2,result)
        metrics,worlds,effects=report.collect_run(root,'CPU_fixture');self.assertEqual(len(metrics),12)
        self.assertTrue(all(m['observed_preserved'] for m in metrics));self.assertTrue(effects)

    def test_palette_uses_one_generated_mapped_color_and_preserves_saved_fields(self):
        state=graph_fixture();before=deepcopy(state);rows=visual.label_rows(state)
        self.assertEqual([r['color'] for r in rows],[visual.OBSERVED,visual.MAPPED,visual.UNMAPPED])
        self.assertEqual(rows[1]['label'],'N01 table');self.assertEqual(rows[0]['label'],'O01 sofa')
        self.assertEqual(state,before);self.assertEqual(state['objects'][1]['visual_color'],'#9c43bb')

    def test_coarse_saved_category_keeps_readable_noun_without_changing_category_id(self):
        state=graph_fixture();obj=state['objects'][0];obj.update(raw_category='furniture',description='sofa',category_id=field(36,'fixture'))
        before=deepcopy(state);label=visual.label_rows(state)[0]
        self.assertEqual(label['label'],'O01 sofa [furniture]');self.assertEqual(label['category'],'furniture');self.assertEqual(state,before)

    def test_front_and_back_projection_have_correct_camera_direction(self):
        state=graph_fixture();front,back=state['objects'][:2]
        self.assertTrue(visual.project_edges(front,'front',90,600,600));self.assertFalse(visual.project_edges(back,'front',90,600,600))
        self.assertTrue(visual.project_edges(back,'back',90,600,600));self.assertFalse(visual.project_edges(front,'back',90,600,600))
        self.assertTrue(visual.project_edges(state['objects'][2],'right',90,600,600))
        self.assertTrue(np.allclose(visual.basis('front'),np.eye(3)))

    def test_ERP_has_exact_two_to_one_ratio_correct_rays_and_unknown_background(self):
        image=Image.new('RGB',(120,120),(17,31,47));erp,mask=visual.erp_background(image,90,320,160)
        self.assertEqual(erp.size,(320,160));self.assertTrue(mask[80,160]);self.assertFalse(mask[80,0])
        self.assertTrue(np.allclose(np.asarray(erp)[80,160],[17,31,47],atol=1));self.assertEqual(tuple(np.asarray(erp)[80,0]),(232,232,232))
        with self.assertRaises(ValueError):visual.erp_background(image,90,300,160)
        uv=visual.erp_uv([[0,0,1],[1,0,0],[0,1,1]],320,160)
        self.assertTrue(np.allclose(uv,[[160,80],[240,80],[160,40]]))

    def test_projection_near_plane_and_canvas_clipping_are_finite(self):
        p,q=visual.line_in_canvas([-100,10],[200,10],100,80);self.assertTrue(np.allclose(p,[0,10]));self.assertTrue(np.allclose(q,[99,10]))
        obj=graph_fixture()['objects'][0];obj['center_local_m']['value']=[.4,0,.1]
        for segment in visual.project_edges(obj,'front',90,600,600):
            self.assertTrue(np.isfinite(segment).all());self.assertTrue(all(0<=p[0]<=599.000001 and 0<=p[1]<=599.000001 for p in segment))

    def test_renderer_exports_all_requested_views_labels_and_no_state_mutation(self):
        state=graph_fixture();before=deepcopy(state);out=self.base/'visual'
        checks=visual.render_world(state,Image.new('RGB',(480,480),'#eeeeee'),90,out)
        for name in ('local_rgb.png','layout_xz.png','local_projection.png','direction_projections.png','scene_3d.png','scene_3d.html','erp_state_map.png','erp_overlay.png','erp_input_projection.png','overview.png','index.html'):self.assertTrue((out/name).exists(),name)
        self.assertEqual(checks['erp_dimensions'],[1600,800]);self.assertEqual(state,before)
        self.assertIn('N01 table',(out/'scene_3d.html').read_text());self.assertNotIn('hidden_table',checks['projected_ids_in_original_view'])
        self.assertIn('hidden_table',checks['direction_projected_ids']['back'])

    def test_pending_report_never_invents_experimental_results(self):
        image=self.base/'input.png';Image.new('RGB',(160,160),'white').save(image)
        with patch.object(report,'new_output',side_effect=create):p=report.build([('unrun_image',None,image)],self.base/'report','CPU pending fixture')
        result=read(p.parent/'results.json');self.assertEqual(result['metrics'],[]);self.assertEqual(result['pending_inputs'],['unrun_image'])
        self.assertIn('尚未',p.read_text());self.assertTrue((p.parent/'group_meeting.html').exists())

    def test_original_run_is_not_an_initialized_suite(self):
        with self.assertRaises(FileNotFoundError):suite.verify_suite(SOURCE)
        self.assertFalse((SOURCE/'phase_records').exists())

    def test_initialized_suite_has_same_frozen_configuration_and_fresh_empty_observers(self):
        root=self.initialized_suite();s=suite.verify_suite(root);self.assertEqual(s['api_model'],'deepseek-ai/DeepSeek-V4.1-Flash')
        self.assertFalse(s['old_candidate_catalog_reused']);self.assertFalse(s['GT_world_used_for_generation'])
        self.assertFalse((root/'scenes/fixture/observation/qwen/observer2d').exists())
        self.assertFalse((root/'scenes/fixture/completion').exists())

    def test_resume_reuses_completed_or_failed_phase_without_new_model_draws(self):
        root=self.initialized_suite()
        class P:
            def __init__(self):self.calls=0
            def ask(inner,image,prompt,path,**kw):
                inner.calls+=1;v=dict(objects=[dict(name='sofa',category='sofa',bbox_1000=[100,200,900,900],on_floor=True)],evidence_description='CPU fixture visible sofa',done=True)
                path.write_text(json.dumps(v));write(path.with_suffix('.json'),v);return v
        p=P()
        with patch.object(suite,'provider',return_value=p):
            self.assertEqual(suite.run_phase(root,'observer2d')['status'],'complete')
            self.assertEqual(suite.run_phase(root,'observer2d')['status'],'complete')
        self.assertEqual(p.calls,1)
        marker=root/'phase_records/fixture_propose_api.json';write(marker,dict(scene='fixture',status='failed',error='fixed-budget CPU failure'))
        with patch.object(suite,'provider',side_effect=AssertionError('Failed phase must not be resampled')):
            self.assertEqual(suite.run_phase(root,'propose','api')['status'],'partial_or_failed')

    def test_fresh_scene_pipeline_rejects_cross_image_catalog_and_unsafe_scene_ids(self):
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG=str(SOURCE)),self.assertRaises(ValueError):suite.initialize_suite(self.base/'illegal')
        manifest=self.base/'bad.jsonl';manifest.write_text(json.dumps(dict(id='../escape',local_rgb='unused',hfov_degrees=90))+'\n')
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG=''),self.assertRaises(ValueError):suite.initialize_suite(self.base/'bad_output',manifest)
        self.assertFalse((self.base/'bad_output').exists())

    def test_single_plan_is_one_world_and_main_only_without_changing_evaluation_mode(self):
        self.assertEqual(suite.output_plan('single'),(1,('end_to_end',)))
        self.assertEqual(suite.output_plan('evaluation'),(4,suite.EXPERIMENTS))
        with self.assertRaises(ValueError):suite.output_plan('single',4)
        with self.assertRaises(ValueError):suite.output_plan('unsupported')

    def test_single_pipeline_executes_only_one_world_and_skips_control_pool(self):
        root=self.fresh_run('single_suite/scenes/sample',single=True);top=self.base/'single_suite';(top/'phase_records').mkdir()
        for backend in ('qwen','api'):suite.estimate(root,backend,CPUProvider())
        cfg=read(root/'run.json');self.assertEqual(cfg['experiments'],['end_to_end']);self.assertEqual(cfg['worlds'],1)
        self.assertEqual(read(root/'call_budget.json')['layout_worlds_per_backend'],1)
        with patch.object(suite,'verify_suite',return_value=dict(scenes=[dict(id='sample')])),patch.object(suite,'provider',return_value=CPUProvider()):
            pooled=suite.run_phase(top,'pool-mechanisms');self.assertTrue(pooled['scenes'][0]['result']['skipped'])
            actual=suite.run_phase(top,'worlds','qwen')
        result=actual['scenes'][0]['result'];self.assertEqual(set(result),{'end_to_end'})
        self.assertEqual(len(result['end_to_end']['worlds'])+len(result['end_to_end']['blocked_worlds']),1)
        self.assertFalse((root/'end_to_end/qwen/completion/world_02').exists())
        self.assertFalse((root/'layout_control/qwen/completion').exists());self.assertFalse((root/'function_free/qwen/completion').exists())


if __name__=='__main__':unittest.main()
