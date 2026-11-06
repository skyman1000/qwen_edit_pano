"""CPU fixtures for actual rich-domain execution; never instantiate providers."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ..common import ROOT,read,write,sha
from . import assembly_organized as model,assembly_organized_experiment as experiment,assembly_organized_geometry as geometry

ACTUAL=ROOT/'qwen_edit_pano/outputs/compositional_completion_20261027_105105_496519'


def make_fixture_source(dest):
    """Two original cases in a new CPU-test fixture, not a new actual run."""
    dest.mkdir();r=read(ACTUAL/'run.json');r['worlds']=2;write(dest/'run.json',r)
    for rel in ('shared/domain.json','shared/mechanisms/qwen.json','shared/mechanisms/api.json'):
        p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ACTUAL/rel,p)
    for branch in ('end_to_end','layout_control'):
        folder=dest/branch;folder.mkdir();shutil.copyfile(ACTUAL/branch/'experiment.json',folder/'experiment.json');shutil.copyfile(ACTUAL/branch/'input.png',folder/'input.png')
        for b in ('qwen','api'):
            (folder/b).mkdir()
            for observer in ('observer2d','observer3d'):shutil.copytree(ACTUAL/branch/b/observer,folder/b/observer)
            source=ACTUAL/branch/b/'completion';target=folder/b/'completion';target.mkdir()
            for name in ('G_obs.json','causal_model.json','support_plane.json','case.json','summary.json','probability_audit.json'):
                shutil.copyfile(source/name,target/name)
            for i in (1,2):
                src=source/f'world_{i:02d}';dst=target/src.name;dst.mkdir()
                for name in ('semantic_world.json','execution.json','numeric_review.json','G_full.json','G_full_candidates.json'):
                    if (src/name).exists():shutil.copyfile(src/name,dst/name)
    return dest


class CPUProvider:
    def __init__(self,all_present=False):self.requests=[];self.all_present=all_present
    def ask(self,image,prompt,path,seed=3407,max_tokens=8192,**kw):
        if image is not None:raise AssertionError('Default paired mode must be text only')
        path=Path(path);self.requests.append(dict(path=str(path),seed=seed,prompt=prompt))
        if '\nROLE_PROGRAMS: ' in prompt:
            h=json.loads(prompt.split('\nH: ')[1].split('\n')[0]);role=prompt.split('\nROLE: ')[1].split('\n')[0]
            present=self.all_present or (h.get('primary_role')==role if 'primary_role' in h else role in h['functions']);options=json.loads(prompt.split('\nALLOWED_CHOICES: ')[1])
            value=dict(choice=next(o['choice'] for o in options if o['state'].startswith('One' if present else 'None')))
        elif '\nPROGRAMS: ' in prompt:
            h=json.loads(prompt.split('\nH: ')[1].split('\n')[0]);role=prompt.split('\nROLE: ')[1].split('\n')[0];options=json.loads(prompt.split('\nALLOWED_CHOICES: ')[1])
            chosen=next((o for o in options if h.get('primary_role')==role and 'rear' in o['regions'].values()),options[0]);value=dict(choice=chosen['choice'])
        elif path.stem.startswith('recognition_'):
            options=json.loads(prompt.split('\nALLOWED_CHOICES: ')[1]);value=dict(choice=next(o['choice'] for o in options if o['profile'].get('primary_role')=='work'))
        elif '_dimensions' in path.stem:
            objects=json.loads(prompt.split('\nFIXED_OBJECTS: ')[1]);value=dict(objects=[])
            for o in objects:
                noun=o['category'].lower().replace('_',' ')
                if noun=='desk' or 'table' in noun:dims=[1.2,.75,.8]
                elif 'bookcase' in noun or 'cabinet' in noun or noun=='shelf':dims=[1.,1.8,.5]
                elif 'media console' in noun:dims=[1.5,.5,.5]
                elif 'chair' in noun:dims=[.6,.9,.6]
                elif 'television' in noun:dims=[.8,.5,.05]
                elif 'lamp' in noun:dims=[.15,.4,.15]
                else:dims=[.16,.08,.12] if o['attachment']['host'] else [.4,.5,.4]
                value['objects'].append(dict(id=o['id'],full_size_m=dims))
        elif '_placement' in path.stem:
            value=json.loads(prompt.split('\nTEMPLATE: ')[1].split('\nFROZEN_DIMENSIONS: ')[0])
            for group in value['groups']:
                group['range_m']=3.;roots=0
                for o in group['objects']:
                    o['yaw_deg']=0.
                    if 'root_offset_m' in o:o['root_offset_m']=[roots*2.,-.5,0.];roots+=1
                    if 'host_offset_uv' in o:o['host_offset_uv']=[0.,0.]
                    if 'host_offset_y_m' in o:o['host_offset_y_m']=0.
                    if 'surface_fraction' in o:o['surface_fraction']=.5
                    if 'hanging_drop_m' in o:o['hanging_drop_m']=.1
        else:raise AssertionError(path.stem)
        write(path.with_suffix('.request.json'),dict(prompt=prompt,seed=seed,max_tokens=max_tokens,image_delivered=False))
        path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value


def toy_world():
    return dict(configuration='desk with two belongings',hypothesis=dict(id='h0',functions=[]),relations=[],spatial_program={'g0':{'region':'rear'}},
        groups=[dict(key='g0',function='work',layout_region='rear')],objects=[dict(id=i,name=i,category=i,variant=i,group='g0',support_model='rigid_box',placement_hint='test',
            attachment=dict(host=None if i=='desk' else 'desk',kind='floor' if i=='desk' else 'top')) for i in ('desk','book','mug')])


class OrganizedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared=tempfile.TemporaryDirectory();cls.source=make_fixture_source(Path(cls.shared.name)/'source')
    @classmethod
    def tearDownClass(cls):cls.shared.cleanup()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.temp=Path(self.tmp.name)
    def prepare(self):
        r=self.temp/'run'
        with patch.dict(os.environ,CAUPANO_LAYOUT_RGB='0'),patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_sdscene_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            experiment.prepare(r,self.source,draws=4)
        return r
    def bundle(self):
        r=self.prepare()
        for b in ('qwen','api'):experiment.mechanisms(r,b,CPUProvider())
        experiment.pool_mechanisms(r);return r

    def test_same_rich_domain_and_new_soft_organization_states(self):
        r=self.prepare();d=read(r/'shared/domain.json');old=read(self.source/'shared/domain.json')
        for k in ('programs','slots','spatial_choices','base_model'):self.assertEqual(d[k],old[k])
        self.assertEqual(len(d['base_model']['component_registry']),70)
        self.assertEqual(len(d['profiles']),5);self.assertTrue(any(h['primary_role']=='work' for h in d['profiles']))
        self.assertEqual(read(r/'call_budget.json')['initial_main_calls_per_backend'],164)
        self.assertEqual(read(r/'call_budget.json')['initial_interface_probe_calls_per_backend'],64)

    def test_four_encodings_are_balanced_without_balancing_answers(self):
        rows=[experiment.binary_aliases(j) for j in range(4)];self.assertEqual(len({tuple(r.items()) for r in rows}),4)
        self.assertEqual([list(r.values())[0] for r in rows].count('1'),2)
        self.assertEqual(sum(r['state_0']=='1' for r in rows),2)
        for r in rows:self.assertEqual(set(r.values()),{'0','1'})
        for v in ({'choice':[]},{'choice':'wrong'},{'present':True},{'choice':None}):
            with self.assertRaises(ValueError):experiment.choice(v,rows[0])

    def test_actual_probabilities_change_with_H_and_Hcut_preserves_marginals(self):
        r=self.bundle();d=read(r/'shared/domain.json');t=read(r/'shared/mechanisms/qwen.json');m=model.compile_model(d,t);cut=model.compile_model(d,t,'function_free')
        hr=next(h['id'] for h in d['profiles'] if h['primary_role']=='reading');hw=next(h['id'] for h in d['profiles'] if h['primary_role']=='work')
        a=model.marginals(m,hr);b=model.marginals(m,hw)
        self.assertAlmostEqual(a['category_presence']['desk'],.1);self.assertAlmostEqual(b['category_presence']['desk'],.9)
        for field in ('object_presence','hidden_count_probabilities','any_declared_region_presence'):
            for k,v in model.marginals(m)[field].items():self.assertAlmostEqual(v,model.marginals(cut)[field][k])
        self.assertEqual(model.marginals(cut,hr)['object_presence'],model.marginals(cut,hw)['object_presence'])

    def test_all_true_mechanisms_remain_all_true_without_artificial_H_effect(self):
        r=self.prepare();experiment.mechanisms(r,'qwen',CPUProvider(all_present=True));d=read(r/'shared/domain.json');t=read(r/'shared/mechanisms/qwen.json');m=model.compile_model(d,t)
        for role in d['slots']:
            for h in d['profiles']:
                self.assertEqual(t['presence'][h['id']+'|'+role]['counts'],{'1':4})
                self.assertAlmostEqual(m['activation_rows'][role][h['id']][1],.9)

    def test_prompt_orders_and_local_seeds_match_across_backends_and_H(self):
        r=self.bundle();a=read(r/'shared/mechanisms/qwen.json');b=read(r/'shared/mechanisms/api.json');self.assertEqual(a['original_parent_prompt_plan'],b['original_parent_prompt_plan'])
        plans=[v for v in a['original_parent_prompt_plan'] if v['stage'].startswith('presence_') and '_work_0' in v['stage']]
        self.assertEqual(len({v['local_seed'] for v in plans}),1)
        probe=read(r/'shared/interface_probe_qwen.json');self.assertTrue(probe['complete']);self.assertEqual(len(probe['rows']),16)
        self.assertEqual(sum(v['counts'].get('1',0) for k,v in probe['rows'].items() if k.endswith('|work')),0)
        self.assertEqual(sum(v['counts'].get('1',0) for k,v in read(r/'shared/source_mechanisms/qwen.json')['presence'].items() if k.endswith('|work')),16)

    def test_pool_smooths_raw_counts_once_and_accepted_cache_is_bound(self):
        r=self.bundle();d=read(r/'shared/domain.json');t=read(r/'shared/mechanisms/pooled.json');m=model.compile_model(d,t)
        self.assertAlmostEqual(m['activation_rows']['work']['h4'][1],8.5/9)
        p=CPUProvider()
        with self.assertRaises(FileExistsError):experiment.mechanisms(r,'qwen',p)
        self.assertEqual(p.requests,[])
        f=next((r/'mechanism_calls/qwen').glob('*.accepted.json'));c=read(f);c['seed']+=1;write(f,c)
        with self.assertRaises(ValueError):experiment.mechanisms(r,'qwen',p)

    def test_invalid_or_missing_rows_never_become_empty_world_defaults(self):
        from .assembly_compositional import raw
        with self.assertRaises(ValueError):raw([None,None])
        r=self.prepare();self.assertFalse(experiment.ready(r,'end_to_end','qwen'));self.assertTrue(experiment.ready(r,'geometry_control','qwen'))
        result=experiment.world(r,'end_to_end','qwen',CPUProvider());self.assertEqual(result['status'],'blocked_dependency')

    def test_joint_packing_separates_siblings_and_reports_infeasibility(self):
        centres=[[0.,0.]]*3;sizes=[[.3,.2],[.2,.2],[.2,.3]]
        p,r=geometry.pack_rectangles(centres,sizes,[[-.5,-.4]]*3,[[.5,.4]]*3,[(0,.2)]*3)
        self.assertIsNotNone(p,r)
        for i in range(3):
            for j in range(i):self.assertTrue(any(abs(p[i,k]-p[j,k])>=(sizes[i][k]+sizes[j][k])/2+9e-5 for k in range(2)))
        p,r=geometry.pack_rectangles([[0.,0.]]*2,[[10.,10.]]*2,[[-.1,-.1]]*2,[[.1,.1]]*2,[(0,1)]*2)
        self.assertIsNone(p);self.assertEqual(r['status'],'infeasible')

    def test_typed_host_coordinates_and_frozen_dimensions_are_not_overwritten(self):
        w=toy_world();dims={o['id']:dict(id=o['id'],full_size_m=[1.2,.75,.8] if o['id']=='desk' else [.1,.1,.1],yaw_deg=90. if o['id']=='desk' else 0.) for o in w['objects']}
        value=geometry.placement_template(w);value['groups'][0]['range_m']=3.
        for row in value['groups'][0]['objects']:
            row['yaw_deg']=90. if row['id']=='desk' else 0.
            if row['id']=='desk':row['root_offset_m']=[0.,0.,0.]
            else:row['host_offset_uv']=[.5,-.5];row['full_size_m']=[9,9,9]
        layout=geometry.placement_answer(value,w,dict(objects=[]),dims,90)
        book=next(o for o in layout['g0']['objects'] if o['id']=='book');self.assertEqual(book['offset_m'],[.3,0.,-.2]);self.assertEqual(book['intrinsic_size_m'],[.1,.1,.1]);self.assertEqual(book['offset_frame'],'host_local')
        bad=deepcopy(value);bad['groups'][0]['objects'][1]['host_offset_uv']=[None,1.]
        with self.assertRaises(ValueError):geometry.placement_answer(bad,w,dict(objects=[]),dims,90)
        with self.assertRaises(ValueError):geometry.dimensions_answer(dict(objects=[]),w)

    def test_frozen_world_replay_preserves_exact_all_source_states(self):
        r=self.prepare();config=read(r/'geometry_control/experiment.json');config['execution_backend']='qwen';d=read(r/'shared/domain.json')
        m=model.prepare_model(d['base_model'],read(r/'shared/case.json'),config,None);worlds=model.sample_worlds(m,999,2)
        self.assertEqual(worlds,[read(self.source/'layout_control/qwen/completion'/f'world_{i:02d}/semantic_world.json') for i in (1,2)])
        with self.assertRaises(ValueError):model.sample_worlds(m,999,1)
        before=sha(self.source/'shared/domain.json');experiment.verify_sources(r);self.assertEqual(sha(self.source/'shared/domain.json'),before)

    def test_cli_errors_never_write_into_an_original_v12_source(self):
        before={str(p):sha(p) for p in self.source.rglob('*') if p.is_file()}
        for args in (['prepare','--source-run',str(self.source)],['world','--backend','qwen','--experiment','end_to_end']):
            v=subprocess.run([sys.executable,'-m','qwen_edit_pano.world_pipeline.completion.assembly_organized_experiment',*args,'--root',str(self.source)],capture_output=True,text=True,cwd=ROOT)
            self.assertNotEqual(v.returncode,0);self.assertFalse((self.source/'phase_errors').exists())
        self.assertEqual(before,{str(p):sha(p) for p in self.source.rglob('*') if p.is_file()})

    def test_end_to_end_and_controls_keep_all_inventory_and_observations(self):
        r=self.bundle();obs=read(r/'context/qwen/observer3d/G_obs.json')
        for branch in experiment.EXPERIMENTS:
            for b in ('qwen','api'):
                result=experiment.world(r,branch,b,CPUProvider());self.assertEqual(len(result['worlds'])+len(result['blocked_worlds']),2,result)
                for row in result['worlds']+result['blocked_worlds']:
                    f=r/branch/b/'completion'/row['folder'];state=read(f/row.get('state_file','G_full.json'));sem=read(f/'semantic_world.json');dims=read(f/'dimension_stage.json')['dimensions'] if sem['objects'] else {}
                    self.assertEqual(state['objects'][:len(obs['objects'])],obs['objects']);self.assertEqual({o['track_id'] for o in state['objects'][len(obs['objects']):]},{o['id'] for o in sem['objects']})
                    for o in state['objects'][len(obs['objects']):]:self.assertEqual(o['intrinsic_size_m']['value'],dims[o['track_id']]['full_size_m'])
        from .assembly_organized_visual import report
        run=read(r/'run.json');run['synthetic_fixture']=True;write(r/'run.json',run);page=report(r);self.assertTrue(page.exists());self.assertIn('CPU 合成',page.read_text())
        self.assertTrue((r/'mechanism_diagnostics.json').exists())


if __name__=='__main__':unittest.main()
