"""Real rich-catalog inputs, CPU mechanism fixtures, no GPU/API calls."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..common import ROOT,read,write,sha
from . import assembly_compositional as model,assembly_compositional_experiment as experiment
from .assembly_world import run
from .world_scm import forward

CATALOG=ROOT/'qwen_edit_pano/outputs/modular_completion_20261025_162311_4103717'
SOURCE=ROOT/'qwen_edit_pano/outputs/holistic_completion_20261026_235517_1710340/end_to_end'


class CPUProgramProvider:
    def __init__(self):self.requests=[]
    def ask(self,image,prompt,path,seed=3407,max_tokens=8192,**kw):
        path=Path(path)
        if image is not None and not path.stem.startswith(('hypotheses','furniture_')):raise AssertionError('Matched downstream mode is text-only')
        self.requests.append(dict(path=str(path),seed=seed,prompt=prompt))
        if path.stem=='hypotheses':value=dict(hypotheses=[dict(description='Possible unseen work program',functions=['work'],assumptions=['Uncertain functional organization'])])
        elif path.stem.startswith('furniture_'):
            d=read(CATALOG/'shared/domain.json');p=d['programs'][d['slots']['work'][1]]
            value=dict(alternatives=[dict(description='Complete possible work program',groups=[dict(function=g['function'],reason=g['reason'],observed_refs=g.get('observed_refs',[]),
                new_objects=[dict(key=o['key'],category=o['category'],variant=o['variant'],reason=o['distinct_instance_reason'],placement=o['placement_hint'],attachment=o['attachment'],affordances=o['affordances']) for o in g['objects']]) for g in p['groups']],
                relations=[dict(v,kind={'in_front_of':'farther_than','behind':'nearer_than'}.get(v['kind'],v['kind'])) for v in p['relations']])])
        elif '\nALTERNATIVE_COMPLETE_PROGRAMS: ' in prompt:
            h=json.loads(prompt.split('\nH: ')[1].split('\n')[0]);role=prompt.split('\nROLE: ')[1].split('\n')[0]
            value=dict(present=role in h['functions'])
        elif '\nPROGRAMS: ' in prompt:
            choices=json.loads(prompt.split('\nALLOWED_CHOICES: ')[1]);chosen=next((v for v in choices if 'rear' in v['regions'].values()),choices[0]);value=dict(choice=chosen['choice'])
        elif path.stem.startswith('recognition_'):
            choices=json.loads(prompt.split('\nALLOWED_CHOICES: ')[1]);value=dict(choice=max(choices,key=lambda v:len(v['profile']['functions']))['choice'])
        elif path.stem.startswith('layout_'):
            template=json.loads(prompt.split('TEMPLATE: ')[1].split('\nFIXED SEMANTIC WORLD:')[0]);world=json.loads(prompt.split('\nFIXED SEMANTIC WORLD: ')[1].split('\n')[0]);objects={o['id']:o for o in world['objects']}
            value=deepcopy(template)
            for g in value['groups']:
                g['range_m']=3.
                roots=0;children={}
                for row in g['objects']:
                    o=objects[row['id']];noun=o['category'].lower().replace('_',' ');host=o['attachment']['host']
                    if 'table' in noun or noun=='desk':dims=[1.2,.75,.8]
                    elif 'bookcase' in noun or 'cabinet' in noun or noun=='shelf':dims=[1.0,1.8,.5]
                    elif 'media console' in noun:dims=[1.5,.5,.5]
                    elif 'chair' in noun:dims=[.6,.9,.6]
                    elif 'television' in noun:dims=[.8,.5,.05]
                    elif 'lamp' in noun:dims=[.15,.4,.15]
                    else:dims=[.16,.08,.12] if host else [.4,.5,.4]
                    row['full_size_m']=dims
                    row['yaw_deg']=0.
                    if o['attachment']['kind'] in ('seat','shelf'):row['surface_fraction']=.5
                    if o['attachment']['kind']=='hang':row['hanging_drop_m']=.1
                    if host:
                        i=children.get(host,0);children[host]=i+1;row['offset_m']=[-.3+.25*i,0,0]
                    else:row['offset_m']=[roots*2.,-.5,0];roots+=1
        else:raise AssertionError(path.stem)
        write(path.with_suffix('.request.json'),dict(prompt=prompt,seed=seed,max_tokens=max_tokens,image_delivered=image is not None))
        path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value


class CompositionalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.temp=Path(self.tmp.name)
    def prepare(self):
        root=self.temp/'run'
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG=str(CATALOG),CAUPANO_LAYOUT_RGB='0'),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_sdscene_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            experiment.prepare(root,SOURCE,worlds=2,draws=2)
        return root
    def bundle(self):
        r=self.prepare()
        for b in ('qwen','api'):experiment.propose(r,b)
        experiment.pool(r)
        for b in ('qwen','api'):experiment.mechanisms(r,b,CPUProgramProvider())
        experiment.pool_mechanisms(r);return r
    def fitted(self,r,branch='end_to_end'):
        d=read(r/'shared/domain.json');c=read(r/branch/'experiment.json');c['execution_backend']='qwen'
        return model.prepare_model(d['base_model'],read(r/'shared/case.json'),c,None)

    def test_rich_catalog_revalidated_without_old_CPT_or_layout(self):
        before=sha(CATALOG/'shared/domain.json');r=self.bundle();d=read(r/'shared/domain.json')
        self.assertEqual(len(d['programs'])-1,15)
        self.assertGreater(sum(len(g['objects']) for p in d['programs'].values() for g in p['groups']),60)
        self.assertEqual(sha(CATALOG/'shared/domain.json'),before)
        self.assertTrue(any(sum(len(g['objects']) for g in p['groups'])==8 for p in d['programs'].values()))

    def test_fresh_program_mode_has_a_complete_executable_path(self):
        root=self.temp/'fresh'
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG='',CAUPANO_LAYOUT_RGB='0'),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_sdscene_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            experiment.prepare(root,SOURCE,worlds=2,draws=2)
        for b in ('qwen','api'):self.assertEqual(experiment.propose(root,b,CPUProgramProvider())['status'],'complete')
        experiment.pool(root)
        for b in ('qwen','api'):experiment.mechanisms(root,b,CPUProgramProvider())
        experiment.pool_mechanisms(root)
        self.assertIsNone(read(root/'run.json')['catalog_source'])
        self.assertEqual(len(read(root/'shared/domain.json')['base_model']['component_registry']),8)
        self.assertTrue(experiment.ready(root,'layout_control','api')['ready'])

    def test_presence_is_independent_of_number_of_variant_options(self):
        r=self.bundle();m=self.fitted(r);d=read(r/'shared/domain.json');t=read(r/'shared/mechanisms/qwen.json')
        for h in m['profiles']:
            for role in m['slots']:
                row=t['presence'][h['id']+'|'+role];expected=(row['counts'].get('1',0)+.5)/(row['valid_draws']+1)
                self.assertAlmostEqual(m['activation_rows'][role][h['id']][1],expected)
        self.assertTrue(all(type(v) is float for v in m['activation_rows']['reading'][m['profiles'][0]['id']]))
        with self.assertRaises(ValueError):model.presence_answer({'present':'true'})

    def test_multiple_complete_programs_and_their_contents_can_coexist(self):
        r=self.bundle();m=self.fitted(r);u={n['id']:.3 for n in m['nodes']};do={}
        for role in m['slots']:
            do['A_'+role]='1';do['M_'+role]=next(n['states'][1] for n in m['nodes'] if n['id']=='M_'+role)
        w=model.active_world(m,forward(m,u,do));self.assertGreater(len(w['objects']),10)
        self.assertEqual(len({o['furniture_state'] for o in w['objects']}),len(m['slots']))
        self.assertTrue(any(o['attachment']['host'] in {v['id'] for v in w['objects']} for o in w['objects']))

    def test_exact_probability_law_and_actual_sampling_agree(self):
        r=self.bundle();m=self.fitted(r);law=model.marginals(m)
        self.assertAlmostEqual(sum(law['hidden_count_probabilities'].values()),1)
        self.assertAlmostEqual(sum(law['expected_category_counts'].values()),law['expected_hidden_instances'])
        worlds=model.sample_worlds(m,count=300)
        self.assertAlmostEqual(sum(len(w['objects']) for w in worlds)/len(worlds),law['expected_hidden_instances'],delta=1.)
        self.assertNotEqual(model.marginals(m,m['profiles'][0]['id'])['object_presence'],model.marginals(m,m['profiles'][-1]['id'])['object_presence'])

    def test_H_cut_preserves_joint_program_and_spatial_law(self):
        r=self.bundle();a=self.fitted(r);b=self.fitted(r,'function_free')
        for field in ('object_presence','hidden_count_probabilities','any_declared_region_presence'):
            for k,p in model.marginals(a)[field].items():self.assertAlmostEqual(p,model.marginals(b)[field][k])
        self.assertEqual(next(n for n in b['nodes'] if n['id']=='J')['parents'],[])
        self.assertEqual(model.marginals(b,b['profiles'][0]['id'])['object_presence'],model.marginals(b,b['profiles'][-1]['id'])['object_presence'])

    def test_spatial_variants_keep_the_same_owned_instance_IDs(self):
        r=self.bundle();m=self.fitted(r);u={n['id']:.3 for n in m['nodes']};fid=m['slots']['reading'][1]
        states=[cid for cid,v in m['spatial_choices'].items() if v['module']==fid];self.assertGreaterEqual(len(states),2)
        first=model.active_world(m,forward(m,u,{'A_reading':'1','M_reading':states[0]}));other=model.active_world(m,forward(m,u,{'A_reading':'1','M_reading':states[-1]}))
        self.assertEqual([o['id'] for o in first['objects']],[o['id'] for o in other['objects']]);self.assertNotEqual(first['spatial_program'],other['spatial_program'])

    def test_same_function_program_can_include_several_distinct_groups(self):
        r=self.bundle();d=read(r/'shared/domain.json');p=deepcopy(d['programs'][d['slots']['reading'][1]]);other=deepcopy(p['groups'][0]);other['key']='extra'
        mapping={o['key']:'extra_'+o['key'] for o in other['objects']}
        for o in other['objects']:
            o['key']=mapping[o['key']];o['attachment']['host']=mapping.get(o['attachment']['host'],o['attachment']['host'])
        p['groups'].append(other);p['relations']=[]
        clean=model.revalidate_program(p,read(r/'shared/case.json'))
        self.assertEqual(len(clean['groups']),2)
        choices,_=model.spatial_choices({'combined':clean},read(r/'shared/case.json'))
        self.assertTrue(any(set(v['regions'].values())=={'left','right'} for v in choices.values()))

    def test_host_removal_preserves_owned_children(self):
        r=self.bundle();m=self.fitted(r);u={n['id']:.3 for n in m['nodes']};do={'A_'+role:'1' for role in m['slots']}
        w=model.active_world(m,forward(m,u,do));child=next(o for o in w['objects'] if o['attachment']['host'] in m['component_registry']);host=m['component_registry'][child['attachment']['host']]
        after=model.active_world(m,forward(m,u,dict(do,**{host['presence_node']:'0'})))
        self.assertIn(child['id'],{o['id'] for o in after['objects']});self.assertIn(child['id'],after['unresolved_installations'])

    def test_backend_and_H_comparisons_have_paired_prompt_orders_and_seeds(self):
        r=self.bundle();a=read(r/'shared/mechanisms/qwen.json');b=read(r/'shared/mechanisms/api.json');self.assertEqual(a['original_parent_prompt_plan'],b['original_parent_prompt_plan'])
        plans=a['original_parent_prompt_plan'];same=[p for p in plans if p['stage'].startswith('variant_') and '_reading_0' in p['stage']]
        self.assertEqual(len({p['local_seed'] for p in same}),1)
        p=CPUProgramProvider()
        # Existing tables cannot be overwritten, but accepted requests are
        # revalidated from the bound cache without another provider call.
        with self.assertRaises(FileExistsError):experiment.mechanisms(r,'qwen',p)
        self.assertEqual(p.requests,[])
        self.assertTrue(list((r/'mechanism_calls/qwen').glob('*.accepted.json')))

    def test_execution_retains_complete_inventory_and_observation(self):
        r=self.bundle();observed=read(r/'context/qwen/observer3d/G_obs.json')
        for exp in experiment.EXPERIMENTS:
            for b in ('qwen','api'):
                result=run(CPUProgramProvider(),r/exp,b)
                self.assertEqual(len(result['worlds'])+len(result['blocked_worlds']),2,result)
                for record in result['worlds']+result['blocked_worlds']:
                    folder=r/exp/b/'completion'/record['folder'];state=read(folder/record.get('state_file','G_full.json'));sem=read(folder/'semantic_world.json')
                    self.assertEqual(state['objects'][:len(observed['objects'])],observed['objects']);self.assertEqual(len(state['objects'])-len(observed['objects']),len(sem['objects']))
                    for o in state['objects'][len(observed['objects']):]:self.assertIn('orientation_yaw_camera_deg',o)


if __name__=='__main__':unittest.main()
