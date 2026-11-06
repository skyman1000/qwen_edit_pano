"""CPU contracts and full synthetic pipeline. No model-quality claim or inference."""
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ..common import read,write
from . import fresh_model as model
from . import fresh_observer as obs
from .fresh_layout import relation_objective
from .fresh_pipeline import initialize,check,DEFAULT_MODEL
from .fresh_review import review,request_record
from .fresh_wilddet import export
from .fresh_world import run
from .world_generation import SYSTEM_PROMPT,MATCHED_OPTIONS
from .world_scm import enumerate_model,abduct,forward


def inventory(two=False):
    rows=[dict(name='bed',category='bed',bbox_1000=[100,100,900,900],on_floor=True)]
    if two:rows.append(dict(name='unknown fixture item',category='fixture_unknown_noun',bbox_1000=[200,200,300,300],on_floor=False))
    return dict(objects=rows,evidence_description='A bed in the photographed area.')


def program(z):
    def object_(key,name,host='floor'):
        return dict(key=key,name=name,category=name,variant=name,identity='new_unobserved',
            distinct_instance_reason='A separate instance in unseen surrounding space.',placement_hint='Behind the original camera.',support_target=host)
    if z=='z0':
        groups=[dict(name='bedside assembly',reason='Plausible unseen furnishings.',objects=[object_('table','side table'),object_('lamp','table lamp','table')]),
            dict(name='separate seating',reason='Optional existing seating.',objects=[object_('chair','chair')])]
        relations=[dict(source='lamp',target='table',kind='requires',reason='This table lamp uses this host.'),
            dict(source='chair',target='table',kind='left_of',reason='One joint camera-coordinate ordering.'),
            dict(source='chair',target='table',kind='near',reason='Same surrounding zone, soft preference.')]
    else:groups=[dict(name='storage',reason='Plausible existing storage.',objects=[object_('wardrobe','wardrobe')])];relations=[]
    return dict(groups=groups,relations=relations)


class FixtureProvider:
    execution_kind='cpu_fixture_not_prediction';generation_profile='matched_v1'
    def __init__(self,two=False):self.two=two
    def ask(self,image,prompt,path,seed=3407,max_tokens=8192,**kw):
        path=Path(path)
        from .world_generation import image_bytes
        import hashlib
        write(path.with_suffix('.request.json'),dict(prompt=prompt,system_prompt=SYSTEM_PROMPT,seed=seed,max_tokens=max_tokens,
            generation=MATCHED_OPTIONS,generation_profile='matched_v1',
            canonical_image_sha256=hashlib.sha256(image_bytes(image)).hexdigest() if image is not None else None))
        if path.stem=='answer':value=inventory(self.two)
        elif path.stem=='configurations':
            value=dict(contexts=[dict(description='Residential sleeping room.')],configurations=[
                dict(description='Sleeping and sitting organization.',visible_expectations='Bed.'),
                dict(description='Sleeping and storage organization.',visible_expectations='Bed beside storage.')])
        elif path.stem.startswith('program_'):value=program(path.stem.removeprefix('program_'))
        elif path.stem=='mechanism_Z':value=dict(rows=[dict(given={'C':'c0'},p_by_state={'z0':.6,'z1':.4})])
        elif path.stem=='mechanism_E':value=dict(rows=[dict(given={'Z':'z0'},p_by_state={'other':.2,'signature':.8}),
            dict(given={'Z':'z1'},p_by_state={'other':.6,'signature':.4})])
        elif path.stem.startswith('layout_'):
            records=json.loads(prompt.split('FIXED INSTANCE CONTRACTS: ')[1].split('\nSELECTED GENERATIVE STATE:')[0]);objects=[]
            for r in records:
                key=r['local_key'];size={'table':(1,.6,.7),'lamp':(.2,.4,.2),'chair':(.6,.8,.6),'wardrobe':(1,2,.5)}[key]
                anchor=r['allowed_anchor_values'][0] if r['support']['kind']=='object' else 'camera'
                objects.append(dict(id=r['id'],width_m=size[0],height_m=size[1],depth_m=size[2],yaw_deg=0,
                    anchor=anchor,offset_m=[0,0,0] if key=='lamp' else [-3,0,-1] if key=='chair' else [3,0,-1]))
            value=dict(objects=objects)
        else:raise AssertionError('Unexpected model call: '+str(path))
        path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return deepcopy(value)


class FreshTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.image=self.root/'rgb.png';Image.new('RGB',(80,60),'white').save(self.image)
    def init(self):
        out=self.root/'experiment'
        with patch('qwen_edit_pano.world_pipeline.completion.fresh_pipeline.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.fresh_pipeline.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            return initialize(out,self.image,90,DEFAULT_MODEL,'cpu_fixture_not_API')
    def observe(self,root,backend,two=False):
        obs.observe(FixtureProvider(two),root,backend)
        source=read(root/backend/'observer2d/G_2d.json');records=[]
        for i,o in enumerate(source['objects']):
            center=np.array([0,0,3] if i==0 else [.3,-1,3]);size=np.array([2,1,2] if i==0 else [.2,.2,.2])
            corners=(center+np.array(list(itertools.product((-1,1),repeat=3)))*size/2).tolist()
            records.append(dict(id=o['track_id'],box3d_native=center.tolist()+[1]*7,corners_native=corners))
        out=root/backend/'observer3d';out.mkdir();write(out/'G_obs.json',export(source,records,backend))
    def test_full_two_independent_branches_and_visuals(self):
        root=self.init()
        for backend,two in [('api',True),('qwen',False)]:
            self.observe(root,backend,two);summary=run(FixtureProvider(),root,backend)
            self.assertEqual(summary['status'],'worlds_exported_needs_review',summary)
            self.assertEqual([w['new_instances'] for w in summary['worlds']],[3,1])
            self.assertAlmostEqual(summary['exported_semantic_mass'],1.)
            base=root/backend/'completion';observed=read(root/backend/'observer3d/G_obs.json')
            for w in summary['worlds']:
                full=read(base/w['folder']/'G_full.json')
                self.assertEqual(full['objects'][:len(observed['objects'])],observed['objects'])
                review_=read(base/w['folder']/'numeric_review.json');self.assertTrue(review_['geometry_contract_passed'])
                with Image.open(base/w['folder']/'world/erp_state_map.png') as im:self.assertEqual(im.width,2*im.height)
            inference=read(base/'inference.json')
            self.assertAlmostEqual(inference['posterior']['marginals']['H']['z0'],.75)
            self.assertAlmostEqual(inference['hidden_posterior_update_TV'],.15)
        page=review(root);audit=read(page.parent/'review.json')
        self.assertTrue(audit['same_2d_prompt']);self.assertTrue(audit['same_2d_canonical_image'])
        self.assertTrue(audit['same_public_generation_parameters'])
        self.assertNotEqual(audit['branches']['api']['observed_2d_count'],audit['branches']['qwen']['observed_2d_count'])
        self.assertEqual(check(root)['protocol'],model.PROTOCOL)
    def test_invalid_2d_boxes_do_not_clip_or_pass(self):
        for box in ([0,0,1001,1000],[500,0,100,1000],[0,0,float('nan'),100]):
            value=inventory();value['objects'][0]['bbox_1000']=box
            with self.assertRaises(ValueError):obs.answer(value)
    def test_unmapped_visible_category_is_preserved(self):
        root=self.init();self.observe(root,'api',True);state=read(root/'api/observer3d/G_obs.json')
        self.assertEqual(len(state['objects']),2);self.assertIsNone(state['objects'][1]['category_id']['value'])
        self.assertEqual(obs.load_case(root,'api')['context'][1]['category'],'fixture_unknown_noun')
    def test_support_references_and_cycles_rejected(self):
        root=self.init();self.observe(root,'qwen');case=obs.load_case(root,'qwen');config=dict(id='z0',description='fixture')
        broken=program('z0');broken['groups'][0]['objects'][0]['support_target']='lamp'
        with self.assertRaises(ValueError):model.program_answer(broken,case,config,dict(groups=4,max_new=24))
        broken=program('z0');broken['relations'][0]['target']='unlisted_host'
        with self.assertRaises(ValueError):model.program_answer(broken,case,config,dict(groups=4,max_new=24))
    def test_no_independent_absence_and_model_relative_intervention(self):
        root=self.init();self.observe(root,'qwen');run(FixtureProvider(),root,'qwen')
        m=read(root/'qwen/completion/causal_model.json')
        self.assertEqual([n['id'] for n in m['nodes']],['C','Z','E','H'])
        self.assertEqual(enumerate_model(m,do={'Z':'z1'})['marginals']['H'],{'z0':0.,'z1':1.})
        noise=abduct(m,{'E':'signature'},17);self.assertEqual(forward(m,noise['noise']),noise['factual_state'])
        program_=read(root/'qwen/completion/joint_programs.json')['z0']
        self.assertEqual(model.geometry_state(program_),{'g0':'1','g1':'1'})
    def test_relation_energy_gradient_and_cross_group(self):
        relations=[dict(source='a',target='b',kind=k) for k in ('near','right_of')]
        evaluate=relation_objective(['a','b'],{},relations)
        x=np.array([-2.,1.,1.,0.]);v,g=evaluate(x);eps=1e-6
        numeric=np.array([(evaluate(x+np.eye(4)[i]*eps)[0]-evaluate(x-np.eye(4)[i]*eps)[0])/(2*eps) for i in range(4)])
        np.testing.assert_allclose(g,numeric,atol=1e-6)
        self.assertGreater(v,evaluate(x-.01*g)[0])
    def test_empty_joint_world_is_kept_without_fake_groups(self):
        root=self.init();self.observe(root,'qwen');case=obs.load_case(root,'qwen')
        p=model.program_answer(dict(groups=[],relations=[]),case,dict(id='z0',description='Empty hypothesis.'),dict(groups=4,max_new=24))
        self.assertEqual(p['new_instances'],0);self.assertEqual(model.geometry_state(p),{})
    def test_geometry_failure_keeps_semantic_mass_and_inventory(self):
        root=self.init();self.observe(root,'qwen')
        from .fresh_world import solve as actual_solve
        def fail_one(case,observed,plane,program_,relations,search):
            if len(program_['groups'])==2:raise ValueError('Synthetic layout failure')
            return actual_solve(case,observed,plane,program_,relations,search)
        with patch('qwen_edit_pano.world_pipeline.completion.fresh_world.solve',side_effect=fail_one):
            summary=run(FixtureProvider(),root,'qwen')
        self.assertEqual(summary['status'],'partial_layouts_needs_review')
        self.assertAlmostEqual(summary['failed_layout_semantic_mass'],.75)
        self.assertAlmostEqual(summary['exported_semantic_mass'],.25)
        self.assertEqual(summary['failures'][0]['new_instances'],3)
    def test_real_API_client_payload_without_network(self):
        from .world_api import SiliconFlowJSON
        import io
        response=dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(inventory())))])
        def opener(request,timeout):
            payload=json.loads(request.data)
            self.assertEqual(payload['response_format'],{'type':'text'})
            self.assertEqual(payload['top_k'],50);self.assertFalse(payload['enable_thinking'])
            self.assertNotIn('seed',payload)
            self.assertEqual(payload['messages'][1]['content'][0]['type'],'image_url')
            return io.BytesIO(json.dumps(response).encode())
        with patch.dict(os.environ,{'SILICONFLOW_API_KEY':'synthetic_test_value'}):
            q=SiliconFlowJSON(opener=opener,stream=False,generation_profile='matched_v1',thinking='off')
            path=self.root/'answer.txt'
            with Image.open(self.image) as image:q.ask(image,obs.PROMPT,path,max_tokens=8192)
        r=request_record(path.with_suffix('.request.json'))
        self.assertEqual(r['prompt'],obs.PROMPT);self.assertTrue(r['penalty_neutral'])
        self.assertEqual(r['generation_profile'],'matched_v1')
    def test_review_accepts_source_updates_but_generation_does_not(self):
        root=self.init();config=read(root/'experiment.json');config['code_sha256']['/synthetic/missing_source.py']='fixture'
        write(root/'experiment.json',config)
        with self.assertRaises(ValueError):check(root)
        page=review(root)
        self.assertIn('/synthetic/missing_source.py',read(page.parent/'review.json')['source_changes_since_init'])
    def test_missing_initialization_is_explained(self):
        with self.assertRaisesRegex(ValueError,'Experiment not initialized'):
            check(self.root/'never_initialized')


if __name__=='__main__':unittest.main()
