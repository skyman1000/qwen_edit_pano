"""Behavioral regressions: coherent plans, immutable observations, visible failures."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from ..common import read, write, field, SCHEMA, FRAME
from . import panorama_plan as workflow, panorama_plan_model as model


def fixture():
    case=dict(hfov=90,image_size=[128,128],context=[dict(id='obs0',category='chair')])
    observed=dict(schema_version=SCHEMA,frame=FRAME,scope='observed',objects=[dict(
        track_id='obs0',category_id=field(3),center_local_m=field([0.,-.5,2.]),
        size_aabb_local_m=field([.5,1.,.5]),bbox2d_visible_xyxy_norm=field([.4,.4,.6,.8]),
        in_primary_room=field(None),local_evidence='visible',raw_category='chair')])
    h=dict(id='h0',room_type='living room',organization='Front seating and two separated storage areas',
           assumptions=['Rear wall and lateral circulation are hypothetical'],evidence_ids=['obs0'])
    plan=dict(room_envelope=dict(bounds_xz_m=[-4,4,-5,5],floor_y_m=-1.,height_m=3.,basis='hypothetical scale'),
        zones=[dict(id='z0',function='storage',region='front',bounds_xz_m=[-3,3,0,4],observed_ids=['obs0'],contents_intent='Observed chair area remains without extra furniture'),
               dict(id='z1',function='storage',region='back',bounds_xz_m=[-3,0,-4,-1],observed_ids=[],contents_intent='Cabinet with book')],
        circulation=[dict(**{'from':'z0','to':'z1'},reason='Central passage')],
        sector_review={d:'Part of the same hypothesized room, with circulation left open' for d in ('front','left','right','back')},visible_unmodeled=[])
    objects=[dict(id='cab',category='cabinet',description='storage cabinet',placement='rear left',reason='rear storage zone',
                  support=dict(kind='floor',host=None),zone_id='z1'),
             dict(id='book',category='book',description='single book',placement='on cabinet',reason='stored reading material',
                  support=dict(kind='top',host='cab'),zone_id='z1')]
    poses=[dict(id='cab',center_m=[-1.5,0,-2.5],full_size_m=[1.,1.,.5],yaw_deg=30),
           dict(id='book',center_m=[-1.5,0,-2.5],full_size_m=[.2,.05,.15],yaw_deg=0)]
    return case,observed,h,plan,objects,poses


class PlanningTests(unittest.TestCase):
    def setUp(self):self.case,self.obs,self.h,self.plan,self.objects,self.poses=fixture()

    def test_coexisting_same_function_zones_and_observed_assignment(self):
        result=model.blueprint_answer(self.plan,self.case)
        self.assertEqual([z['function'] for z in result['zones']],['storage','storage'])
        for refs in ([],['obs0','obs0'],['nonexistent']):
            changed=deepcopy(self.plan);changed['zones'][0]['observed_ids']=refs
            with self.assertRaises(ValueError):model.blueprint_answer(changed,self.case)

    def test_empty_is_allowed_but_must_explain(self):
        self.assertEqual(model.inventory_answer(dict(objects=[],empty_reason='circulation'),self.case,self.plan['zones'][0],[]),[])
        with self.assertRaises(ValueError):model.inventory_answer(dict(objects=[],empty_reason=''),self.case,self.plan['zones'][0],[])

    def test_no_observed_redeclaration_or_host_cycles(self):
        changed=deepcopy(self.objects);changed[0]['id']='obs0'
        with self.assertRaises(ValueError):model.inventory_answer(dict(objects=changed),self.case,self.plan['zones'][1],[])
        changed=deepcopy(self.objects);changed[0]['support']=dict(kind='top',host='book')
        with self.assertRaises(ValueError):model.inventory_answer(dict(objects=changed),self.case,self.plan['zones'][1],[])

    def test_cross_zone_dependency_is_retained(self):
        earlier=deepcopy(self.objects[:1]);earlier[0]['zone_id']='z0'
        result=model.inventory_answer(dict(objects=self.objects[1:]),self.case,self.plan['zones'][1],earlier)
        self.assertEqual(result[0]['support']['host'],'cab')

    def test_host_intervention_changes_installation_not_ownership(self):
        baseline=model.installation(self.objects,['obs0'])
        changed=model.installation(self.objects,['obs0'],['cab'])
        self.assertTrue(baseline['installed']['book'])
        self.assertFalse(changed['installed']['book'])
        self.assertEqual(changed['inventory_ids'],baseline['inventory_ids'])
        self.assertTrue(changed['installed']['obs0'])

    def test_numeric_layout_preserves_every_instance_and_uses_yaw_aabb(self):
        before=deepcopy(self.obs)
        full,review=workflow.compile_boxes(self.case,self.obs,self.plan,self.objects,self.poses)
        self.assertEqual(self.obs,before)
        self.assertEqual(full['objects'][0],self.obs['objects'][0])
        self.assertEqual({o['track_id'] for o in full['objects']},{'obs0','cab','book'})
        self.assertEqual(review['observation_conflict_ids'],[])
        self.assertAlmostEqual(full['objects'][2]['center_local_m']['value'][1],.025)
        self.assertGreater(full['objects'][1]['size_aabb_local_m']['value'][0],1.)
        self.assertFalse(full['world_state_ready'])

    def test_layout_conflicts_are_diagnosed_without_moving_or_deleting(self):
        poses=deepcopy(self.poses)
        for p in poses:p['center_m']=[0,0,2]
        full,review=workflow.compile_boxes(self.case,self.obs,self.plan,self.objects,poses)
        self.assertTrue(review['observation_conflict_ids'])
        self.assertEqual(len(full['objects']),3)
        self.assertEqual(full['objects'][1]['center_local_m']['value'][2],2)
        self.assertFalse(full['observation_compatible_box_proxy'])

    def test_numeric_batch_rejects_missing_id_and_nonfinite(self):
        with self.assertRaises(ValueError):model.metric_answer(dict(objects=self.poses[:1]),self.objects)
        bad=deepcopy(self.poses);bad[0]['center_m'][0]=float('nan')
        with self.assertRaises(ValueError):model.metric_answer(dict(objects=bad),self.objects)

    def test_request_cache_no_model_on_resume_and_failed_not_redrawn(self):
        calls=[]
        class Client:
            def ask(inner,image,prompt,path,**kwargs):
                calls.append(kwargs['seed']);path.write_text('{}');return dict(ok=1)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            requests=workflow.Requests(lambda:Client(),folder,22,None)
            self.assertEqual(requests.ask('stage','question',lambda v:v['ok']),1)
            cached=workflow.Requests(lambda:self.fail('Model loaded on cached request'),folder,22,None)
            self.assertEqual(cached.ask('stage','question',lambda v:v['ok']),1)
            with self.assertRaises(ValueError):cached.ask('stage','different question',lambda v:v)
            def invalid(v):raise ValueError('bad contract')
            with self.assertRaises(ValueError):requests.ask('broken','question',invalid)
            self.assertEqual(len(calls),3)
            with self.assertRaises(ValueError):cached.ask('broken','question',invalid)
            (folder/'stage_attempt0.txt').write_text('changed')
            with self.assertRaises(ValueError):cached.ask('stage','question',lambda v:v)

    def test_end_to_end_planning_then_layout_and_no_model_resume(self):
        case,obs,h,plan,objects,poses=self.case,self.obs,self.h,self.plan,self.objects,self.poses
        calls=[]
        class Client:
            def ask(inner,image,prompt,path,**kwargs):
                calls.append(path.name)
                if path.name.startswith('hypotheses'):value=dict(hypotheses=[h])
                elif path.name.startswith('blueprint'):value=plan
                elif path.name.startswith('inventory_z0'):value=dict(objects=[],empty_reason='Preserved observed area')
                elif path.name.startswith('inventory_z1'):value=dict(objects=objects,empty_reason='')
                elif path.name.startswith('semantic_review'):value=dict(plausibility='plausible',reason='Coherent hypothetical room',zones=[dict(id=z['id'],assessment='consistent') for z in plan['zones']],issues=[])
                elif path.name.startswith('metric_'):value=dict(objects=poses)
                else:raise AssertionError(path.name)
                path.write_text(json.dumps(value));return value
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'scenes'/'scene0';ob=base/'observation';ob.mkdir(parents=True)
            Image.new('RGB',(128,128),'white').save(ob/'input.png');write(ob/'G_obs.json',obs)
            c=dict(case,image=str(ob/'input.png'),state=str(ob/'G_obs.json'))
            suite=dict(scenes=[dict(id='scene0',hfov=90)],hypotheses=1,seed=77)
            with patch.object(workflow,'check',return_value=suite),patch.object(workflow,'load_case',return_value=c):
                planned=workflow.run(root,layout=False,client_factory=lambda *args:Client())
                folder=base/'world_01'
                self.assertEqual(planned['execution_failures'],0)
                self.assertTrue((folder/'blueprint.svg').exists())
                self.assertFalse((folder/'G_full.json').exists())
                self.assertEqual(len(calls),5)
                result=workflow.run(root,layout=True,client_factory=lambda *args:Client())
                self.assertEqual(result['records'][0]['new_objects'],2)
                self.assertTrue((folder/'views/scene_3d.html').exists())
                self.assertEqual(read(folder/'G_full.json')['objects'][0],obs['objects'][0])
                self.assertEqual(len(calls),6)
                workflow.run(root,layout=True,client_factory=lambda *args:self.fail('Cached resume loaded model'))
                self.assertEqual(len(calls),6)

    def test_metric_failure_keeps_plan_inventory_and_visualization(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            write(folder/'status.json',dict(status='failed',stage='layout',error='fixture failure'))
            workflow.visual.page(folder,self.h,self.plan,self.obs,90,self.objects)
            html=(folder/'index.html').read_text()
            self.assertIn('blueprint.svg',html)
            self.assertIn('fixture failure',html)
            self.assertIn('book',html)


if __name__=='__main__':unittest.main()
