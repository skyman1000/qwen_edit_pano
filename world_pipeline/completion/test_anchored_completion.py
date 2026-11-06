"""Contract/geometry tests only; no claim about a real model's scene quality."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ..common import read, sha, write, write_rows
from ..oracle_model import tensorize
from .anchored_completion import CAMERA_ID, PROTOCOL, compile_layout, prompt
from .anchored_examples import PROTOCOL as BANK_PROTOCOL, example, messages, select
from .semantic_world import make_graph, run
from .test_layout_examples import source_room
from .test_semantic_world import fixture


def proposal():
    return dict(room_type='bedroom',scene_hypothesis='A plausible room continuation.',additions=[
        dict(id='new_cabinet',category='cabinet',size=[1.2,1.8,.6],anchor_id='observed_0000',
             offset_m=[-3.,.2,-4.],description='cabinet',reason='Storage beside the observed furniture.',
             unseen_reason='outside_fov')])


class AnchoredCompletionTests(unittest.TestCase):
    def test_existing_anchor_changes_all_three_coordinates_without_moving_evidence(self):
        obs,_,layout=fixture(); original=deepcopy(layout)
        result,report=compile_layout(proposal(),obs,layout,8)
        np.testing.assert_allclose(result['additions'][0]['center'],[-3.,-.5,-1.])
        self.assertEqual(layout,original)
        self.assertFalse(report['offscreen_translation'])
        self.assertFalse(report['floor_snapping'])

    def test_continuous_full_surrounding_space_and_no_forced_outside_translation(self):
        obs,_,layout=fixture()
        for angle in np.linspace(-np.pi,np.pi,25):
            raw=proposal(); obj=raw['additions'][0]
            centre=[3*float(np.sin(angle)),.35,3*float(np.cos(angle))]
            obj.update(anchor_id=CAMERA_ID,offset_m=centre)
            answer,_=compile_layout(raw,obs,layout,8)
            np.testing.assert_allclose(answer['additions'][0]['center'],centre,atol=0)

    def test_forward_references_are_allowed_and_output_order_is_preserved(self):
        obs,_,layout=fixture();raw=proposal()
        child=deepcopy(raw['additions'][0]);child.update(id='child',anchor_id='new_cabinet',offset_m=[1.,2.,3.])
        raw['additions'].insert(0,child)
        result,_=compile_layout(raw,obs,layout,8)
        self.assertEqual([o['id']for o in result['additions']],['child','new_cabinet'])
        np.testing.assert_allclose(result['additions'][0]['center'],[-2.,1.5,2.])

    def test_missing_self_cyclic_ids_and_two_position_definitions_are_rejected(self):
        obs,_,layout=fixture()
        for mode in ('missing','self','cycle','second_position','duplicate','invalid_offset','invalid_size','category'):
            with self.subTest(mode=mode):
                raw=proposal();obj=raw['additions'][0]
                if mode=='missing':obj['anchor_id']='not_committed_in_objects'
                elif mode=='self':obj['anchor_id']=obj['id']
                elif mode=='cycle':
                    child=deepcopy(obj);child.update(id='child',anchor_id=obj['id']);obj['anchor_id']='child'
                    raw['additions'].append(child)
                elif mode=='second_position':obj['center']=[0.,0.,0.]
                elif mode=='duplicate':raw['additions'].append(deepcopy(obj))
                elif mode=='invalid_offset':obj['offset_m'][0]=float('nan')
                elif mode=='invalid_size':obj['size'][0]=0.
                else:obj['category']='spaceship'
                with self.assertRaises(ValueError):compile_layout(raw,obs,layout,8)

    def test_overlap_is_reported_without_separating_or_dropping_boxes(self):
        obs,_,layout=fixture();raw=proposal();raw['additions'][0]['offset_m']=[0.,0.,0.]
        result,report=compile_layout(raw,obs,layout,8)
        self.assertTrue(report['overlap_report']['pairs'])
        self.assertEqual(result['additions'][0]['center'],layout['objects'][0]['center'])
        with self.assertRaisesRegex(ValueError,'budget'):compile_layout(raw,obs,layout,0)

    def test_raw_labels_use_the_official_coarse_vocabulary_not_native_model_aliases(self):
        obs,_,layout=fixture()
        for label,canonical in [('nightstand','chest_of_drawers'),('wardrobe','furniture'),('clock','objects')]:
            raw=proposal();raw['additions'][0]['category']=label
            result,_=compile_layout(raw,obs,layout,8)
            self.assertEqual(result['additions'][0]['category'],canonical)
            self.assertEqual(result['additions'][0]['raw_category'],label)

    def test_integrated_state_preserves_ids_provenance_and_exposes_failed_projection(self):
        obs,_,layout=fixture();state=make_graph(obs,layout,'test','wilddet3d_camera_aabb_center','observed')
        state['camera_geometry']=dict(levelled=False,gravity_verified=False,floor_verified=False)
        raw=proposal();raw['additions'][0].update(category='nightstand',anchor_id=CAMERA_ID,offset_m=[0.,0.,3.])
        calls=[]
        class Model:
            def ask(self,image,question,path,**kwargs):
                calls.append(path.stem)
                self_test.assertNotIn('photographed bed',question)
                return deepcopy(raw)
        self_test=self
        with tempfile.TemporaryDirectory() as tmp:
            summary=run(Model(),Image.new('RGB',(768,768)),90.,Path(tmp),'test',
                        reused=obs,fixed_observed_state=state,completion_mode='anchored')
            graph=read(Path(tmp)/'G_full.json');new=graph['objects'][-1]
            self.assertEqual(calls,['anchored_scene'])
            self.assertEqual(graph['objects'][:len(obs)],state['objects'])
            self.assertEqual(graph['camera_geometry'],state['camera_geometry'])
            self.assertEqual(new['center_local_m']['value'],[0.,0.,3.])
            self.assertEqual(new['category_id']['value'],13)
            self.assertEqual(new['raw_category'],'nightstand')
            self.assertEqual(new['layout_parameters']['anchor_id'],CAMERA_ID)
            self.assertGreater(summary['outside_fov_failed'],0)
            self.assertEqual(summary['quality_status'],'needs_completion_revision')
            self.assertEqual(int(tensorize(graph,160)[2].sum()),len(obs)+2)

    def test_camera_y_is_not_floor_snapped_when_legacy_height_prior_changes(self):
        obs,_,layout=fixture();state=make_graph(obs,layout,'test','fixed','observed')
        class Model:
            def ask(self,*args,**kwargs):return proposal()
        centres=[]
        for height in (.7,1.8):
            with tempfile.TemporaryDirectory() as tmp:
                run(Model(),Image.new('RGB',(768,768)),90.,Path(tmp),'test',reused=obs,
                    fixed_observed_state=state,completion_mode='anchored',camera_height=height)
                centres.append(read(Path(tmp)/'G_full.json')['objects'][-1]['center_local_m']['value'])
        self.assertEqual(centres[0],centres[1])


class AnchoredExamplesTests(unittest.TestCase):
    def bank(self,directory):
        source=source_room();source['vocabulary_sha256']='test'
        gp=directory/'G_full.json';write(gp,source)
        row=dict(id='train_view',split='train',graph=str(gp),graph_sha256=sha(gp),
                 paired=dict(scene_id='train_house',local_condition=dict(sha256='train_photo')))
        manifest=directory/'samples.jsonl';write_rows(manifest,[row])
        item=example(row,source,90.)
        bank=dict(protocol=BANK_PROTOCOL,output_protocol=PROTOCOL,frame=source['frame'],
                  vocabulary_sha256='test',manifest=str(manifest),manifest_sha256=sha(manifest),
                  hfov_deg=90.,examples=[item],selection='visible furniture overlap')
        path=directory/'pool.json';write(path,bank)
        return path,bank,row

    def test_examples_reconstruct_original_gt_centres_and_select_only_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,bank,row=self.bank(Path(tmp));obs,_,_=fixture()
            selected=select(path,'test',90.,'another_photo',obs)
            item=selected['examples'][0]
            result,_=compile_layout(item['output'],item['input']['fixed_observed'],
                                   dict(objects=item['input']['fixed_observed']),8)
            gt={o['track_id']:o for o in read(row['graph'])['objects']}
            for obj,source_id in zip(result['additions'],item['source_object_ids']['additions']):
                np.testing.assert_allclose(obj['center'],gt[source_id]['center_local_m']['value'],atol=1e-12)
            self.assertEqual([m['role']for m in messages(selected)],['user','assistant'])
            self.assertEqual(select(path,'test',90.,'train_photo',obs)['examples'],[])

    def test_unrelated_furniture_gets_no_forced_bedroom_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_,_=self.bank(Path(tmp))
            bank=select(path,'test',90.,'other',[dict(category='bathtub')])
            self.assertTrue(bank['no_match'])

    def test_tampered_coordinates_and_validation_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,bank,row=self.bank(Path(tmp));obs,_,_=fixture()
            bank['examples'][0]['output']['additions'][0]['offset_m'][0]+=.3;write(path,bank)
            with self.assertRaisesRegex(ValueError,'differs from source'):select(path,'test',90.,'other',obs)
            path,bank,row=self.bank(Path(tmp));row['split']='val'
            write_rows(bank['manifest'],[row]);bank['manifest_sha256']=sha(bank['manifest']);write(path,bank)
            with self.assertRaisesRegex(ValueError,'training source'):select(path,'test',90.,'other',obs)

    def test_selected_demonstrations_reach_the_actual_model_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);path,_,_=self.bank(directory);obs,_,layout=fixture()
            bank=select(path,'test',90.,'other',obs)
            state=make_graph(obs,layout,'test','fixed','observed');output=directory/'run';output.mkdir()
            captured=[]
            class Model:
                def ask(self,image,question,path,**kwargs):
                    captured.extend(kwargs['examples']);return proposal()
            result=run(Model(),Image.new('RGB',(768,768)),90.,output,'test',reused=obs,
                       fixed_observed_state=state,completion_mode='anchored',layout_examples=bank)
            self.assertEqual(result['layout_example_count'],1)
            self.assertEqual(captured,messages(bank))


if __name__=='__main__':
    unittest.main()
