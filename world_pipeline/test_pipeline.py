"""Meaningful CPU tests: frame reflections, OBB geometry, occlusion, data leakage, matching."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from .common import SCHEMA, FRAME, ROOT, field, sha, read, write, validate_graph
from .geometry import alignment_basis, box_geometry, MeshScene, erp_rays, local_rays, sample_erp, visible_box
from .observer import parse_prediction, input_rows, prompt
from .evaluate_observer import match
from . import prepare_gt, evaluate_observer


class GeometryTests(unittest.TestCase):
    def test_alignment_composition_matches_image_roll_and_mirror(self):
        # Compare every pixel, including seam; reflection/roll order is observable.
        image = np.arange(8*16).reshape(8,16)
        for mirror in (False,True):
            for shift in (0,3,12):
                a=dict(mirror=mirror,roll_fraction=shift/16)
                actual=sample_erp(image,erp_rays(8,16)@alignment_basis(a).T,nearest=True)
                expected=np.roll(image[:,::-1] if mirror else image,shift,axis=1)
                np.testing.assert_array_equal(actual,expected)

    def test_obb_transform_not_old_axis_size(self):
        a=np.pi/4
        basis=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        obj=dict(center_world=[4,2,8],axes_world=np.eye(3).tolist(),radii=[1,2,3])
        center,size=box_geometry(obj,np.array([1,1,1]),basis)
        np.testing.assert_allclose(size,[4*np.sqrt(2),4,4*np.sqrt(2)])
        np.testing.assert_allclose(np.array(center)@basis.T+[1,1,1],obj['center_world'])
        reflected=basis@np.diag([-1,1,1])
        c2,s2=box_geometry(obj,np.array([1,1,1]),reflected)
        np.testing.assert_allclose(c2,np.array(center)*[-1,1,1])
        np.testing.assert_allclose(size,s2)

    def test_box_pixel_edges(self):
        m=np.zeros((10,20),dtype=bool);m[2:5,4:8]=True
        self.assertEqual(visible_box(m),[.2,.2,.4,.5])
        self.assertIsNone(visible_box(np.zeros_like(m)))


def make_mesh(root):
    region=root/'regions';linked=root/'links'
    region.mkdir();linked.mkdir()
    # Large front plane (instance1, z2) occludes smaller plane (instance2, z4).
    xyz=np.array([[-4,-4,2],[4,-4,2],[4,4,2],[-4,4,2],[-1,-1,4],[1,-1,4],[1,1,4],[-1,1,4]],dtype='float32')
    triangles=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]],dtype='uint32')
    np.savez(region/'r.npz',vertices_world=xyz,triangles=triangles)
    np.savez(linked/'i.npz',face_instance_id=np.array([1,1,2,2]),face_instance_valid=np.ones(4,dtype=bool))
    write(region/'region_index.json',dict(category_mapping_sha256='fixture',regions=[dict(region_id=0,array_file='r.npz')]))
    for path in (region/'region_validation.json',linked/'object_link_validation.json'):
        write(path,dict(status='PASS',scan_id='house',errors=[]))
    links=linked/'object_links.json'
    write(links,dict(scan_id='house',region_directory=str(region),category_mapping_sha256='fixture',
                     region_files=[dict(region_id=0,array_file='i.npz')]))
    return links


class MeshTests(unittest.TestCase):
    def test_real_cpu_raycast_occlusion_and_invalid_occluder(self):
        with tempfile.TemporaryDirectory() as tmp:
            links=make_mesh(Path(tmp));scene=MeshScene(links,1)
            hit=scene.cast(np.zeros(3),np.eye(3),local_rays(32))
            self.assertTrue(np.all(hit['instance']==1))
            self.assertFalse(np.any(hit['instance']==2))
            self.assertAlmostEqual(float(hit['depth'][16,16]),2*np.sqrt(1+2*(1/32)**2),places=5)
            # Unknown front surface still occludes labeled back surface.
            gid=next(iter(scene.labels));ids,valid=scene.labels[gid];valid[:2]=False
            hit=scene.cast(np.zeros(3),np.eye(3),local_rays(32))
            self.assertTrue(np.all(hit['instance']==-1))
            self.assertTrue(hit['hit_valid'].all())

    def test_export_fixture_and_no_room_leak_in_obs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); links=make_mesh(root);scene=MeshScene(links,1)
            old=root/'old';world=old/'world_state_pilot/house/view';geo=old/'geometry_pilot/house/view'
            world.mkdir(parents=True);geo.mkdir(parents=True)
            op=old/'object_link_pilot/house';op.mkdir(parents=True);write(op/'object_links.json',read(links))
            state=dict(sample_id='house_view',camera=dict(camera_to_world=np.eye(4).tolist(),position_world=[0,0,0]),
                       room=dict(supervision_eligible=True,region_id=0,type='b',extent=[-5,-5,-1,5,5,3]),
                       layout=dict(floor_z=-1,ceiling_z=None))
            write(world/'world_state.json',state)
            source=dict(id=1,house_object_id=0,category_id=3,center_world=[0,0,2],axes_world=np.eye(3).tolist(),
                        radii=[4,4,.1],region_id=0,room_supervision_eligible=True,
                        supervision_mask=dict(category=True,position=True,size=True,existence=True))
            write(world/'objects.json',dict(objects=[source]))
            write(geo/'geometry_report.json',dict(sample_id='house_view',checks=dict(ok=True),review_flags=[],
                  erp_sampling='pixel_centers',erp_to_world_basis_candidate=np.eye(3).tolist(),origin_world_candidate=[0,0,0]))
            image=np.stack([*np.meshgrid(np.arange(1024)%256,np.arange(512)%256)[::-1],np.full((512,1024),128)],axis=-1).astype('uint8')
            Image.fromarray(image).save(geo/'rgb_erp.png')
            old_hit=scene.cast(np.zeros(3),np.eye(3),erp_rays(512,1024))
            np.save(geo/'instance_erp.npy',old_hit['instance']);np.save(geo/'instance_valid.npy',old_hit['instance_valid'])
            local=Image.fromarray(sample_erp(image,local_rays(1024)))
            target=Image.fromarray(image).resize((2048,1024))
            row=dict(id='house_view',scene_id='house',source_view_id='view',split='train',
                     target_alignment=dict(mirror=False,roll_fraction=0))
            output=root/'out';output.mkdir()
            align=dict(mirror=False,roll_fraction=0,score=1,margin=1)
            args=argparse.Namespace(min_score=-1,min_margin=0)
            with patch.object(prepare_gt,'OLD',old),patch.object(prepare_gt,'local_image',return_value=local), \
                    patch.object(prepare_gt,'target_image',return_value=target), \
                    patch.object(prepare_gt,'estimate_alignment',return_value=align):
                manifest,observer=prepare_gt.export_one(row,output,scene,args,'fixture')
            directory=output/manifest['directory']
            obs=read(directory/'G_obs.json');full=read(directory/'G_full.json')
            self.assertEqual(len(obs['objects']),1)
            self.assertEqual(len(full['objects']),1)
            self.assertTrue(all(f['value'] is None for f in obs['room'].values()))
            self.assertIsNone(full['room']['ceiling_plane_local']['value'])
            self.assertFalse(manifest['ready_for_training'])
            self.assertNotIn('world_state',observer)
            self.assertTrue((directory/'review.jpg').is_file())


class ObserverTests(unittest.TestCase):
    def test_duplicate_json_keys_are_rejected(self):
        for text in (
            '{"room_type":null,"objects":[],"objects":[]}',
            '{"room_type":null,"objects":[{"category_id":4,"bbox2d":[1,2,3,4],"bbox2d":[2,3,4,5]}]}',
        ):
            with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
                parse_prediction(text,dict(id='s',split='train'),'v',bbox_scale=1000)

    def test_baseline_prompt_preserves_original_shape(self):
        text=prompt({'3':'chair'},1000,'baseline_v1')
        self.assertIn('normalized to [0,1000]',text)
        self.assertIn('"bbox2d":[100,200,300,400]',text)
        self.assertNotIn('compact JSON',text)

    def test_explicit_thousand_scale(self):
        text=json.dumps(dict(room_type=None,objects=[dict(category_id=4,bbox2d=[100,200,900,1000])]))
        row=dict(id='s',split='train')
        graph=parse_prediction(text,row,'v',bbox_scale=1000)
        self.assertEqual(graph['objects'][0]['bbox2d_visible_xyxy_norm']['value'],[.1,.2,.9,1.])
        with self.assertRaises(ValueError):
            parse_prediction(text,row,'v')
        for box in ([0,0,1001,1000],[900,0,100,1000],[0,0,float('nan'),1000]):
            bad=json.dumps(dict(room_type=None,objects=[dict(category_id=4,bbox2d=box)]))
            with self.assertRaises(ValueError):
                parse_prediction(bad,row,'v',bbox_scale=1000)

    def parse(self, objects):
        return parse_prediction(json.dumps(dict(room_type='bedroom',objects=objects)),dict(id='s',split='train'),'v')

    def test_unknown_geometry_and_parser_failure(self):
        graph=self.parse([dict(category_id=3,bbox2d=[0,0,.5,.5])])
        self.assertIsNone(graph['objects'][0]['center_local_m']['value'])
        self.assertIsNone(graph['room']['type']['value'])
        with self.assertRaises(ValueError):self.parse([dict(category_id=3,bbox2d=[0,0,100,100])])
        with self.assertRaises(ValueError):self.parse([dict(category_id=True,bbox2d=[0,0,.5,.5])])
        with self.assertRaises(ValueError):self.parse([dict(category_id=3,bbox2d=[0,0,float('nan'),1])])
        graph['objects'][0]['center_local_m']=field([0,0,0]);validate_graph(graph)
        graph['objects'][0]['center_local_m']['valid']=False
        with self.assertRaises(ValueError):validate_graph(graph)

    def test_leaky_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'inputs.jsonl'
            path.write_text(json.dumps(dict(id='s',split='train',caption='hidden sofa'))+'\n')
            with self.assertRaises(ValueError):input_rows(path)

    def test_duplicate_and_wrong_category_predictions(self):
        gt=self.parse([dict(category_id=3,bbox2d=[0,0,.5,.5])])['objects']
        pred=self.parse([dict(category_id=3,bbox2d=[0,0,.5,.5]),dict(category_id=3,bbox2d=[0,0,.5,.5]),
                         dict(category_id=4,bbox2d=[0,0,.5,.5])])['objects']
        self.assertEqual(len(match(gt,pred)),1)
        self.assertEqual(match(gt,pred[2:]),[])
        self.assertEqual(match(gt,[]),[])

    def test_matching_maximizes_valid_count_before_iou(self):
        objects=self.parse([dict(category_id=3,bbox2d=[0,0,.5,.5])]*2)['objects']
        with patch.object(evaluate_observer,'iou',side_effect=[.99,.5,.5,0]):
            self.assertEqual(len(match(objects,objects)),2)

    def test_evaluation_review_gate_and_split_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);gt=root/'gt';pred=root/'pred';gt.mkdir();pred.mkdir()
            (gt/'observer_inputs.jsonl').write_text('fixture inputs')
            write(gt/'vocabulary.json',{'fixture':True})
            records=[]
            for sid,split in [('s1','train'),('s2','val')]:
                (gt/sid).mkdir();(pred/sid).mkdir()
                graph=self.parse([dict(category_id=3,bbox2d=[0,0,.5,.5])]);graph.update(id=sid,split=split)
                write(gt/sid/'G_obs.json',graph)
                records.append(dict(id=sid,split=split,directory=sid,review_flags=[],
                                    gt_sha256={'G_obs':sha(gt/sid/'G_obs.json')}))
                if sid=='s1':
                    prediction=self.parse([dict(category_id=3,bbox2d=[0,0,.5,.5])]*2)
                    prediction.update(id=sid,split=split)
                    write(pred/sid/'G_obs.json',prediction)
                write(pred/sid/'status.json',dict(id=sid,status='parsed' if sid=='s1' else 'invalid_prediction'))
            (gt/'gt_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            write(gt/'EXPORT_COMPLETE.json',dict(gt_manifest_sha256=sha(gt/'gt_manifest.jsonl')))
            write(pred/'COMPLETE.json',{})
            write(pred/'run_config.json',dict(inputs_sha256=sha(gt/'observer_inputs.jsonl'),
                  vocabulary_sha256=sha(gt/'vocabulary.json'),selected_ids=['s1','s2']))
            review=root/'review.json';write(review,{})
            out=root/'metrics'
            argv=['eval','--gt',str(gt),'--predictions',str(pred),'--review',str(review),'--output',str(out)]
            with patch('sys.argv',argv):
                with self.assertRaises(ValueError):evaluate_observer.main()
            self.assertFalse(out.exists())
            write(review,{sid:{'decision':'approved'} for sid in ['s1','s2']})
            def create(path):
                path.mkdir();return path
            with patch('sys.argv',argv),patch.object(evaluate_observer,'new_output',side_effect=create):
                evaluate_observer.main()
            result=read(out/'metrics.json')['by_split']
            self.assertEqual(result['train']['tp'],1)
            self.assertEqual(result['train']['fp'],1)
            self.assertEqual(result['val']['fn'],1)
            self.assertEqual(result['val']['invalid_predictions'],1)


if __name__=='__main__':
    unittest.main()
