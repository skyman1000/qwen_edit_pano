"""Data provenance/visibility regression tests, not tests of model quality."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from ..common import FRAME, field, sha, write, write_rows
from .layout_examples import PROTOCOL, load, messages, room_example


def source_room():
    def obj(oid, cat, center, visible, primary=True):
        return dict(track_id=oid, category_id=field(cat), center_local_m=field(center),
            size_aabb_local_m=field([1., 1., 1.]), in_primary_room=field(primary),
            bbox2d_visible_xyxy_norm=field([.2,.2,.8,.8] if visible else None),
            local_evidence='visible' if visible else 'unobserved', visibility=dict(erp_pixels=100))
    return dict(split='train', frame=FRAME, vocabulary_sha256='test', room=dict(type=field('b')),
        objects=[obj('bed',11,[0,-.7,3],True), obj('lamp',28,[1.,0.,3.],True),
                 obj('chair',3,[-2,-.8,-2],False), obj('cabinet',7,[2,-.3,-2],False),
                 obj('neighbour',5,[3,0,-3],False,False), obj('in_view',14,[0,0,4],False)])


class LayoutExamplesTests(unittest.TestCase):
    def fixture(self, directory):
        graph_path=directory/'G_full.json'
        write(graph_path, source_room())
        row=dict(id='training_house_view', split='train', graph=str(graph_path), graph_sha256=sha(graph_path),
                 paired=dict(scene_id='training_house',local_condition=dict(sha256='training_photo')))
        manifest=directory/'samples.jsonl'
        write_rows(manifest,[row])
        example=room_example(row,source_room(),90.)
        bank=dict(protocol=PROTOCOL,frame=FRAME,vocabulary_sha256='test',
                  manifest=str(manifest),manifest_sha256=sha(manifest),examples=[example])
        path=directory/'examples.json';write(path,bank)
        return path,bank,row

    def test_examples_are_primary_room_and_outside_camera_without_moving_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,bank,_=self.fixture(Path(tmp))
            loaded=load(path,'test',90.,'different_photo')
            example=loaded['examples'][0]
            self.assertEqual(example['source_object_ids']['additions'],['chair','cabinet'])
            self.assertEqual(example['output']['additions'][0]['center'],[-2,-.8,-2])
            pairs=messages(loaded)
            self.assertEqual([m['role']for m in pairs],['user','assistant'])
            self.assertEqual(loaded,bank)

    def test_target_photo_and_tampered_coordinates_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,bank,_=self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError,'own GT demonstration'):
                load(path,'test',90.,'training_photo')
            bank['examples'][0]['output']['additions'][0]['center'][0]+=1
            write(path,bank)
            with self.assertRaisesRegex(ValueError,'differs from its source'):
                load(path,'test',90.,'different_photo')

    def test_validation_sources_and_wrong_fov_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,bank,row=self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError,'FOV differs'):
                load(path,'test',60.,'different_photo')
            row['split']='val'
            write_rows(bank['manifest'],[row]);bank['manifest_sha256']=sha(bank['manifest']);write(path,bank)
            with self.assertRaisesRegex(ValueError,'independent training buildings'):
                load(path,'test',90.,'different_photo')

    def test_source_geometry_change_and_frame_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,bank,row=self.fixture(Path(tmp))
            graph=source_room();graph['objects'][0]['center_local_m']['value'][2]=4
            write(row['graph'],graph)
            with self.assertRaisesRegex(ValueError,'provenance differs'):
                load(path,'test',90.,'different_photo')
            bank['frame']='world_frame';write(path,bank)
            with self.assertRaisesRegex(ValueError,'protocol/frame/vocabulary'):
                load(path,'test',90.,'different_photo')


if __name__=='__main__':
    unittest.main()
