"""CPU contract/geometry/integration tests; synthetic outputs do not test VLM quality."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from ..common import read, write, validate_graph
from .off_fov_completion import (PROTOCOL,code_hashes,contents_prompt,context_prompt,
                                 load_semantics,normalized_contents,run,validate_world)
from .off_fov_geometry import check,layout_answer,solve
from .scene_completion import map_plan,view_check
from .test_causal_world_completion import Model,contents as old_contents,factors,make_case


def contents():
    result=old_contents()
    for o in result['additions']:o['off_fov_region']='behind_camera'
    return result


def box(center=(0.,0.,-3.),size=(.8,1.5,.4)):
    return dict(id='new1',center=list(center),size=list(size),reason='One hypothesized position outside the input.')


def item(center=(0.,0.,-3.),size=(.8,1.5,.4),shift=.75,region=None):
    c=contents();plan=dict(additions=c['additions'])
    plan['additions'][0]['off_fov_region']=region or ('behind_camera' if center[2]<0 else
        'right' if center[0]>0 else 'above' if center[1]>0 else 'behind_camera')
    return layout_answer(dict(objects=[box(center,size)]),plan,map_plan(plan,'semantic_lexicon_v1'),shift)[0]


class RetryModel(Model):
    def __init__(self,repair):
        super().__init__(content=contents(),box=dict(objects=[box([0.,0.,3.])]))
        self.repair=repair

    def ask(self,image,prompt,path,**kwargs):
        if path.stem.startswith('boxes_retry'):self.answers[path.stem]=deepcopy(self.repair)
        return super().ask(image,prompt,path,**kwargs)


class GeometryTests(unittest.TestCase):
    def test_behind_camera_and_front_sideways_are_valid_but_front_is_not(self):
        for c,expected in [([0.,0.,-3.],True),([4.,0.,2.],True),([0.,0.,3.],False)]:
            r=check([item(c)],dict(objects=[]),[],90,[100,100])
            self.assertEqual(r['numeric_constraints_passed'],expected)

    def test_center_outside_is_insufficient_for_large_box(self):
        r=check([item([3.,0.,2.],[3.,1.,1.])],dict(objects=[]),[],90,[100,100])
        self.assertTrue(r['objects'][0]['intersects_input_frustum'])
        self.assertFalse(r['numeric_constraints_passed'])

    def test_vertical_fov_depends_on_aspect_and_camera_inside_is_rejected(self):
        self.assertFalse(view_check([0,2,2],[.2,.2,.2],90,[200,100])['intersects_input_box_proxy'])
        self.assertTrue(view_check([0,2,2],[.2,.2,.2],90,[100,200])['intersects_input_box_proxy'])
        r=check([item([0.,0.,0.],[1.,1.,1.])],dict(objects=[]),[],90,[100,100])
        self.assertTrue(r['objects'][0]['camera_inside_box'])

    def test_solver_uses_only_declared_local_domain_and_preserves_sizes(self):
        source=[item([2.5,0.,2.],[1.,1.,1.])]
        before=deepcopy(source)
        solved,report=solve(source,dict(objects=[]),[],90,[100,100])
        self.assertFalse(report['before']['numeric_constraints_passed'])
        self.assertTrue(report['after']['numeric_constraints_passed'])
        self.assertEqual(source,before)
        self.assertEqual(solved[0]['size'],source[0]['size'])
        for k in range(3):
            self.assertLessEqual(abs(solved[0]['center'][k]-source[0]['center'][k]),.75)
        self.assertEqual(solved[0]['off_fov_region'],'right')

    def test_unsatisfied_domain_is_not_moved_to_back_of_camera(self):
        source=[item([0.,0.,3.],shift=0.)]
        solved,report=solve(source,dict(objects=[]),[],90,[100,100])
        self.assertEqual(solved,source)
        self.assertFalse(report['after']['numeric_constraints_passed'])

    def test_size_and_movement_schema_do_not_allow_hacks(self):
        for change in (dict(size=[0,1,1]),dict(center=[float('nan'),0,2]),
                       dict(center_bounds_m=[[-3,0,-3],[3,0,-3]]),dict(unseen_reason='occluded')):
            a=box();a.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                layout_answer(dict(objects=[a]),dict(additions=contents()['additions']),
                              map_plan(contents(),'semantic_lexicon_v1'))

    def test_contradictory_relations_fail_and_support_is_not_certified(self):
        obs=dict(objects=[dict(track_id='obs',center_local_m={'value':[0,0,-3]},
                               size_aabb_local_m={'value':[1,1,1]})])
        obj=item([2.,0.,-3.])
        relations=[dict(source='new1',target='obs',kind=k,reason='Hypothesis')
                   for k in ('left_of','right_of')]
        self.assertGreater(check([obj],obs,relations,90,[100,100])['failed_relation_proxies'],0)
        relations=[dict(source='new1',target='obs',kind='supported_by',reason='Unmeasured')]
        r=check([item([.9,0.,-3.])],obs,relations,90,[100,100])
        self.assertEqual(r['unverified_relations'],1)
        self.assertFalse(r['physical_geometry_verified'])
        far=check([obj],obs,relations,90,[100,100])
        self.assertEqual(far['failed_relation_proxies'],1)

    def test_search_bounds_are_program_owned_and_no_model_arithmetic_is_needed(self):
        obj=item([.92,.04,3.67],[.25,1.8,.8],region='right')
        self.assertAlmostEqual(obj['center_bounds_m'][1][1],.79)
        self.assertEqual(obj['bounds_source'],'program_trust_region_not_scene_measurement')
        self.assertFalse(check([obj],dict(objects=[]),[],90,[100,100])['numeric_constraints_passed'])

    def test_outside_box_in_wrong_declared_region_is_not_a_valid_fix(self):
        obj=item([4.,0.,2.],region='behind_camera')
        r=check([obj],dict(objects=[]),[],90,[100,100])
        self.assertFalse(r['objects'][0]['intersects_input_frustum'])
        self.assertFalse(r['objects'][0]['intended_region_passed'])
        self.assertFalse(r['numeric_constraints_passed'])

    def test_unrelated_aabb_overlap_blocks_candidate(self):
        a=item();b=deepcopy(a);b['id']='new2'
        r=check([a,b],dict(objects=[]),[],90,[100,100])
        self.assertEqual(r['unrelated_aabb_overlaps'],1)
        self.assertFalse(r['numeric_constraints_passed'])


class PipelineTests(unittest.TestCase):
    def execute(self,root,c=None,b=None):
        case,observed=make_case(root);case['image_size']=[64,64]
        output=root/'out';output.mkdir()
        write(output/'config.json',dict(protocol=PROTOCOL,case=case,max_objects=256,
              reasoning='hypothesis',code_sha256=code_hashes(),max_distance=20.))
        model=Model(content=contents() if c is None else c,box=dict(objects=[b or box()]))
        record=run(model,case,output,layout_retries=0)
        return case,observed,output,record

    def test_full_off_fov_state_preserves_observations_and_visualizes_without_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            case,observed,out,r=self.execute(Path(tmp))
            self.assertEqual(r['status'],'numeric_world_ready_needs_semantic_review')
            self.assertFalse(r['geometry_verified']);self.assertFalse(r['semantics_verified'])
            folder=out/case['id'];full=read(folder/'G_full.json')
            self.assertEqual(full['objects'][:len(observed['objects'])],observed['objects'])
            hidden=validate_graph(read(folder/'G_out.json'))
            self.assertEqual(hidden['objects'],full['objects'][len(observed['objects']):])
            self.assertEqual(hidden['input_camera']['hfov_deg'],90.)
            self.assertTrue(r['world_state_ready'])
            self.assertFalse(r['adapter_connected'])
            self.assertFalse((folder/'G_full_adapter.json').exists())
            self.assertTrue((folder/'world/scene_3d.html').is_file())
            self.assertTrue((folder/'world/erp_state_map.png').is_file())
            self.assertNotIn('不进入Adapter子集', (folder/'world/index.html').read_text())

    def test_inside_fov_preserved_for_diagnosis_but_no_adapter_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            case,observed,out,r=self.execute(Path(tmp),b=box([0.,0.,3.]))
            self.assertEqual(r['status'],'world_blocked')
            folder=out/case['id']
            self.assertFalse((folder/'G_full_adapter.json').exists())
            self.assertEqual(len(read(folder/'G_full.json')['objects']),len(observed['objects'])+1)
            self.assertFalse(r['world_state_ready'])

    def test_unmapped_fine_category_is_kept_and_does_not_block_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=contents();c['additions'][0].update(name='rug',object_type='rug')
            case,_,out,r=self.execute(Path(tmp),c=c)
            self.assertEqual(r['status'],'numeric_world_ready_needs_semantic_review')
            self.assertEqual(r['unmapped_new'],1)
            obj=read(out/case['id']/'G_full.json')['objects'][-1]
            self.assertEqual(obj['object_type'],'rug')
            self.assertEqual(obj['description'],'rug')
            self.assertIsNone(obj['category_id']['value'])

    def test_zero_new_objects_is_not_a_completion_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=contents();c['additions']=[]
            _,_,_,r=self.execute(Path(tmp),c=c)
            self.assertEqual(r['status'],'world_blocked')
            self.assertIn('no_selected_outside_instances',r['blockers'])

    def test_observed_change_and_capacity_overflow_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            case,observed,out,_=self.execute(Path(tmp))
            full=read(out/case['id']/'G_full.json')
            items=read(out/case['id']/'layout_solved.json')['objects']
            bad=deepcopy(full);bad['objects'][0]['center_local_m']['value'][0]+=1
            with self.assertRaisesRegex(ValueError,'Observed'):validate_world(observed,bad,items,256)
            with self.assertRaisesRegex(ValueError,'capacity'):validate_world(observed,full,items,len(observed['objects']))

    def test_solved_layout_and_world_cannot_disagree(self):
        with tempfile.TemporaryDirectory() as tmp:
            case,observed,out,_=self.execute(Path(tmp))
            full=read(out/case['id']/'G_full.json')
            items=read(out/case['id']/'layout_solved.json')['objects']
            full['objects'][-1]['center_local_m']['value'][0]+=1
            with self.assertRaisesRegex(ValueError,'geometry'):validate_world(observed,full,items,256)

    def test_new_semantics_reuse_checks_raw_requests_and_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            case,_,out,_=self.execute(Path(tmp))
            f,c,hashes=load_semantics(out,case,256,'hypothesis')
            self.assertTrue(f['factors']);self.assertEqual(c['additions'][0]['id'],'new1')
            self.assertTrue(hashes)
            (out/case['id']/'contents.txt').write_text('{}')
            with self.assertRaisesRegex(ValueError,'binding'):load_semantics(out,case,256,'hypothesis')

    def test_old_occlusion_semantics_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'config.json',dict(protocol='hypothesized_mechanisms_hidden_world_v2'))
            with self.assertRaisesRegex(ValueError,'legacy'):
                load_semantics(root,{'id':'case'},256,'hypothesis')

    def test_failed_off_fov_v1_semantics_cannot_be_reused_as_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'config.json',dict(protocol='outside_input_frustum_world_v1'))
            with self.assertRaisesRegex(ValueError,'scope'):load_semantics(root,{'id':'case'},256,'hypothesis')

    def test_semantics_must_declare_region_and_threshold_for_near(self):
        c=contents();c['additions'][0].pop('off_fov_region')
        with self.assertRaisesRegex(ValueError,'off_fov_region'):
            normalized_contents(c,factors(),dict(context=[]),'hypothesis')
        c=contents();c['relations']=[dict(source='new1',target='obs',kind='near',reason='Hypothesis')]
        with self.assertRaisesRegex(ValueError,'max_distance_m'):
            normalized_contents(c,factors(),dict(context=[{'id':'obs'}]),'hypothesis')

    def test_numeric_feedback_retries_geometry_without_rerunning_semantics_or_changing_observed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,observed=make_case(root);case['image_size']=[64,64]
            out=root/'out';out.mkdir()
            model=RetryModel(dict(objects=[box()]))
            r=run(model,case,out,layout_retries=1)
            self.assertEqual(model.calls,['context','contents','boxes','boxes_retry1'])
            self.assertTrue(r['world_state_ready'])
            self.assertEqual(r['selected_layout_attempt'],1)
            folder=out/case['id'];self.assertTrue((folder/'layout_attempt0_review.json').is_file())
            self.assertEqual(read(folder/'G_full.json')['objects'][:len(observed['objects'])],observed['objects'])
            self.assertEqual(read(folder/'layout_proposal.json')['objects'][0]['center'],[0.,0.,3.])

    def test_correction_cannot_shrink_boxes_and_original_failed_world_is_still_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,_=make_case(root);case['image_size']=[64,64]
            out=root/'out';out.mkdir()
            r=run(RetryModel(dict(objects=[box(size=(.1,.1,.1))])),case,out,layout_retries=1)
            self.assertEqual(r['status'],'world_blocked')
            self.assertFalse(r['world_state_ready'])
            folder=out/case['id']
            self.assertIn('fixed sizes',read(folder/'layout_attempt1_review.json')['error'])
            self.assertTrue((folder/'world/scene_3d.html').is_file())

    def test_prompt_scope_is_outside_not_occluded_and_not_room_design(self):
        case=dict(hfov=90.,context=[],image_size=[64,64])
        p=contents_prompt(case,factors())
        self.assertNotIn('or be genuinely occluded',p)
        self.assertIn('OUTSIDE THE INPUT CAMERA FIELD OF VIEW',p)
        self.assertIn('not equipment needed',p)
        self.assertIn('factors (exactly an empty array)',context_prompt(case,'plain'))


if __name__=='__main__':unittest.main()
