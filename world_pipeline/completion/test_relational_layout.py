"""CPU relation/search regressions; synthetic specs do not validate Qwen quality."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ..common import ROOT, read, write
from ..oracle_model import tensorize
from .audit_semantic_layout import project_aabb
from .relational_layout import (NoLayout, apply_revision, candidates, export, intersect,
                                relation_holds, revision_prompt, run, solve, validate_spec, validate_revision)
from .test_semantic_probe import state


def item(oid='new_chair', anchor='observed_0000', **updates):
    result = dict(id=oid,name='armchair',raw_category='chair',category='chair',size=[.7,.9,.7],
        anchor_id=anchor,relation='near',gap_m=.3,height_anchor_id='observed_0000',
        height_alignment='bottom',height_offset_m=0.,reason='Additional seating related to the visible furniture.')
    result.update(updates)
    return result


def proposal(*objects):
    return dict(scene_type='living room',scene_hypothesis='Additional seating outside this partial view.',
                additions=list(objects or [item()]))


class RelationalLayoutTests(unittest.TestCase):
    def test_reference_edges_drive_height_and_no_absolute_floor_is_used(self):
        ref = dict(id='reference',center=[0.,-.7,3.],size=[2.,1.,2.])
        obj = item(); rng = np.random.default_rng(0)
        a = candidates(obj,ref,ref,[],90,(768,768),15,1.,rng)
        shifted = dict(ref,center=[0.,1.67,3.])
        b = candidates(obj,shifted,shifted,[],90,(768,768),15,1.,rng)
        self.assertTrue(a and b)
        self.assertTrue(all(abs(p['center'][1]-(-.75))<1e-10 for p in a))
        self.assertTrue(all(abs(p['center'][1]-1.62)<1e-10 for p in b))

    def test_same_reference_can_produce_left_right_front_and_back_without_group_translation(self):
        ref = dict(id='reference',center=[0.,0.,0.],size=[1.,1.,1.])
        for relation,axis,sign in [('left_of',0,-1),('right_of',0,1),('in_front_of',2,-1),('behind',2,1)]:
            options = candidates(item(relation=relation),ref,ref,[],90,(768,768),15,1.,np.random.default_rng(0))
            self.assertTrue(options)
            self.assertTrue(all(relation_holds(p,ref,relation) for p in options))
            self.assertTrue(all(sign*p['center'][axis]>0 for p in options))
            self.assertTrue(all(not project_aabb(p['center'],p['size'],90,(768,768))['entire_aabb_inside_input_frustum'] for p in options))

    def test_joint_search_handles_new_object_anchors_without_moving_observed_encoder_inputs(self):
        observed = state(); original = deepcopy(observed)
        spec = validate_spec(proposal(item(),item('new_table','new_chair',name='side table',category='table',
                        raw_category='side table',size=[.5,.6,.5])),observed,6)
        selected, report = solve(spec,observed,90,(768,768))
        all_candidates, full, unmapped = export(observed,spec,selected)
        self.assertEqual(observed,original)
        self.assertEqual(full['objects'][:len(observed['objects'])],observed['objects'])
        self.assertFalse(unmapped)
        self.assertEqual(selected[1]['anchor_id'],'new_chair')
        self.assertFalse(intersect(selected[0],selected[1]))
        self.assertFalse(report['geometry_verified'])
        for before,after in zip(tensorize(observed,160),tensorize(full,160)):
            self.assertTrue(before[:,:len(observed['objects'])+1].equal(after[:,:len(observed['objects'])+1]))
        self.assertEqual(full['objects'][-1]['center_local_m']['source'],'cpu_relation_search')

    def test_infeasible_search_retains_specification_and_never_drops_or_pushes_objects_far_away(self):
        observed = state(); observed['objects'] = observed['objects'][:1]
        observed['objects'][0]['center_local_m']['value']=[0,0,10]
        observed['objects'][0]['size_aabb_local_m']['value']=[.1,.1,.1]
        spec = validate_spec(proposal(item(size=[.1,.1,.1],gap_m=0)),observed,6)
        original = deepcopy(spec)
        with self.assertRaises(NoLayout) as failed: solve(spec,observed,90,(768,768),gap_slack=0)
        self.assertGreater(failed.exception.report['stages'][-1]['candidate_pools'][0]['counts']['input_fov'],0)
        self.assertEqual(spec,original)

    def test_category_legality_does_not_allow_known_rug_to_become_furniture(self):
        observed = state()
        raw = proposal(item(raw_category='rug',category='furniture',name='geometric rug'))
        validated = validate_spec(raw,observed,6)
        self.assertIsNone(validated['additions'][0]['category'])
        self.assertEqual(validated['additions'][0]['category_review']['model_category'],'furniture')
        self.assertEqual(validated['additions'][0]['category_review']['official_mpcat40_id'],2)
        self.assertEqual(raw['additions'][0]['category'],'furniture')
        selected,_ = solve(validated,observed,90,(768,768))
        candidates_state,full,unmapped = export(observed,validated,selected)
        self.assertEqual(len(unmapped),1)
        self.assertEqual(len(full['objects']),len(observed['objects']))
        self.assertEqual(len(candidates_state['objects']),len(observed['objects'])+1)

    def test_actual_rug_label_and_exact_aliases_do_not_abort_the_scene(self):
        for category in ('rug','furniture',None):
            spec = validate_spec(proposal(item(raw_category='rug',category=category)),state(),6)
            self.assertIsNone(spec['additions'][0]['category'])
        spec = validate_spec(proposal(item(raw_category='floor lamp',category='floor lamp')),state(),6)
        self.assertEqual(spec['additions'][0]['category'],'lighting')
        spec = validate_spec(proposal(item(category='unrecognized_label')),state(),6)
        self.assertIsNone(spec['additions'][0]['category'])

    def test_unknown_and_cyclic_references_are_reported_not_invented(self):
        for raw in (proposal(item(anchor='missing')),proposal(item(anchor='new_chair')),
                    proposal(item(anchor='new_table'),item('new_table','new_chair'))):
            with self.assertRaises(ValueError): validate_spec(raw,state(),6)

    def test_separate_functional_area_can_use_camera_without_becoming_an_observed_accessory(self):
        observed = state()
        spec = validate_spec(proposal(item(anchor='__camera__',gap_m=2.)),observed,6)
        selected,_ = solve(spec,observed,90,(768,768))
        self.assertEqual(selected[0]['anchor_id'],'__camera__')
        self.assertTrue(np.isfinite(selected[0]['center']).all())
        with self.assertRaisesRegex(ValueError,'Camera roots'):
            validate_spec(proposal(item(anchor='__camera__',relation='behind')),observed,6)

    def test_lamp_on_new_table_uses_its_actual_box_top_without_vertical_search_or_floor_guess(self):
        observed = state()
        table = item('new_table',anchor='__camera__',gap_m=2.,name='side table',raw_category='side table',
                     category='table',size=[.7,.6,.7])
        lamp = item('new_lamp',anchor='new_table',name='table lamp',raw_category='lamp',category='lighting',
                    size=[.25,.45,.25],relation='on_top',gap_m=0)
        for k in ('height_anchor_id','height_alignment','height_offset_m'): lamp.pop(k)
        lamp['height_anchor_id'] = ['unused_for_on_top']
        spec = validate_spec(proposal(table,lamp),observed,6)
        selected,_ = solve(spec,observed,90,(768,768))
        table,lamp = selected
        self.assertTrue(np.all(abs(np.array(table['center'])[::2]-np.array(lamp['center'])[::2])
                                  +np.array(lamp['size'])[::2]/2<=np.array(table['size'])[::2]/2+1e-7))
        self.assertAlmostEqual(lamp['center'][1]-.45/2,table['center'][1]+.6/2)
        self.assertEqual(lamp['gap_deviation_m'],0.)
        self.assertFalse(intersect(table,lamp))

    def test_actual_kitchen_top_relation_has_offcentre_candidates_and_still_reports_cooktop_conflict(self):
        root = ROOT/'qwen_edit_pano/outputs/relational_world_val2_run001/8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029'
        if not root.exists(): self.skipTest('Local diagnostic result not available')
        observed = read(root/'G_obs.json')
        spec = validate_spec(read(root/'relative_scene.json'),observed,6)
        rack = spec['additions'][0]
        parent = observed['objects'][7]
        reference = dict(center=parent['center_local_m']['value'],size=parent['size_aabb_local_m']['value'])
        counts = {}
        options = candidates(rack,reference,None,[],90,(768,768),15,1.,np.random.default_rng(0),counts)
        self.assertEqual(counts['sampled'],25)
        self.assertTrue(options)
        for p in options:
            self.assertTrue(np.all(abs(np.array(p['center'])[::2]-np.array(reference['center'])[::2])
                                  +np.array(p['size'])[::2]/2<=np.array(reference['size'])[::2]/2+1e-7))
            self.assertEqual(p['gap_deviation_m'],0.)
        with self.assertRaises(NoLayout) as failed: solve(spec,observed,90,(768,768))
        self.assertEqual(failed.exception.report['failed_object'],'addition_0002')

    def test_feedback_revision_cannot_delete_or_resize_fixed_contents(self):
        observed = state(); initial = validate_spec(proposal(),observed,6)
        removed = proposal(); removed['additions']=[]
        with self.assertRaisesRegex(ValueError,'fixed object list'):
            validate_revision(removed,initial,observed,6)
        resized = proposal(); resized['additions'][0]['size'][0] += .1
        with self.assertRaisesRegex(ValueError,'fixed object list'):
            validate_revision(resized,initial,observed,6)

    def test_sparse_patch_keeps_omitted_objects_and_only_owns_placement_fields(self):
        observed=state(); initial=validate_spec(proposal(),observed,6); original=deepcopy(initial)
        patch=dict(placements={'new_chair':dict(anchor_id='__camera__',gap_m=2)},withdrawals={})
        revised=apply_revision(patch,initial,observed,6)
        self.assertEqual(initial,original)
        self.assertEqual(revised['additions'][0]['size'],initial['additions'][0]['size'])
        self.assertEqual(revised['additions'][0]['anchor_id'],'__camera__')
        unchanged=apply_revision(dict(placements={},withdrawals={}),initial,observed,6)
        self.assertEqual(unchanged['additions'],initial['additions'])
        for bad in (dict(placements={'renamed':{}},withdrawals={}),
                    dict(placements={'observed_0000':{}},withdrawals={}),
                    dict(placements={'new_chair':dict(size=[1,1,1])},withdrawals={}),
                    dict(placements={'new_chair':{}},withdrawals={'new_chair':'duplicate'}),
                    dict(placements={},withdrawals={'new_chair':''})):
            with self.assertRaises(ValueError): apply_revision(bad,initial,observed,6)

    def test_explicit_withdrawal_keeps_original_registry_and_never_repairs_dangling_references(self):
        observed=state()
        raw=proposal(item(),item('new_table','new_chair'))
        initial=validate_spec(raw,observed,6)
        bad=dict(placements={},withdrawals={'new_chair':'Already present in the image.'})
        with self.assertRaisesRegex(ValueError,'reference'):
            apply_revision(bad,initial,observed,6)
        patch=dict(placements={},withdrawals={'new_table':'Duplicates visible table.'})
        revised=apply_revision(patch,initial,observed,6)
        self.assertEqual([o['id'] for o in revised['additions']],['new_chair'])
        self.assertEqual(revised['withdrawn_hypotheses'][0]['hypothesis'],initial['additions'][1])
        self.assertFalse(revised['withdrawn_hypotheses'][0]['semantics_verified'])
        self.assertEqual(len(initial['additions']),2)

    def test_sparse_patch_changes_one_object_and_retains_other_identity_geometry_and_references(self):
        observed=state()
        initial=validate_spec(proposal(item(),item('new_table','new_chair',category='table',
                                    raw_category='table',size=[.5,.4,.5])),observed,6)
        patch=dict(placements={'new_chair':dict(gap_m=.8)},withdrawals={})
        revised=apply_revision(patch,initial,observed,6)
        self.assertEqual(revised['additions'][1],initial['additions'][1])
        self.assertEqual(revised['additions'][0],dict(initial['additions'][0],gap_m=.8))

    def test_actual_kitchen_reply_did_withdraw_and_renumber_not_only_change_placement(self):
        root=ROOT/'qwen_edit_pano/outputs/relational_world_val2_run002/8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029'
        if not root.exists(): self.skipTest('Local diagnostic result not available')
        observed=read(root/'G_obs.json')
        initial=validate_spec(read(root/'relative_scene.json'),observed,6)
        with self.assertRaisesRegex(ValueError,'missing_ids.*addition_0003'):
            apply_revision(read(root/'placement_revision.json'),initial,observed,6)
        # A manually specified withdrawal tests the interface, NOT Qwen quality.
        patch=dict(placements={'addition_0001':{},'addition_0003':{}},
                   withdrawals={'addition_0002':'The photograph already contains a cooking surface.'})
        revised=apply_revision(patch,initial,observed,6)
        selected,_=solve(revised,observed,90,(768,768))
        self.assertEqual([p['id'] for p in selected],['addition_0001','addition_0003'])
        self.assertEqual(revised['withdrawn_hypotheses'][0]['hypothesis']['name'],'induction cooktop')

    def test_actual_run003_sparse_reply_finishes_on_cpu_without_rewriting_the_answer(self):
        source=ROOT/'qwen_edit_pano/outputs/relational_world_val2_run003'
        sid='8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029'
        if not (source/sid/'placement_revision.json').exists():
            self.skipTest('Local diagnostic result not available')
        case=read(source/sid/'input.json')
        original=read(source/sid/'placement_revision.json')
        observed=read(source/sid/'G_obs.json')
        initial=validate_spec(read(source/sid/'relative_scene.json'),observed,6)
        self.assertEqual(original['placements'],{})
        revised=apply_revision(original,initial,observed,6)
        self.assertEqual(revised['additions'],[initial['additions'][0],initial['additions'][2]])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertTrue(run(None,[case],root,replay=source,apply_saved_revision=True))
            dest=root/sid; report=read(dest/'summary.json'); full=read(dest/'G_full.json')
            self.assertTrue(report['placement_revision_reused'])
            self.assertEqual(report['status'],'needs_content_review')
            self.assertEqual(report['hypotheses_withdrawn'],1)
            self.assertEqual(report['hypothesized'],2)
            self.assertEqual(len(full['objects']),18)
            self.assertEqual(full['objects'][:16],observed['objects'])
            self.assertEqual(read(dest/'placement_revision.json'),original)
            self.assertEqual(read(dest/'withdrawn_hypotheses.json')[0]['withdrawal_reason'],
                             original['withdrawals']['addition_0002'])
            for before,after in zip(tensorize(observed,160),tensorize(full,160)):
                self.assertTrue(before[:,:17].equal(after[:,:17]))
            replay=root/'replay'; replay.mkdir()
            self.assertTrue(run(None,[case],replay,replay=root))
            self.assertEqual(read(replay/sid/'G_full.json'),full)
            # A mismatched saved request must not be applied to this registry.
            import shutil
            copied=root/'wrong_source'; shutil.copytree(source/sid,copied/sid)
            request=read(copied/sid/'placement_revision.request.json')
            request['prompt']=request['prompt'].replace('PREVIOUS VALIDATED SPECIFICATION:','WRONG SPECIFICATION:')
            write(copied/sid/'placement_revision.request.json',request)
            failed=root/'failed'; failed.mkdir()
            self.assertFalse(run(None,[case],failed,replay=copied,apply_saved_revision=True))
            self.assertFalse((failed/sid/'G_full.json').exists())

    def test_repair_prompt_has_no_competing_full_generation_request(self):
        observed=state(); observed['objects']=observed['objects'][:1]
        observed['objects'][0]['center_local_m']['value']=[0,0,10]
        observed['objects'][0]['size_aabb_local_m']['value']=[.1,.1,.1]
        spec=validate_spec(proposal(item(size=[.1,.1,.1],gap_m=0)),observed,6)
        with self.assertRaises(NoLayout) as failed: solve(spec,observed,90,(768,768))
        text=revision_prompt(dict(hfov=90,context=[],prompt='RETURN_FULL_ADDITIONS'),spec,failed.exception)
        self.assertNotIn('RETURN_FULL_ADDITIONS',text)
        self.assertIn('placements',text)
        self.assertIn('semantic reason',text)

    def test_patch_withdrawal_is_reviewable_not_a_successfully_placed_object_and_replays_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); observed=state(); observed['objects']=observed['objects'][:1]
            observed['objects'][0]['center_local_m']['value']=[0,0,10]
            observed['objects'][0]['size_aabb_local_m']['value']=[.1,.1,.1]
            state_path=root/'state.json'; write(state_path,observed)
            image_path=root/'input.png'; Image.new('RGB',(64,64)).save(image_path)
            case=dict(id='case',image=str(image_path),state=str(state_path),hfov=90,prompt='test',source_sha256={})
            initial=proposal(item('retained',size=[.1,.1,.1],gap_m=0,category='floor lamp',raw_category='floor lamp'),
                             item('withdrawn',size=[.1,.1,.1],gap_m=0))
            patch=dict(placements={'retained':dict(anchor_id='__camera__',gap_m=2)},
                       withdrawals={'withdrawn':'The image already contains this chair instance.'})
            class Model:
                def ask(self,image,prompt,path,**kwargs): return initial if path.stem=='relative_scene' else patch
            self.assertTrue(run(Model(),[case],root))
            report=read(root/'case/summary.json')
            self.assertEqual(report['status'],'needs_content_review')
            self.assertEqual(report['hypotheses_withdrawn'],1)
            self.assertEqual(report['hypothesized'],1)
            self.assertEqual(report['planned_before_revision'],2)
            self.assertEqual(read(root/'case/G_full.json')['objects'][:1],observed['objects'])
            self.assertEqual(read(root/'case/placement_revision.json'),patch)
            self.assertEqual(read(root/'case/withdrawn_hypotheses.json')[0]['hypothesis']['id'],'withdrawn')
            self.assertIn('withdrawn_hypotheses.json',(root/'case/index.html').read_text())
            replay=root/'replay'; replay.mkdir()
            self.assertTrue(run(None,[case],replay,replay=root))
            self.assertEqual(read(root/'case/G_full.json'),read(replay/'case/G_full.json'))
            self.assertEqual(read(root/'case/withdrawn_hypotheses.json'),read(replay/'case/withdrawn_hypotheses.json'))

    def test_bounded_feedback_reuses_observation_and_replays_accepted_revision_without_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); observed = state(); observed['objects']=observed['objects'][:1]
            observed['objects'][0]['center_local_m']['value']=[0,0,10]
            observed['objects'][0]['size_aabb_local_m']['value']=[.1,.1,.1]
            state_path=root/'state.json'; write(state_path,observed)
            image_path=root/'input.png'; Image.new('RGB',(64,64)).save(image_path)
            case=dict(id='case',image=str(image_path),state=str(state_path),hfov=90,prompt='original question',source_sha256={})
            initial=proposal(item(size=[.1,.1,.1],gap_m=0))
            corrected=deepcopy(initial); corrected['additions'][0].update(anchor_id='__camera__',gap_m=2.)
            calls=[]
            class Model:
                def ask(self,image,prompt,path,**kwargs):
                    calls.append((path.stem,prompt))
                    return initial if path.stem=='relative_scene' else corrected
            self.assertTrue(run(Model(),[case],root))
            self.assertEqual([c[0] for c in calls],['relative_scene','placement_revision'])
            self.assertIn('COMPUTED FAILURE FEEDBACK',calls[1][1])
            self.assertEqual(read(root/'case/relative_scene.json'),initial)
            self.assertEqual(read(root/'case/revision_report.json')['status'],'accepted')
            replay=root/'replay'; replay.mkdir()
            newer_case=dict(case,prompt='new compiler instructions; evidence unchanged')
            self.assertTrue(run(None,[newer_case],replay,replay=root))
            self.assertEqual(read(root/'case/G_full.json'),read(replay/'case/G_full.json'))
            self.assertEqual(read(replay/'case/source_input.json')['prompt'],'original question')

    def test_feedback_stops_after_one_failed_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); observed=state(); observed['objects']=observed['objects'][:1]
            observed['objects'][0]['center_local_m']['value']=[0,0,10]
            observed['objects'][0]['size_aabb_local_m']['value']=[.1,.1,.1]
            state_path=root/'state.json'; write(state_path,observed)
            image_path=root/'input.png'; Image.new('RGB',(64,64)).save(image_path)
            case=dict(id='case',image=str(image_path),state=str(state_path),hfov=90,prompt='test',source_sha256={})
            calls=[]
            class Model:
                def ask(self,*args,**kwargs): calls.append(1); return proposal(item(size=[.1,.1,.1],gap_m=0))
            self.assertFalse(run(Model(),[case],root))
            self.assertEqual(len(calls),2)
            self.assertFalse((root/'case/G_full.json').exists())
            self.assertEqual(read(root/'case/revision_report.json')['status'],'failed')

    def test_real_white_sofa_numbers_cannot_put_aligned_coffee_table_on_ceiling(self):
        # Actual frozen WildDet3D observation, manually specified relation.
        # This is a numeric regression, NOT a new Qwen inference result.
        path = ROOT/'qwen_edit_pano/outputs/scene_box_layout_val2_run001/zsNo4HB9uLZ_ed8645386a0a4de2bfc486e0a1d02468/G_obs.json'
        if not path.exists(): self.skipTest('Local diagnostic result not available')
        observed = read(path)
        spec = validate_spec(proposal(item(name='coffee table',raw_category='coffee table',category='table',
            size=[.8,.4,.8],relation='in_front_of',gap_m=.3)),observed,6)
        selected,_ = solve(spec,observed,90,(768,768))
        sofa = observed['objects'][0]
        expected_y = sofa['center_local_m']['value'][1]-sofa['size_aabb_local_m']['value'][1]/2+.2
        self.assertAlmostEqual(selected[0]['center'][1],expected_y)
        self.assertLess(selected[0]['center'][1],-1.)
        self.assertTrue(relation_holds(selected[0],dict(center=sofa['center_local_m']['value'],size=sofa['size_aabb_local_m']['value']),'in_front_of'))

    def test_runner_uses_one_8b_request_and_writes_actual_colored_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'G_obs.json'; write(source,state())
            image_path = root/'input.png'; Image.new('RGB',(64,64)).save(image_path)
            case = dict(id='test',image=str(image_path),state=str(source),hfov=90,prompt='test',source_sha256={})
            calls=[]
            class Model:
                def ask(self,image,prompt,path,**kwargs): calls.append(path.name); return proposal()
            self.assertTrue(run(Model(),[case],root))
            self.assertEqual(calls,['relative_scene.txt'])
            self.assertTrue((root/'test/erp_state_map.png').exists())
            self.assertIn('CPU', (root/'test/index.html').read_text())
            self.assertEqual(read(root/'test/summary.json')['hypothesized'],1)
            replay = root/'replay'; replay.mkdir()
            self.assertTrue(run(None,[case],replay,replay=root))
            self.assertEqual(read(root/'test/G_full.json'),read(replay/'test/G_full.json'))
            self.assertEqual(len(calls),1)


if __name__=='__main__': unittest.main()
