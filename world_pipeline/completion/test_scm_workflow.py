"""Regression of run003 failures and stage reuse; no real-model quality claim."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from ..common import read,write
from .conditional_world_geometry import compile_layout
from .scene_completion import map_plan
from .scm_checkpoint import validate_checkpoint
from .scm_typed_layout import compile_dependency_layout,feasible_supports,numeric_body,solve_center
from .scm_world import code_hashes,run
from .scm_world_audit import audit
from .test_causal_world_completion import make_case
from .test_scm_typed import fake_choice
from .test_scm_world import SyntheticModel,SyntheticScorer

DATA=Path(__file__).with_name('testdata')


class ActualFailureTests(unittest.TestCase):
    def test_all_real_numeric_answers_without_reason_are_accepted_without_invented_rationale(self):
        records=read(DATA/'scm_run003_first_proposals.json')['records']
        self.assertEqual(len(records),3)
        for record in records:
            raw=record['body'];before=deepcopy(raw);body=numeric_body(raw)
            self.assertNotIn('reason',raw);self.assertEqual(body['size'],raw['size'])
            self.assertIn('No model rationale supplied',body['reason']);self.assertEqual(raw,before)
        body=numeric_body(dict(azimuth_deg=0,range_m=1,size=[1,1,1]))
        self.assertIsNone(body['center_y_m'])
        floor=solve_center(body,dict(kind='floor',target=None),dict(floor_y_m=-2),{},90,[64,64])
        self.assertEqual(floor['center'][1],-1.5)
        with self.assertRaisesRegex(ValueError,'explicit camera-Y'):
            solve_center(body,dict(kind='unspecified',target=None),dict(floor_y_m=-2),{},90,[64,64])

    def test_real_cushion_cannot_use_visible_sofa_but_can_use_sampled_hidden_couch(self):
        data=read(DATA/'scm_run003_geometry_context.json');case=data['case'];obs=data['observed'];plane=data['plane']
        before=deepcopy(obs);plan=deepcopy(data['plan'])
        plan['additions']=[o for o in plan['additions'] if o['id'] in ('scm_new_0001','scm_new_0007')]
        parent,child=[o['id'] for o in plan['additions']]
        raw=read(DATA/'scm_run003_first_proposals.json')['records'][1]['body']
        bodies={parent:numeric_body(dict(azimuth_deg=180,range_m=3,size=[1.8,.8,.9])),child:numeric_body(raw)}
        from .scm_typed_layout import initial_boxes
        choices,rejected=feasible_supports(child,bodies,plan,{},initial_boxes(case,obs),plane,case)
        self.assertFalse(any(c['target']=='observed_0000' for c in choices))
        self.assertTrue(any(r['choice']['target']=='observed_0000' for r in rejected))
        host=next(c for c in choices if c['target']==parent)
        selected={parent:dict(kind='floor',target=None),child:host}
        solved,answer,certs,components=compile_dependency_layout(case,obs,plan,plane,bodies,selected)
        items,review=compile_layout(answer,solved,map_plan(solved,'semantic_lexicon_v1'),obs,plane,case['hfov'],case['image_size'])
        self.assertTrue(review['geometry_contract_passed']);self.assertEqual(len(items),2)
        self.assertLess(solved['placement_dependency_order'].index(parent),solved['placement_dependency_order'].index(child))
        self.assertEqual(obs,before);self.assertEqual(len(components),1)

    def test_component_solver_does_not_trap_child_with_locally_hidden_floor_host(self):
        case=dict(context=[],hfov=90,image_size=[64,64]);obs=dict(objects=[]);plane=dict(floor_y_m=-2)
        objects=[dict(id=i,name=i,object_type='furniture',support=dict(kind='unspecified',target=None)) for i in ('host','child')]
        plan=dict(additions=objects,relations=[],placement_dependency_order=['host','child'])
        bodies={'host':numeric_body(dict(azimuth_deg=0,range_m=.5,size=[.5,.5,.5],center_y_m=-1.75)),
            'child':numeric_body(dict(azimuth_deg=0,range_m=.5,size=[.3,2.,.3],center_y_m=-.5))}
        local=solve_center(bodies['host'],dict(kind='floor',target=None),plane,{},90,[64,64])
        self.assertEqual(local['off_fov_certificate_region'],'below')
        selected=dict(host=dict(kind='floor',target=None),child=dict(kind='object',target='host'))
        _,_,certs,_=compile_dependency_layout(case,obs,plan,plane,bodies,selected)
        self.assertNotEqual(certs['host']['off_fov_certificate_region'],'below')
        self.assertEqual(certs['host']['off_fov_certificate_region'],certs['child']['off_fov_certificate_region'])

    def test_future_support_reference_allowed_but_cycle_excluded(self):
        case=dict(context=[],hfov=90,image_size=[64,64]);plane=dict(floor_y_m=-2)
        plan=dict(additions=[dict(id=i,name=i) for i in ('a','b')])
        bodies={i:numeric_body(dict(azimuth_deg=180,range_m=3,size=[1,1,1])) for i in ('a','b')}
        first,_=feasible_supports('a',bodies,plan,{}, {},plane,case)
        self.assertTrue(any(c['target']=='b' for c in first))
        second,_=feasible_supports('b',bodies,plan,{'a':dict(kind='object',target='b')},{},plane,case)
        self.assertFalse(any(c['target']=='a' for c in second))


class CheckpointTests(unittest.TestCase):
    def test_layout_can_reuse_validated_semantics_without_any_new_semantic_LM_call(self):
        class Model(SyntheticModel):
            def ask(self,image,prompt,path,**options):
                if path.stem.startswith('metric_'):
                    # Essential geometry only: no prose and no unnecessary Y.
                    value=dict(azimuth_deg=180,range_m=3,size=[.4,.5,.4])
                    path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value
                if getattr(self,'forbid_semantics',False):raise AssertionError('Semantic model must be reused')
                return super().ask(image,prompt,path,**options)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,obs=make_case(root);case['image_size']=[64,64]
            args=dict(generic_types=3,max_count=3,worlds=1,seed=0,max_objects=256,layout_backend='typed',mechanism_scoring='continuation')
            source=root/'source';source.mkdir()
            with patch('qwen_edit_pano.world_pipeline.completion.scm_typed_layout.finite_choice',fake_choice):
                first=run(Model(case),SyntheticScorer(),case,source,max_count=3,generic_types=3,worlds=1,seed=0,layout_backend='typed')
            self.assertTrue(first['world_state_ready'],first)
            write(source/'config.json',dict(case=case,code_sha256=code_hashes(),arguments=args))
            self.assertEqual(audit(source)['status'],'scm_postflight_contract_passed')
            validate_checkpoint(source,case,3,3)
            dest=root/'reused';dest.mkdir();model=Model(case);model.forbid_semantics=True
            with patch('qwen_edit_pano.world_pipeline.completion.scm_typed_layout.finite_choice',fake_choice):
                second=run(model,None,case,dest,max_count=3,generic_types=3,worlds=1,seed=0,layout_backend='typed',semantic_source=source)
            self.assertTrue(second['world_state_ready'],second)
            self.assertEqual(read(source/'case/hypothesis_01/posterior_sample.json'),read(dest/'case/hypothesis_01/posterior_sample.json'))
            self.assertEqual(read(dest/'case/G_full.json')['objects'][:len(obs['objects'])],obs['objects'])
            write(dest/'config.json',dict(case=case,code_sha256=code_hashes(),arguments=args))
            self.assertEqual(audit(dest)['status'],'scm_postflight_contract_passed')
            broken=deepcopy(case);broken['hfov']=70
            with self.assertRaises(ValueError):validate_checkpoint(source,broken,3,3)
            data=read(source/'case/domain.normalized.json');data['object_types']=['changed'];write(source/'case/domain.normalized.json',data)
            with self.assertRaises(ValueError):validate_checkpoint(source,case,3,3)


if __name__=='__main__':unittest.main()
