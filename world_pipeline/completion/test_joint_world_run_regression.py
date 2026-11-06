"""Regressions for the actual repaired_001 run, not evidence of new LM quality."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from PIL import Image

from ..common import ROOT,read,write
from . import joint_world as runner,joint_world_model as model,joint_world_geometry as geometry
from . import test_joint_world as fixture
from .joint_world_replay import case_for
from .test_joint_world_contracts import Replies
from .assembly_experiment import stage_seed

RUN=ROOT/'qwen_edit_pano/outputs/joint_world_repaired_001'
LOUNGE='8194nk5LbLH_c9e8dc09263e4d0da77d16de0ecddd39'
BED='zsNo4HB9uLZ_ead481533f834704bd489d3d44b6a03a'
BATH='x8F5xyUWy9e_3a127fd2c5f34d978cba253145dae7d6'


def comp(sid):return RUN/'scenes'/sid/'joint_completion'


class RunRegressionTests(unittest.TestCase):
    setUp=fixture.JointWorldTests.setUp
    compiled=fixture.JointWorldTests.compiled

    def propose(self,value):
        picture=self.base/'input.png';Image.new('RGB',(32,32)).save(picture)
        case=dict(self.case,image=str(picture))
        def ask(name,image,prompt,validator,max_tokens=8192):
            return validator(fixture.hypotheses() if name=='hypotheses' else value)
        runner.propose(case,dict(objects=[]),self.base,{},ask)
        return case,read(self.base/'domain.json')

    def test_nonempty_proposals_do_not_get_a_synthetic_empty_competitor(self):
        case,d=self.propose(fixture.programs_reply())
        self.assertEqual(list(d['programs']),['f_h0_0'])
        calls=[]
        def ask(name,image,prompt,validator,max_tokens=8192):
            calls.append(name);return validator(dict(choice='h0',reason='Plausible whole-world assumption.'))
        runner.mechanisms(case,dict(objects=[]),self.base,{},ask,4)
        self.assertEqual(calls,['recognition_0','recognition_1','recognition_2','recognition_3'])
        scm=read(self.base/'causal_model.json');w=model.sample_worlds(scm,17,4)
        self.assertTrue(all(len(v['objects'])==4 for v in w))

    def test_explicit_LM_empty_is_legal_without_nonempty_quota(self):
        _,d=self.propose(dict(alternatives=[dict(description='Unseen rear is assumed to be an unfurnished circulation space.',groups=[],relations=[])]))
        p=d['programs']['f_h0_0'];self.assertFalse(p.get('proposal_failed',False));self.assertEqual(p['groups'],[])
        self.assertEqual(model.reference_program_law(d['programs'])['probabilities'],[1.])

    def test_reference_retains_genuine_empty_and_failed_states_without_resampling(self):
        row=model.reference_program_law(dict(full={},empty={},failed={}))
        self.assertEqual(row['probabilities'],[1/3]*3);self.assertFalse(row['estimated_from_RGB'])
        self.assertEqual(row['states'],['full','empty','failed'])

    def test_all_five_saved_domains_reference_ablation_preserves_whole_inventory(self):
        counts={};suite=read(RUN/'suite.json')
        for s in suite['scenes']:
            d=read(comp(s['id'])/'domain.json');old=read(comp(s['id'])/'empirical_tables.json')
            programs={fid:p for fid,p in d['programs'].items() if p.get('origin')!='explicit_model_absence_alternative_not_a_failed_proposal'}
            tables=dict(recognition=old['recognition'],programs={h['id']:model.reference_program_law(
                {fid:p for fid,p in programs.items() if p['hypothesis']['id']==h['id']}) for h in d['hypotheses']})
            m=model.compile_model(d['hypotheses'],programs,tables)
            expected=sum(len(g['objects']) for p in programs.values() for g in p['groups'])
            self.assertEqual(len(m['component_registry']),expected)
            worlds=model.sample_worlds(m,stage_seed(suite['seed'],s['id']),suite['worlds'])
            self.assertEqual(len(worlds),suite['worlds'])
            for w in worlds:
                counts[w['proposal_failed']]=counts.get(w['proposal_failed'],0)+1
                if not w['proposal_failed']:self.assertGreater(len(w['objects']),0)
        self.assertEqual(sum(counts.values()),20)

    def test_saved_front_origin_is_a_proposal_not_an_early_visibility_failure(self):
        w,obs,_,_=fixture.JointWorldTests.layout_fixture(self)
        raw=read(comp(BED)/'calls/world_03_origins.json')
        origins=geometry.origins_answer(raw,w,90,['g0'])
        origins.update(geometry.origins_answer(dict(groups=[dict(id='g1',center_xz_m=[-3.,-4.])]),w,90,['g1']))
        dims={o['id']:dict(full_size_m=[.8,.6,.5] if not o['attachment']['host'] else [.1,.12,.1]) for o in w['objects']}
        poses={o['id']:dict(id=o['id'],yaw_deg=25.,**({'root_offset_m':[0.,0.,0.]} if not o['attachment']['host'] else {'host_offset_uv':[.2,.1]})) for o in w['objects']}
        layout=geometry.convert(w,obs,dims,origins,poses,90)
        self.assertTrue(layout['g0']['origin_initialization']['changed'])
        solved=geometry.compile_layout(self.case,obs,w,layout,dict(floor_y_m=-1.5))
        self.assertTrue(solved['review']['geometry_contract_passed'],solved['review'])
        for g in layout.values():
            for row in g['objects']:self.assertEqual(row['intrinsic_size_m'],dims[row['id']]['full_size_m'])

    def test_per_group_requests_use_exact_templates_and_compact_world(self):
        w,obs,_,_=fixture.JointWorldTests.layout_fixture(self)
        dims={o['id']:dict(full_size_m=[.8,.6,.5] if not o['attachment']['host'] else [.1,.12,.1]) for o in w['objects']}
        requests=[];provider=fixture.CPUProvider()
        def ask(name,image,prompt,validator,max_tokens=8192):
            requests.append((name,prompt));return validator(provider.reply(prompt,'world_01_'+name))
        layout,raw=geometry.infer(self.case,obs,w,dims,dict(floor_y_m=-1.5),None,ask)
        self.assertEqual([n for n,p in requests if n.startswith('origins_')],['origins_g0','origins_g1'])
        for name,prompt in requests:
            self.assertNotIn('"structural_state"',prompt);self.assertNotIn('"noise"',prompt)
            if name.startswith('origins_'):
                template=fixture.after(prompt,'\nORIGIN_TEMPLATE: ')
                self.assertEqual(len(template['groups']),1)
                self.assertEqual(template['groups'][0]['id'],name.removeprefix('origins_'))
        self.assertEqual(set(raw['poses']),{o['id'] for o in w['objects']})

    def test_saved_missing_group_reply_is_valid_only_for_the_requested_group(self):
        w,_,_,_=fixture.JointWorldTests.layout_fixture(self)
        raw=read(comp(LOUNGE)/'calls/world_04_origins.json')
        self.assertEqual(set(geometry.origins_answer(raw,w,90,['g0'])),{'g0'})
        with self.assertRaisesRegex(ValueError,'g1'):geometry.origins_answer(raw,w,90,['g1'])

    def test_saved_null_floor_cannot_advance_to_origins(self):
        case,obs=case_for(LOUNGE);saved=read(comp(LOUNGE)/'calls/world_04_floor.json')
        w=read(comp(LOUNGE)/'world_04/semantic_world.json')
        p=Replies([saved,saved]);ask=runner.asker(p,dict(seed=3407),self.base/'calls')
        with self.assertRaises(ValueError):geometry.floor_hypothesis(case,obs,w,None,ask)
        self.assertEqual(len(p.prompts),2)
        self.assertTrue((self.base/'calls/floor_repair.validation.json').exists())
        # Independent unresolved-value check even after fixing the copied basis.
        with self.assertRaisesRegex(ValueError,'Unresolved floor'):
            geometry.floor_answer(dict(saved,basis='Camera tilt prevents a defensible floor assumption.'),obs,require_resolved=True)

    def test_real_wall_host_error_and_schema_echo_are_both_reported(self):
        case,_=case_for(BATH);d=read(comp(BATH)/'domain.json');raw=read(comp(BATH)/'calls/programs_h0_repair.json')
        with self.assertRaises(ValueError) as caught:model.programs_answer(raw,case,d['hypotheses'][0],require_concrete=True)
        self.assertIn('placeholder',str(caught.exception));self.assertIn('wall requires host=null',str(caught.exception))

    def test_all_five_floor_choice_domains_use_exact_predicted_bottoms(self):
        for s in read(RUN/'suite.json')['scenes']:
            case,obs=case_for(s['id']);choices=geometry.floor_candidates(obs)
            self.assertTrue(choices)
            known={o['track_id']:o for o in obs['objects']}
            for cid,c in choices.items():
                oid=c['anchor_ids'][0];o=known[oid]
                y=o['center_local_m']['value'][1]-o['size_aabb_local_m']['value'][1]/2
                self.assertEqual(y,c['floor_y_m']);self.assertLess(y,0.)
                p=geometry.floor_choice_answer(dict(choice=cid,basis='Test-only contact assumption, not verified from RGB.'),obs,choices)
                self.assertEqual(p['floor_y_m'],y);self.assertFalse(p['floor_measured'])
            for cid in ('unresolved','not_declared'):
                with self.assertRaises(ValueError):geometry.floor_choice_answer(dict(choice=cid,basis='Unknown floor.'),obs,choices)

    def test_initialization_of_varied_fixed_assemblies_passes_actual_frustum_checks(self):
        import numpy as np
        from .assembly_spatial import instantiate
        from .scene_completion import view_check
        w,obs,_,_=fixture.JointWorldTests.layout_fixture(self);rng=np.random.default_rng(51)
        for trial in range(30):
            raw=dict(groups=[dict(id=g['key'],center_xz_m=rng.uniform(-6.,6.,2).tolist()) for g in w['groups']])
            origins=geometry.origins_answer(raw,w,90)
            dims={o['id']:dict(full_size_m=([float(rng.uniform(.4,1.8)),.6,float(rng.uniform(.4,1.8))] if not o['attachment']['host'] else [.1,.12,.1])) for o in w['objects']}
            poses={o['id']:dict(id=o['id'],yaw_deg=float(rng.uniform(-180.,180.)),
                **({'root_offset_m':[float(rng.uniform(-1.,1.)),0.,float(rng.uniform(-1.,1.))]} if not o['attachment']['host'] else {'host_offset_uv':[.2,.1]})) for o in w['objects']}
            layout=geometry.convert(w,obs,dims,origins,poses,90)
            boxes,_,_,_=instantiate(w,layout,obs,dict(floor_y_m=-1.5),90)
            self.assertEqual(set(boxes),set(dims))
            for b in boxes.values():
                check=view_check(b['center'],b['size'],90,[512,512])
                self.assertFalse(check['intersects_input_box_proxy'],(trial,b,check))
                self.assertFalse(check['camera_inside_box_proxy'])
            for group in layout.values():
                for row in group['objects']:self.assertEqual(row['intrinsic_size_m'],dims[row['id']]['full_size_m'])


if __name__=='__main__':unittest.main()
