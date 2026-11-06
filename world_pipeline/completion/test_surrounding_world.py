"""Regressions for actual sparse pilot and v3 expansion/edit contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ..common import read,write
from .conditional_world_core import inventory_key
from .surrounding_inventory import blueprint_answer,normalize,expand_answer,review_answer,build_catalog
from .surrounding_world import run,code_hashes
from .surrounding_world_audit import audit
from .test_causal_world_completion import make_case


def group():
    return dict(id='g1',description='Possible off-camera storage area',off_camera_region='behind this camera',
                assumption='Room may extend into that region; not observed.',evidence_ids=[])


def obj(oid='n1',name='cabinet',support=None):
    return dict(id=oid,name=name,object_type=name,group_id='g1',reason='A compatible optional hypothesis, not known to exist.',
                placement='Behind the original camera.',support=support or dict(kind='floor',target=None))


def proposal():
    return dict(worlds=[dict(description='One possible existing surrounding storage arrangement.',additions=[obj()])])


def keep(plan):
    return dict(decisions=[dict(id=o['id'],action='retain',reason='Coherent with this assumed world; uncertain.',
                               replacement=None) for o in plan['additions']])


class UnitTests(unittest.TestCase):
    def plan(self):return normalize(proposal(),dict(context=[]),'z1',[group()],2,256)[0]

    def test_actual_broad_category_collision_is_fixed(self):
        worlds=[dict(additions=[dict(object_type='furniture',name=n)])
                for n in ('wall','gray fabric armchair','gray storage ottoman')]
        self.assertEqual(len({inventory_key(w) for w in worlds}),3)

    def test_broad_adapter_category_keeps_specific_open_name(self):
        value=proposal();value['worlds'][0]['additions'][0]['object_type']='furniture'
        plan=normalize(value,dict(context=[]),'z1',[group()],2,256)[0]
        self.assertEqual(plan['additions'][0]['object_type'],'cabinet')

    def test_room_envelope_is_not_a_furniture_instance(self):
        value=proposal();value['worlds'][0]['additions'][0].update(name='wall',object_type='furniture')
        with self.assertRaisesRegex(ValueError,'envelope'):normalize(value,dict(context=[]),'z1',[group()],2,256)

    def test_attachment_uses_existing_unmodeled_branch_without_losing_declaration(self):
        for kind in ('wall','ceiling'):
            value=proposal();value['worlds'][0]['additions'][0]['support']=dict(kind=kind,target=None)
            before=deepcopy(value);plan=normalize(value,dict(context=[]),'z1',[group()],2,256)[0]
            o=plan['additions'][0]
            self.assertEqual(o['support'],dict(kind='unspecified',target=None))
            self.assertEqual(o['support_declaration'],dict(kind=kind,target=None))
            answer=dict(additions=[],coverage=[dict(group_id='g1',reason='Considered.')],stop_reason='Stop.')
            self.assertEqual(expand_answer(answer,plan,dict(context=[]),'z1',256),plan)
            self.assertEqual(value,before)

    def test_attachment_cannot_disguise_invalid_instance_reference(self):
        value=proposal();value['worlds'][0]['additions'][0]['support']=dict(kind='wall',target='invented_wall')
        with self.assertRaisesRegex(ValueError,'target must be null'):
            normalize(value,dict(context=[]),'z1',[group()],2,256)

    def test_expansion_preserves_old_objects_and_support_parents(self):
        p=self.plan();before=deepcopy(p)
        answer=dict(additions=[obj('n2','table lamp',dict(kind='object',target=p['additions'][0]['id']))],
                    coverage=[dict(group_id='g1',reason='Considered furniture and new supporting surfaces.')],stop_reason='No further coherent additions proposed.')
        result=expand_answer(answer,p,dict(context=[]),'z1',256)
        self.assertEqual(result['additions'][0],p['additions'][0]);self.assertEqual(p,before)
        self.assertEqual(result['additions'][1]['support']['target'],'z1_w1_n1')

    def test_no_forced_count_or_occupancy(self):
        p=self.plan();answer=dict(additions=[],coverage=[dict(group_id='g1',reason='No further coherent items.')],stop_reason='Stop.')
        self.assertEqual(expand_answer(answer,p,dict(context=[]),'z1',256),p)
        self.assertEqual(blueprint_answer(dict(groups=[],extent_assumption='No further groups hypothesized.'),dict(context=[]))['groups'],[])

    def test_coverage_cannot_be_silently_omitted(self):
        with self.assertRaisesRegex(ValueError,'every group'):
            expand_answer(dict(additions=[],coverage=[],stop_reason='Stop.'),self.plan(),dict(context=[]),'z1',256)

    def test_review_cannot_silently_drop_instance(self):
        with self.assertRaisesRegex(ValueError,'each instance'):
            review_answer(dict(decisions=[]),self.plan(),dict(context=[]),'z1',256)

    def test_uncertainty_alone_is_not_a_withdrawal_ground(self):
        p=self.plan();answer=keep(p);answer['decisions'][0].update(action='withdraw',grounds='uncertain',evidence_ids=[])
        with self.assertRaisesRegex(ValueError,'substantive'):review_answer(answer,p,dict(context=[]),'z1',256)

    def test_removed_support_parent_needs_explicit_child_revision(self):
        value=proposal();value['worlds'][0]['additions'].append(obj('n2','lamp',dict(kind='object',target='n1')))
        p=normalize(value,dict(context=[]),'z1',[group()],2,256)[0];answer=keep(p)
        answer['decisions'][0].update(action='withdraw',grounds='incompatible_world',evidence_ids=[])
        with self.assertRaises(ValueError):review_answer(answer,p,dict(context=[]),'z1',256)

    def test_second_world_ids_stay_stable_after_expansion(self):
        value=proposal();value['worlds']*=2
        p=normalize(value,dict(context=[]),'z1',[group()],2,256)[1]
        answer=dict(additions=[],coverage=[dict(group_id='g1',reason='Considered.')],stop_reason='Stop.')
        result=expand_answer(answer,p,dict(context=[]),'z1',256)
        self.assertEqual(result,p);self.assertEqual(result['id'],'z1_w2')


class FixtureModel:
    def __init__(self,anchor):self.anchor=anchor;self.calls=[]
    def ask(self,image,prompt,path,**options):
        self.calls.append(path.stem)
        if path.stem=='contexts':
            v=dict(contexts=[dict(description='Possible living room use.',reason='An assumed explanation.',evidence_ids=[])],floor_anchor_ids=[self.anchor])
        elif path.stem.startswith('blueprint'):
            v=dict(groups=[group()],extent_assumption='Unobserved space may extend behind this camera.')
        elif path.stem.startswith('contents'):v=proposal()
        elif path.stem.startswith('expand'):
            v=dict(additions=[obj('n2','table lamp',dict(kind='object',target='z1_w1_n1'))],
                   coverage=[dict(group_id='g1',reason='Furniture and its off-camera support surface considered.')],stop_reason='No further proposals.')
        elif path.stem.startswith('review'):
            v=dict(decisions=[dict(id='z1_w1_n'+str(i),action='retain',reason='Compatible but uncertain.',replacement=None) for i in (1,2)])
        else:
            v=dict(objects=[dict(id='z1_w1_n1',azimuth_deg=180.,range_m=3.,size=[1.,1.,.5],reason='Off-camera cabinet.'),
                            dict(id='z1_w1_n2',azimuth_deg=180.,range_m=3.,size=[.2,.4,.2],reason='Lamp above cabinet.')])
        path.write_text(json.dumps(v));write(path.with_suffix('.json'),v)
        write(path.with_suffix('.request.json'),dict(prompt=prompt,**options));return v


class FixtureScorer:
    def score(self,description,parents,states):
        values=[1.] if len(states)==1 else [.8,.2]
        return dict(probabilities=dict(zip([s['id'] for s in states],values)),
                    permutation_max_total_variation=0.,synthetic_test=True)


class IntegrationTests(unittest.TestCase):
    def test_end_to_end_expansion_review_export_visualization_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,observed=make_case(root);case['image_size']=[64,64]
            out=root/'out';out.mkdir();model=FixtureModel(case['context'][0]['id'])
            record=run(model,lambda *args:FixtureScorer(),case,out,context_budget=1,per_context=1,worlds=2)
            self.assertEqual(record['status'],'candidates_exported_needs_review',record)
            counts=record['proposal_coverage']['stage_counts'][0]
            self.assertEqual([counts[k] for k in ('initial','expanded','reviewed')],[1,2,2])
            full=read(out/'case/G_full.json')
            self.assertEqual(full['objects'][:len(observed['objects'])],observed['objects'])
            self.assertEqual(len(full['objects']),len(observed['objects'])+2)
            self.assertTrue((out/'case/hypothesis_01/world/scene_3d.html').is_file())
            write(out/'config.json',dict(case=case,code_sha256=code_hashes(),
                arguments=dict(worlds=2,per_context=1,max_objects=256,proposal_mode='expanded')))
            self.assertEqual(audit(out)['status'],'postflight_contract_passed')
            catalog=read(out/'case/world_catalog.json');catalog['worlds'].pop(0)
            write(out/'case/world_catalog.json',catalog)
            with self.assertRaises(ValueError):audit(out)

    def test_legacy_proposals_no_longer_merge_distinct_or_same_category_worlds(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);case=dict(context=[],hfov=90.)
            class Legacy:
                def ask(self,image,prompt,path,**options):
                    value=dict(worlds=[dict(description='Alternative '+str(i),additions=[obj(name=name)])
                                      for i,name in enumerate(('armchair','ottoman'))])
                    for w in value['worlds']:w['additions'][0]['object_type']='furniture'
                    path.write_text(json.dumps(value));return value
            catalog=build_catalog(Legacy(),None,case,[dict(id='z1',description='possible room')],{},folder,
                                  dict(seed=1),2,256,mode='legacy')
            self.assertEqual([w['id'] for w in catalog],['z1_w1','z1_w2','empty'])
            self.assertFalse(read(folder/'proposal_coverage.json')['category_based_deduplication'])


if __name__=='__main__':unittest.main()
