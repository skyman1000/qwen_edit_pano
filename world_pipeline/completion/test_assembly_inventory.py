"""Contracts and generative semantics for inventory / installation SCM v2."""
from copy import deepcopy
import unittest

import numpy as np

from . import assembly_inventory as model,assembly_spatial as spatial
from . import assembly_model as legacy
from . import test_assembly as fixtures
from .world_scm import forward
from .conditional_world_geometry import support_plane


class InventoryTests(unittest.TestCase):
    setUp=fixtures.AssemblyTests.setUp
    initialize=fixtures.AssemblyTests.initialize
    def model(self):
        zs=model.hypotheses_answer(fixtures.hypotheses());fs=model.furniture_answer(fixtures.furniture(),self.case,zs[0])
        f={f'f{i}':v for i,v in enumerate(fs)};c={fid:self.joint(v) for fid,v in f.items()}
        return model.build_model(zs,f,c)
    def joint(self,f):
        raw=fixtures.contents(f);gid=raw['bundles'][0]['group'];raw=dict(alternatives=raw['bundles'][0]['alternatives'])
        for a in raw['alternatives']:
            for o in a['objects']:o['group']=gid
        return model.contents_answer(raw,self.case,f)
    def rich_world(self,m=None):
        m=m or self.model();u={n['id']:0. for n in m['nodes']};w=model.active_world(m,forward(m,u));w['noise']=u;return w
    def test_inventory_survives_host_removal_installation_does_not(self):
        m=self.model();w=self.rich_world(m);objects={o['key']:o for o in w['objects']}
        after=model.active_world(m,forward(m,w['noise'],{objects['table']['presence_node']:'0'}))
        bykey={o['key']:o for o in after['objects']}
        self.assertNotIn('table',bykey);self.assertIn('lamp',bykey)
        self.assertFalse(bykey['lamp']['installation_ready']);self.assertTrue(bykey['plant']['installation_ready'])
        nodes={n['id']:n for n in m['nodes']}
        self.assertNotIn(objects['table']['presence_node'],nodes[objects['lamp']['presence_node']]['parents'])
        self.assertIn(objects['table']['installation_node'],nodes[objects['lamp']['installation_node']]['parents'])
        self.assertTrue(model.intervention_diagnostics(m,w)['queries'])
    def test_world_contents_is_one_joint_variable_not_independent_group_counts(self):
        m=self.model();self.assertEqual([n['id'] for n in m['nodes'] if n['id'].startswith('T_')],['T_f0','T_f1'])
        self.assertTrue(all(not w['unresolved_installations'] for w in model.sample_worlds(m)))
        self.assertFalse(m['calibrated_posterior_defined'])
    def test_group_is_derived_from_host_without_cross_group_rejection(self):
        z=model.hypotheses_answer(fixtures.hypotheses())[0]
        f=model.furniture_answer(fixtures.furniture(),self.case,z)[0]
        f['groups'].append(dict(key='g1',function='Storage.',reason='Possible unseen storage.',objects=[fixtures.object_('cabinet','cabinet',aff=['top'])]))
        mug=fixtures.object_('mug','mug','top','cabinet');mug['group']='g0'
        out=model.contents_answer(dict(alternatives=[dict(description='Mug on this exact cabinet.',objects=[mug])]),self.case,f)
        self.assertEqual(out[0]['objects'][0]['group'],'g1');self.assertEqual(out[0]['objects'][0]['attachment']['host'],'cabinet')
    def test_single_installation_source_handles_reported_on_hang(self):
        o=fixtures.object_('cover','loose fabric cover','hang','lamp')
        r=dict(source='cover',target='lamp',kind='on',reason='Attached on this lamp.')
        rel,binding=legacy.relations_answer([r],{'cover','lamp'},[o])
        self.assertEqual(rel,[]);self.assertEqual(o['attachment']['kind'],'hang')
        self.assertFalse(binding[0]['physical_installation_verified'])
    def test_host_local_offset_rotates_by_host_not_child_yaw(self):
        w=self.rich_world();raw=fixtures.metric(w)
        for g in raw['groups']:
            for r in g['objects']:r['offset_frame']='camera'
        objects={o['key']:o for o in w['objects']};rows={o['id']:o for g in raw['groups'] for o in g['objects']}
        rows[objects['table']['id']]['yaw_deg']=-90
        rows[objects['lamp']['id']].update(offset_m=[.25,0,0],offset_frame='host_local',yaw_deg=17)
        layout=spatial.layout_answer(raw,w,90)
        boxes,_,_,_=spatial.instantiate(w,layout,self.observed,support_plane(self.observed,['obs_000']),90)
        delta=np.array(boxes[objects['lamp']['id']]['center'])-boxes[objects['table']['id']]['center']
        self.assertAlmostEqual(delta[0],0.);self.assertAlmostEqual(delta[2],.25)
    def test_numeric_solver_can_separate_roots_in_same_semantic_group(self):
        w=self.rich_world();raw=fixtures.metric(w)
        for g in raw['groups']:
            for r in g['objects']:
                r['offset_frame']='camera'
                if r['id'].endswith('_chair'):r['offset_m']=[0,0,0]
        clean=spatial.layout_answer(raw,w,90);plane=support_plane(self.observed,['obs_000'])
        result=spatial.compile_layout(self.case,self.observed,w,clean,plane)
        self.assertEqual(result['review']['program_solver']['method'],'articulated_support_tree')
        self.assertTrue(result['review']['geometry_contract_passed'])
        self.assertTrue(result['review']['program_solver']['initial_overlaps'])
        self.assertFalse(result['review']['program_solver']['final_overlaps'])
        self.assertTrue(all(r['horizontal_contact_proxy_passed'] is not False for r in result['review']['attachment_review']))
    def test_offset_frame_must_be_explicit(self):
        w=self.rich_world()
        with self.assertRaisesRegex(ValueError,'offset_frame'):spatial.layout_answer(fixtures.metric(w),w,90)
        raw=fixtures.metric(w)
        for g in raw['groups']:
            for r in g['objects']:r['offset_frame']='host_local'
        with self.assertRaisesRegex(ValueError,'Roots'):spatial.layout_answer(raw,w,90)
    def test_empty_furniture_does_not_force_content_or_forbid_independent_items(self):
        zs=model.hypotheses_answer(fixtures.hypotheses());f=dict(description='Existing surroundings with no extra large furniture.',hypothesis=zs[0],groups=[],relations=[])
        plant=fixtures.object_('plant','plant');plant['group']='g_free'
        alts=model.contents_answer(dict(alternatives=[dict(description='A possible floor plant.',objects=[plant]),
            dict(description='Genuinely empty hidden inventory.',objects=[])]),self.case,f)
        m=model.build_model(zs,{'f0':f},{'f0':alts});samples=model.sample_worlds(m)
        self.assertEqual(sorted(len(w['objects']) for w in samples),[0,1])
    def test_missing_host_never_deletes_or_reassigns_item(self):
        f=model.furniture_answer(fixtures.furniture(),self.case,model.hypotheses_answer(fixtures.hypotheses())[0])[0]
        mug=fixtures.object_('mug','mug','top','unknown_table');original=deepcopy(mug)
        with self.assertRaisesRegex(ValueError,'unknown_table'):
            model.contents_answer(dict(alternatives=[dict(description='Invalid host.',objects=[mug])]),self.case,f)
        self.assertEqual(mug,original)
    def test_unresolved_intervention_does_not_export_fake_positions(self):
        m=self.model();w=self.rich_world(m);table=next(o for o in w['objects'] if o['key']=='table')
        after=model.active_world(m,forward(m,w['noise'],{table['presence_node']:'0'}))
        with self.assertRaisesRegex(ValueError,'Unresolved installation'):
            spatial.compile_layout(self.case,self.observed,after,{},support_plane(self.observed,['obs_000']))


if __name__=='__main__':unittest.main()
