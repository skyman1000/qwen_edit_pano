"""Synthetic CPU verification; no paid API, GPU model, or quality claims."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ..common import read,write,sha
from . import assembly_model as scm,assembly_layout as layout
from .assembly_experiment import initialize,check,restart,SeededProvider
from .assembly_world import run
from .assembly_review import review,suite
from . import test_fresh as fresh_tests
from .fresh_observer import load_case
from .fresh_pipeline import DEFAULT_MODEL
from .conditional_world_geometry import support_plane
from .world_scm import forward
from .world_generation import SYSTEM_PROMPT,MATCHED_OPTIONS,image_bytes
from .world_scm_runner import ask_valid


def object_(key,name,kind='floor',host=None,aff=()):
    return dict(key=key,name=name,category=name,variant=name,distinct_instance_reason='Separate existing unseen instance.',
        placement_hint='Behind camera in same surrounding room.',attachment=dict(kind=kind,host=host),affordances=list(aff))


def hypotheses():return dict(hypotheses=[dict(description='Existing living and reading room.',functions=['reading','relaxing'],assumptions=['An unphotographed activity area.'])])


def furniture():
    objects=[object_('table','side table',aff=['top']),object_('chair','chair',aff=['seat']),
        object_('shelf','bookshelf',aff=['shelf','top']),object_('hooks','coat hooks','wall',aff=['hanging'])]
    return dict(alternatives=[dict(description='Joint reading and storage area.',groups=[dict(key='reading',function='reading and storage',reason='Possible existing surrounding area.',objects=objects)],
        relations=[dict(source='chair',target='table',kind='near',reason='Coherent functional pair.')]),
        dict(description='Separate wardrobe storage area.',groups=[dict(key='storage',function='storage',reason='Possible storage.',objects=[object_('wardrobe','wardrobe',aff=['shelf'])])],relations=[])])


def contents(f):
    if any(o['key']=='table' for g in f['groups'] for o in g['objects']):
        plant=object_('plant','plant');plant['category']='fixture_unmapped_noun'
        objects=[object_('lamp','table lamp','top','table'),object_('book','book','shelf','shelf'),
            object_('bag','bag','hang','hooks'),object_('cushion','cushion','seat','chair'),plant]
        alts=[dict(description='Lived-in installed contents.',objects=objects,relations=[
            dict(source='lamp',target='table',kind='on',reason='Same already declared tabletop host.'),
            dict(source='plant',target='chair',kind='next_to',reason='Nearby independent accessory.')]),
            dict(description='Only a book on storage.',objects=[object_('book','book','shelf','shelf')],relations=[])]
    else:alts=[dict(description='Bare storage exterior.',objects=[],relations=[])]
    return dict(bundles=[dict(group=f['groups'][0]['key'],alternatives=alts)])


def metric(world,angle=180,radius=3):
    dims={'table':[1,.7,.6],'chair':[.5,.9,.5],'shelf':[.7,2,.4],'hooks':[.4,.1,.1],
        'wardrobe':[1.3,2,.6],'lamp':[.2,.4,.2],'book':[.2,.2,.15],'bag':[.3,.4,.1],
        'cushion':[.4,.1,.4],'plant':[.3,.6,.3]}
    offsets={'table':[0,0,0],'chair':[1.2,0,0],'shelf':[-1.5,0,0],'hooks':[0,1.2,1.2],
        'wardrobe':[0,0,0],'plant':[2.,0,0]}
    groups=[]
    for gid,objects in layout.grouped(world).items():
        rows=[]
        for o in objects:
            k=o['key'];s=dims[k]
            rows.append(dict(id=o['id'],width_m=s[0],height_m=s[1],depth_m=s[2],yaw_deg=0,
                offset_m=offsets.get(k,[0,0,0]),surface_fraction=.45 if o['attachment']['kind'] in ('shelf','seat') else None,
                hanging_drop_m=.05 if o['attachment']['kind']=='hang' else None))
        groups.append(dict(id=gid,azimuth_deg=angle,range_m=radius,objects=rows))
    return dict(groups=groups)


class GrammarFixture:
    execution_kind='synthetic_CPU_fixture';generation_profile='matched_v1'
    def __init__(self):self.furniture={};self.worlds={}
    def ask(self,image,prompt,path,seed=3407,max_tokens=8192,**kw):
        import hashlib
        path=Path(path)
        write(path.with_suffix('.request.json'),dict(prompt=prompt,system_prompt=SYSTEM_PROMPT,seed=seed,max_tokens=max_tokens,
            generation=MATCHED_OPTIONS,generation_profile='matched_v1',canonical_image_sha256=hashlib.sha256(image_bytes(image)).hexdigest() if image is not None else None))
        if path.stem=='hypotheses':value=hypotheses()
        elif path.stem.startswith('furniture_'):
            value=furniture()
            if prompt.startswith('Complete the UNOBSERVED surrounding world'):
                # Same two concrete joint scene alternatives, with contents
                # already included. No second content proposal in this model.
                z=scm.hypotheses_answer(hypotheses())[0]
                normalized=scm.furniture_answer(value,dict(context=[]),z)
                for raw,f in zip(value['alternatives'],normalized):
                    extras=contents(f)['bundles'][0]['alternatives'][0]
                    raw['relations']+=extras['relations']
                    raw['groups'][0]['objects']+=extras['objects']
                    for group in raw['groups']:
                        group['observed_refs']=[];group['new_objects']=group.pop('objects')
                        for o in group['new_objects']:
                            o['reason']=o.pop('distinct_instance_reason');o['placement']=o.pop('placement_hint')
        elif path.stem.startswith('contents_'):
            payload=json.loads(prompt.split('DECLARED PARENTS (whole configuration, furniture variants, host affordances, fixed observed identities): ')[1])
            value=contents(payload['furniture'])
            if prompt.startswith('Propose 1..2 alternative COMPLETE JOINT inventories'):
                gid=value['bundles'][0]['group'];value=dict(alternatives=value['bundles'][0]['alternatives'])
                for alt in value['alternatives']:
                    for o in alt['objects']:o['group']=gid
        elif path.stem.startswith('layout_'):
            fixed=json.loads(prompt.split('FIXED SEMANTIC WORLD: ')[1].split('\nFROZEN OBSERVATIONS:')[0])
            # Reconstruct only the wire's grouping metadata, not private state.
            for o in fixed['objects']:o['key']=next(k for k in ('table','chair','shelf','hooks','wardrobe','lamp','book','bag','cushion','plant') if o['id'].endswith('_'+k))
            fixed['groups']=[dict(key=g) for g in dict.fromkeys(o['group'] for o in fixed['objects'])]
            value=metric(fixed)
            if prompt.startswith('Supply relative metric attributes'):
                for g in value['groups']:
                    g.pop('azimuth_deg');g['region']='rear'
                    for o in g['objects']:
                        o['full_size_m']=[o.pop(k) for k in ('width_m','height_m','depth_m')]
            elif 'offset_frame' in prompt:
                for g in value['groups']:
                    for o in g['objects']:o['offset_frame']='camera'
        else:raise AssertionError('Unexpected query '+path.stem)
        path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return deepcopy(value)


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.base=fresh_tests.FreshTests();self.base.setUp();self.addCleanup(self.base.doCleanups)
        self.temp=Path(self.base.root);self.image=self.base.image
        self.root=self.initialize(self.temp/'new')
        self.base.observe(self.root,'qwen');self.case=load_case(self.root,'qwen');self.observed=read(self.case['state'])
    def initialize(self,out,source=None):
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
            patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            return initialize(out,self.image,90,DEFAULT_MODEL,'synthetic_CPU_fixture',observations_from=source)
    def model(self):
        zs=scm.hypotheses_answer(hypotheses());fs=scm.furniture_answer(furniture(),self.case,zs[0]);f={f'f{i}':v for i,v in enumerate(fs)}
        c={fid:scm.contents_answer(contents(v),self.case,v) for fid,v in f.items()}
        return scm.build_model(zs,f,c)
    def rich_world(self,m=None):
        m=m or self.model();u={n['id']:0. for n in m['nodes']};w=scm.active_world(m,forward(m,u));w['noise']=u;return w
    def test_host_intervention_changes_only_causal_descendants(self):
        m=self.model();w=self.rich_world(m);objects={o['key']:o for o in w['objects']}
        state=forward(m,w['noise'],{objects['table']['presence_node']:'0'});after=scm.active_world(m,state);keys={o['key'] for o in after['objects']}
        self.assertNotIn('table',keys);self.assertNotIn('lamp',keys)
        self.assertTrue({'plant','book','bag','cushion','chair','shelf'}<=keys)
        self.assertEqual(forward(m,w['noise']),w['structural_state'])
        self.assertTrue(scm.intervention_diagnostics(m,w)['queries'])
    def test_furniture_intervention_recomputes_contents(self):
        m=self.model();w=self.rich_world(m);s=forward(m,w['noise'],{'F':'f1'});out=scm.active_world(m,s)
        self.assertEqual([o['key'] for o in out['objects']],['wardrobe'])
        self.assertFalse(m['calibrated_posterior_defined']);self.assertFalse(m['RGB_likelihood_defined'])
    def test_attachment_heights_and_host_envelope_overlap(self):
        w=self.rich_world();answer=layout.layout_answer(metric(w),w,90);plane=support_plane(self.observed,['obs_000'])
        result=layout.compile_layout(self.case,self.observed,w,answer,plane);items={o['id']:o for o in result['items']};objects={o['key']:o for o in w['objects']}
        bag=items[objects['bag']['id']];hooks=items[objects['hooks']['id']]
        self.assertAlmostEqual(bag['center'][1]+bag['size'][1]/2,hooks['center'][1]-.05)
        book=items[objects['book']['id']];shelf=items[objects['shelf']['id']]
        self.assertLess(book['center'][1],shelf['center'][1]+shelf['size'][1]/2)
        self.assertTrue(result['review']['program_solver']['intentional_host_envelope_overlaps'])
        self.assertTrue(result['review']['geometry_contract_passed'])
    def test_rigid_collision_repair_preserves_internal_offsets(self):
        w=self.rich_world();answer=layout.layout_answer(metric(w),w,90);plane=support_plane(self.observed,['obs_000'])
        boxes,_,cert,fixed=layout.instantiate(w,answer,self.observed,plane,90);contracts={o['id']:o for o in w['objects']}
        obstacle=deepcopy(self.observed);obstacle['objects'][0]['center_local_m']['value']=boxes[w['objects'][0]['id']]['center'][:]
        repaired,review_=layout.rigid_repair(self.case,obstacle,boxes,contracts,cert,fixed,w['relations'])
        self.assertTrue(review_['initial_overlaps'])
        ref=w['objects'][0]['id']
        for oid in boxes:
            np.testing.assert_allclose(np.array(repaired[oid]['center'])-repaired[ref]['center'],np.array(boxes[oid]['center'])-boxes[ref]['center'],atol=1e-6)
    def test_off_frustum_parameterization_and_identity(self):
        w=self.rich_world();plane=support_plane(self.observed,['obs_000'])
        with self.assertRaises(ValueError):layout.layout_answer(metric(w,angle=0),w,90)
        a=layout.layout_answer(metric(w,angle=80,radius=.01),w,90);result=layout.compile_layout(self.case,self.observed,w,a,plane)
        self.assertTrue(result['review']['geometry_contract_passed'])
        self.assertEqual(len(result['items']),len(w['objects']))
    def test_attachment_cycles_and_unknown_hosts_rejected(self):
        z=scm.hypotheses_answer(hypotheses())[0];f=furniture();f['alternatives'][0]['groups'][0]['objects'][0]['attachment']={'kind':'top','host':'lamp'}
        with self.assertRaises(ValueError):scm.furniture_answer(f,self.case,z)
    def test_complete_two_comparisons_and_visualization(self):
        self.base.observe(self.root,'api',True)
        for backend in ('api','qwen'):
            s=run(GrammarFixture(),self.root,backend)
            self.assertEqual(s['status'],'worlds_exported_needs_review',s)
            self.assertEqual(len(s['worlds']),2)
            for w in s['worlds']:
                p=self.root/backend/'completion'/w['folder'];full=read(p/'G_full.json');observed=read(self.root/backend/'observer3d/G_obs.json')
                self.assertEqual(full['objects'][:len(observed['objects'])],observed['objects'])
                self.assertEqual(len(full['objects'])-len(observed['objects']),w['new_instances'])
                with Image.open(p/'world/erp_state_map.png') as im:self.assertEqual(im.width,2*im.height)
        a=review(self.root);self.assertFalse(read(a.parent/'review.json')['observed_state_SHA_equal'])
        self.assertTrue(read(a.parent/'review.json')['common_public_parameters_passed'])
        html=a.read_text()
        self.assertEqual(html.count('<iframe class="scene"'),4)
        self.assertLess(html.index('<iframe'),html.index('<pre>'))
        self.assertIn('<details><summary>状态字段',html)
        self.assertIn('/world/layout_xz.png',html)
        self.assertIn('/world/erp_state_map.png',html)
        controlled=self.initialize(self.temp/'controlled',self.root)
        self.assertEqual(sha(controlled/'api/observer3d/G_obs.json'),sha(controlled/'qwen/observer3d/G_obs.json'))
        for b in ('api','qwen'):self.assertEqual(run(GrammarFixture(),controlled,b)['status'],'worlds_exported_needs_review')
        combined=suite(self.root,controlled);self.assertTrue(combined.is_file());check(controlled)
        self.assertEqual(combined.read_text().count('<iframe class="scene"'),8)
        digest=sha(combined);second=suite(self.root,controlled)
        self.assertNotEqual(second,combined);self.assertEqual(sha(combined),digest)
    def test_no_per_group_automatic_absence(self):
        m=self.model()
        for node in m['nodes']:
            if node['id'].startswith('T_'):
                for fid,ps in node['cpt'].items():
                    self.assertEqual(ps[0],0. if fid in node['id'] else 1.)
    def test_experiment_seed_is_recorded(self):
        class Recorder:
            def ask(self,*args,**kwargs):return kwargs
        p=SeededProvider(Recorder(),9000)
        self.assertEqual(p.ask(None,'',None,seed=3407)['seed'],9000)
        self.assertEqual(p.ask(None,'',None,seed=3408)['seed'],9001)
    def test_geometry_failure_preserves_sample(self):
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_world.geometry.compile_layout',side_effect=ValueError('Synthetic numerical failure')):
            summary=run(GrammarFixture(),self.root,'qwen')
        self.assertEqual(summary['status'],'no_world_exported');self.assertEqual(len(summary['failures']),2)
        self.assertTrue(all(f['proposed_instances']>0 for f in summary['failures']))
        self.assertTrue((self.root/'qwen/completion/structural_samples.json').is_file())
    def test_empty_hypothesis_remains_a_valid_candidate(self):
        z=scm.hypotheses_answer(hypotheses());f={'f0':dict(description='An actual empty surroundings hypothesis.',hypothesis=z[0],groups=[],relations=[])}
        m=scm.build_model(z,f,{'f0':{}});worlds=scm.sample_worlds(m,count=3)
        self.assertEqual(len(worlds),1);self.assertEqual(worlds[0]['objects'],[])
        solved=layout.compile_layout(self.case,self.observed,worlds[0],{},support_plane(self.observed,['obs_000']))
        self.assertTrue(solved['review']['geometry_contract_passed'])

    def test_contents_prompt_explicitly_matches_spatial_contract(self):
        f=scm.furniture_answer(furniture(),self.case,scm.hypotheses_answer(hypotheses())[0])[0]
        prompt=scm.contents_prompt(self.case,f)
        self.assertIn(scm.RELATION_CONTRACT,prompt)
        self.assertIn('ONLY NEW content',prompt)
        self.assertIn('not a table or shelf',prompt)
        self.assertEqual(scm.OBJECT_SCHEMA['affordances'],[])

    def test_real_relation_spellings_bind_without_changing_identity(self):
        rows=[dict(source='mug_001',target='book_001',kind=k,reason='Adjacency.') for k in ('beside','next_to','adjacent_to')]
        for row in rows:
            original=deepcopy(row);out,bindings=scm.relations_answer([row],{'mug_001','book_001'})
            self.assertEqual(out,[{**row,'kind':'near'}]);self.assertEqual(row,original)
            self.assertEqual(bindings[0]['original_kind'],row['kind'])
            self.assertEqual(bindings[0]['canonical_kind'],'near')

    def test_support_relation_is_checked_against_attachment_not_above(self):
        lamp=object_('lamp','table lamp','top','table')
        r=dict(source='lamp',target='table',kind='on',reason='Already declared tabletop attachment.')
        out,bindings=scm.relations_answer([r],{'lamp','table'},[lamp])
        self.assertEqual(out,[]);self.assertEqual(bindings[0]['attachment'],lamp['attachment'])
        for kind in ['floor','free']:
            invalid=deepcopy(lamp);invalid['attachment']=dict(kind=kind,host=None if kind!='hang' else 'table')
            with self.assertRaisesRegex(ValueError,'must agree'):
                scm.relations_answer([r],{'lamp','table'},[invalid])
        hanging=deepcopy(lamp);hanging['attachment']=dict(kind='hang',host='table')
        self.assertEqual(scm.relations_answer([r],{'lamp','table'},[hanging])[0],[])
        with self.assertRaisesRegex(ValueError,'must agree'):
            scm.relations_answer([{**r,'kind':'on_top_of'}],{'lamp','table'},[hanging])
        with self.assertRaisesRegex(ValueError,'must agree'):
            scm.relations_answer([{**r,'target':'chair'}],{'lamp','chair','table'},[lamp])

    def test_bad_relation_errors_identify_path_and_choices(self):
        r=dict(source='mug',target='book',kind='unknown',reason='Test.')
        with self.assertRaisesRegex(ValueError,r'bundles\[g0\].*unsupported kind.*Allowed spatial kinds'):
            scm.relations_answer([r],{'mug','book'},path='bundles[g0].relations')
        with self.assertRaisesRegex(ValueError,'invalid endpoints.*DISTINCT exact IDs'):
            scm.relations_answer([{**r,'target':'guessed_book'}],{'mug','book'})
        for invalid in (None,[],{},123):
            with self.assertRaises(ValueError):scm.relations_answer([{**r,'source':invalid}],{'mug','book'})

    def test_cross_group_hosts_report_all_invalid_alternatives(self):
        f=scm.furniture_answer(furniture(),self.case,scm.hypotheses_answer(hypotheses())[0])[0]
        f['groups'].append(dict(key='g1',function='Independent lighting.',reason='Possible surroundings.',
            objects=[object_('foreign_lamp','floor lamp')]))
        raw=contents(f);raw['bundles'].append(dict(group='g1',alternatives=[dict(description='Bare surroundings.',objects=[],relations=[])]))
        for a in raw['bundles'][0]['alternatives']:
            book=next(o for o in a['objects'] if o['key']=='book')
            book['attachment']=dict(kind='top',host='foreign_lamp')
        original=deepcopy(raw)
        with self.assertRaises(ValueError) as ctx:scm.contents_answer(raw,self.case,f)
        error=str(ctx.exception)
        self.assertIn('bundles[g0].alternatives[0]',error);self.assertIn('bundles[g0].alternatives[1]',error)
        self.assertIn('foreign_lamp',error);self.assertIn('IN THIS GROUP/ALTERNATIVE',error)
        self.assertEqual(raw,original)

    def test_content_collision_error_identifies_existing_records(self):
        with self.assertRaisesRegex(ValueError,'existing IDs.*table.*NEW content'):
            scm.objects_answer([object_('table','table')],set(),{'table'},path='bundles[g0].objects')

    def test_interface_retry_receives_actionable_error(self):
        r=dict(source='mug',target='book',kind='in_the_middle',reason='Test.')
        class Provider:
            def __init__(self):self.prompts=[]
            def ask(self,image,prompt,path,**kw):
                self.prompts.append(prompt)
                value=dict(relations=[r if len(self.prompts)==1 else {**r,'kind':'near'}])
                path.write_text(json.dumps(value));return value
        q=Provider();out=ask_valid(q,None,scm.RELATION_CONTRACT,self.temp/'test_relation.txt',
            lambda v:scm.relations_answer(v['relations'],{'mug','book'}))
        self.assertEqual(len(q.prompts),2);self.assertIn("unsupported kind 'in_the_middle'",q.prompts[1])
        self.assertIn('Allowed spatial kinds',q.prompts[1]);self.assertEqual(out[0][0]['kind'],'near')

    def test_restart_preserves_both_observers_without_reusing_semantics(self):
        self.base.observe(self.root,'api',True)
        write(self.root/'qwen/observer3d/summary.json',dict(status='completed',note='Fixture'))
        write(self.root/'qwen/saved_failure.json',dict(status='failed',error='Original preserved.'))
        digests={str(p.relative_to(self.root)):sha(p) for p in self.root.rglob('*') if p.is_file()}
        out=self.temp/'restart'
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            restart(out,self.root)
        config=check(out);self.assertFalse(config['observation_reuse']['semantic_mechanisms_reused'])
        self.assertEqual(config['experiment_kind'],check(self.root)['experiment_kind'])
        for backend in ('api','qwen'):
            self.assertFalse((out/backend/'completion').exists())
            for name,h in config['observation_reuse']['observer_file_sha256'][backend].items():
                self.assertEqual(sha(out/backend/name),h)
        self.assertNotEqual(sha(out/'api/observer3d/G_obs.json'),sha(out/'qwen/observer3d/G_obs.json'))
        for name,h in digests.items():self.assertEqual(sha(self.root/name),h)

    def test_failed_world_review_does_not_invent_boxes(self):
        self.base.observe(self.root,'api',True)
        for b in ('api','qwen'):
            p=self.root/b/'completion';p.mkdir()
            write(p/'summary.json',dict(status='failed',worlds=[],failed_stage='contents_f2',error='Invalid relation endpoints/kind'))
        p=review(self.root);html=p.read_text()
        self.assertNotIn('<iframe',html);self.assertIn('没有生成 G_full 三维布局',html)
        self.assertIn('contents_f2',html)


if __name__=='__main__':unittest.main()
