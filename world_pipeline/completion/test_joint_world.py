"""CPU invariants and synthetic full-pipeline execution, never model quality evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ..common import ROOT,read,write,sha
from . import joint_world as runner,joint_world_model as model,joint_world_geometry as geometry
from .world_scm import forward

SOURCE=ROOT/'qwen_edit_pano/outputs/local_observation_fixed_002'


def obj(key,host=None):
    return dict(key=key,name='side table' if host is None else 'mug',category='table' if host is None else 'cup',
        variant='small wooden side table' if host is None else 'ceramic mug',distinct_instance_reason='A distinct unseen instance.',
        placement_hint='Belongs to this hypothesized activity zone.',
        attachment=dict(kind='floor' if host is None else 'top',host=host),affordances=['top'] if host is None else [])


def hypotheses():
    return dict(hypotheses=[dict(description='Two coexisting reading places outside the photographed view.',
        functions=['reading'],assumptions=['Unobserved seating may exist.'],floor_y_m=-1.5,
        floor_basis='Explicit uncertain camera-Y level hypothesis.',bounds_xz_m=None,zones=[
        dict(id='z1',function='reading',reason='Independent place at the right rear.',center_xz_m=[3.,-2.],observed_refs=[]),
        dict(id='z2',function='reading',reason='Another place at the left rear.',center_xz_m=[-3.,-4.],observed_refs=[])])])


def programs_reply():
    groups=[]
    for i in ('1','2'):
        items=[]
        for v in (obj('r'+i),obj('c'+i,'r'+i)):
            v=dict(v,reason=v['distinct_instance_reason'],placement=v['placement_hint'])
            items.append(v)
        groups.append(dict(function='reading',reason='A distinct plausible place.',observed_refs=[],new_objects=items))
    return dict(alternatives=[dict(description='Both places coexist as a complete delta.',groups=groups,
        relations=[dict(source='r1',target='r2',kind='right_of',reason='Continuous global layout order.')])])


def after(prompt,marker):return json.JSONDecoder().raw_decode(prompt.split(marker,1)[1].lstrip())[0]


class CPUProvider:
    def reply(self,prompt,name):
        if name=='hypotheses':answer=hypotheses()
        elif name.startswith('programs_'):answer=programs_reply()
        elif '_sizes_' in name:
            template=after(prompt,'DIMENSION_TEMPLATE: ')
            answer=dict(objects=[dict(id=o['id'],full_size_m=[.8,.6,.5] if o['id'].endswith(('_r1','_r2')) else [.1,.12,.1]) for o in template['objects']])
        elif name.startswith('recognition_'):answer=dict(choice='h0',reason='Image-compatible alternative.')
        elif name.startswith('program_h0_'):answer=dict(choice='f_h0_0',reason='The whole joint arrangement is plausible.')
        elif name.endswith('_floor'):
            if '\nFLOOR_CHOICES: ' in prompt:
                choices=after(prompt,'\nFLOOR_CHOICES: ')
                answer=dict(choice=next(iter(choices)),basis='CPU fixture assumes this observed bottom contacts the floor; not a quality inference.')
            else:answer=dict(anchor_ids=[],floor_y_m=-1.5,basis='Explicit uncertain synthetic floor hypothesis.')
        elif '_origins_' in name:
            w=after(prompt,'\nWORLD: ')
            template=after(prompt,'\nORIGIN_TEMPLATE: ');wanted={r['id'] for r in template['groups']}
            answer=dict(groups=[dict(id=g['key'],center_xz_m=[3.,-2.] if i==0 else [-3.,-4.]) for i,g in enumerate(w['groups']) if g['key'] in wanted])
        elif '_poses_' in name:
            contracts=after(prompt,'\nREQUIRED_FIELDS: ');placements=[]
            for oid,fields in contracts.items():
                r=dict(id=oid,yaw_deg=25.)
                if 'root_offset_m' in fields:r['root_offset_m']=[0.,0.,0.]
                else:r['host_offset_uv']=[.2,.1]
                placements.append(r)
            answer=dict(placements=placements)
        else:raise AssertionError('Unexpected local-model request: '+name)
        return answer

    def ask(self,image,prompt,path,**kwargs):
        answer=self.reply(prompt,Path(path).stem)
        # Like the real client, this fixture requires the caller to own mkdir.
        # Creating parents here previously concealed a production I/O failure.
        Path(path).write_text(json.dumps(answer))
        return answer


def transport_fixture():
    """Run the REAL client for every full-pipeline call; replace only its engine."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    import numpy as np
    from .world_generation import MatchedLocalJSON
    fixture=CPUProvider()
    class Batch(dict):
        @property
        def input_ids(self):return self['input_ids']
        def to(self,device):return self
    class Processor:
        def apply_chat_template(self,messages,**kwargs):
            self.prompt=messages[-1]['content'][-1]['text'];return self.prompt
        def __call__(self,**kwargs):return Batch(input_ids=np.array([[1,2]]))
        def batch_decode(self,*args,**kwargs):
            p=self.prompt
            if 'DIMENSION_TEMPLATE: ' in p:name='world_01_sizes_0'
            elif '\nREQUIRED_FIELDS: ' in p:name='world_01_poses_0'
            elif p.startswith('Propose a continuous origin'):name='world_01_origins_g0'
            elif p.startswith('Resolve a camera-Y-level FLOOR'):name='world_01_floor'
            elif '\nALTERNATIVES: ' in p:
                choices=after(p,'\nALTERNATIVES: ')
                name='recognition_0' if 'h0' in choices else 'program_h0_0'
            elif p.startswith('Propose 1..2 concrete COMPLETE hidden-world deltas'):name='programs_h0'
            else:name='hypotheses'
            return [json.dumps(fixture.reply(p,name))]
    class Model:
        device='cpu';generation_config=SimpleNamespace(eos_token_id=9)
        def generate(self,**kwargs):return np.array([[1,2,3,9]])
    engine=SimpleNamespace(processor=Processor(),model=Model(),
        torch=SimpleNamespace(manual_seed=lambda seed:None,inference_mode=nullcontext))
    return MatchedLocalJSON('/unused/CPU_engine_only',engine=engine)


class JointWorldTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='joint_world_cpu_',dir=ROOT/'research/audits');self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.case=dict(context=[],hfov=90,image_size=[512,512],observed_support_domains={})
        self.h=model.hypotheses_answer(hypotheses(),self.case)[0]
        self.p=model.programs_answer(programs_reply(),self.case,self.h)['valid'][0]

    def test_actual_local_client_writes_fresh_call_directory_and_reuses_cache(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        import numpy as np
        from PIL import Image
        from .world_generation import MatchedLocalJSON
        generations=[]
        class Batch(dict):
            @property
            def input_ids(inner):return inner['input_ids']
            def to(inner,device):return inner
        class Processor:
            def apply_chat_template(inner,*args,**kwargs):return 'CPU transport fixture'
            def __call__(inner,**kwargs):return Batch(input_ids=np.array([[1,2]]))
            def batch_decode(inner,*args,**kwargs):return ['{"ok":true}']
        class Model:
            device='cpu';generation_config=SimpleNamespace(eos_token_id=9)
            def generate(inner,**kwargs):
                generations.append(1);return np.array([[1,2,3,9]])
        engine=SimpleNamespace(processor=Processor(),model=Model(),
            torch=SimpleNamespace(manual_seed=lambda seed:None,inference_mode=nullcontext))
        # Real transport, request/raw/parsed/generation writes and cache; only
        # the numerical model engine is replaced. No checkpoint or API calls.
        client=MatchedLocalJSON('/unused/CPU_engine_only',engine=engine)
        calls=self.base/'fresh_completion'/'calls';self.assertFalse(calls.exists())
        ask=runner.asker(client,dict(seed=3407),calls);picture=Image.new('RGB',(8,8),'white')
        for _ in range(2):self.assertEqual(ask('hypotheses',picture,'Return {"ok":true}',lambda v:v),dict(ok=True))
        self.assertEqual(len(generations),1)
        for suffix in ('.request.json','.generation.json','.json','.txt'):
            self.assertTrue((calls/('hypotheses'+suffix)).is_file())
        self.assertFalse(read(calls/'hypotheses.request.json')['text_only'])

    def compiled(self,failed=False):
        empty=dict(description='Observed-only alternative',hypothesis=deepcopy(self.h),groups=[],relations=[])
        programs={'full':deepcopy(self.p),'empty':empty};hs=[self.h]
        tables=dict(recognition=model.predictive(['h0']*4,['h0']),programs={'h0':model.predictive(['full']*4,['full','empty'])})
        if failed:
            h=deepcopy(self.h);h['id']='h1';hs.append(h)
            programs['failed']=dict(empty,hypothesis=h,proposal_failed=True)
            tables['recognition']=model.predictive(['h0']*3+['h1'],['h0','h1'])
            tables['programs']['h1']=dict(states=['failed'],probabilities=[1.])
        return model.compile_model(hs,programs,tables)

    def test_joint_program_retains_two_same_function_zones_and_cross_zone_relation(self):
        m=self.compiled();w=model.sample_worlds(m,2,1)[0]
        self.assertEqual(len(w['objects']),4);self.assertEqual(len(w['groups']),2)
        self.assertEqual({g['function'] for g in w['groups']},{'reading'})
        self.assertEqual(w['relations'][0]['kind'],'right_of')
        for o in w['objects']:
            if o['key'].startswith('c'):
                self.assertIn(o['attachment']['host'],{v['id'] for v in w['objects']})

    def test_failed_hypothesis_keeps_probability_and_empty_is_distinct(self):
        m=self.compiled(failed=True);samples=model.sample_worlds(m,13,100)
        self.assertEqual(len(samples),100)
        self.assertTrue(any(w['proposal_failed'] for w in samples))
        self.assertTrue(any(not w['objects'] and not w['proposal_failed'] for w in samples))
        z=next(n for n in m['nodes'] if n['id']=='Z');self.assertAlmostEqual(z['cpt']['root'][1],.3)
        f=next(n for n in m['nodes'] if n['id']=='F');self.assertEqual(f['cpt']['h1'],[0.,0.,1.])

    def test_proposal_failure_is_preserved_in_domain_not_replaced_by_empty(self):
        from PIL import Image
        case=dict(self.case,image=str(self.base/'input.png'))
        Image.new('RGB',(32,32),'white').save(case['image'])
        def ask(name,image,prompt,validator,max_tokens=8192):
            if name=='hypotheses':return validator(hypotheses())
            raise ValueError('Synthetic proposal failure; never real model evidence')
        result=runner.propose(case,dict(objects=[]),self.base,{},ask)
        self.assertEqual(result['status'],'partial');d=read(self.base/'domain.json')
        self.assertEqual(list(d['programs']),['f_h0'])
        self.assertTrue(d['programs']['f_h0']['proposal_failed'])
        self.assertEqual(d['hypotheses'][0]['functions'],['reading'])

    def test_complete_delta_cannot_invent_host_or_redeclare_observed(self):
        bad=programs_reply();bad['alternatives'][0]['groups'][0]['new_objects'][1]['attachment']['host']='invented'
        with self.assertRaises(ValueError):model.programs_answer(bad,self.case,self.h)
        case=dict(self.case,context=[dict(id='r1')])
        with self.assertRaisesRegex(ValueError,'existing IDs'):model.programs_answer(programs_reply(),case,self.h)

    def test_host_intervention_invalidates_installation_without_erasing_child(self):
        m=self.compiled();w=model.sample_worlds(m,2,1)[0];host=next(o for o in w['objects'] if o['key']=='r1')
        child=next(o for o in w['objects'] if o['key']=='c1')
        state=forward(m,w['noise'],{host['presence_node']:'0'})
        self.assertEqual(state[child['presence_node']],'1');self.assertEqual(state[child['installation_node']],'0')

    def layout_fixture(self):
        w=model.sample_worlds(self.compiled(),2,1)[0];obs=dict(objects=[])
        dims={o['id']:dict(full_size_m=[.8,.6,.5] if o['key'].startswith('r') else [.1,.12,.1]) for o in w['objects']}
        origins=geometry.origins_answer(dict(groups=[dict(id='g0',center_xz_m=[3.,-2.]),dict(id='g1',center_xz_m=[-3.,-4.])]),w,90)
        poses={o['id']:dict(id=o['id'],yaw_deg=25.,**({'root_offset_m':[0.,0.,0.]} if not o['attachment']['host'] else {'host_offset_uv':[.2,.1]})) for o in w['objects']}
        layout=geometry.convert(w,obs,dims,origins,poses,90)
        return w,obs,layout,origins

    def test_continuous_origins_survive_compiler_and_floor_is_explicit(self):
        w,obs,layout,origins=self.layout_fixture()
        solved=geometry.compile_layout(self.case,obs,w,layout,dict(floor_y_m=-1.5),policy='strict')
        self.assertTrue(solved['review']['geometry_contract_passed'],solved['review'])
        self.assertNotIn(origins['g0']['azimuth_deg'],(-90.,90.,180.))
        self.assertAlmostEqual(solved['review']['assembly_certificates']['g0']['azimuth_deg'],origins['g0']['azimuth_deg'])
        for o in solved['items']:
            if next(v for v in w['objects'] if v['id']==o['id'])['attachment']['kind']=='floor':
                self.assertAlmostEqual(o['center'][1]-o['size'][1]/2,-1.5)
        blocked=geometry.compile_layout(self.case,obs,w,layout,dict(floor_y_m=None))
        self.assertFalse(blocked['review']['geometry_contract_passed'])
        self.assertTrue(any('unresolved_floor' in e for e in blocked['review']['joint_contract_errors']))

    def test_hypothesized_extent_and_full_box_above_are_checked(self):
        w,obs,layout,_=self.layout_fixture();w['hypothesis']['bounds_xz_m']=[-1.,1.,-1.,1.]
        a,b=[o['id'] for o in w['objects'] if not o['attachment']['host']]
        w['relations']=[dict(source=a,target=b,kind='above',reason='Contradictory floor-root ordering')]
        solved=geometry.compile_layout(self.case,obs,w,layout,dict(floor_y_m=-1.5),policy='strict')
        errors=solved['review']['joint_contract_errors']
        self.assertTrue(any('outside_hypothesized_extent' in e for e in errors))
        self.assertTrue(any('unsatisfied_declared_relation' in e for e in errors))

    def test_full_phases_with_fixed_real_observation_and_synthetic_LM_replies(self):
        source=self.base/'source';source.mkdir();suite=read(SOURCE/'suite.json');suite['scenes']=suite['scenes'][:1]
        write(source/'suite.json',suite);sid=suite['scenes'][0]['id']
        shutil.copytree(SOURCE/'scenes'/sid/'observation',source/'scenes'/sid/'observation')
        before={str(p):sha(p) for p in source.rglob('*.json')}
        root=self.base/'run'
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG=''),patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}):
            runner.prepare(root,source,worlds=2,draws=4)
        with patch.object(runner,'provider',return_value=transport_fixture()):
            for name in runner.PHASES:
                result=runner.phase(root,name);self.assertIn(result['status'],runner.COMPLETED,result)
        result=runner.report(root);self.assertIn(result['status'],('complete_needs_review','complete_with_warnings'))
        comp=root/'scenes'/sid/'joint_completion';observed=read(source/'scenes'/sid/'observation/qwen/observer3d/G_obs.json')
        for dest in sorted(comp.glob('world_*')):
            full=read(dest/'G_full.json');self.assertEqual(full['objects'][:16],observed['objects'])
            self.assertEqual(len(full['objects']),20)
            self.assertTrue((dest/'views/scene_3d.html').exists())
            self.assertEqual(read(dest/'views/visual_validation.json')['labels'],20)
        self.assertEqual(before,{str(p):sha(p) for p in source.rglob('*.json')})
        self.assertTrue((root/'index.html').exists())
        # Resume does not rerun successful or failed scene phases.
        with patch.object(runner,'provider',side_effect=AssertionError('unexpected replay')):
            for name in runner.PHASES:runner.phase(root,name)


if __name__=='__main__':unittest.main()
