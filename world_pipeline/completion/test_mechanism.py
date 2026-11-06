"""v8 exact-law, causal, interface, paired-pipeline tests; synthetic CPU only."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ..common import read,write,sha
from . import assembly_mechanism as model,assembly_mechanism_experiment as experiment
from . import assembly_mechanism_geometry as geometry,assembly_realization as v7
from .test_realization import RealizationFixture,raw_proposals
from . import test_assembly as fixture
from .assembly_world import run
from .assembly_mechanism_visual import inventory_law,report
from .world_scm import validate,forward
from .world_generation import image_bytes


def proposals():
    raw=raw_proposals();short=deepcopy(raw['alternatives'][0]);short['description']='Same functional roots, without optional reading books.'
    for group in short['groups']:
        if group['function']=='reading':group['new_objects']=[o for o in group['new_objects'] if o['category']!='book']
    raw['alternatives'].append(short)
    return raw


class MechanismFixture(RealizationFixture):
    def ask(self,image,prompt,path,seed=3407,max_tokens=8192,**kwargs):
        path=Path(path);name=path.stem
        if name.startswith('furniture_'):value=proposals()
        elif name.startswith('reference_'):
            if image is not None or 'CURRENT_PARENT_H' in prompt:raise AssertionError('Reference must see D only')
            menu=json.loads(prompt.split('\nAVAILABLE_REALIZATION_CHANNELS: ')[1].split('\n')[0]);j=int(name.split('_')[-1])
            active=[v['id'] for v in menu if v['realization_mode']=='observed_extension' and j>0]
            value=dict(active_channels=active)
        elif name.startswith('direct_'):
            if image is None or 'CURRENT_PARENT_H' in prompt or 'PROFILE_DOMAIN' in prompt:raise AssertionError('Direct needs O,D and no H')
            menu=json.loads(prompt.split('\nNEUTRAL_SLOT_DOMAIN: ')[1].split('\n')[0]);j=int(name.split('_')[-1])
            value=dict(choices={v['slot']:('none' if j==0 or (j==2 and int(v['slot'][1:])%2) else v['options'][1+(j%max(1,len(v['options'])-1))]['id']) for v in menu})
        elif name.startswith('layout_'):
            value=super().ask(image,prompt,path,seed=seed,max_tokens=max_tokens,**kwargs)
            for g in value['groups']:g.pop('region',None)
            path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return deepcopy(value)
        else:return super().ask(image,prompt,path,seed=seed,max_tokens=max_tokens,**kwargs)
        write(path.with_suffix('.request.json'),dict(prompt=prompt,seed=seed,max_tokens=max_tokens,
            canonical_image_sha256=hashlib.sha256(image_bytes(image)).hexdigest() if image is not None else None))
        path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return deepcopy(value)


class MechanismTests(unittest.TestCase):
    setUp=fixture.AssemblyTests.setUp
    initialize=fixture.AssemblyTests.initialize

    def prepare(self,name='mechanism'):
        root=self.temp/name
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_sdscene_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            experiment.prepare(root,self.root,worlds=2,draws=4)
        return root

    def bundle(self):
        root=self.prepare()
        for b in ('qwen','api'):
            result=experiment.propose(root,b,MechanismFixture(b));self.assertEqual(result['status'],'complete',result)
        experiment.pool(root)
        for b in ('qwen','api'):experiment.mechanisms(root,b,MechanismFixture(b))
        experiment.pool_mechanisms(root)
        return root

    def fitted(self,policy='functional'):
        root=self.bundle();d=read(root/'shared/domain.json')
        m=model.prepare_model(d['base_model'],read(root/'shared/case.json'),dict(sdscene_bundle=str(root/'shared'),mechanism_source='qwen',mechanism_policy=policy),None)
        return root,m

    def test_reference_shrinkage_matches_dense_predictive_and_has_no_fixed_point_one_floor(self):
        channels=['x','y'];reference=v7.estimate_activation(['00']*4,channels,1)
        activation={'h':model.raw_activation(['00']*4,channels)}
        components,rows,refs=model.activation_rows(activation,reference,channels,1)
        p_on=sum(rows['h'][j]*components[j][0] for j in components)
        self.assertAlmostEqual(p_on,.02)
        for a in map(''.join,itertools.product('01',repeat=2)):
            actual=sum(rows['h'][j]*((.5**2) if j=='prior' else float(j==a)) for j in components)
            q0=(4*float(a=='00')+1/4)/5
            self.assertAlmostEqual(actual,(4*float(a=='00')+q0)/5)
        with self.assertRaisesRegex(ValueError,'No valid'):model.raw_activation([None],channels)
        with self.assertRaises(ValueError):model.activation_rows(activation,reference,channels,0)

    def test_partial_alternatives_preserve_legal_whole_program_and_do_not_invent_relations(self):
        raw=proposals();raw['alternatives'][1]['groups'][-1]['observed_refs']=[];before=deepcopy(raw)
        h=model.profile('z',['reading'])
        with self.assertRaises(ValueError):v7.furniture_answer(raw,self.case,h)
        accepted=model.furniture_answer(raw,self.case,h)
        expected=v7.furniture_answer(dict(alternatives=[raw['alternatives'][0]]),self.case,h)[0]
        for k,v in expected.items():self.assertEqual(accepted[0][k],v)
        self.assertEqual(len(accepted),1);self.assertEqual(raw,before)
        self.assertEqual(accepted[0]['proposal_validation']['rejected'][0]['alternative'],1)
        raw['alternatives'][0]['groups'][-1]['observed_refs']=[]
        with self.assertRaisesRegex(ValueError,'No legal complete alternative'):model.furniture_answer(raw,self.case,h)

    def test_partial_reply_is_accepted_without_repair_and_all_rejections_remain_auditable(self):
        root=self.prepare();p=MechanismFixture();original=p.ask
        def mixed(image,prompt,path,**kw):
            value=original(image,prompt,path,**kw)
            if Path(path).stem.startswith('furniture_'):
                value['alternatives'][1]['groups'][-1]['realization_mode']='invalid'
                Path(path).write_text(json.dumps(value));write(Path(path).with_suffix('.json'),value)
            return value
        p.ask=mixed;result=experiment.propose(root,'qwen',p)
        self.assertEqual(result['status'],'complete');self.assertGreater(result['partial_alternative_replies_accepted'],0)
        self.assertFalse(list((root/'proposals/qwen/calls').glob('furniture_*_repair.request.json')))
        records=read(root/'proposals/qwen/alternative_validation.json')
        self.assertTrue(all(len(r['rejected'])==1 for r in records.values()))

    def test_all_invalid_proposals_get_one_repair_each_and_remaining_stages_continue(self):
        root=self.prepare();p=MechanismFixture();original=p.ask;seen=[]
        def invalid(image,prompt,path,**kw):
            value=original(image,prompt,path,**kw);name=Path(path).stem
            if name.startswith('furniture_'):
                seen.append(name)
                for alt in value['alternatives']:alt['groups'][-1]['realization_mode']='invalid'
                Path(path).write_text(json.dumps(value));write(Path(path).with_suffix('.json'),value)
            return value
        p.ask=invalid;result=experiment.propose(root,'qwen',p)
        self.assertEqual(result['status'],'failed');self.assertEqual(result['programs'],0)
        stages={s.removesuffix('_repair') for s in seen}
        self.assertEqual(len(stages),2)
        for s in stages:self.assertEqual(seen.count(s),1);self.assertEqual(seen.count(s+'_repair'),1)

    def test_pooled_counts_are_combined_before_reference_shrinkage_and_budget_is_exact(self):
        root=self.bundle();tables=[read(root/'shared/mechanisms'/(b+'.json')) for b in ('qwen','api')];pooled=read(root/'shared/mechanisms/pooled.json')
        self.assertEqual(pooled['reference']['valid_draws'],8)
        self.assertAlmostEqual(pooled['reference']['prior_weight'],1/9)
        for h,row in pooled['activation'].items():
            self.assertEqual(row['valid_draws'],8)
            for a,n in row['counts'].items():self.assertEqual(n,sum(t['activation'][h]['counts'].get(a,0) for t in tables))
        budget=read(root/'call_budget.json')
        for b in ('qwen','api'):
            plan=read(root/'shared/mechanisms'/(b+'.json'))['original_parent_prompt_plan']
            self.assertEqual(sum('prompt_sha256' in p for p in plan),budget['mechanism_initial_calls_per_backend'])
        self.assertEqual(sum(budget['stages_per_backend'].values()),budget['mechanism_initial_calls_per_backend'])

    def test_functional_and_direct_exact_joint_laws_match_independent_formulas(self):
        root,m=self.fitted();d=read(root/'shared/domain.json');t=m['empirical_tables'];alpha=t['concentration'];ref=t['reference'];k0=ref['valid_draws'];eta=ref['concentration']
        direct=model.prepare_model(d['base_model'],read(root/'shared/case.json'),dict(sdscene_bundle=str(root/'shared'),mechanism_source='qwen',mechanism_policy='direct'),None)
        actual=inventory_law(m);actual_direct=inventory_law(direct);q=model.marginals(m)['profile_probabilities']
        for values in itertools.product(*m['slots'].values()):
            raw=json.dumps(values,separators=(',',':'));mask=''.join(str(int(v!='none')) for v in values)
            q0=(ref['counts'].get(mask,0)+eta/2**len(values))/(k0+eta);prob=0.;base=q0
            for s,states in zip(values,m['slots'].values()):
                if s!='none':base/=len(states)-1
            for h,qh in q.items():
                counts=t['activation'][h];weight=(counts['counts'].get(mask,0)+alpha*q0)/(counts['valid_draws']+alpha)
                for r,v in zip(m['slots'],values):
                    if v!='none':weight*=t['variants'][h+'|'+r]['probabilities'][v]
                prob+=qh*weight
            self.assertAlmostEqual(actual.get(raw,0),prob)
            expected=(t['direct']['counts'].get(raw,0)+alpha*base)/(t['direct']['valid_draws']+alpha)
            self.assertAlmostEqual(actual_direct.get(raw,0),expected)
        for fitted in (m,direct):
            law=model.marginals(fitted);self.assertAlmostEqual(sum(law['hidden_count_probabilities'].values()),1)
            self.assertAlmostEqual(sum(law['expected_category_counts'].values()),law['expected_hidden_instances'])
            self.assertLess(law['category_presence']['book'],law['expected_category_counts']['book'])

    def test_sampling_matches_exact_counts_and_presence_with_null_and_duplicates_retained(self):
        root,m=self.fitted('direct');law=model.marginals(m);samples=model.sample_worlds(m,count=1600);hist={};presence=dict.fromkeys(law['object_presence'],0)
        for w in samples:
            self.assertEqual(forward(m,w['noise']),w['structural_state'])
            n=str(len(w['objects']));hist[n]=hist.get(n,0)+1
            for o in w['objects']:presence[o['id']]+=1
        self.assertGreater(hist.get('0',0),0);self.assertEqual(len(samples),1600)
        for n,p in law['hidden_count_probabilities'].items():self.assertAlmostEqual(hist.get(n,0)/1600,p,delta=.04)
        for k,p in law['object_presence'].items():self.assertAlmostEqual(presence[k]/1600,p,delta=.04)

    def test_h_changes_functional_inventory_but_not_direct_or_h_cut_inventory(self):
        root,m=self.fitted();domain=read(root/'shared/domain.json');case=read(root/'shared/case.json')
        empty=next(h['id'] for h in m['profiles'] if not h['functions']);reading=next(h['id'] for h in m['profiles'] if h['functions']==['reading'])
        a,b=model.marginals(m,empty),model.marginals(m,reading)
        self.assertGreater(b['activation_probabilities']['reading__new_group']-a['activation_probabilities']['reading__new_group'],.2)
        for policy in ('direct','function_free'):
            cut=model.prepare_model(domain['base_model'],case,dict(sdscene_bundle=str(root/'shared'),mechanism_source='qwen',mechanism_policy=policy),None)
            self.assertTrue(all('H' not in n['parents'] for n in cut['nodes'] if n['id']!='H'))
            first=model.marginals(cut,empty);second=model.marginals(cut,reading)
            self.assertEqual(first['object_presence'],second['object_presence'])
            if policy=='function_free':
                for k,p in model.marginals(m)['object_presence'].items():self.assertAlmostEqual(model.marginals(cut)['object_presence'][k],p)
        noise={n['id']:.7 for n in m['nodes']}
        x=model.active_world(m,forward(m,noise,{'H':empty}));y=model.active_world(m,forward(m,noise,{'H':reading}))
        self.assertNotEqual([o['id'] for o in x['objects']],[o['id'] for o in y['objects']])

    def test_host_removal_keeps_owned_child_and_invalidates_installation(self):
        root,m=self.fitted();r='reading__new_group';state=m['slots'][r][1]
        do={'A_'+c:'0' for c in m['slots']};do.update({'A_'+r:'1','M_'+r:state})
        noise={n['id']:.3 for n in m['nodes']};world=model.active_world(m,forward(m,noise,do))
        host=next(o for o in world['objects'] if o['category']=='side table');child=next(o for o in world['objects'] if o['attachment']['host']==host['id'])
        after=model.active_world(m,forward(m,noise,dict(do,**{host['presence_node']:'0'})))
        self.assertIn(child['id'],[o['id'] for o in after['objects']]);self.assertIn(child['id'],after['unresolved_installations'])

    def test_uncertainty_means_match_applied_conditional_activation_not_real_world_intervals(self):
        root,m=self.fitted();u=model.uncertainty(m)
        for row in u['rows']:
            p=model.marginals(m,row['profile'])['activation_probabilities'][row['channel']]
            self.assertAlmostEqual(row['mean'],p)
            lo,hi=row['conditional_interval_95'];self.assertTrue(0<=lo<=row['mean']<=hi<=1)
        uniform=model.uniform_reference_ablation(m);self.assertNotEqual(uniform['activation_probabilities'],model.marginals(m)['activation_probabilities'])

    def test_no_parent_leakage_paired_prompts_and_api_identity_are_bound(self):
        root=self.bundle();tables=[read(root/'shared/mechanisms'/(b+'.json')) for b in ('qwen','api')]
        self.assertEqual(tables[0]['original_parent_prompt_plan'],tables[1]['original_parent_prompt_plan'])
        run_config=read(root/'run.json');self.assertEqual(run_config['api_model'],'deepseek-ai/DeepSeek-V4.1-Flash');self.assertFalse(run_config['training_performed'])
        self.assertFalse(read(root/'call_budget.json')['direct_and_functional_budgets_matched'])
        for b in ('qwen','api'):
            for p in (root/'mechanism_calls'/b).glob('*.request.json'):
                req=read(p)
                if p.name.startswith(('reference_','activation_','variant_')):self.assertIsNone(req['canonical_image_sha256'])
                if p.name.startswith('reference_'):self.assertNotIn('CURRENT_PARENT_H',req['prompt'])
                if p.name.startswith('direct_'):
                    self.assertNotIn('CURRENT_PARENT_H',req['prompt']);self.assertNotIn('PROFILE_DOMAIN',req['prompt'])
                    self.assertEqual(req['canonical_image_sha256'],run_config['common_image_sha256'])
        for h in set(tables[0]['recognition']['probabilities'])-set(read(root/'shared/domain.json')['recognition_profile_ids']):
            self.assertEqual(tables[0]['recognition']['probabilities'][h],0)

    def test_direct_validator_rejects_missing_slots_and_wrong_channel_variants(self):
        domain=dict(slots={'a':['none','x'],'b':['none','y']})
        self.assertEqual(model.direct_answer(dict(choices={'k0':'none','k1':'none'}),domain),'["none","none"]')
        for value in [dict(choices={'k0':'x'}),dict(choices={'k0':'y','k1':'none'}),dict(choices={'k0':'x','k1':'y'},explanation='extra')]:
            with self.assertRaises(ValueError):model.direct_answer(value,domain)
        with self.assertRaises(ValueError):model.raw_direct([None],domain)

    def test_program_owned_directions_never_silently_convert_front(self):
        world=dict(objects=[dict(id='x',group='g',attachment=dict(kind='floor',host=None))],groups=[dict(key='g')])
        raw=dict(groups=[dict(id='g',range_m=2,objects=[dict(id='x',full_size_m=[1,1,1],yaw_deg=0,offset_m=[0,0,0])])])
        clean=geometry.layout_answer(raw,world,90);self.assertEqual(clean['g']['region_domain'],['left','right','rear'])
        raw['groups'][0]['region']='front'
        with self.assertRaisesRegex(ValueError,'program-owned'):geometry.layout_answer(raw,world,90)
        world['objects'][0].update(name='chair',category='chair',variant='chair',placement_hint='outside original view')
        world.update(configuration='fixture',hypothesis={},relations=[])
        prompt=geometry.layout_prompt(self.case,self.observed,world,{})
        template=json.loads(prompt.split('TEMPLATE: ')[1].split('\nFIXED SEMANTIC WORLD:')[0])
        self.assertNotIn('region',template['groups'][0])
        self.assertNotIn('The region is an initial numeric hypothesis',prompt)
        self.assertNotIn('For each group choose region',prompt)

    def test_all_eight_branches_export_or_report_every_sample_and_visualize_taxonomy(self):
        root=self.bundle();observed=read(self.root/'qwen/observer3d/G_obs.json')
        for exp in experiment.EXPERIMENTS:
            for b in ('qwen','api'):
                result=run(MechanismFixture(b),root/exp,b);self.assertNotEqual(result['status'],'failed',result)
                self.assertEqual(len(result['worlds'])+len(result['blocked_worlds'])+len(result['failures']),2)
                self.assertFalse(result['failures'],result)
                for record in result['worlds']+result['blocked_worlds']:
                    path=root/exp/b/'completion'/record['folder']/record.get('state_file','G_full.json');full=read(path)
                    self.assertEqual(full['objects'][:len(observed['objects'])],observed['objects'])
                    for obj in full['objects'][len(observed['objects']):]:self.assertIn('taxonomy_review',obj)
        paths=[root/'layout_control'/b/'completion/structural_samples.json' for b in ('qwen','api')]
        self.assertEqual(sha(paths[0]),sha(paths[1]));page=report(root);self.assertTrue(page.is_file())
        audit=read(page.parent/'review.json');self.assertEqual(len(audit['branch_status']),8)
        self.assertTrue(audit['comparisons']['layout_control_structural_samples_identical'])
        for b in ('qwen','api'):self.assertLess(audit['v8_probability_diagnostics'][b]['direct_control']['max_do_H_instance_probability_delta'],1e-9)

    def test_all_dependency_failures_still_generate_review_page(self):
        root=self.prepare()
        for exp in experiment.EXPERIMENTS:
            for b in ('qwen','api'):experiment.block(root,exp,b)
        page=report(root);self.assertIn('blocked_dependency',page.read_text())

    def test_domain_hash_mismatch_and_empty_raw_counts_are_not_silent_fallbacks(self):
        root=self.bundle();path=root/'shared/mechanisms/qwen.json';t=read(path);t['domain_sha256']='bad';write(path,t)
        d=read(root/'shared/domain.json');self.assertFalse(experiment.ready(root,'end_to_end','qwen')['ready'])
        with self.assertRaisesRegex(ValueError,'binding'):model.prepare_model(d['base_model'],read(root/'shared/case.json'),dict(sdscene_bundle=str(root/'shared'),mechanism_source='qwen'),None)
        with self.assertRaises(ValueError):model._count_row(dict(counts={'0':0},valid_draws=0))


if __name__=='__main__':unittest.main()
