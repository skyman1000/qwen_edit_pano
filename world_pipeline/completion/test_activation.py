"""Meaningful CPU fixtures; no real model/API calls or quality claims."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from ..common import read, write, sha
from . import assembly_activation as model, assembly_activation_experiment as experiment
from . import assembly_activation_geometry as geometry, assembly_modular_geometry as old_geometry
from . import test_assembly as fixture
from .test_modular import ModularFixture, proposals
from .assembly_world import run
from .assembly_taxonomy import annotate, review, vocabulary
from .semantic_layout_visual import color
from .world_scm import validate, forward
from .world_generation import image_bytes, MATCHED_OPTIONS, SYSTEM_PROMPT


class ActivationFixture(ModularFixture):
    def ask(self,image,prompt,path,seed=3407,max_tokens=8192,**kwargs):
        path = Path(path); name = path.stem
        if not name.startswith(('activation_','variant_')):
            return super().ask(image,prompt,path,seed=seed,max_tokens=max_tokens,**kwargs)
        if image is not None: raise AssertionError('Undeclared image in inventory mechanism')
        h = json.loads(prompt.split('\nCURRENT_PARENT_H: ')[1].split('\n')[0]); j = int(name.split('_')[-1])
        if name.startswith('activation_'):
            roles = json.loads(prompt.split('\nAVAILABLE_ROLES: ')[1].split('\n')[0])
            # Joint negative correlation between reading and storage in fixture.
            active = ['reading'] if 'reading' in h['functions'] else ['storage']
            if j==0: active = []
            if self.backend=='api' and j==1: active = ['background']
            value = dict(active_roles=[r for r in active if r in roles])
        else:
            menu = json.loads(prompt.split('\nALLOWED_VARIANTS: ')[1]); value = dict(choice=menu[j%len(menu)]['id'])
        write(path.with_suffix('.request.json'),dict(prompt=prompt,seed=seed,max_tokens=max_tokens,
            system_prompt=SYSTEM_PROMPT,generation=MATCHED_OPTIONS,canonical_image_sha256=None))
        path.write_text(json.dumps(value)); write(path.with_suffix('.json'),value); return deepcopy(value)


class ActivationTests(unittest.TestCase):
    setUp = fixture.AssemblyTests.setUp
    initialize = fixture.AssemblyTests.initialize

    def bundle(self):
        root = self.temp/'activation'
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}), \
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]), \
             patch('qwen_edit_pano.world_pipeline.completion.assembly_sdscene_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            experiment.prepare(root,self.root,worlds=3,draws=4)
        for backend in ('qwen','api'):
            result = experiment.propose(root,backend,ActivationFixture(backend)); self.assertEqual(result['status'],'complete',result)
        experiment.pool(root)
        for backend in ('qwen','api'): experiment.mechanisms(root,backend,ActivationFixture(backend))
        experiment.pool_mechanisms(root); return root

    def fitted(self,policy='functional'):
        root = self.bundle(); d = read(root/'shared/domain.json'); c = read(root/'shared/case.json')
        m = model.prepare_model(d['base_model'],c,dict(sdscene_bundle=str(root/'shared'),mechanism_source='per_backend',
            execution_backend='qwen',activation_policy=policy),None)
        return root,m

    def test_joint_activation_and_probability_flow_are_normalized_and_not_independent(self):
        root,m = self.fitted(); validate(m); law = model.marginals(m)
        self.assertAlmostEqual(sum(law['hidden_count_probabilities'].values()),1.)
        self.assertAlmostEqual(sum(law['expected_category_counts'].values()),law['expected_hidden_instances'])
        roles = list(m['slots']); r = roles.index('reading'); s = roles.index('storage')
        both = sum(p for a,p in law['activation_pattern_probabilities'].items() if a[r]==a[s]=='1')
        self.assertLess(both,law['activation_probabilities']['reading']*law['activation_probabilities']['storage'])
        self.assertLess(law['category_presence']['book'],law['expected_category_counts']['book'])
        hist = {}; samples = model.sample_worlds(m,count=1600)
        for w in samples:
            self.assertEqual(forward(m,w['noise']),w['structural_state'])
            n = str(len(w['objects'])); hist[n] = hist.get(n,0)+1
        for n,p in law['hidden_count_probabilities'].items(): self.assertAlmostEqual(hist.get(n,0)/len(samples),p,delta=.045)
        self.assertTrue(any(not w['objects'] for w in samples))

    def test_function_interventions_change_actual_hidden_presence_with_same_noise(self):
        root,m = self.fitted(); without = next(h['id'] for h in m['profiles'] if not h['functions'])
        with_role = next(h['id'] for h in m['profiles'] if h['functions']==['reading'])
        a,b = model.marginals(m,without),model.marginals(m,with_role)
        self.assertGreater(b['category_presence']['chair']-a['category_presence']['chair'],.4)
        changed = 0
        for w in model.sample_worlds(m,count=60):
            x = model.active_world(m,forward(m,w['noise'],{'H':without}))
            y = model.active_world(m,forward(m,w['noise'],{'H':with_role}))
            changed += [o['id'] for o in x['objects']] != [o['id'] for o in y['objects']]
        self.assertGreater(changed,10)

    def test_h_cut_preserves_joint_activation_and_instance_marginals_but_removes_h_edges(self):
        root,m = self.fitted(); d = read(root/'shared/domain.json'); c = read(root/'shared/case.json')
        cut = model.prepare_model(d['base_model'],c,dict(sdscene_bundle=str(root/'shared'),mechanism_source='qwen',activation_policy='function_free'),None)
        a,b = model.marginals(m),model.marginals(cut)
        for field in ('object_presence','activation_pattern_probabilities','assembly_probabilities'):
            for k,p in a[field].items(): self.assertAlmostEqual(p,b[field][k])
        self.assertTrue(all('H' not in n['parents'] for n in cut['nodes'] if n['id']!='H'))
        laws = [model.marginals(cut,h['id']) for h in cut['profiles']]
        self.assertTrue(all(law['category_presence']==laws[0]['category_presence'] for law in laws))

    def test_activation_prior_does_not_depend_on_variant_cardinality(self):
        from .assembly_sdscene import estimate_table
        roles = ['reading','storage']; patterns = model.activation_states(roles)
        row = estimate_table(['11']*4,patterns,1.)['probabilities']
        p = sum(p for a,p in row.items() if a[0]=='0')
        self.assertAlmostEqual(p,.1) # alpha / (2*(K+alpha)), no J variant count.
        domain = dict(common_inventory=[],programs={},slots={'reading':['none','a']})
        first = model.activation_prompt(domain,model.profile('h0',[]),roles)
        domain['slots']['reading'] += ['b','c','d']
        self.assertEqual(first,model.activation_prompt(domain,model.profile('h0',[]),roles))

    def test_parent_only_prompts_and_matched_profile_support(self):
        root = self.bundle(); d = read(root/'shared/domain.json')
        self.assertTrue(all(any(p['role']==r for p in d['matched_profile_pairs']) for r in d['slots']))
        a = read(root/'shared/mechanisms/qwen.json'); b = read(root/'shared/mechanisms/api.json')
        self.assertEqual(a['original_parent_prompt_plan'],b['original_parent_prompt_plan'])
        for p in (root/'mechanism_calls/qwen').glob('activation_*.request.json'):
            v = read(p); self.assertIsNone(v['canonical_image_sha256']); self.assertNotIn('ALLOWED_VARIANTS',v['prompt'])
            self.assertNotIn('proposal_source',v['prompt'])
        self.assertEqual(read(root/'run.json')['api_model'],'deepseek-ai/DeepSeek-V4.1-Flash')

    def test_numeric_repair_sees_all_missing_values_and_does_not_fill_defaults(self):
        root,m = self.fitted(); world = next(w for w in model.sample_worlds(m,count=30) if w['objects'])
        from .conditional_world_geometry import support_plane
        prompt = geometry.layout_prompt(self.case,self.observed,world,support_plane(self.observed,['obs_000']))
        raw = json.loads(prompt.split('TEMPLATE: ')[1].split('\nFIXED SEMANTIC WORLD:')[0])
        before = deepcopy(raw)
        with self.assertRaises(ValueError) as err: geometry.layout_answer(raw,world,90)
        for field in ('range_m','full_size_m','yaw_deg','offset_m'): self.assertIn(field,str(err.exception))
        self.assertEqual(raw,before)

    def test_only_explicit_depth_names_are_exposed_and_parser_errors_agree(self):
        h = model.profile('z0',['reading']); prompt = model.furniture_prompt(self.case,h)
        self.assertNotIn('in_front_of',prompt); self.assertNotIn('behind',prompt)
        raw = proposals(); raw['alternatives'][0]['relations'] = [dict(source='table',target='chair',kind='behind',reason='test')]
        with self.assertRaises(ValueError) as err: model.furniture_answer(raw,self.case,h)
        self.assertNotIn('in_front_of',str(err.exception)); self.assertNotIn('behind',str(err.exception))

    def test_taxonomy_colors_distinguish_raw_membership_from_supported_mapping_and_preserve_observations(self):
        self.assertEqual(len(vocabulary()['categories']),37)
        self.assertTrue(review('chair')['raw_label_in_dataset_table'])
        self.assertFalse(review('armchair')['raw_label_in_dataset_table'])
        before = deepcopy(self.observed); state = deepcopy(before)
        source = before['objects'][0]
        for i,(noun,cid) in enumerate([('chair',3),('armchair',3),('not_a_real_dataset_category',None)]):
            obj = deepcopy(source); obj.update(track_id='new_'+str(i),local_evidence='unknown',raw_category=noun)
            obj['category_id'].update(value=cid,valid=cid is not None); state['objects'].append(obj)
        display = annotate(state)
        self.assertEqual(display['objects'][:len(before['objects'])],before['objects']); self.assertEqual(self.observed,before)
        self.assertEqual([color(o) for o in display['objects'][-3:]],['#e88c13','#9c43bb','#c3314d'])

    def test_collision_failure_retains_inventory_in_diagnostic_and_host_removal_preserves_child(self):
        root,m = self.fitted(); noise = {n['id']:.2 for n in m['nodes']}; role = 'reading'
        world = model.active_world(m,forward(m,noise,{'M_'+role:next(s for s in m['slots'][role] if s!='none')}))
        table = next(o for o in world['objects'] if o['key']=='table'); lamp = next(o for o in world['objects'] if o['key']=='lamp')
        after = model.active_world(m,forward(m,noise,{'M_'+role:table['furniture_state'],table['presence_node']:'0'}))
        self.assertIn(lamp['id'],after['unresolved_installations']); self.assertIn(lamp['id'],[o['id'] for o in after['objects']])
        fake = dict(review=dict(program_solver=dict(final_overlaps=[dict(a='x',b='y')]),geometry_contract_passed=True,
            layout_proxy_passed=True,hard_violations=0),items=[dict(id='x'),dict(id='y')])
        with patch.object(old_geometry,'compile_layout',return_value=deepcopy(fake)):
            result = geometry.compile_layout({}, {'objects':[]}, {'relations':[]}, {}, {})
        self.assertFalse(result['review']['geometry_contract_passed']); self.assertEqual(result['items'],fake['items'])

    def test_full_six_branch_execution_and_visual_comparison(self):
        root = self.bundle(); original = read(self.root/'qwen/observer3d/G_obs.json')
        for experiment_name in ('end_to_end','layout_control','function_free'):
            for backend in ('qwen','api'):
                s = run(ActivationFixture(backend),root/experiment_name,backend)
                self.assertNotEqual(s['status'],'failed',s)
                self.assertEqual(len(s['worlds'])+len(s['failures']),3)
                for record in s['worlds']+s['blocked_worlds']:
                    full = read(root/experiment_name/backend/'completion'/record['folder']/record.get('state_file','G_full.json'))
                    self.assertEqual(full['objects'][:len(original['objects'])],original['objects'])
                    self.assertTrue(all('taxonomy_review' in o for o in full['objects'][len(original['objects']):]))
        paths = [root/'layout_control'/b/'completion/structural_samples.json' for b in ('qwen','api')]
        self.assertEqual(sha(paths[0]),sha(paths[1]))
        from .assembly_activation_visual import report
        page = report(root); self.assertTrue(page.is_file()); self.assertTrue((page.parent/'review.json').is_file())
        before = sha(page); again = report(root); self.assertNotEqual(page,again); self.assertEqual(sha(page),before)

    def test_partial_proposal_summary_reports_saved_valid_candidates(self):
        root = self.temp/'partial'
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}), \
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]), \
             patch('qwen_edit_pano.world_pipeline.completion.assembly_sdscene_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            experiment.prepare(root,self.root)
        p = ActivationFixture('qwen'); original = p.ask
        def fail_late(image,prompt,path,**kw):
            if 'belongings' in str(path): raise ValueError('deliberate later failure')
            return original(image,prompt,path,**kw)
        p.ask = fail_late; result = experiment.propose(root,'qwen',p)
        self.assertEqual(result['status'],'failed'); self.assertTrue(result['partial_success'])
        self.assertEqual(result['programs'],len(read(root/'proposals/qwen/programs.json'))); self.assertGreater(result['programs'],0)

    def test_role_intervention_removes_only_selected_role_with_same_noise(self):
        root,m = self.fitted(); noise = {n['id']:.1 for n in m['nodes']}
        before = model.active_world(m,forward(m,noise,{'A':'1'*len(m['slots'])}))
        do = {'A':'1'*len(m['slots']),'A_reading':'0'}
        after = model.active_world(m,forward(m,noise,do))
        expected = {o['id'] for o in before['objects'] if m['state_slot'][o['furniture_state']]!='reading'}
        self.assertEqual(expected,{o['id'] for o in after['objects']})

    def test_v6_shell_blocks_missing_dependencies_and_still_reports(self):
        binary = self.temp/'bin'; binary.mkdir(); events = self.temp/'events'; shim = binary/'fake_python'
        shim.write_text('''#!/usr/bin/env python3
import os,sys
from pathlib import Path
args=sys.argv[1:];action=args[args.index('-m')+2];root=Path(args[args.index('--root')+1]);root.mkdir(exist_ok=True)
assert os.environ['SILICONFLOW_API_BASE']=='https://api.siliconflow.cn/v1'
with open(os.environ['V6_TEST_EVENTS'],'a') as f:f.write(action+'\\n')
if action in ('propose','pool','ready'):sys.exit(1)
'''); shim.chmod(0o755)
        gpu = binary/'srun'; gpu.write_text('''#!/usr/bin/env python3
import os,sys
with open(os.environ['V6_TEST_EVENTS'],'a') as f:f.write('gpu\\n')
sys.exit(1)
'''); gpu.chmod(0o755)
        env = dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],CAUPANO_QWEN_PYTHON=str(shim),
            CAUPANO_ACTIVATION_OUTPUT=str(self.temp/'shell_out'),V6_TEST_EVENTS=str(events),SILICONFLOW_API_KEY='fixture_only')
        env.pop('CAUPANO_BASELINE_ROOT',None)
        script = Path(__file__).resolve().parents[2]/'scripts/activation_completion.sh'
        result = subprocess.run(['bash',str(script),'compare','FAKE_NODE',str(self.root)],env=env,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,1,result.stderr); lines = events.read_text().splitlines()
        self.assertEqual(lines.count('gpu'),1); self.assertEqual(lines.count('block'),6); self.assertEqual(lines.count('report'),1)
        self.assertNotIn('mechanisms',lines); self.assertNotIn('world',lines)


if __name__=='__main__': unittest.main()
