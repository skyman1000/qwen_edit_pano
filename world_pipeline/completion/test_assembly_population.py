"""Whole inventory, numeric causal influence, ablations and actual export path."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import numpy as np

from ..common import read,write,sha
from . import assembly_population as pop
from . import test_assembly as fixture
from .test_assembly_needs import mechanism
from .assembly_experiment import model_info,restart
from .assembly_world import run
from .world_scm import validate,forward,key


def proposals():
    raw=mechanism()
    first=dict(description='A possible additional reading/storage area and incidental plant.',groups=raw['groups'],relations=[])
    second=deepcopy(first);second['description']='A small separate storage surface and book.'
    second['groups'][0]['new_objects']=[o for o in second['groups'][0]['new_objects'] if o['key'] in ('table','book')]
    return dict(alternatives=[first,second])


class PopulationFixture(fixture.GrammarFixture):
    def ask(self,image,prompt,path,**kwargs):
        path=Path(path)
        if path.stem.startswith('furniture_'):
            import hashlib
            from .world_generation import image_bytes
            value=proposals();write(path.with_suffix('.request.json'),dict(prompt=prompt,
                canonical_image_sha256=hashlib.sha256(image_bytes(image)).hexdigest()))
            path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value
        return super().ask(image,prompt,path,**kwargs)


class PopulationTests(unittest.TestCase):
    setUp=fixture.AssemblyTests.setUp
    initialize=fixture.AssemblyTests.initialize

    def model(self,policy='posterior',observed='floor lamp',raw=None):
        z=pop.hypotheses_answer(fixture.hypotheses())
        rows=pop.furniture_answer(raw or proposals(),self.case,z[0])
        case=deepcopy(self.case);case['context']=[dict(id='obs_000',description=observed,category='furniture')]
        return pop.prepare_model(pop.build_model(z,{f'f{i}':row for i,row in enumerate(rows)}),
            case,dict(population_sampling=policy),None)

    def test_proper_distributions_and_explicit_null_with_no_count_floor(self):
        model=self.model();validate(model);m=pop.marginals(model)
        self.assertAlmostEqual(sum(m['program_probabilities'].values()),1.)
        self.assertAlmostEqual(sum(m['hidden_count_probabilities'].values()),1.)
        self.assertGreater(m['empty_probability'],0.);self.assertLess(m['empty_probability'],1.)
        self.assertAlmostEqual(m['expected_hidden_instances'],sum(float(n)*p for n,p in m['hidden_count_probabilities'].items()))
        self.assertTrue(all(o['presence_node'].startswith('E_') for o in model['component_registry'].values()))

    def test_functional_count_mechanism_is_normalized_and_does_not_reward_unlimited_duplicates(self):
        import math
        p=pop.parameters();h={r:'0' for r in pop.ROLES};h['reading']='1'
        ratios=[pop._log_count_ratio(n,p,h,1,'books') for n in range(81)]
        self.assertLess(ratios[-1],max(ratios)-.1)
        mass=sum(math.exp(pop._log_nb(n,p['base_means']['books'],p['dispersion'])+
            pop._log_count_ratio(n,p,h,1,'books')) for n in range(201))
        self.assertAlmostEqual(mass,1.,places=8)
        self.assertEqual(pop._log_count_ratio(80,p,h,1,'books',False),0.)

    def test_positive_report_changes_the_sampling_law_without_inventing_absences(self):
        model=self.model();prior=pop.marginals(model,False);post=pop.marginals(model)
        self.assertGreater(abs(prior['program_probabilities']['f0']-post['program_probabilities']['f0']),1e-4)
        self.assertEqual(model['observed_counts']['books'],0)
        nodes={n['id']:n for n in model['nodes']};nsmall=model['program_count_state']['f1'];nnull=model['program_count_state']['f_null']
        for q in nodes['Q_report']['states']:
            # Book/table are unreported; small and null have same lighting count.
            self.assertEqual(nodes['O_positive_counts']['cpt'][key([nsmall,q])],
                             nodes['O_positive_counts']['cpt'][key([nnull,q])])
        self.assertEqual(pop.family('table lamp'),'lighting');self.assertEqual(pop.family('bookcase'),'storage')

    def test_function_mechanism_changes_distribution_and_background_support_survives_ablation(self):
        model=self.model();ablated=self.model('no_function')
        self.assertEqual(model['scene_program_states'],ablated['scene_program_states'])
        self.assertEqual(model['population_parameters'],ablated['population_parameters'])
        a=pop.marginals(model);b=pop.marginals(ablated)
        self.assertGreater(max(abs(a['program_probabilities'][k]-b['program_probabilities'][k]) for k in a['program_probabilities']),.001)
        self.assertTrue(all(v>0 for v in b['program_probabilities'].values()))
        nodes={n['id']:n for n in ablated['nodes']}
        for probs in nodes['N_population']['cpt'].values():
            self.assertTrue(np.allclose(probs,[1/len(probs)]*len(probs)))

    def test_adding_duplicate_proposals_does_not_increase_mass(self):
        z=pop.hypotheses_answer(fixture.hypotheses());rows=pop.furniture_answer(proposals(),self.case,z[0])
        base=pop.build_model(z,{'f0':rows[0],'f1':rows[1]})
        duplicate=dict(deepcopy(rows[0]),description='A paraphrase with the exact same structured inventory.')
        repeated=pop.build_model(z,{'f0':rows[0],'f1':rows[1],'duplicate':duplicate})
        self.assertEqual(repeated['proposal_bindings']['duplicate'],'f0')
        self.assertEqual(set(base['furniture_states']),set(repeated['furniture_states']))
        p=pop.prepare_model(base,self.case,{},None);q=pop.prepare_model(repeated,self.case,{},None)
        self.assertEqual(pop.marginals(p),pop.marginals(q))

    def test_actual_samples_match_exact_program_probabilities_and_replay_noise(self):
        model=self.model();worlds=pop.sample_worlds(model,73,900);law=pop.marginals(model)
        for fid,p in law['program_probabilities'].items():
            empirical=sum(w['structural_state']['F']==fid for w in worlds)/len(worlds)
            self.assertLess(abs(empirical-p),.06)
        self.assertLess(len({tuple(o['id'] for o in w['objects']) for w in worlds}),len(worlds))
        for w in worlds[:5]:
            self.assertEqual(forward(model,w['noise']),w['structural_state'])
            self.assertEqual(w['structural_state']['O_positive_counts'],'report')
        reference=self.model('reference');draws=pop.sample_worlds(reference,73,900)
        for fid in reference['furniture_states']:
            self.assertLess(abs(sum(w['structural_state']['F']==fid for w in draws)/len(draws)-1/3),.06)
        ref_law=pop.marginals(reference,True,True)
        for role in pop.ROLES:
            empirical=sum(w['functional_state'][role]=='1' for w in draws)/len(draws)
            self.assertLess(abs(empirical-ref_law['role_presence'][role]),.06)

    def test_functional_intervention_uses_same_noise_and_can_change_inventory(self):
        model=self.model();worlds=pop.sample_worlds(model,31,80);changed=[]
        for w in worlds:
            if w['structural_state']['H_reading']=='1':
                after=forward(model,w['noise'],{'H_reading':'0'})
                changed.append(after['F']!=w['structural_state']['F'])
        self.assertTrue(any(changed))
        rich=next(w for w in worlds if len(w['objects'])==5)
        host=next(o for o in rich['objects'] if o['key']=='table')
        state=forward(model,rich['noise'],{host['presence_node']:'0'});after=pop.active_world(model,state)
        self.assertTrue(any(o['key']=='book' for o in after['objects']))
        self.assertTrue(after['unresolved_installations'])

    def test_empty_proposals_are_recorded_as_support_failure_not_proof_of_no_hidden_objects(self):
        raw=dict(alternatives=[dict(description='Only the observed region described.',groups=[],relations=[])])
        model=self.model(raw=raw)
        self.assertTrue(model['candidate_support']['all_nonempty_proposals_missing'])
        self.assertEqual(len(model['furniture_states']),1)
        self.assertAlmostEqual(pop.marginals(model)['empty_probability'],1.)
        self.assertEqual([len(w['objects']) for w in pop.sample_worlds(model,count=3)],[0,0,0])

    def test_empty_end_to_end_domain_is_not_reported_as_successful_completion(self):
        class EmptyFixture(PopulationFixture):
            def ask(inner,image,prompt,path,**kwargs):
                value=super().ask(image,prompt,path,**kwargs)
                if Path(path).stem.startswith('furniture_'):
                    value=dict(alternatives=[dict(description='Observed-only response.',groups=[],relations=[])])
                    Path(path).write_text(json.dumps(value));write(Path(path).with_suffix('.json'),value)
                return value
        config=read(self.root/'experiment.json');config.update(model_info('population'),worlds=1)
        write(self.root/'experiment.json',config);result=run(EmptyFixture(),self.root,'qwen')
        self.assertEqual(result['status'],'empty_candidate_domain_no_completion')
        self.assertFalse(result['world_state_ready'])
        out=self.root/'qwen/completion'
        self.assertFalse(read(out/'world_01/G_full.json')['world_state_ready'])
        from .assembly_functional_status import candidate_status
        self.assertFalse(candidate_status(self.root,'qwen')['ready'])

    def test_qwen_shell_skips_both_dependent_GPU_jobs_after_upstream_failure(self):
        import os
        import subprocess
        script=Path(__file__).resolve().parents[2]/'scripts/population_completion.sh'
        binary=self.temp/'fakebin';binary.mkdir();events=self.temp/'events.txt';shim=binary/'fake_python'
        shim.write_text('''#!/usr/bin/env python3
import os,sys
from pathlib import Path
args=sys.argv[1:]
def val(flag):return args[args.index(flag)+1]
def event(name):
    with open(os.environ['PC_TEST_EVENTS'],'a') as stream:stream.write(name+'\\n')
if '-m' in args:
    module=args[args.index('-m')+1]
    if module.endswith('assembly_experiment'):
        (Path(val('--root'))/'qwen').mkdir(parents=True);event('restart')
    elif module.endswith('assembly_functional_status'):
        action=args[args.index('-m')+2];event(action)
        if action=='check':sys.exit(1)
    elif module.endswith('assembly_pilot_visual'):event('visual')
    else:raise RuntimeError(args)
elif args[0]=='-':event('manifest')
else:raise RuntimeError(args)
''');shim.chmod(0o755)
        gpu=binary/'srun';gpu.write_text('''#!/usr/bin/env python3
import os,sys
with open(os.environ['PC_TEST_EVENTS'],'a') as stream:stream.write('gpu\\n')
sys.exit(1)
''');gpu.chmod(0o755)
        env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],CAUPANO_QWEN_PYTHON=str(shim),
            CAUPANO_POPULATION_OUTPUT=str(self.temp/'shell_output'),PC_TEST_EVENTS=str(events))
        env.pop('SILICONFLOW_API_KEY',None)
        result=subprocess.run(['bash',str(script),'compare','FAKE_NODE',str(self.root),'qwen'],
            env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,1,result.stderr)
        lines=events.read_text().splitlines()
        self.assertEqual(lines.count('gpu'),1);self.assertEqual(lines.count('block'),2)
        self.assertIn('has failures',result.stderr);self.assertNotIn('Set SILICONFLOW_API_KEY',result.stderr)

    def test_fixed_budget_proposals_include_empty_replies_without_retry_until_nonempty(self):
        z=pop.hypotheses_answer(fixture.hypotheses());calls=[]
        def ask(name,image,prompt,validator):
            calls.append((name,prompt))
            return validator(dict(alternatives=[dict(description='Empty candidate.',groups=[],relations=[])]))
        out=self.temp/'proposals';out.mkdir()
        programs=pop.propose_programs(self.case,z,None,ask,out)
        self.assertEqual(len(calls),2);self.assertEqual(len(programs),2)
        self.assertTrue(all('minimum count' in prompt for _,prompt in calls))
        self.assertTrue(all('requirements' not in json.dumps(proposals()) for _ in [0]))

    def test_end_to_end_same_domain_ablations_preserve_observations_and_generate_actual_visuals(self):
        config=read(self.root/'experiment.json');config.update(model_info('population'),worlds=2,population_sampling='posterior')
        write(self.root/'experiment.json',config);result=run(PopulationFixture(),self.root,'qwen')
        self.assertEqual(result['status'],'worlds_exported_needs_review',result)
        self.assertTrue(result['count_likelihood_defined']);self.assertTrue(result['posterior_sampling_used'])
        first=self.root/'qwen/completion';self.assertTrue((first/'probability_audit.html').is_file())
        self.assertEqual(len(list((first/'calls').glob('furniture_*.request.json'))),2)
        self.base.observe(self.root,'api',True)
        roots={'population_posterior':self.root}
        for label,policy in [('joint_reference','reference'),('population_no_function','no_function')]:
            with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
                 patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
                second=restart(self.temp/label,self.root,'population')
            config=read(second/'experiment.json');config.update(population_sampling=policy,semantic_mechanisms_from=str(self.root))
            write(second/'experiment.json',config);r=run(PopulationFixture(),second,'qwen')
            self.assertEqual(r['status'],'worlds_exported_needs_review',r);roots[label]=second
            self.assertFalse((second/'qwen/completion/calls/hypotheses.json').exists())
            self.assertEqual(read(second/'qwen/completion/furniture_mechanisms.json'),read(first/'furniture_mechanisms.json'))
        gallery=self.temp/'gallery';gallery.mkdir()
        for label,root in roots.items():
            shutil.copytree(root,gallery/label)
            s=read(root/'qwen/completion/summary.json')
            for record in s['worlds']:
                state=read(root/'qwen/completion'/record['folder']/'G_full.json')
                self.assertEqual(state['objects'][:len(self.observed['objects'])],self.observed['objects'])
                for obj in state['objects'][len(self.observed['objects']):]:self.assertIn('generative_presence_probability',obj)
        write(gallery/'pilot.json',dict(methods=['joint_reference','population_posterior','population_no_function'],comparison='population_v3_fixed_domain'))
        from .assembly_pilot_visual import build
        page=build(gallery)
        self.assertTrue(read(page.parent/'review.json')['probability_ablation_check']['verified'])
        self.assertTrue((page.parent/'synchronized_3d.html').is_file())

    def test_real_joint_proposals_have_nonzero_support_and_numeric_function_effect(self):
        source=Path(__file__).resolve().parents[2]/'outputs/causal_completion_pilot_20261024_220629_380006_3407/joint/qwen/completion'
        if not source.is_dir():self.skipTest('Saved real joint domain unavailable')
        path=source/'furniture_mechanisms.json';before=sha(path)
        base=pop.build_model(read(source/'hypotheses.json'),read(path));case=read(source/'case.json')
        a=pop.prepare_model(base,case,{},None);b=pop.prepare_model(base,case,dict(population_sampling='no_function'),None)
        ma=pop.marginals(a);mb=pop.marginals(b)
        self.assertGreater(ma['expected_hidden_instances'],0.)
        self.assertGreater(max(abs(ma['program_probabilities'][f]-mb['program_probabilities'][f]) for f in ma['program_probabilities']),.001)
        self.assertEqual(sha(path),before)

    def test_saved_joint_raw_domain_is_revalidated_and_wrong_evidence_rejected(self):
        source=Path(__file__).resolve().parents[2]/'outputs/causal_completion_pilot_20261024_220629_380006_3407/joint'
        if not source.is_dir():self.skipTest('Saved joint run unavailable')
        case=read(source/'qwen/completion/case.json')
        hypotheses,programs,reuse=pop.load_joint_domain(case,source,'qwen')
        self.assertGreater(len(programs),0);self.assertTrue(reuse['current_parser_revalidated'])
        self.assertFalse(reuse['old_geometry_reused']);self.assertFalse(reuse['old_probabilities_reused'])
        changed=deepcopy(case);changed['hfov']=80
        with self.assertRaisesRegex(ValueError,'camera/context/support'):
            pop.load_joint_domain(changed,source,'qwen')

    def test_saved_joint_candidate_entry_runs_new_probability_and_new_layout(self):
        source=self.root;config=read(source/'experiment.json');config.update(model_info('joint'),worlds=1)
        write(source/'experiment.json',config);original=run(fixture.GrammarFixture(),source,'qwen')
        self.assertEqual(original['status'],'worlds_exported_needs_review',original)
        path=source/'qwen/completion/furniture_mechanisms.json';before=sha(path)
        self.base.observe(source,'api',True)
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            target=restart(self.temp/'population_cached',source,'population',73,1)
        config=read(target/'experiment.json');config.update(candidate_programs_from=str(source))
        write(target/'experiment.json',config);result=run(PopulationFixture(),target,'qwen')
        self.assertEqual(result['status'],'worlds_exported_needs_review',result)
        out=target/'qwen/completion';self.assertTrue(read(out/'candidate_source.json')['current_parser_revalidated'])
        self.assertFalse((out/'calls/hypotheses.json').exists())
        self.assertFalse(list((out/'calls').glob('furniture_*.json')))
        self.assertTrue((out/'structural_samples.json').is_file());self.assertEqual(sha(path),before)

    def test_supervised_fit_keeps_validation_out_and_refuses_scene_leakage(self):
        from .assembly_population_fit import fit
        zero={r:0 for r in pop.ROLES};one={r:1 for r in pop.ROLES}
        low={k:1 for k in pop.FAMILIES};high={k:2 for k in pop.FAMILIES}
        records=[dict(scene_id='a',split='train',roles=zero,full_counts=low),
            dict(scene_id='b',split='train',roles=one,full_counts=high),
            dict(scene_id='c',split='val',roles=one,full_counts=high)]
        p,report=fit(records)
        changed=deepcopy(records);changed[-1]['full_counts']={k:9 for k in pop.FAMILIES}
        self.assertEqual(p,fit(changed)[0]);self.assertFalse(report['hidden_inventory_calibrated'])
        self.assertEqual(p['population_scales'],[1.])
        changed[-1]['scene_id']='a'
        with self.assertRaisesRegex(ValueError,'scene_id'):fit(changed)


if __name__=='__main__':unittest.main()
