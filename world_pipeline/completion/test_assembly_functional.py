"""Probability semantics and the actual export path; CPU, no model calls."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from . import assembly_functional as f
from . import test_assembly as fixture
from .test_assembly_needs import mechanism, NeedsFixture
from .assembly_world import run
from .assembly_experiment import model_info, restart
from .world_scm import forward, validate
from ..common import read, write


def functional_mechanism():
    raw=mechanism()
    for need,role in zip(raw['requirements'],('reading','storage','background')):need['role']=role
    return raw


class FunctionalFixture(NeedsFixture):
    def ask(self,image,prompt,path,**kwargs):
        result=super().ask(image,prompt,path,**kwargs)
        if Path(path).stem.startswith('furniture_'):
            result=functional_mechanism()
            Path(path).write_text(json.dumps(result));write(Path(path).with_suffix('.json'),result)
        return result


class FunctionalTests(unittest.TestCase):
    setUp=fixture.AssemblyTests.setUp
    initialize=fixture.AssemblyTests.initialize

    def model(self, cues=True):
        z=f.hypotheses_answer(fixture.hypotheses())
        program=f.furniture_answer(functional_mechanism(),self.case,z[0])[0]
        case=deepcopy(self.case)
        case['context']=[dict(id='obs_000',category='book',description='book')] if cues else []
        return f.prepare_model(f.build_model(z,{'f0':program}),case,{},None)

    def test_exact_posterior_matches_direct_bayes_and_changes_object_probability(self):
        m=self.model();p=f.DEFAULTS
        numerator=denominator=0.
        for reading in (0,1):
            for storage in (0,1):
                prior=(.30 if reading else .70)*(.55 if storage else .45)
                a=1-(1-.08)*(1-.75*reading)*(1-.35*storage)
                likelihood=.01+.99*.45*.85*a
                denominator+=prior*likelihood;numerator+=reading*prior*likelihood
        post=f.marginals(m);prior=f.marginals(m,False)
        self.assertAlmostEqual(post['role_presence']['reading'],numerator/denominator)
        expected=.5*(p['need_activation']['inactive']+
            (p['need_activation']['active']-p['need_activation']['inactive'])*numerator/denominator)
        self.assertAlmostEqual(post['object_presence']['z0_lamp'],expected)
        self.assertGreater(post['object_presence']['z0_lamp'],prior['object_presence']['z0_lamp']+.05)
        self.assertAlmostEqual(sum(r['probability'] for r in f.role_posterior(m)['rows']),1.)

    def test_no_detection_is_unknown_and_unrelated_cue_does_not_force_demand(self):
        m=self.model(False)
        self.assertEqual(f.marginals(m)['object_presence'],f.marginals(m,False)['object_presence'])
        report=f.observed_cues(dict(context=[dict(id='o',category='unknown',description='unknown')]))
        self.assertEqual(report['positive_cues'],{})
        self.assertEqual(report['missing_cues'],'unobserved_not_absent')

    def test_deduplicate_cue_family_and_single_detection(self):
        c=dict(id='o1',category='bookcase',description='bookcase')
        d=dict(c,id='o2')
        report=f.observed_cues(dict(context=[c,d]))
        self.assertEqual(list(report['positive_cues']),['books'])
        self.assertEqual(report['positive_cues']['books']['source_ids'],['o1','o2'])

    def test_do_need_off_changes_numeric_marginals_preserves_shared_objects(self):
        m=self.model();post=f.marginals(m);after=f.marginals(m,do={'R_z0_reading':'0'})
        self.assertEqual(after['object_presence']['z0_lamp'],0.)
        self.assertGreater(after['object_presence']['z0_book'],0.)
        self.assertEqual(after['object_presence']['z0_plant'],post['object_presence']['z0_plant'])
        self.assertEqual(after['role_presence'],post['role_presence'])

    def test_actual_sampler_matches_exact_posterior_and_prior_on_same_catalog(self):
        m=self.model()
        for use in (False,True):
            worlds=f.sample_worlds(m,seed=73,count=1800,use_evidence=use)
            empirical=sum(any(o['id']=='z0_lamp' for o in w['objects']) for w in worlds)/len(worlds)
            exact=f.marginals(m,use)['object_presence']['z0_lamp']
            self.assertLess(abs(empirical-exact),.04)
            self.assertLess(len({tuple(o['id'] for o in w['objects']) for w in worlds}),len(worlds))
        w=f.sample_worlds(m,seed=21,count=1)[0]
        self.assertEqual(forward(m,w['noise']),w['structural_state'])
        self.assertEqual(w['structural_state']['O_books'],'1')
        self.assertEqual(f.sample_worlds(m,21,1),f.sample_worlds(m,21,1))

    def test_role_is_required_and_schema_has_no_history_variable(self):
        z=f.hypotheses_answer(fixture.hypotheses())[0]
        with self.assertRaisesRegex(ValueError,'role'):
            f.furniture_answer(mechanism(),self.case,z)
        prompt=f.furniture_prompt(self.case,z)
        schema=json.loads(prompt.split('SINGLE RESPONSE SCHEMA: ')[1].split('\nSCENE HYPOTHESIS:')[0])
        self.assertIn('role',schema['requirements'][0]);self.assertNotIn('alternatives',schema)
        m=self.model();self.assertFalse(m['resident_history_inferred'])
        self.assertFalse(m['calibrated_posterior_defined']);self.assertFalse(m['RGB_likelihood_defined'])
        self.assertTrue(m['cue_likelihood_defined']);validate(m)

    def test_activity_role_as_kind_is_normalized_explicitly_without_editing_raw(self):
        raw=functional_mechanism();raw['requirements'][0]['kind']='reading';before=deepcopy(raw)
        z=f.hypotheses_answer(fixture.hypotheses())[0]
        clean=f.furniture_answer(raw,self.case,z)[0]
        self.assertEqual(raw,before)
        self.assertEqual(clean['requirements'][0]['kind'],'activity')
        self.assertEqual(clean['requirements'][0]['role'],'reading')
        self.assertEqual(clean['interface_normalizations'][0]['original'],'reading')
        raw['requirements'][0]['kind']='dining'
        with self.assertRaisesRegex(ValueError,'requirement.kind'):
            f.furniture_answer(raw,self.case,z)

    def test_real_failed_qwen_reply_revalidates_without_inventing_inventory(self):
        folder=Path(__file__).resolve().parents[2]/'outputs/functional_completion_20261024_235203_723560/posterior/qwen/completion'
        if not folder.is_dir():self.skipTest('Saved Qwen failure unavailable')
        case=read(folder/'case.json');hypothesis=f.hypotheses_answer(read(folder/'calls/hypotheses.json'))[0]
        for name in ('furniture_z0.json','furniture_z0_repair.json'):
            raw=read(folder/'calls'/name);before=deepcopy(raw)
            clean=f.furniture_answer(raw,case,hypothesis)[0]
            self.assertEqual(raw,before)
            self.assertEqual(clean['requirements'][0]['kind'],'activity')
            self.assertEqual(sum(len(g['objects']) for g in clean['groups']),0)
            self.assertEqual(clean['interface_normalizations'][0]['original'],'social')

    def test_upstream_failure_blocks_dependency_and_gallery_shows_original_error(self):
        from .assembly_functional_status import candidate_status,mark_blocked,dependency_error
        from .assembly_pilot_visual import build
        import shutil
        class FailedFixture(FunctionalFixture):
            def ask(inner,image,prompt,path,**kwargs):
                if Path(path).stem.startswith('furniture_'):raise ValueError('controlled candidate failure')
                return super().ask(image,prompt,path,**kwargs)
        config=read(self.root/'experiment.json');config.update(model_info('functional'),worlds=2)
        write(self.root/'experiment.json',config)
        result=run(FailedFixture(),self.root,'qwen')
        self.assertEqual(result['status'],'failed')
        check=candidate_status(self.root,'qwen')
        self.assertFalse(check['ready']);self.assertIn('controlled candidate failure',dependency_error(check))
        self.base.observe(self.root,'api',True)
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            prior=restart(self.temp/'prior',self.root,'functional')
        blocked=mark_blocked(prior,self.root,'qwen')
        self.assertEqual(blocked['status'],'blocked_upstream_failed');self.assertFalse(blocked['model_loaded'])
        gallery=self.temp/'failed_gallery';gallery.mkdir()
        shutil.copytree(self.root,gallery/'posterior');shutil.copytree(prior,gallery/'prior')
        write(gallery/'pilot.json',dict(methods=['prior','posterior']))
        page=build(gallery,'qwen')
        self.assertIn('controlled candidate failure',page.read_text())
        review=read(page.parent/'review.json')
        self.assertFalse(review['probability_ablation_check']['verified'])
        self.assertEqual(review['probability_ablation_check']['missing_model_branches'],['prior','posterior'])
        self.assertEqual(review['worlds'],[])

    def test_qwen_shell_without_api_key_skips_prior_gpu_when_posterior_failed(self):
        import os
        import subprocess
        script=Path(__file__).resolve().parents[2]/'scripts/functional_completion.sh'
        binary=self.temp/'fakebin';binary.mkdir();events=self.temp/'shell_events.txt'
        shim=binary/'fake_python'
        shim.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
def value(flag):return args[args.index(flag)+1]
def event(name):
    with open(os.environ['FC_TEST_EVENTS'],'a') as stream:stream.write(name+'\\n')
if '-m' in args:
    module=args[args.index('-m')+1]
    if module.endswith('assembly_experiment'):
        root=Path(value('--root'));(root/'qwen').mkdir(parents=True);event('restart')
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
with open(os.environ['FC_TEST_EVENTS'],'a') as stream:stream.write('gpu\\n')
sys.exit(1)
''');gpu.chmod(0o755)
        env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],CAUPANO_QWEN_PYTHON=str(shim),
            CAUPANO_FUNCTIONAL_OUTPUT=str(self.temp/'shell_output'),FC_TEST_EVENTS=str(events))
        env.pop('SILICONFLOW_API_KEY',None)
        result=subprocess.run(['bash',str(script),'compare','FAKE_NODE',str(self.root),'qwen'],
            env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,1,result.stderr)
        calls=events.read_text().splitlines()
        self.assertEqual(calls.count('gpu'),1,calls)
        self.assertIn('check',calls);self.assertIn('block',calls);self.assertIn('visual',calls)
        self.assertIn('has failures',result.stderr)
        self.assertNotIn('Set SILICONFLOW_API_KEY',result.stderr)

    def test_fit_uses_train_only_and_refuses_duplicate_scene_splits(self):
        from .assembly_functional_fit import fit
        roles={r:0 for r in f.ROLES}
        rows=[dict(scene_id='train1',split='train',roles=roles,cue_reports={'books':0}),
              dict(scene_id='val1',split='val',roles={r:1 for r in f.ROLES},cue_reports={'books':1})]
        p,report=fit(rows)
        self.assertAlmostEqual(p['role_priors']['reading'],1/3)
        self.assertFalse(report['calibrated'])
        self.assertEqual(report['validation_cue_report_metrics_given_audited_roles']['n'],1)
        changed=deepcopy(rows);changed[1]['cue_reports']['books']=0;changed[1]['roles']=roles
        self.assertEqual(fit(changed)[0],p)
        changed[1]['scene_id']='train1'
        with self.assertRaisesRegex(ValueError,'scene_id'):fit(changed)

    def test_shared_instance_marginal_accounts_for_correlated_roles(self):
        m=self.model();post=f.role_posterior(m);expected=0.
        # Book belongs to reading corner OR storage, never counted twice.
        for row in post['rows']:
            p1=.5*(.12+.66*int(row['roles']['reading']))
            p2=.12+.66*int(row['roles']['storage'])
            expected+=row['probability']*(1-(1-p1)*(1-p2))
        actual=f.marginals(m)['object_presence']['z0_book']
        self.assertAlmostEqual(actual,expected)
        self.assertLessEqual(actual,1.)

    def test_explicit_parameter_weights_really_change_inventory_law(self):
        z=f.hypotheses_answer(fixture.hypotheses())
        program=f.furniture_answer(functional_mechanism(),self.case,z[0])[0]
        base=f.build_model(z,{'f0':program});p=deepcopy(f.DEFAULTS)
        p['strategy_weights']={'B_z0_reading':dict(existing=9.,corner=1.)}
        changed=f.prepare_model(base,self.case,dict(functional_parameters=p),None)
        standard=f.prepare_model(base,self.case,{},None)
        self.assertAlmostEqual(f.marginals(changed)['object_presence']['z0_lamp'],
                               f.marginals(standard)['object_presence']['z0_lamp']/5.)
        p['strategy_weights']={'made_up_node':dict(s=1)}
        with self.assertRaisesRegex(ValueError,'Unknown strategy'):
            f.prepare_model(base,self.case,dict(functional_parameters=p),None)

    def test_end_to_end_probability_ablation_reuses_catalog_and_preserves_observations(self):
        config=read(self.root/'experiment.json');config.update(model_info('functional'),worlds=2)
        write(self.root/'experiment.json',config)
        a=run(FunctionalFixture(),self.root,'qwen')
        self.assertEqual(a['status'],'worlds_exported_needs_review')
        self.assertTrue(a['posterior_sampling_used'])
        first=self.root/'qwen/completion'
        self.assertTrue((first/'probability_audit.html').is_file())
        self.base.observe(self.root,'api',True)
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            second=restart(self.temp/'prior',self.root,'functional')
        config=read(second/'experiment.json');config.update(functional_sampling='prior',semantic_mechanisms_from=str(self.root))
        write(second/'experiment.json',config)
        b=run(FunctionalFixture(),second,'qwen');dest=second/'qwen/completion'
        self.assertEqual(b['status'],'worlds_exported_needs_review');self.assertFalse(b['posterior_sampling_used'])
        self.assertTrue((dest/'candidate_reuse.json').is_file())
        self.assertFalse((dest/'calls/hypotheses.json').exists())
        self.assertEqual(read(dest/'furniture_mechanisms.json'),read(first/'furniture_mechanisms.json'))
        for folder,summary in ((first,a),(dest,b)):
            for row in summary['worlds']:
                full=read(folder/row['folder']/'G_full.json')
                self.assertEqual(full['objects'][:len(self.observed['objects'])],self.observed['objects'])
        config['semantic_mechanisms_from']=str(second);write(second/'experiment.json',config)
        # Confirm a future restart never inherits an old candidate reuse link.
        from .assembly_pilot_visual import build
        import shutil
        gallery=self.temp/'gallery';gallery.mkdir()
        shutil.copytree(self.root,gallery/'posterior');shutil.copytree(second,gallery/'prior')
        write(gallery/'pilot.json',dict(methods=['prior','posterior']))
        page=build(gallery,'qwen');self.assertTrue(page.is_file())
        self.assertIn('相同候选目录',page.read_text())
        model_path=gallery/'prior/qwen/completion/causal_model.json';model=read(model_path)
        model['functional_parameters']['detection']['recall']=.2;write(model_path,model)
        with self.assertRaisesRegex(ValueError,'exact candidate grammar and numerical parameters'):
            build(gallery,'qwen')


if __name__=='__main__':unittest.main()
