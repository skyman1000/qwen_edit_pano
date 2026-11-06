"""CPU fixtures verify probability flow and paired comparisons, not model quality."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ..common import read,write,sha
from . import assembly_sdscene as sd,assembly_sdscene_experiment as experiment
from . import assembly_sdscene_geometry as geometry,assembly_population as population
from . import test_assembly as fixture
from .test_assembly_population import proposals
from .assembly_world import run
from .assembly_experiment import stage_seed
from .world_generation import image_bytes,MATCHED_OPTIONS,SYSTEM_PROMPT
from .world_scm import forward,validate


class SceneFixture(fixture.GrammarFixture):
    def __init__(self,backend='qwen'):
        super().__init__();self.backend=backend
    def ask(self,image,prompt,path,seed=3407,max_tokens=8192,**kwargs):
        path=Path(path);name=path.stem
        if name.startswith('layout_'):return super().ask(image,prompt,path,seed=seed,max_tokens=max_tokens,**kwargs)
        write(path.with_suffix('.request.json'),dict(prompt=prompt,system_prompt=SYSTEM_PROMPT,seed=seed,
            max_tokens=max_tokens,generation=MATCHED_OPTIONS,
            canonical_image_sha256=hashlib.sha256(image_bytes(image)).hexdigest() if image is not None else None))
        if name=='hypotheses':
            value=fixture.hypotheses();value['hypotheses'][0]['functions']=['reading','social']
        elif name.startswith('furniture_'):value=proposals()
        elif name.startswith('recognition_'):
            profiles=json.loads(prompt.split('\nPROFILE_DOMAIN: ')[1].split('\nOBSERVED_ANCHORS: ')[0])
            choices=sorted(h['id'] for h in profiles)
            value=dict(choice=choices[0 if self.backend=='api' else -1])
        elif name.startswith('mechanism_'):
            self.assert_parent_only(image,prompt)
            h=json.loads(prompt.split('\nCURRENT_PARENT_H: ')[1].split('\nCOMMON_INVENTORY')[0])
            menu=json.loads(prompt.split('\nALLOWED_VALUE_DOMAIN: ')[1]);choices=sorted(x['id'] for x in menu)
            # Different functional rows, and a second backend with a different
            # categorical law; neither table is secretly hand set by production.
            choice=choices[-1] if not h['functions'] else choices[0 if self.backend=='qwen' else min(1,len(choices)-1)]
            value=dict(choice=choice)
        else:raise AssertionError(name)
        path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return deepcopy(value)
    @staticmethod
    def assert_parent_only(image,prompt):
        if image is not None or 'OBSERVED_ANCHORS' in prompt or 'earlier votes' in prompt:
            raise AssertionError('Nonparent evidence delivered to inventory mechanism')


class SceneTests(unittest.TestCase):
    setUp=fixture.AssemblyTests.setUp
    initialize=fixture.AssemblyTests.initialize

    def prepare(self,name='sd'):
        # Deliberately different original API observation must be ignored.
        self.base.observe(self.root,'api',True)
        out=self.temp/name
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_sdscene_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            return experiment.prepare(out,self.root,worlds=3,draws=3)

    def bundle(self):
        root=self.prepare()
        for backend in ('qwen','api'):
            result=experiment.propose(root,backend,SceneFixture(backend));self.assertEqual(result['status'],'complete',result)
        experiment.pool(root)
        for backend in ('qwen','api'):experiment.mechanisms(root,backend,SceneFixture(backend))
        experiment.pool_mechanisms(root);return root

    def model(self):
        root=self.bundle();domain=read(root/'shared/domain.json');case=read(root/'shared/case.json')
        model=sd.prepare_model(domain['base_model'],case,dict(sdscene_bundle=str(root/'shared'),
            mechanism_source='per_backend',execution_backend='qwen'),None)
        return root,model

    def test_empirical_probabilities_use_counts_with_one_symmetric_smoothing_mass(self):
        row=sd.estimate_table(['a','a','b',None],['a','b'],1.)
        self.assertEqual(row['counts'],dict(a=2,b=1));self.assertEqual(row['invalid_draws'],1)
        self.assertEqual(row['probabilities'],dict(a=.625,b=.375))
        with self.assertRaisesRegex(ValueError,'No valid'):sd.estimate_table([None],['a','b'])
        for value in ({'choice':'bad'},{'choice':'a','explanation':'longer'},None):
            with self.assertRaises(ValueError):sd.choice_answer(value,['a','b'])

    def test_taxonomy_preserves_raw_labels_but_recovers_concrete_nouns(self):
        objects=[dict(category='furniture',variant=n) for n in ('low-profile storage cabinet','modern desk','ergonomic office chair')]
        objects.append(dict(category='lighting',variant='desk lamp'));before=deepcopy(objects)
        counts,bindings=population.counts(objects)
        self.assertEqual({k:v for k,v in counts.items() if v},dict(storage=1,tables=1,seating=1,lighting=1))
        self.assertTrue(all(b['noun_source']=='variant' for b in bindings));self.assertEqual(objects,before)
        self.assertEqual(population.family('side_table'),'tables');self.assertEqual(population.family('TV stand'),'tables')

    def test_strict_domain_rejects_generic_nouns_and_contradictory_mounts(self):
        h=sd.hypotheses_answer(dict(hypotheses=[dict(description='Uncertain reading.',functions=['reading'],assumptions=['Unseen area.'])]))[0]
        for category,variant,kind in [('furniture','side table','floor'),('bookcase','freestanding wooden bookcase','wall'),
                                      ('shelf','wall-mounted shelf','floor')]:
            raw=proposals();item=raw['alternatives'][0]['groups'][0]['new_objects'][0]
            item.update(category=category,variant=variant,attachment=dict(kind=kind,host=None))
            with self.assertRaises(ValueError):sd.furniture_answer(raw,self.case,h)

    def test_same_frozen_observation_and_identical_parent_prompts_for_both_models(self):
        root=self.bundle()
        for exp in ('context','end_to_end','layout_control'):
            self.assertEqual(sha(root/exp/'api/observer3d/G_obs.json'),sha(self.root/'qwen/observer3d/G_obs.json'))
            self.assertEqual(sha(root/exp/'api/observer3d/G_obs.json'),sha(root/exp/'qwen/observer3d/G_obs.json'))
        tables=[read(root/'shared/mechanisms'/(b+'.json')) for b in ('qwen','api')]
        self.assertEqual(tables[0]['original_parent_prompt_plan'],tables[1]['original_parent_prompt_plan'])
        for backend in ('qwen','api'):
            requests=[read(p) for p in (root/'mechanism_calls'/backend).glob('*.request.json')]
            mechanism=[r for r in requests if 'CURRENT_PARENT_H' in r['prompt']]
            self.assertTrue(mechanism);self.assertTrue(all(r['canonical_image_sha256'] is None for r in mechanism))
            self.assertTrue(all('qwen_' not in r['prompt'] and 'api_' not in r['prompt'] for r in mechanism))
            recognition=[r for r in requests if 'PROFILE_DOMAIN' in r['prompt']]
            self.assertEqual(len({r['seed'] for r in recognition}),3)
            self.assertTrue(all(r['canonical_image_sha256']==read(root/'run.json')['common_image_sha256'] for r in recognition))

    def test_functional_mechanism_changes_actual_sampling_and_do_replays_noise(self):
        root,model=self.model();validate(model);law=sd.marginals(model);nodes={n['id']:n for n in model['nodes']}
        self.assertFalse(model['count_likelihood_defined']);self.assertFalse(model['RGB_likelihood_defined'])
        self.assertEqual(nodes['F']['parents'],['H']);self.assertEqual(nodes['N_population']['parents'],['F'])
        h0,h1=nodes['H']['states'];p0=sd.marginals(model,h0);p1=sd.marginals(model,h1)
        self.assertGreater(max(abs(p0['program_probabilities'][f]-p1['program_probabilities'][f]) for f in nodes['F']['states']),.1)
        self.assertEqual(p0['profile_probabilities'][h0],1.)
        for j,f in enumerate(nodes['F']['states']):
            expected=sum(nodes['H']['cpt']['root'][i]*nodes['F']['cpt'][h][j] for i,h in enumerate(nodes['H']['states']))
            self.assertAlmostEqual(law['program_probabilities'][f],expected)
        self.assertTrue(all(0<=p<=1 for p in law['category_presence'].values()))
        self.assertAlmostEqual(sum(law['expected_category_counts'].values()),law['expected_hidden_instances'])
        worlds=sd.sample_worlds(model,count=3000);counts={f:0 for f in nodes['F']['states']}
        for world in worlds:
            self.assertEqual(forward(model,world['noise']),world['structural_state'])
            counts[world['structural_state']['F']]+=1
        for f,n in counts.items():self.assertAlmostEqual(n/len(worlds),law['program_probabilities'][f],delta=.035)
        world=sd.active_world(model,forward(model,{n['id']:.1 for n in model['nodes']},{'H':h1,'F':nodes['F']['states'][0]}))
        world['noise']={n['id']:.1 for n in model['nodes']}
        table=next(o for o in world['objects'] if o['key']=='table');lamp=next(o for o in world['objects'] if o['key']=='lamp')
        after=sd.active_world(model,forward(model,world['noise'],{'H':h1,'F':nodes['F']['states'][0],table['presence_node']:'0'}))
        self.assertIn(lamp['id'],[o['id'] for o in after['objects']]);self.assertIn(lamp['id'],after['unresolved_installations'])

    def test_full_paired_pipeline_exports_probability_and_identical_layout_controls(self):
        root=self.bundle();observed=read(self.root/'qwen/observer3d/G_obs.json')
        for exp in ('end_to_end','layout_control'):
            for backend in ('qwen','api'):
                result=run(SceneFixture(backend),root/exp,backend)
                self.assertEqual(result['status'],'worlds_exported_needs_review',result)
                self.assertTrue(result['empirical_mechanisms_used']);self.assertFalse(result['count_likelihood_defined'])
                for record in result['worlds']:
                    full=read(root/exp/backend/'completion'/record['folder']/'G_full.json')
                    self.assertEqual(full['objects'][:len(observed['objects'])],observed['objects'])
                    self.assertTrue(all(0<o['generative_presence_probability']<=1 for o in full['objects'][len(observed['objects']):]))
        paths=[root/'layout_control'/b/'completion/structural_samples.json' for b in ('qwen','api')]
        self.assertEqual(sha(paths[0]),sha(paths[1]))
        page=experiment.report(root);self.assertTrue(page.is_file())
        self.assertTrue(read(root/'comparison.json')['layout_control']['structural_samples_identical'])
        self.assertGreater(read(root/'mechanism_diagnostics.json')['qwen']['max_profile_mechanism_TV'],0.)
        self.assertTrue(experiment.report(root).is_file()) # rebuilding preserves previous gallery
        model=read(root/'layout_control/qwen/completion/causal_model.json');world=sd.sample_worlds(model,count=1)[0]
        from .conditional_world_geometry import support_plane
        prompt=geometry.layout_prompt(self.case,observed,world,support_plane(observed,['obs_000']))
        template=json.loads(prompt.split('\nTEMPLATE: ')[1].split('\nFIXED SEMANTIC WORLD: ')[0])
        for group in template['groups']:
            self.assertIsNone(group['range_m'])
            self.assertTrue(all(o['offset_m']==[None,None,None] and o['yaw_deg'] is None for o in group['objects']))

    def test_missing_mechanisms_are_blocked_without_fabricated_worlds(self):
        root=self.prepare()
        for exp in ('end_to_end','layout_control'):
            for backend in ('qwen','api'):
                self.assertFalse(experiment.ready(root,exp,backend)['ready'])
                status=experiment.block(root,exp,backend);self.assertFalse(status['world_state_ready'])
        self.assertTrue(experiment.report(root).is_file())
        self.assertFalse(list(root.glob('*/api/completion/world_*/G_full.json')))

    def test_changed_domain_or_observation_cannot_reuse_frozen_mechanisms(self):
        root,model=self.model();domain=read(root/'shared/domain.json');case=read(root/'shared/case.json')
        domain['profiles'][0]['description']='Modified after estimation.';write(root/'shared/domain.json',domain)
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            sd.prepare_model(domain['base_model'],case,dict(sdscene_bundle=str(root/'shared'),
                mechanism_source='per_backend',execution_backend='qwen'),None)
        case['hfov']=case['hfov']+1
        with self.assertRaisesRegex(ValueError,'camera mismatch'):sd.load_domain(case,dict(sdscene_bundle=str(root/'shared')))

    def test_failed_proposals_skip_dependent_GPU_work_and_still_build_report(self):
        binary=self.temp/'fakebin';binary.mkdir();events=self.temp/'events.txt';shim=binary/'fake_python'
        shim.write_text('''#!/usr/bin/env python3
import os,sys
from pathlib import Path
args=sys.argv[1:];module=args[args.index('-m')+1];action=args[args.index('-m')+2]
root=Path(args[args.index('--root')+1]);root.mkdir(exist_ok=True)
if action=='prepare':
    assert args[args.index('--api-model')+1]=='deepseek-ai/DeepSeek-V4.1-Flash'
    assert os.environ['SILICONFLOW_API_BASE']=='https://api.siliconflow.cn/v1'
with open(os.environ['SD_TEST_EVENTS'],'a') as stream:stream.write(action+'\\n')
if action in ('propose','pool','ready'):sys.exit(1)
''');shim.chmod(0o755)
        gpu=binary/'srun';gpu.write_text('''#!/usr/bin/env python3
import os,sys
with open(os.environ['SD_TEST_EVENTS'],'a') as stream:stream.write('gpu\\n')
sys.exit(1)
''');gpu.chmod(0o755)
        env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],CAUPANO_QWEN_PYTHON=str(shim),
            CAUPANO_SDSCENE_OUTPUT=str(self.temp/'shell_output'),SD_TEST_EVENTS=str(events),SILICONFLOW_API_KEY='offline_fixture_key',
            SILICONFLOW_MODEL='stale_wrong_model',SILICONFLOW_API_BASE='https://stale.invalid/v1')
        script=Path(__file__).resolve().parents[2]/'scripts/sdscene_compare.sh'
        result=subprocess.run(['bash',str(script),'compare','FAKE_NODE',str(self.root)],env=env,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,1,result.stderr);lines=events.read_text().splitlines()
        self.assertEqual(lines.count('gpu'),1);self.assertEqual(lines.count('block'),4)
        self.assertEqual(lines.count('report'),1);self.assertNotIn('mechanisms',lines)

    def test_shell_requires_key_before_launching_any_experiment(self):
        env=dict(os.environ);env.pop('SILICONFLOW_API_KEY',None)
        script=Path(__file__).resolve().parents[2]/'scripts/sdscene_compare.sh'
        result=subprocess.run(['bash',str(script),'compare','FAKE_NODE',str(self.root)],env=env,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,2);self.assertIn('SILICONFLOW_API_KEY',result.stderr)


if __name__=='__main__':unittest.main()
