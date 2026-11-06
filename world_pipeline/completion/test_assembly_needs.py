"""Mechanism semantics and synthetic end-to-end export, without model calls."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from . import assembly_needs as needs, assembly_delta as delta
from . import test_assembly as fixture
from .world_scm import forward
from .assembly_world import run
from .assembly_experiment import restart, model_info
from ..common import read, write


def mechanism():
    def item(key, category, kind='floor', host=None, aff=()):
        return dict(key=key, category=category, variant=category, reason='A separate possible unseen instance.',
            placement='Unphotographed surrounding area.', attachment=dict(kind=kind, host=host), affordances=list(aff))
    def strategy(key, ids, description):
        return dict(key=key, description=description, object_keys=ids, observed_refs=['obs_000'], relations=[])
    return dict(description='Reading, storage and incidental belongings.', groups=[dict(
        function='Surrounding reading/storage area.', reason='Uncertain organization beyond the local view.',
        observed_refs=['obs_000'], new_objects=[item('table','side table',aff=['top']),
            item('lamp','table lamp','top','table'),item('book','book','top','table'),
            item('chair','chair',aff=['seat']),item('plant','plant')])], requirements=[
        dict(key='reading', purpose='An intended reading use.', kind='activity', evidence_refs=['obs_000'],
            assumptions=['Possible reading use.'], strategies=[
                strategy('existing', [], 'Observed inventory suffices; no additional reading setup.'),
                strategy('corner', ['table','lamp','chair','book'], 'Separate shared reading/storage corner.')]),
        dict(key='storage', purpose='Book storage.', kind='storage', evidence_refs=[],
            assumptions=['A stored book.'], strategies=[strategy('shared', ['table','book'], 'Book stored on shared table.')]),
        dict(key='incidental', purpose='An incidental plant.', kind='background', evidence_refs=[],
            assumptions=['Ordinary lived-in belongings.'], strategies=[
                strategy('none', [], 'No extra background instance.'),strategy('plant', ['plant'], 'An incidental plant.')])])


class NeedsFixture(fixture.GrammarFixture):
    def ask(self, image, prompt, path, **kwargs):
        path = Path(path)
        if path.stem.startswith('furniture_'):
            from .world_generation import image_bytes
            import hashlib
            write(path.with_suffix('.request.json'),dict(prompt=prompt,
                canonical_image_sha256=hashlib.sha256(image_bytes(image)).hexdigest()))
            value=mechanism();path.write_text(json.dumps(value));write(path.with_suffix('.json'),value)
            return value
        return super().ask(image,prompt,path,**kwargs)


class BaselineFixture(fixture.GrammarFixture):
    def ask(self,image,prompt,path,**kwargs):
        original=prompt
        if Path(path).stem.startswith('furniture_'):
            prompt=prompt[prompt.index('Complete the UNOBSERVED surrounding world'):]
        value=super().ask(image,prompt,path,**kwargs)
        request=Path(path).with_suffix('.request.json');record=read(request)
        record['prompt']=original;write(request,record)
        return value


class NeedsTests(unittest.TestCase):
    setUp=fixture.AssemblyTests.setUp
    initialize=fixture.AssemblyTests.initialize

    def model(self):
        z=needs.hypotheses_answer(fixture.hypotheses())
        p=needs.furniture_answer(mechanism(),self.case,z[0])[0]
        return needs.build_model(z,{'f0':p})

    def world(self):
        m=self.model();noise={n['id']:.1 for n in m['nodes']}
        noise['B_z0_reading']=.9;noise['B_z0_incidental']=.9
        w=needs.active_world(m,forward(m,noise));w['noise']=noise
        return m,w

    def test_need_removal_preserves_shared_instances_and_unrelated_belongings(self):
        m,w=self.world();after=needs.active_world(m,forward(m,w['noise'],{'R_z0_reading':'0'}))
        self.assertEqual({o['key'] for o in after['objects']},{'table','book','plant'})
        self.assertEqual({o['key'] for o in w['objects']},{'table','book','plant','chair','lamp'})
        self.assertEqual(len(w['objects']),len({o['id'] for o in w['objects']}))

    def test_removing_host_preserves_inventory_but_invalidates_installation(self):
        m,w=self.world();after=needs.active_world(m,forward(m,w['noise'],{'E_z0_table':'0'}))
        self.assertEqual({o['key'] for o in after['objects']},{'book','lamp','chair','plant'})
        self.assertEqual(set(after['unresolved_installations']),{'z0_book','z0_lamp'})

    def test_semantic_conditioning_is_different_from_do(self):
        m,w=self.world();post=needs.semantic_posterior(m,{'E_z0_lamp':'1'})
        self.assertAlmostEqual(post['evidence_probability'],.5)
        self.assertTrue(all(r['state']['B_z0_reading']=='corner' for r in post['rows']))
        noise=deepcopy(w['noise']);noise['B_z0_reading']=.1
        after=forward(m,noise,{'E_z0_lamp':'1'})
        self.assertEqual(after['B_z0_reading'],'existing')

    def test_semantic_abduction_replays_and_uses_evidence_conditioned_noise(self):
        m=self.model();factual=needs.abduct_semantics(m,{'E_z0_lamp':'1'},seed=22)
        self.assertEqual(forward(m,factual['noise']),factual['factual_state'])
        self.assertEqual(factual['factual_state']['B_z0_reading'],'corner')
        after=forward(m,factual['noise'],{'R_z0_reading':'0'})
        self.assertEqual(after['E_z0_lamp'],'0');self.assertEqual(after['E_z0_book'],'1')
        with self.assertRaisesRegex(ValueError,'zero probability'):
            needs.semantic_posterior(m,{'E_z0_book':'0'})

    def test_iid_reference_samples_retain_duplicate_worlds(self):
        worlds=needs.sample_worlds(self.model(),count=20)
        self.assertEqual(len(worlds),20)
        self.assertLess(len({tuple(o['id'] for o in w['objects']) for w in worlds}),20)

    def test_realization_requires_its_own_support_ancestors(self):
        raw=mechanism();raw['requirements'][0]['strategies'][1]['object_keys'].remove('table')
        with self.assertRaisesRegex(ValueError,'support ancestor'):
            needs.furniture_answer(raw,self.case,needs.hypotheses_answer(fixture.hypotheses())[0])

    def test_model_never_calls_reference_law_a_visual_posterior(self):
        m=self.model()
        for key in ('real_causal_structure_identified','calibrated_posterior_defined','RGB_likelihood_defined'):
            self.assertFalse(m[key])
        self.assertEqual(len(needs.semantic_posterior(m,{})['rows']),4)
        self.assertAlmostEqual(needs.semantic_posterior(m,{})['evidence_probability'],1.)

    def test_real_failed_run_drives_a_single_unambiguous_prompt_schema(self):
        hypothesis=needs.hypotheses_answer(fixture.hypotheses())[0]
        prompt=needs.furniture_prompt(self.case,hypothesis)
        self.assertNotIn('"alternatives"',prompt)
        self.assertNotIn('Complete the UNOBSERVED surrounding world',prompt)
        schema=json.loads(prompt.split('SINGLE RESPONSE SCHEMA: ')[1].split('\nSCENE HYPOTHESIS:')[0])
        self.assertEqual(set(schema),{'description','groups','requirements'})
        source=Path(__file__).resolve().parents[2]/'outputs/causal_completion_pilot_20261024_220629_380006_3407/needs/qwen/completion/calls/furniture_z0_repair.json'
        if source.is_file():
            raw=read(source);before=deepcopy(raw)
            with self.assertRaisesRegex(ValueError,'shared catalog'):
                needs.furniture_answer(raw,self.case,hypothesis)
            self.assertEqual(raw,before)

    def test_multiple_scene_hypotheses_have_disjoint_catalogs(self):
        z0=needs.hypotheses_answer(fixture.hypotheses())[0];z1=dict(deepcopy(z0),id='z1')
        p0=needs.furniture_answer(mechanism(),self.case,z0)[0]
        p1=needs.furniture_answer(mechanism(),self.case,z1)[0]
        m=needs.build_model([z0,z1],{'f0':p0,'f1':p1})
        posterior=needs.semantic_posterior(m,{'E_z0_lamp':'1'})
        self.assertAlmostEqual(posterior['evidence_probability'],.25)
        self.assertTrue(all(r['state']['Z']=='z0' and r['state']['E_z1_lamp']=='0' for r in posterior['rows']))

    def test_real_hang_alias_is_annotated_without_changing_raw_reply(self):
        source=Path(__file__).resolve().parents[2]/'outputs/assembly_rgb_compare_20261024_211032_189937/qwen/completion/calls/furniture_z0.json'
        if not source.is_file():self.skipTest('Saved real Qwen reply unavailable in this checkout')
        raw=read(source);before=deepcopy(raw)
        context=read(source.parents[1]/'case.json')
        z=read(source.parents[1]/'hypotheses.json')[0]
        parsed=delta.furniture_answer(raw,context,z)
        records=[o for p in parsed for g in p['groups'] for o in g['objects'] if o['key']=='wall_hooks_001']
        self.assertTrue(records)
        self.assertEqual(records[0]['affordances'],['hanging'])
        self.assertEqual(records[0]['interface_normalizations'][0]['original'],'hang')
        self.assertEqual(raw,before)

    def test_end_to_end_needs_exports_frozen_observation_and_actual_boxes(self):
        config=read(self.root/'experiment.json');config.update(model_info('needs'),worlds=3)
        write(self.root/'experiment.json',config);before=deepcopy(self.observed)
        result=run(NeedsFixture(),self.root,'qwen')
        self.assertEqual(result['status'],'worlds_exported_needs_review')
        self.assertEqual(result['model_variant'],needs.MODEL_VARIANT)
        out=self.root/'qwen/completion'
        evidence=read(out/'calls/hypotheses.request.json')['canonical_image_sha256']
        self.assertEqual(read(out/'calls/furniture_z0.request.json')['canonical_image_sha256'],evidence)
        for row in result['worlds']:
            full=read(out/row['folder']/'G_full.json')
            self.assertEqual(full['objects'][:len(before['objects'])],before['objects'])
            self.assertTrue((out/row['folder']/'world/scene_3d.html').is_file())
            self.assertTrue((out/row['folder']/'world/erp_state_map.png').is_file())

    def test_restart_selects_model_seed_and_world_count_without_reusing_semantics(self):
        self.base.observe(self.root,'api',True)
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            out=restart(self.temp/'needs_restart',self.root,'needs',73,4)
        config=read(out/'experiment.json')
        self.assertEqual(config['completion_model'],'needs');self.assertEqual(config['seed'],73)
        self.assertEqual(config['worlds'],4);self.assertFalse((out/'qwen/completion').exists())
        self.assertEqual(read(out/'qwen/observer3d/G_obs.json'),self.observed)

    def test_pilot_preparation_and_report_verify_matched_observations(self):
        from . import assembly_pilot as pilot
        self.base.observe(self.root,'api',True)
        def create(p):p.mkdir();return p
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=create),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_pilot.new_output',side_effect=create):
            out=pilot.prepare(self.temp/'pilot',self.root)
        result=run(BaselineFixture(),out/'baseline','qwen')
        self.assertEqual(result['status'],'worlds_exported_needs_review')
        page=pilot.report(out)
        self.assertTrue(page.is_file());data=read(out/'comparison.json')
        self.assertEqual(len(data['records']),6)
        changed=read(out/'needs/qwen/observer3d/G_obs.json');changed['objects'][0]['description']='Changed evidence'
        write(out/'needs/qwen/observer3d/G_obs.json',changed)
        with self.assertRaisesRegex(ValueError,'G_obs differs'):
            pilot.report(out)

    def test_all_observed_realization_can_export_an_empty_hidden_delta(self):
        class EmptyFixture(NeedsFixture):
            def ask(inner,image,prompt,path,**kwargs):
                result=super().ask(image,prompt,path,**kwargs)
                if Path(path).stem.startswith('furniture_'):
                    result=dict(description='No additional hidden instances assumed.',groups=[],requirements=[])
                    Path(path).write_text(json.dumps(result));write(Path(path).with_suffix('.json'),result)
                return result
        config=read(self.root/'experiment.json');config.update(model_info('needs'),worlds=1)
        write(self.root/'experiment.json',config)
        result=run(EmptyFixture(),self.root,'qwen')
        self.assertEqual(result['status'],'worlds_exported_needs_review')
        self.assertEqual(result['worlds'][0]['new_instances'],0)
        full=read(self.root/'qwen/completion/world_01/G_full.json')
        self.assertEqual(full['objects'],self.observed['objects'])


if __name__=='__main__':unittest.main()
