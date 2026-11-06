"""Behavioral regressions for generative inference, API wire format and exports."""
from copy import deepcopy
from contextlib import redirect_stdout,redirect_stderr
import io
import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import numpy as np
from PIL import Image

from ..common import read,write
from .test_causal_world_completion import make_case
from .world_api import APIRequestError,SiliconFlowJSON
from .world_scm import abduct,enumerate_model,forward,independent_groups,mechanism_strengths,validate
from .world_scm_fixture import FixtureProvider,fixture_domain
from .world_scm_runner import (ask_valid,execute,geometry_records,geometry_response,grounding_prompt,
    grounding_response,domain_response,layout_prompt,load_checkpoint,observed_geometry,rank_worlds,require_instance_ids,source_hashes)
from .world_scm_runner import mechanism_prompt,mechanism_response
from .world_generation import MATCHED_OPTIONS,SYSTEM_PROMPT,MatchedLocalJSON,image_bytes
from .world_scm_spec import build_model,domain_answer,geometry_answer,table_answer


def tiny_model():
    return dict(nodes=[dict(id='C',states=['c0','c1'],parents=[],cpt={'root':[.5,.5]}),
        dict(id='S',states=['s0','s1'],parents=['C'],cpt={'c0':[.9,.1],'c1':[.1,.9]}),
        dict(id='E',states=['other','signature'],parents=['C','S'],cpt={
            'c0|s0':[.1,.9],'c0|s1':[.7,.3],'c1|s0':[.1,.9],'c1|s1':[.7,.3]}),
        dict(id='g0',states=['0','1'],parents=['C','S'],cpt={k:[.5,.5] for k in ['c0|s0','c0|s1','c1|s0','c1|s1']}),
        dict(id='g1',states=['0','1'],parents=['C','S','g0'],cpt={
            f'{c}|{s}|{g}':([.9,.1] if g=='0' else [.1,.9]) for c in ['c0','c1'] for s in ['s0','s1'] for g in ['0','1']})])


class SCMTests(unittest.TestCase):
    def test_observations_update_latent_distribution_by_bayes_rule(self):
        model=tiny_model();prior=enumerate_model(model);posterior=enumerate_model(model,{'E':'signature'})
        self.assertAlmostEqual(prior['marginals']['C']['c0'],.5)
        self.assertAlmostEqual(posterior['evidence_probability'],.6)
        self.assertAlmostEqual(posterior['marginals']['C']['c0'],.7)
        self.assertAlmostEqual(sum(r['probability'] for r in posterior['rows']),1.)

    def test_do_replaces_mechanism_instead_of_conditioning_on_it(self):
        model=tiny_model();conditioned=enumerate_model(model,{'S':'s1'});intervened=enumerate_model(model,do={'S':'s1'})
        self.assertAlmostEqual(conditioned['marginals']['C']['c1'],.9)
        self.assertAlmostEqual(intervened['marginals']['C']['c1'],.5)
        self.assertAlmostEqual(enumerate_model(model,do={'g0':'1'})['marginals']['g1']['1'],.9)
        self.assertAlmostEqual(enumerate_model(model,do={'g0':'0'})['marginals']['g1']['1'],.1)

    def test_abducted_noise_reproduces_factual_and_preserves_non_descendants(self):
        model=tiny_model();sample=abduct(model,{'E':'signature'},72)
        self.assertEqual(forward(model,sample['noise']),sample['factual_state'])
        other=forward(model,sample['noise'],{'g0':'1'})
        self.assertEqual([other[k] for k in ['C','S','E']],[sample['factual_state'][k] for k in ['C','S','E']])
        self.assertEqual(abduct(model,{'E':'signature'},72),sample)

    def test_independent_ablation_keeps_marginals_and_changes_joint_dependence(self):
        model=tiny_model();cut=independent_groups(model)
        before=enumerate_model(model,{'E':'signature'});after=enumerate_model(cut,{'E':'signature'})
        self.assertEqual(next(n for n in cut['nodes'] if n['id']=='g1')['parents'],['C','S'])
        for name in ['C','S','g0','g1']:
            for state,p in before['marginals'][name].items():self.assertAlmostEqual(p,after['marginals'][name][state])
        def both(rows):return sum(r['probability'] for r in rows if r['state']['g0']=='1' and r['state']['g1']=='1')
        self.assertAlmostEqual(both(before['rows']),.45);self.assertAlmostEqual(both(after['rows']),.25)

    def test_invalid_tables_and_zero_likelihood_are_explicit_errors(self):
        model=tiny_model();model['nodes'][0]['cpt']['root']=[.7,.7]
        with self.assertRaises(ValueError):validate(model)
        model=tiny_model();model['nodes'][2]['cpt']={k:[1.,0.] for k in model['nodes'][2]['cpt']}
        with self.assertRaisesRegex(ValueError,'zero mass'):enumerate_model(model,{'E':'signature'})

    def test_declared_edge_strength_is_measured_in_actual_CPT(self):
        strengths=mechanism_strengths(tiny_model())
        edge=next(r for r in strengths if r['parent']=='g0' and r['child']=='g1')
        self.assertAlmostEqual(edge['max_CPT_total_variation'],.8)
        irrelevant=next(r for r in strengths if r['parent']=='C' and r['child']=='g0')
        self.assertEqual(irrelevant['max_CPT_total_variation'],0.)

    def test_merged_world_uses_highest_mass_latent_explanation_for_layout(self):
        posterior=dict(rows=[dict(state={'C':'c0','g0':'1'},probability=.1),dict(state={'C':'c1','g0':'1'},probability=.9)])
        ranked=rank_worlds(posterior,dict(groups=[dict(id='g0')]))
        self.assertEqual(len(ranked),1);self.assertEqual(ranked[0]['state']['C'],'c1')
        self.assertEqual(ranked[0]['probability'],1.)


class InstanceInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.case,_=make_case(self.root);self.case['image_size']=[64,64]
        self.domain=domain_answer(fixture_domain(),self.case)
        self.state={'C':'c0','S':'s0','E':'signature','g0':'1','g1':'1'}

    def test_grounding_template_covers_every_supplied_identity(self):
        self.case['context']=[dict(id=f'observed_{i:04d}',description='visible item') for i in range(9)]
        # The inventory formatter also reads category/box; isolate only those
        # independent payload fields to focus this regression on identity scope.
        with patch('qwen_edit_pano.world_pipeline.completion.world_scm_runner.visible_inventory',return_value=[]):
            prompt=grounding_prompt(self.case)
        template=json.loads(prompt.split('REQUIRED OUTPUT TEMPLATE: ',1)[1].split('\nFROZEN OBSERVED INVENTORY:',1)[0])
        self.assertEqual([r['id'] for r in template['instances']],[r['id'] for r in self.case['context']])
        self.assertTrue(all(r['variant'] is None and r['on_floor'] is None for r in template['instances']))

    def test_partial_grounding_reports_exact_missing_ids_and_fabricates_nothing(self):
        self.case['context']=[dict(id=f'observed_{i:04d}') for i in range(9)]
        reply=dict(instances=[dict(id='observed_0006',variant='tray',on_floor=False),
                              dict(id='observed_0008',variant='ottoman',on_floor=True)],evidence_description='Tray on ottoman.')
        before=deepcopy(reply)
        with self.assertRaises(ValueError) as error:grounding_response(reply,self.case)
        details=json.loads(str(error.exception))
        self.assertEqual(details['missing_ids'],[f'observed_{i:04d}' for i in [0,1,2,3,4,5,7]])
        self.assertEqual(details['returned_count'],2);self.assertEqual(reply,before)

    def test_identity_errors_distinguish_missing_unknown_duplicate_and_wrong_shape(self):
        for value in [None,dict(objects=None),dict(objects=[dict(id=1)])]:
            with self.assertRaises(ValueError):require_instance_ids(value,'objects',['a','b'])
        with self.assertRaises(ValueError) as error:
            require_instance_ids(dict(objects=[dict(id='a'),dict(id='a'),dict(id='c')]),'objects',['a','b'])
        details=json.loads(str(error.exception))
        self.assertEqual(details['missing_ids'],['b']);self.assertEqual(details['unknown_ids'],['c'])
        self.assertEqual(details['duplicate_ids'],['a'])

    def geometry_reply(self):
        return FixtureProvider(self.case).ask(None,layout_prompt(self.domain,self.state,self.case,{}),self.root/'layout_01.txt')

    def test_layout_template_has_actual_prefixed_ids_and_literal_anchor_domains(self):
        prompt=layout_prompt(self.domain,self.state,self.case,{})
        template=json.loads(prompt.split('REQUIRED OUTPUT TEMPLATE: ',1)[1].split('\nFIXED INSTANCE CONTRACTS:',1)[0])
        records=geometry_records(self.domain,self.state,self.case)
        self.assertEqual([r['id'] for r in template['objects']],['g0_table','g0_light','g1_shelf','g1_plant'])
        light=next(r for r in records if r['id']=='g0_light')
        self.assertEqual(light['allowed_anchor_values'],['g0_table'])
        self.assertEqual(template['objects'][1]['anchor'],'g0_table')
        self.assertNotIn('camera/originalObservedID/localKeyINTHESAMEGROUP',prompt)

    def test_metric_layout_receives_exact_frozen_anchor_coordinates_without_changing_state(self):
        before=read(self.case['state']);geometry=observed_geometry(self.case)
        self.assertEqual(geometry,[dict(id=o['track_id'],center_m=o['center_local_m']['value'],
            full_AABB_size_m=o['size_aabb_local_m']['value']) for o in before['objects']])
        prompt=layout_prompt(self.domain,self.state,self.case,{})
        actual=json.loads(prompt.split('FROZEN OBSERVED CAMERA GEOMETRY (predicted metres, not GT; use for observed anchors): ',1)[1].split('\nCAMERA AND LEVEL PRIOR:',1)[0])
        self.assertEqual(actual,geometry);self.assertEqual(read(self.case['state']),before)

    def test_unprefixed_ids_are_rejected_with_exact_expected_ids(self):
        value=self.geometry_reply()
        for row in value['objects']:row['id']=row['id'][3:]
        with self.assertRaises(ValueError) as error:geometry_response(value,self.domain,self.state,self.case)
        details=json.loads(str(error.exception))
        self.assertEqual(details['unknown_ids'],['table','light','shelf','plant'])
        self.assertEqual(details['missing_ids'],['g0_table','g0_light','g1_shelf','g1_plant'])

    def test_concatenated_anchor_is_rejected_instead_of_guessed(self):
        value=self.geometry_reply();value['objects'][0]['anchor']='camera/originalObservedID/observed_0000'
        with self.assertRaises(ValueError) as error:geometry_response(value,self.domain,self.state,self.case)
        details=json.loads(str(error.exception));self.assertEqual(details['field'],'anchor')
        self.assertIn('camera',details['allowed_values']);self.assertEqual(value['objects'][0]['anchor'],details['received'])

    def test_reference_cycle_is_repaired_before_numerical_layout(self):
        value=self.geometry_reply();value['objects'][0]['anchor']='light'
        with self.assertRaisesRegex(ValueError,'[Cc]ycl|anchor'):
            geometry_response(value,self.domain,self.state,self.case)

    def test_valid_wire_response_has_identical_geometry_to_existing_compiler(self):
        value=self.geometry_reply();before=deepcopy(value)
        self.assertEqual(geometry_response(value,self.domain,self.state,self.case),geometry_answer(value,self.domain,self.state,self.case))
        self.assertEqual(value,before)

    def test_qualified_anchor_binds_to_exact_same_host_without_changing_numeric_fields(self):
        value=self.geometry_reply();before=deepcopy(value)
        value['objects'][1]['anchor']='g0_table'
        normalized,converted=geometry_response(value,self.domain,self.state,self.case)
        expected,_=geometry_answer(before,self.domain,self.state,self.case)
        self.assertEqual(normalized,expected)
        self.assertEqual(converted['g0_light']['reference_binding'],dict(wire_anchor='g0_table',canonical_anchor='g0_table',compiler_anchor='table'))
        self.assertEqual(value['objects'][1]['anchor'],'g0_table')

    def test_qualified_cross_group_or_unknown_anchor_is_not_suffix_matched(self):
        for anchor in ['g1_table','g1_shelf','object/table','g0_unknown']:
            value=self.geometry_reply();value['objects'][1]['anchor']=anchor
            with self.assertRaises(ValueError):geometry_response(value,self.domain,self.state,self.case)

    def test_domain_root_and_nested_container_errors_are_repairable_value_errors(self):
        values=[[dict(description='A list is not a complete domain')],None]
        for key in ['contexts','organizations','groups']:
            raw=fixture_domain();raw[key]=['incorrect record'];values.append(raw)
        for field in ['modes','parents']:
            raw=fixture_domain();raw['groups'][0][field]=[{}] if field=='parents' else [1];values.append(raw)
        raw=fixture_domain();raw['groups'][0]['modes'][0]['objects'][0]['support']=[];values.append(raw)
        for raw in values:
            with self.assertRaises(ValueError):domain_response(raw,self.case,dict(groups=4,max_new=24))
        self.assertEqual(domain_response(fixture_domain(),self.case,dict(groups=4,max_new=24)),self.domain)

    def test_target_only_support_round_trips_local_hosts_surfaces_and_observed_hosts(self):
        budgets=dict(groups=4,max_new=24)
        for target in ['floor','wall','ceiling','unspecified',self.case['context'][0]['id']]:
            raw=fixture_domain()
            for group in raw['groups']:
                for mode in group['modes']:
                    for obj in mode['objects']:
                        support=obj.pop('support');obj['support_target']=support['target'] or support['kind']
            raw['groups'][1]['modes'][0]['objects'][1]['support_target']=target
            before=deepcopy(raw);bindings=[]
            normalized=domain_response(raw,self.case,budgets,bindings=bindings)
            expected=deepcopy(self.domain)
            expected['groups'][1]['modes'][1]['objects'][1]['support']=(dict(kind=target,target=None) if target in
                ['floor','wall','ceiling','unspecified'] else dict(kind='object',target=target))
            self.assertEqual(normalized,expected);self.assertEqual(raw,before)
            self.assertEqual(len(bindings),4)
            self.assertEqual(normalized['groups'][0]['modes'][1]['objects'][1]['support'],dict(kind='object',target='table'))

    def test_legacy_host_category_binds_only_to_that_exact_host_and_keeps_semantics(self):
        raw=fixture_domain();light=raw['groups'][0]['modes'][0]['objects'][1]
        light['support']['kind']='table';before=deepcopy(raw);bindings=[]
        self.assertEqual(domain_response(raw,self.case,dict(groups=4,max_new=24),bindings=bindings),self.domain)
        self.assertEqual(bindings,[dict(path='groups[0].modes[0].objects[1]',source=dict(support=light['support']),
            canonical_support=dict(kind='object',target='table'),rule='exact_referenced_host_category')])
        self.assertEqual(raw,before)
        observed=self.case['context'][0];light['support']=dict(kind=observed['category'],target=observed['id'])
        result=domain_response(raw,self.case,dict(groups=4,max_new=24))
        self.assertEqual(result['groups'][0]['modes'][1]['objects'][1]['support'],dict(kind='object',target=observed['id']))

    def test_unknown_self_cross_group_or_ambiguous_support_never_guesses_a_host(self):
        invalid=[dict(support_target='sofa'),dict(support_target='shelf'),dict(support_target='light'),
            dict(support_target=None),dict(support=dict(kind='chair',target='table')),
            dict(support=dict(kind='sofa',target='unknown')),
            dict(support=dict(kind='floor',target='table')),
            dict(support_target='wall',support=dict(kind='object',target='table'))]
        for fields in invalid:
            with self.subTest(fields=fields):
                raw=fixture_domain();obj=raw['groups'][0]['modes'][0]['objects'][1];obj.pop('support');obj.update(fields)
                before=deepcopy(raw);bindings=[]
                with self.assertRaises(ValueError) as error:domain_response(raw,self.case,dict(groups=4,max_new=24),bindings=bindings)
                details=json.loads(str(error.exception));self.assertEqual(details['errors'][0]['path'],'groups[0].modes[0].objects[1]')
                self.assertIn('table',details['errors'][0]['allowed_instance_targets'])
                self.assertNotIn('shelf',details['errors'][0]['allowed_instance_targets'])
                self.assertEqual(raw,before);self.assertEqual(bindings,[])

    def test_all_support_errors_are_reported_together_and_reference_cycles_still_fail(self):
        raw=fixture_domain()
        for obj in raw['groups'][0]['modes'][0]['objects']:obj['support']=dict(kind='wrong',target='unknown')
        with self.assertRaises(ValueError) as error:domain_response(raw,self.case,dict(groups=4,max_new=24))
        self.assertEqual(len(json.loads(str(error.exception))['errors']),2)
        raw=fixture_domain()
        for obj,target in zip(raw['groups'][0]['modes'][0]['objects'],['light','table']):
            obj.pop('support');obj['support_target']=target
        with self.assertRaisesRegex(ValueError,'[Cc]ycl'):domain_response(raw,self.case,dict(groups=4,max_new=24))

    def test_repair_feedback_ends_with_all_errors_and_repeated_invalid_output_stops(self):
        invalid=fixture_domain();invalid['groups'][0]['modes'][0]['objects'][0]['support_target']='unknown'
        prompts=[]
        class Provider:
            def ask(inner,image,prompt,path,**kwargs):
                prompts.append(prompt);path.write_text(json.dumps(invalid));return deepcopy(invalid)
        with self.assertRaises(ValueError):ask_valid(Provider(),None,'Complete domain',self.root/'domain.txt',
            lambda v:domain_response(v,self.case,dict(groups=4,max_new=24)))
        self.assertEqual(len(prompts),2)
        correction=prompts[1].split('CORRECTION REQUIRED:',1)[1]
        self.assertIn('groups[0].modes[0].objects[0]',correction);self.assertIn('allowed_instance_targets',correction)
        self.assertLess(prompts[1].index('PREVIOUS REJECTED ANSWER'),prompts[1].index('CORRECTION REQUIRED:'))
        self.assertTrue((self.root/'domain_repair.validation.json').exists())

    def test_top_level_list_enters_one_repair_request_instead_of_attribute_error(self):
        prompts=[];valid=fixture_domain()
        class Provider:
            def ask(inner,image,prompt,path,**kwargs):
                prompts.append(prompt);value=[dict(description='partial')] if len(prompts)==1 else valid
                path.write_text(json.dumps(value));return value
        result=ask_valid(Provider(),None,'Return domain object with contexts, organizations, groups',self.root/'domain.txt',
            lambda v:domain_response(v,self.case,dict(groups=4,max_new=24)))
        self.assertEqual(result,self.domain);self.assertEqual(len(prompts),2)
        self.assertIn('top-level JSON must be an object',prompts[1])

    def test_single_retry_receives_specific_identity_differences_and_complete_template(self):
        valid=self.geometry_reply();invalid=deepcopy(valid);invalid['objects'][0]['id']='table';prompts=[]
        class Provider:
            def ask(inner,image,prompt,path,**kwargs):
                prompts.append(prompt);value=invalid if len(prompts)==1 else valid
                path.write_text(json.dumps(value));return value
        result=ask_valid(Provider(),None,layout_prompt(self.domain,self.state,self.case,{}),self.root/'request.txt',
            lambda v:geometry_response(v,self.domain,self.state,self.case))
        self.assertEqual(len(prompts),2)
        self.assertIn('"missing_ids": ["g0_table"]',prompts[1]);self.assertIn('"unknown_ids": ["table"]',prompts[1])
        self.assertIn('REQUIRED OUTPUT TEMPLATE:',prompts[1])
        self.assertEqual(result,geometry_answer(valid,self.domain,self.state,self.case))


class MechanismInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.payload=dict(variable='C',states=[dict(id='c0',description='a'),dict(id='c1',description='b')],
            declared_parent_domains={},required_given_rows=[{}],response_schema=dict(rows=[dict(given={},p=['old placeholder'])]))

    def test_actual_single_state_failure_has_separate_mass_and_cardinality_errors(self):
        self.payload['states']=self.payload['states'][:1]
        for ps,message in [(['0.95'],'only declared state'),(['0.6','0.4'],'exactly one probability')]:
            raw=dict(rows=[dict(given={},p=ps)]);before=deepcopy(raw);bindings=[]
            with self.assertRaises(ValueError) as error:mechanism_response(raw,self.payload,bindings=bindings)
            detail=json.loads(str(error.exception))
            self.assertEqual(detail['variable'],'C');self.assertEqual(detail['required_state_ids'],['c0'])
            self.assertEqual(detail['required_state_count'],1);self.assertIn(message,detail['error'])
            self.assertEqual(raw,before);self.assertEqual(bindings,[])

    def test_keyed_probabilities_bind_by_ID_and_quoted_number_values_match_legacy_CPT(self):
        for raw in [dict(rows=[dict(given={},p_by_state={'c1':'0.4','c0':'6e-1'})]),
                    dict(rows=[dict(given={},p=['0.6','0.4'])]),dict(rows=[dict(given={},p=[.6,.4])])]:
            before=deepcopy(raw);bindings=[];canonical=mechanism_response(raw,self.payload,bindings=bindings)
            self.assertEqual(canonical,dict(rows=[dict(given={},p=[.6,.4])]))
            self.assertEqual(table_answer(canonical,dict(states=['c0','c1'],parents=[]),[{}]),{'root':[.6,.4]})
            self.assertEqual(raw,before)
        self.assertEqual(bindings,[])

    def test_invalid_values_states_containers_and_arbitrary_weights_are_never_filled_or_rescaled(self):
        invalid=[None,[],dict(rows=None),dict(rows=[[]]),dict(rows=[dict(given={},p=[.6])]),
            dict(rows=[dict(given={},p_by_state={'c0':1})]),dict(rows=[dict(given={},p_by_state={'c0':.6,'other':.4})]),
            dict(rows=[dict(given={},p=[.6,.4],p_by_state={'c0':.4,'c1':.6})]),
            dict(rows=[dict(given={'C':'c0'},p=[.6,.4])])]
        for values in [[.2,.3],['60%','40%'],[True,False],['NaN',1.],['Infinity',0.],[float('nan'),1.],
                       [float('inf'),0.],[-.1,1.1],['0.6 probability','.4'],[None,1.],['1/2','1/2'],[10**400,0]]:
            invalid.append(dict(rows=[dict(given={},p=values)]))
        for raw in invalid:
            with self.subTest(raw=raw):
                bindings=[]
                with self.assertRaises(ValueError):mechanism_response(raw,self.payload,bindings=bindings)
                self.assertEqual(bindings,[])

    def test_parent_assignments_can_be_reordered_but_not_omitted_duplicated_or_invented(self):
        self.payload.update(variable='S',declared_parent_domains={'C':[dict(id='c0'),dict(id='c1')]},
            required_given_rows=[{'C':'c0'},{'C':'c1'}])
        raw=dict(rows=[dict(given={'C':'c1'},p_by_state={'c1':.7,'c0':.3}),dict(given={'C':'c0'},p=[.6,.4])])
        canonical=mechanism_response(raw,self.payload)
        self.assertEqual(table_answer(canonical,dict(states=['c0','c1'],parents=['C']),self.payload['required_given_rows']),
            {'c1':[.3,.7],'c0':[.6,.4]})
        for given in [{'C':'c1'},{'C':'unknown'},{'C':'c0','extra':'x'},{}]:
            bad=deepcopy(raw);bad['rows'][1]['given']=given
            with self.assertRaises(ValueError):mechanism_response(bad,self.payload)

    def test_prompt_template_has_every_state_as_a_key_and_no_quoted_probability_placeholder(self):
        prompt=mechanism_prompt(self.payload);wire=json.loads(prompt.split('\n')[-1])
        self.assertEqual(wire['state_count'],2);self.assertEqual(wire['state_order'],['c0','c1'])
        self.assertEqual(wire['response_schema'],dict(rows=[dict(given={},p_by_state=dict(c0=None,c1=None))]))
        self.assertNotIn('old placeholder',prompt);self.assertIn('JSON NUMBER',prompt)


class APITests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        env=patch.dict(os.environ,{'CAUPANO_TEST_API_KEY':'secret-test-value'});env.start();self.addCleanup(env.stop)

    def test_image_json_wire_format_key_not_in_artifacts_and_response_model_recorded(self):
        requests=[]
        def fake(request,timeout):
            requests.append(request)
            return io.BytesIO(json.dumps(dict(model='served-1017',usage={'total_tokens':12},choices=[dict(
                finish_reason='stop',message={'content':'{"answer":1}','reasoning_content':'not required'})])).encode())
        client=SiliconFlowJSON(key_env='CAUPANO_TEST_API_KEY',opener=fake,stream=False)
        answer=client.ask(Image.new('RGB',(32,32)),'Return JSON',self.root/'answer.txt')
        payload=json.loads(requests[0].data)
        self.assertEqual(answer,{'answer':1});self.assertEqual(payload['model'],'deepseek-ai/DeepSeek-V4.1-Flash')
        self.assertTrue(payload['messages'][1]['content'][0]['image_url']['url'].startswith('data:image/jpeg;base64,'))
        self.assertNotIn('logprobs',payload);self.assertNotIn('enable_thinking',payload)
        for path in self.root.iterdir():
            self.assertNotIn('secret-test-value',path.read_text());self.assertNotIn('not required',path.read_text())
        self.assertEqual(read(self.root/'answer.response.json')['model'],'served-1017')

    def test_matched_API_recipe_sends_explicit_common_controls_and_image_hash(self):
        requests=[]
        def fake(request,timeout):
            requests.append(request);return io.BytesIO(b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"ok\\":true}"}}]}')
        client=SiliconFlowJSON(key_env='CAUPANO_TEST_API_KEY',opener=fake,stream=False,generation_profile='matched_v1')
        picture=Image.new('RGB',(32,32),'red');client.ask(picture,'Return JSON',self.root/'matched.txt')
        payload=json.loads(requests[0].data)
        self.assertEqual([payload[k] for k in ['temperature','top_p','top_k']],[.4,.9,50])
        self.assertFalse(payload['enable_thinking']);self.assertEqual(payload['messages'][0]['content'],SYSTEM_PROMPT)
        self.assertEqual(payload['response_format'],dict(type='text'))
        self.assertEqual(read(self.root/'matched.request.json')['canonical_image_sha256'],hashlib.sha256(image_bytes(picture)).hexdigest())
        self.assertNotIn('seed',payload)

    def test_matched_local_recipe_uses_same_system_image_and_explicit_sampling(self):
        import torch
        from types import SimpleNamespace
        captured={}
        class Batch(dict):
            @property
            def input_ids(inner):return inner['input_ids']
            def to(inner,device):return inner
        class Processor:
            def apply_chat_template(inner,messages,**kwargs):captured['messages']=messages;return 'formatted'
            def __call__(inner,**kwargs):captured['inputs']=kwargs;return Batch(input_ids=torch.tensor([[1,2]]))
            def batch_decode(inner,*args,**kwargs):return ['{"ok":true}']
        class Model:
            device='cpu';generation_config=SimpleNamespace(eos_token_id=9)
            def generate(inner,**kwargs):captured['generation']=kwargs;return torch.tensor([[1,2,3,9]])
        engine=SimpleNamespace(torch=torch,processor=Processor(),model=Model())
        client=MatchedLocalJSON('/unused/cpu_fake',engine=engine);picture=Image.new('RGB',(32,32),'red')
        self.assertEqual(client.ask(picture,'Return JSON',self.root/'local.txt'),dict(ok=True))
        self.assertEqual(captured['messages'][0]['content'],SYSTEM_PROMPT)
        for k,v in MATCHED_OPTIONS.items():self.assertEqual(captured['generation'][k],v)
        self.assertEqual(read(self.root/'local.request.json')['canonical_image_sha256'],hashlib.sha256(image_bytes(picture)).hexdigest())

    def test_http_error_is_not_retried_and_does_not_echo_body(self):
        calls=[]
        def fake(request,timeout):
            calls.append(1);raise urllib.error.HTTPError(request.full_url,401,'auth',{},io.BytesIO(b'secret-test-value'))
        client=SiliconFlowJSON(key_env='CAUPANO_TEST_API_KEY',opener=fake)
        with self.assertRaisesRegex(RuntimeError,'HTTP 401') as exc:client.ask(None,'JSON',self.root/'q.txt')
        self.assertNotIn('secret-test-value',str(exc.exception));self.assertEqual(len(calls),1)

    def test_truncated_answers_preserved_and_never_accepted(self):
        def fake(request,timeout):return io.BytesIO(b'{"choices":[{"finish_reason":"length","message":{"content":"{\\\"a\\\":1}"}}]}')
        client=SiliconFlowJSON(key_env='CAUPANO_TEST_API_KEY',opener=fake,stream=False)
        with self.assertRaisesRegex(ValueError,'incomplete'):client.ask(None,'JSON',self.root/'q.txt')
        self.assertTrue((self.root/'q.txt').exists());self.assertFalse((self.root/'q.json').exists())

    def test_model_list_and_request_budget(self):
        def fake(request,timeout):return io.BytesIO(b'{"data":[{"id":"deepseek-ai/DeepSeek-V4.1-Flash"}]}')
        client=SiliconFlowJSON(key_env='CAUPANO_TEST_API_KEY',max_calls=1,opener=fake)
        self.assertTrue(client.check_model()['listed'])
        with self.assertRaisesRegex(RuntimeError,'budget'):client.check_model()

    def test_stream_collects_final_json_usage_and_ignores_reasoning_text(self):
        requests=[]
        events=[dict(id='request-test',model='served-1017',choices=[dict(index=0,delta={'reasoning_content':'private reasoning'},finish_reason=None)]),
            dict(choices=[dict(index=0,delta={'content':'{"中文":'},finish_reason=None)]),
            dict(choices=[dict(index=0,delta={'content':'1}'},finish_reason=None)]),
            dict(choices=[dict(index=0,delta={},finish_reason='stop')]),dict(choices=[],usage={'total_tokens':44})]
        wire=b': heartbeat\n\n'+b''.join(('data: '+json.dumps(e,ensure_ascii=False)+'\n\n').encode() for e in events)+b'data: [DONE]\n\n'
        def fake(request,timeout):requests.append(request);return io.BytesIO(wire)
        client=SiliconFlowJSON(key_env='CAUPANO_TEST_API_KEY',opener=fake)
        value=client.ask(None,'Return JSON',self.root/'stream.txt')
        self.assertEqual(value,{'中文':1});self.assertTrue(json.loads(requests[0].data)['stream'])
        response=read(self.root/'stream.response.json');self.assertEqual(response['usage'],{'total_tokens':44})
        for path in self.root.iterdir():self.assertNotIn('private reasoning',path.read_text())

    def test_interrupted_stream_is_preserved_and_not_accepted_even_if_json_complete(self):
        event=dict(choices=[dict(delta={'content':'{"a":1}'},finish_reason='stop')])
        def fake(request,timeout):return io.BytesIO(('data: '+json.dumps(event)+'\n\n').encode())
        client=SiliconFlowJSON(key_env='CAUPANO_TEST_API_KEY',opener=fake)
        with self.assertRaisesRegex(APIRequestError,'incomplete_stream'):client.ask(None,'JSON',self.root/'broken.txt')
        self.assertEqual((self.root/'broken.partial.txt').read_text(),'{"a":1}')
        self.assertFalse((self.root/'broken.json').exists())
        self.assertEqual(read(self.root/'broken.transport_error.json')['category'],'incomplete_stream')

    def test_timeout_mid_stream_has_sanitized_diagnostics_and_no_retry(self):
        class Interrupted:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def __iter__(self):
                event=dict(choices=[dict(delta={'content':'{"a":'},finish_reason=None)])
                yield ('data: '+json.dumps(event)+'\n').encode();yield b'\n'
                raise TimeoutError('sensitive error detail must not be logged')
        calls=[]
        def fake(request,timeout):calls.append(1);return Interrupted()
        client=SiliconFlowJSON(key_env='CAUPANO_TEST_API_KEY',opener=fake)
        with self.assertRaisesRegex(APIRequestError,'timeout'):client.ask(None,'JSON',self.root/'slow.txt')
        info=read(self.root/'slow.transport_error.json');self.assertEqual(info['exception_type'],'TimeoutError')
        self.assertEqual(info['timeout_seconds'],600);self.assertEqual(len(calls),1)
        self.assertTrue(info['trace_id']);self.assertFalse((self.root/'slow.json').exists())
        for path in self.root.iterdir():self.assertNotIn('sensitive error detail',path.read_text())


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.case,self.obs=make_case(self.root);self.case['image_size']=[64,64]

    def test_preflight_and_bad_output_do_not_initialize_provider(self):
        from . import world_scm_runner as runner
        for bad in [False,True]:
            output=Path('1') if bad else self.root/'qwen_edit_pano/outputs/new_run'
            args=['world_scm_runner','--output',str(output),'--preflight']
            with patch('qwen_edit_pano.world_pipeline.common.ROOT',self.root),patch('sys.argv',args),\
                 patch.object(runner,'load_case',return_value=self.case),patch.object(runner,'SiliconFlowJSON') as api,\
                 redirect_stdout(io.StringIO()) as stdout,redirect_stderr(io.StringIO()):
                if bad:
                    with self.assertRaises(SystemExit) as exc:runner.main()
                    self.assertEqual(exc.exception.code,2)
                else:
                    runner.main();self.assertEqual(json.loads(stdout.getvalue())['network_started'],False)
                    self.assertFalse(output.exists())
                api.assert_not_called()

    def test_semantic_domain_and_geometric_axes_are_separate(self):
        domain=domain_answer(fixture_domain(),self.case)
        raw=dict(objects=[dict(id='g0_table',width_m=1.8,height_m=.005,depth_m=1.2,yaw_deg=90.,anchor='camera',offset_m=[0,0,-3]),
                          dict(id='g0_light',width_m=.3,height_m=.5,depth_m=.3,yaw_deg=0.,anchor='table',offset_m=[0,0,0])])
        converted,_=geometry_answer(raw,domain,{'g0':'1','g1':'0'},self.case)
        self.assertTrue(np.allclose(converted['groups'][0]['modes'][0]['objects'][0]['size_m'],[1.2,.005,1.8]))
        invalid=fixture_domain();invalid['groups'][0]['modes'][0]['objects'][0]['identity']='relocated_visible'
        with self.assertRaises(ValueError):domain_answer(invalid,self.case)
        invalid=fixture_domain();invalid['groups'][0]['modes'][0]['objects'][0]['support']={'kind':'wall','target':self.case['context'][0]['id']}
        with self.assertRaisesRegex(ValueError,'must be null'):domain_answer(invalid,self.case)

    def test_dense_parent_tables_are_chunked_without_dropping_states_or_edges(self):
        raw=fixture_domain();raw['contexts']=[dict(description=f'test context {i}') for i in range(3)]
        raw['organizations']=[dict(description=f'test organization {i}') for i in range(3)]
        template=raw['groups'][0];raw['groups']=[]
        for gi in range(4):
            group=deepcopy(template);group['parents']=list(range(gi));group['name']=f'test group {gi}'
            group['modes'].append(deepcopy(group['modes'][0]));group['modes'][1]['description']='test alternative'
            raw['groups'].append(group)
        domain=domain_answer(raw,self.case);queries=[]
        def ask(name,payload,validator):
            queries.append((name,len(payload['required_given_rows'])))
            n=len(payload['states'])
            return validator(dict(rows=[dict(given=r,p=[1/n]*n) for r in payload['required_given_rows']]))
        model=build_model(domain,'test evidence',ask)
        g3=next(n for n in model['nodes'] if n['id']=='g3')
        self.assertEqual(len(g3['cpt']),243);self.assertEqual(g3['parents'],['C','S','g0','g1','g2'])
        self.assertTrue(all(n<=24 for _,n in queries))

    def test_full_execution_and_cpu_replay_preserve_observations_and_weights(self):
        out=self.root/'first';out.mkdir();q=FixtureProvider(self.case);budgets=dict(groups=4,max_new=24)
        write(out/'config.json',dict(case=self.case,budgets=budgets,code_sha256=source_hashes()))
        record=execute(q,self.case,out,budgets)
        self.assertTrue(record['world_state_ready'],record)
        self.assertTrue(record['semantic_inference_complete']);self.assertTrue(record['no_geometric_renormalization'])
        self.assertEqual(record['layout_failures'],[])
        self.assertAlmostEqual(record['exported_semantic_mass']+record['unattempted_semantic_mass'],1.)
        self.assertNotAlmostEqual(record['latent_prior']['C']['c0'],record['latent_posterior']['C']['c0'])
        folder=out/self.case['id']
        for h in record['worlds']:
            full=read(folder/h['folder']/'G_full.json');self.assertEqual(full['objects'][:len(self.obs['objects'])],self.obs['objects'])
            with Image.open(folder/h['folder']/'world/erp_state_map.png') as image:self.assertEqual(image.width,2*image.height)
        self.assertTrue(all(not r['image_used'] for r in q.requests if r['name'].startswith('mechanism_')))
        second=self.root/'replay';second.mkdir()
        replay=execute(None,self.case,second,budgets,replay=folder)
        self.assertTrue(replay['world_state_ready'],replay)
        self.assertEqual(read(second/self.case['id']/'G_full.json'),read(folder/'G_full.json'))
        self.assertEqual(read(second/self.case['id']/'posterior.json'),read(folder/'posterior.json'))

    def test_target_only_wire_executes_exports_and_reloads_the_same_semantic_model(self):
        q=FixtureProvider(self.case);normal=q.ask;budgets=dict(groups=4,max_new=24)
        def target_only(image,prompt,path,**kwargs):
            raw=normal(image,prompt,path,**kwargs)
            if path.stem=='domain':
                for group in raw['groups']:
                    for mode in group['modes']:
                        for obj in mode['objects']:
                            support=obj.pop('support');obj['support_target']=support['target'] or support['kind']
                path.write_text(json.dumps(raw));write(path.with_suffix('.json'),raw)
            return raw
        out=self.root/'target_only';out.mkdir()
        write(out/'config.json',dict(protocol='generative_semantic_world_scm_v1',case=self.case,budgets=budgets,code_sha256=source_hashes()))
        with patch.object(q,'ask',side_effect=target_only),redirect_stdout(io.StringIO()):record=execute(q,self.case,out,budgets)
        self.assertTrue(record['world_state_ready'],record);self.assertEqual(record['layout_failures'],[])
        folder=out/self.case['id'];saved=read(folder/'domain.json')
        self.assertEqual(saved,domain_answer(fixture_domain(),self.case))
        self.assertEqual(len(read(folder/'domain_binding.json')['bindings']),4)
        checkpoint=load_checkpoint(out,allow_transport_updates=True)
        self.assertEqual(checkpoint['domain'],saved)
        self.assertEqual(enumerate_model(checkpoint['model'],{'E':'signature'}),read(folder/'posterior.json'))
        self.assertEqual(read(folder/'G_full.json')['objects'][:len(self.obs['objects'])],self.obs['objects'])

    def test_grounding_and_domain_survive_a_later_mechanism_failure(self):
        q=FixtureProvider(self.case);normal=q.ask
        def fail_mechanism(image,prompt,path,**kwargs):
            if path.stem.startswith('mechanism_'):raise APIRequestError('synthetic unavailable mechanism')
            return normal(image,prompt,path,**kwargs)
        out=self.root/'mechanism_failure';out.mkdir()
        with patch.object(q,'ask',side_effect=fail_mechanism),redirect_stdout(io.StringIO()):
            record=execute(q,self.case,out,dict(groups=4,max_new=24))
        self.assertEqual(record['status'],'failed');self.assertEqual(record['failed_stage'],'mechanism_C')
        folder=out/self.case['id']
        for name in ['grounding.json','domain.json','domain_binding.json']:self.assertTrue((folder/name).is_file())
        self.assertFalse((folder/'causal_model.json').exists());self.assertFalse(record['world_state_ready'])

    def test_single_state_C_and_S_need_no_model_request_and_keyed_mechanisms_replay(self):
        q=FixtureProvider(self.case);normal=q.ask
        def keyed_singleton(image,prompt,path,**kwargs):
            raw=normal(image,prompt,path,**kwargs)
            if path.stem=='domain':
                for k in ['contexts','organizations']:raw[k]=raw[k][:1]
            elif path.stem.startswith('mechanism_'):
                payload=json.loads(prompt.split('\n')[-1]);ids=[s['id'] for s in payload['states']]
                for row in raw['rows']:
                    ps=row.pop('p');row['p_by_state']={k:str(v) for k,v in reversed(list(zip(ids,ps)))}
            path.write_text(json.dumps(raw));write(path.with_suffix('.json'),raw)
            return raw
        out=self.root/'singletons';out.mkdir();budgets=dict(groups=4,max_new=24)
        write(out/'config.json',dict(protocol='generative_semantic_world_scm_v1',case=self.case,budgets=budgets,code_sha256=source_hashes()))
        with patch.object(q,'ask',side_effect=keyed_singleton),redirect_stdout(io.StringIO()):record=execute(q,self.case,out,budgets)
        self.assertTrue(record['world_state_ready'],record);self.assertEqual(record['layout_failures'],[])
        names=[r['name'] for r in q.requests];self.assertNotIn('mechanism_C',names);self.assertNotIn('mechanism_S',names)
        self.assertEqual(record['latent_domain_sizes'],dict(C=1,S=1));self.assertAlmostEqual(record['max_hidden_marginal_update_TV'],0.)
        folder=out/self.case['id'];model=read(folder/'causal_model.json')
        self.assertEqual(model['nodes'][0]['cpt'],{'root':[1.]});self.assertEqual(model['nodes'][1]['cpt'],{'c0':[1.]})
        for name in ['C','S']:
            self.assertTrue((folder/f'calls/mechanism_{name}.deterministic.json').is_file())
            self.assertFalse((folder/f'calls/mechanism_{name}.json').exists())
        checkpoint=load_checkpoint(out);self.assertEqual(checkpoint['model'],model)
        replay=self.root/'singletons_replay';replay.mkdir()
        with redirect_stdout(io.StringIO()):again=execute(None,self.case,replay,budgets,replay=folder)
        self.assertTrue(again['world_state_ready'],again)
        self.assertEqual(read(replay/self.case['id']/'posterior.json'),read(folder/'posterior.json'))
        self.assertEqual(read(replay/self.case['id']/'G_full.json'),read(folder/'G_full.json'))

    def test_resume_domain_does_not_reuse_failed_mechanisms_or_repeat_grounding_and_domain(self):
        q=FixtureProvider(self.case);normal=q.ask;budgets=dict(groups=4,max_new=24)
        def fail_C(image,prompt,path,**kwargs):
            if path.stem=='mechanism_C':
                write(path.with_name('mechanism_C_repair.json'),dict(rows=[dict(given={},p=['bad'])]))
                raise APIRequestError('synthetic failure before semantic model')
            return normal(image,prompt,path,**kwargs)
        out=self.root/'failed_C';out.mkdir()
        write(out/'config.json',dict(protocol='generative_semantic_world_scm_v1',case=self.case,budgets=budgets,code_sha256=source_hashes()))
        with patch.object(q,'ask',side_effect=fail_C),redirect_stdout(io.StringIO()):record=execute(q,self.case,out,budgets)
        self.assertEqual(record['failed_stage'],'mechanism_C')
        checkpoint=load_checkpoint(out,allow_transport_updates=True,domain_only=True)
        self.assertIsNone(checkpoint['model']);self.assertTrue(checkpoint['provenance']['reused_domain_checkpoint'])
        second=self.root/'domain_resumed';second.mkdir();fresh=FixtureProvider(self.case)
        with redirect_stdout(io.StringIO()):resumed=execute(fresh,self.case,second,budgets,checkpoint=checkpoint)
        self.assertTrue(resumed['world_state_ready'],resumed);self.assertFalse(resumed['reused_semantic_checkpoint'])
        names=[r['name'] for r in fresh.requests]
        self.assertNotIn('grounding',names);self.assertNotIn('domain',names);self.assertIn('mechanism_C',names)
        self.assertFalse((second/self.case['id']/'calls/mechanism_C_repair.json').exists())
        self.assertEqual(read(second/self.case['id']/'domain.json'),read(out/self.case['id']/'domain.json'))
        manifest=read(out/'artifact_manifest.json')
        self.assertEqual(manifest,{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})

    def test_domain_only_continuation_reelicits_CPTs_and_replans_even_completed_layouts(self):
        out=self.root/'complete';out.mkdir();budgets=dict(groups=4,max_new=24)
        write(out/'config.json',dict(protocol='generative_semantic_world_scm_v1',case=self.case,budgets=budgets,code_sha256=source_hashes()))
        with redirect_stdout(io.StringIO()):execute(FixtureProvider(self.case),self.case,out,budgets)
        checkpoint=load_checkpoint(out,domain_only=True);second=self.root/'fresh_mechanisms';second.mkdir();fresh=FixtureProvider(self.case)
        with redirect_stdout(io.StringIO()):record=execute(fresh,self.case,second,budgets,checkpoint=checkpoint)
        self.assertTrue(record['world_state_ready'],record)
        self.assertEqual(record['reused_layout_responses'],[])
        names=[r['name'] for r in fresh.requests];self.assertIn('mechanism_C',names);self.assertIn('layout_01',names)
        self.assertNotIn('domain',names)

    def test_geometry_failure_does_not_change_semantic_posterior(self):
        q=FixtureProvider(self.case);out=self.root/'failure';out.mkdir()
        def fail(*args,**kwargs):raise ValueError('explicit synthetic layout failure')
        with patch('qwen_edit_pano.world_pipeline.completion.world_scm_runner.solve_assignment',side_effect=fail):
            record=execute(q,self.case,out,dict(groups=4,max_new=24))
        self.assertFalse(record['world_state_ready']);self.assertTrue(record['semantic_inference_complete'])
        self.assertTrue(record['no_geometric_renormalization'])
        self.assertAlmostEqual(record['failed_layout_semantic_mass']+record['unattempted_semantic_mass'],1.)
        posterior=read(out/self.case['id']/'posterior.json');self.assertAlmostEqual(sum(r['probability'] for r in posterior['rows']),1.)

    def test_layout_transport_failure_is_isolated_and_continuation_reuses_semantics(self):
        budgets=dict(groups=4,max_new=24);q=FixtureProvider(self.case);normal=q.ask
        def interrupted(image,prompt,path,**kwargs):
            if path.stem=='layout_01':raise APIRequestError('timeout',timeout_seconds=180)
            return normal(image,prompt,path,**kwargs)
        out=self.root/'transport_failure';out.mkdir()
        write(out/'config.json',dict(protocol='generative_semantic_world_scm_v1',case=self.case,budgets=budgets,code_sha256=source_hashes()))
        with patch.object(q,'ask',side_effect=interrupted):record=execute(q,self.case,out,budgets)
        self.assertTrue(record['semantic_inference_complete']);self.assertTrue(record['world_state_ready'])
        self.assertEqual(record['layout_failures'][0]['failure_kind'],'transport')
        self.assertEqual(record['layout_failures'][0]['world'],1)
        before=read(out/self.case['id']/'posterior.json')
        checkpoint=load_checkpoint(out,allow_transport_updates=True)
        second=self.root/'continued';second.mkdir();fresh=FixtureProvider(self.case)
        continued=execute(fresh,self.case,second,budgets,checkpoint=checkpoint)
        self.assertTrue(continued['world_state_ready']);self.assertTrue(continued['reused_semantic_checkpoint'])
        self.assertEqual([r['name'] for r in fresh.requests],['layout_01'])
        self.assertIn('layout_03',continued['reused_layout_responses'])
        self.assertEqual(read(second/self.case['id']/'posterior.json'),before)
        self.assertEqual(continued['layout_failures'],[])

    def test_all_layout_transport_failures_retain_complete_semantic_state(self):
        q=FixtureProvider(self.case);normal=q.ask
        def interrupted(image,prompt,path,**kwargs):
            if path.stem.startswith('layout_'):raise APIRequestError('connection_error')
            return normal(image,prompt,path,**kwargs)
        out=self.root/'network_down';out.mkdir()
        with patch.object(q,'ask',side_effect=interrupted):record=execute(q,self.case,out,dict(groups=4,max_new=24))
        # The empty alternative still exports without a network call. Nonempty
        # failures are retained at their original semantic weights.
        self.assertTrue(record['semantic_inference_complete'])
        self.assertTrue(all(r['failure_kind']=='transport' for r in record['layout_failures']))
        self.assertAlmostEqual(record['exported_semantic_mass']+record['failed_layout_semantic_mass']+record['unattempted_semantic_mass'],1.)

    def test_continuation_allows_transport_updates_but_rejects_changed_semantic_engine(self):
        for name in ['world_api.py','world_scm.py']:
            out=self.root/name;out.mkdir();hashes=source_hashes()
            path=next(p for p in hashes if Path(p).name==name);hashes[path]='0'*64
            write(out/'config.json',dict(protocol='generative_semantic_world_scm_v1',case=self.case,
                budgets=dict(groups=4,max_new=24),code_sha256=hashes))
            with redirect_stdout(io.StringIO()):execute(FixtureProvider(self.case),self.case,out,dict(groups=4,max_new=24))
            if name=='world_api.py':
                checkpoint=load_checkpoint(out,allow_transport_updates=True)
                self.assertEqual(checkpoint['provenance']['changed_transport_sources'],[path])
            else:
                with self.assertRaises(ValueError):load_checkpoint(out,allow_transport_updates=True)
            with self.assertRaises(ValueError):load_checkpoint(out,allow_transport_updates=False)

    def test_continuation_replans_layout_that_had_json_but_failed_geometry(self):
        from .world_scm_runner import solve_assignment as normal
        out=self.root/'invalid_layout';out.mkdir();budgets=dict(groups=4,max_new=24)
        write(out/'config.json',dict(protocol='generative_semantic_world_scm_v1',case=self.case,budgets=budgets,code_sha256=source_hashes()))
        calls=[]
        def fail_first(*args,**kwargs):
            calls.append(1)
            if len(calls)==1:raise ValueError('synthetic invalid geometry')
            return normal(*args,**kwargs)
        with patch('qwen_edit_pano.world_pipeline.completion.world_scm_runner.solve_assignment',side_effect=fail_first),redirect_stdout(io.StringIO()):
            execute(FixtureProvider(self.case),self.case,out,budgets)
        self.assertTrue((out/self.case['id']/'calls/layout_01.json').exists())
        checkpoint=load_checkpoint(out,allow_transport_updates=True)
        second=self.root/'replanned';second.mkdir();provider=FixtureProvider(self.case)
        with redirect_stdout(io.StringIO()):record=execute(provider,self.case,second,budgets,checkpoint=checkpoint)
        self.assertEqual([r['name'] for r in provider.requests],['layout_01'])
        self.assertEqual(record['layout_failures'],[])


if __name__=='__main__':unittest.main()
