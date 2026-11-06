"""CPU logical/integration checks. Synthetic replies never certify VLM quality."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from ..common import read, sha, write
from ..oracle_model import tensorize
from .causal_contract import (diagnostics, execute, typed_graph, validate_candidates,
                              validate_factors)
from .causal_world_completion import (CASE_ID, PROTOCOL, LEGACY_PROTOCOL, LEGACY_CORE_SHA256,
                                     boxes_answer, boxes_prompt, code_hashes,
                                     contents_prompt, context_prompt, load_pilot,
                                     load_semantics, relation_checks, run, selected_plan)
from .scene_completion import map_plan
from .semantic_probe import state_context
from .test_semantic_probe import state


def factors(ids=()):
    return dict(scene_type='indoor room', scene_hypothesis='One hypothesized surrounding room.', factors=[
        dict(id='h1', kind='activity', statement='An optional activity is present.', assumed=True,
             evidence_ids=list(ids), reason='This is an inferred possibility.', parents=[]),
        dict(id='h2', kind='preference', statement='An independent occupant preference.', assumed=True,
             evidence_ids=[], reason='An explicitly unobserved assumption.', parents=[])])


def contents():
    return dict(additions=[dict(id='new1', name='hidden bookshelf', object_type='bookshelf',
        reason='A plausible storage choice.', placement='Behind this camera.',
        grounds=[['h1'], ['h2']], requires=[])], relations=[])


def response_boxes(ids=('new1',)):
    return dict(objects=[dict(id=oid, center=[3.*i, 0., -3.], size=[.8, 1.5, .4],
                              reason='A hypothesized corner outside the photograph.', unseen_reason='outside_fov')
                         for i, oid in enumerate(ids)])


class Model:
    def __init__(self, context=None, content=None, box=None):
        self.answers = dict(context=context or factors(), contents=content or contents(),
                            boxes=box or response_boxes())
        self.calls = []

    def ask(self, image, prompt, path, **kwargs):
        self.calls.append(path.stem)
        result = deepcopy(self.answers[path.stem])
        path.write_text(json.dumps(result))
        write(path.with_suffix('.json'), result)
        write(path.with_suffix('.request.json'), dict(prompt=prompt, **kwargs))
        write(path.with_suffix('.generation.json'), dict(finish_reason='eos', synthetic_test=True))
        return result


def make_case(root):
    observed = state()
    observer = root/'observer'; observer.mkdir()
    image = root/'image.png'; Image.new('RGB', (64, 64)).save(image)
    image.replace(observer/'projection_boxes.png')
    image = observer/'projection_boxes.png'
    state_path = observer/'G_obs.json'; write(state_path, observed)
    case = dict(id='case', image=str(image), state=str(state_path), inventory=str(state_path),
                hfov=90., context=state_context(observed),
                source_sha256={str(image):sha(image), str(state_path):sha(state_path)})
    return case, observed


class MechanismTests(unittest.TestCase):
    def validated(self, c=None, f=None):
        f = validate_factors(f or factors(), ['obs'])
        return f, validate_candidates(c or contents(), f, ['obs'])

    def test_multiple_independent_grounds_survive_one_intervention(self):
        f, c = self.validated()
        before = deepcopy((f, c))
        self.assertEqual(execute(f, c, ['obs'], {'h1':False})['selected_ids'], ['new1'])
        self.assertEqual(execute(f, c, ['obs'], {'h1':False, 'h2':False})['selected_ids'], [])
        self.assertEqual((f, c), before)
        d = diagnostics(f, c, ['obs'], execute(f, c, ['obs']))
        self.assertFalse(d['causal_gain_validated'])
        self.assertEqual(d['results'][0]['removed'], [])

    def test_do_replaces_equation_and_cuts_parents_without_editing_observation(self):
        f = factors(); f['factors'][0]['assumed'] = False
        f['factors'][1]['parents'] = ['h1']
        c = contents(); c['additions'][0]['grounds'] = [['h2']]
        f, c = self.validated(c, f)
        self.assertFalse(execute(f, c, ['obs'])['factor_values']['h2'])
        self.assertFalse(selected_plan(f,execute(f,c,['obs']))['factor_values']['h2'])
        v = execute(f, c, ['obs'], {'h2':True})
        self.assertEqual(v['selected_ids'], ['new1'])
        self.assertFalse(v['factor_values']['h1'])
        with self.assertRaises(ValueError): execute(f, c, ['obs'], {'obs':False})

    def test_object_requirement_is_executable_but_spatial_edge_is_not_existence(self):
        c = contents(); c['additions'][0]['grounds'] = [['h1']]
        other = deepcopy(c['additions'][0]); other.update(id='new2', grounds=[[]], requires=['new1'])
        c['additions'].append(other)
        f, c = self.validated(c)
        self.assertEqual(execute(f,c,['obs'],{'h1':False})['selected_ids'], [])
        c['additions'][1]['requires'] = []
        c['relations'] = [dict(source='new2', target='new1', kind='supported_by', reason='Proposed support.')]
        f, c = self.validated(c)
        v = execute(f,c,['obs'],{'h1':False})
        self.assertEqual(v['selected_ids'], ['new2'])
        self.assertEqual(len(v['inactive_relations']),1)

    def test_invalid_refs_duplicate_instance_self_cycles_and_placement_cycles_rejected(self):
        for kind in ('factor_cycle', 'factor_unknown', 'observed_factor_id', 'nonbool', 'unknown_ground',
                     'duplicate_new', 'object_cycle', 'unknown_relation', 'placement_cycle', 'numeric_semantics'):
            f, c = factors(), contents()
            if kind == 'factor_cycle':
                f['factors'][0]['parents'] = ['h2']; f['factors'][1]['parents'] = ['h1']
            elif kind == 'factor_unknown': f['factors'][0]['evidence_ids'] = ['fake_obs']
            elif kind == 'observed_factor_id': f['factors'][0]['id'] = 'obs'
            elif kind == 'nonbool': f['factors'][0]['assumed'] = 1
            elif kind == 'unknown_ground': c['additions'][0]['grounds'] = [['h404']]
            elif kind == 'duplicate_new': c['additions'] *= 2
            elif kind == 'object_cycle': c['additions'][0]['requires'] = ['new1']
            elif kind == 'unknown_relation': c['relations'] = [dict(source='new1',target='other',kind='near',reason='Assumed')]
            elif kind == 'placement_cycle':
                other=deepcopy(c['additions'][0]);other['id']='new2';c['additions'].append(other)
                c['relations'] = [dict(source='new1', target='new2', kind='inside', reason='Assumed'),
                                  dict(source='new2', target='new1', kind='supported_by', reason='Assumed')]
            else: c['additions'][0]['center'] = [1,2,3]
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.validated(c,f)

    def test_unconditional_or_empty_world_is_allowed_not_forced_minimum(self):
        f = validate_factors(dict(scene_type='room',scene_hypothesis='An unspecified room.', factors=[]), ['obs'])
        c = contents(); c['additions'][0]['grounds'] = [[]]
        c = validate_candidates(c,f,['obs'])
        self.assertEqual(execute(f,c,['obs'])['selected_ids'],['new1'])
        c['additions'] = []
        self.assertEqual(execute(f,c,['obs'])['selected_ids'],[])

    def test_graph_labels_evidence_and_spatial_as_not_causal(self):
        f = factors(['obs']);f,c = self.validated(f=f)
        graph = typed_graph(f,c,['obs'],execute(f,c,['obs']))
        self.assertFalse(graph['identified_causal_graph'])
        self.assertTrue(any(e['type']=='evidence_for_not_cause' for e in graph['edges']))

    def test_on_top_of_is_preserved_and_orders_support_before_dependent(self):
        c=contents();other=deepcopy(c['additions'][0]);other.update(id='new2',requires=['new1'])
        c['additions'].append(other)
        c['relations']=[dict(source='new2',target='new1',kind='on_top_of',reason='On the top surface.')]
        f,c=self.validated(c)
        v=execute(f,c,['obs'])
        self.assertEqual(v['relations'][0]['kind'],'on_top_of')
        self.assertLess(v['placement_order'].index('new1'),v['placement_order'].index('new2'))
        checks=relation_checks(v['relations'],dict(objects=[]),[
            dict(id='new1',center=[0,0,-3],size=[1,1,1]),
            dict(id='new2',center=[0,1,-3],size=[.2,.2,.2])])
        self.assertIsNone(checks[0]['proxy_passed'])
        self.assertFalse(checks[0]['physical_relation_verified'])
        c['relations'].append(dict(source='new1',target='new2',kind='on_top_of',reason='Contradictory support.'))
        with self.assertRaisesRegex(ValueError,'cycle'):self.validated(c)

    def test_relation_errors_identify_index_field_and_bad_value(self):
        for change,fragment in [(dict(target='missing'),'relations[0].target=\'missing\''),
                                (dict(kind='on_moon'),'relations[0].kind=\'on_moon\''),
                                (dict(target='new1'),'self edge')]:
            c=contents();r=dict(source='new1',target='obs',kind='near',reason='A hypothesis.');r.update(change)
            c['relations']=[r]
            with self.subTest(change=change),self.assertRaises(ValueError) as error:self.validated(c)
            self.assertIn(fragment,str(error.exception))


class IntegrationTests(unittest.TestCase):
    def test_open_vocab_full_state_and_adapter_projection_keep_observed_encoder_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case, observed=make_case(root)
            c=contents();other=deepcopy(c['additions'][0])
            other.update(id='new2',name='hypothetical rug', object_type='rug', grounds=[[]]);c['additions'].append(other)
            c['relations']=[dict(source='new1',target='new2',kind='near',reason='A proposed adjacency.')]
            model=Model(content=c,box=response_boxes(('new1','new2')))
            out=root/'out';out.mkdir()
            self.assertTrue(run(model,case,out))
            folder=out/'case'
            semantic=read(folder/'G_full_semantic.json'); adapter=read(folder/'G_full_adapter_proposal.json')
            self.assertEqual(model.calls,['context','contents','boxes'])
            n=len(observed['objects'])
            self.assertEqual(semantic['objects'][:n],observed['objects'])
            self.assertEqual(adapter['objects'][:n],observed['objects'])
            self.assertEqual(len(semantic['objects']),n+2);self.assertEqual(len(adapter['objects']),n+1)
            self.assertEqual(semantic['objects'][-1]['description'],'hypothetical rug')
            self.assertIsNone(semantic['objects'][-1]['category_id']['value'])
            self.assertEqual(len(semantic['world_relations']),1)
            self.assertEqual(adapter['world_relations'],[])
            self.assertEqual(len(adapter['relations_omitted_by_category_projection']),1)
            for a,b in zip(tensorize(observed,160),tensorize(adapter,160)):
                self.assertTrue(a[:,:n+1].equal(b[:,:n+1]))
            summary=read(folder/'summary.json')
            self.assertFalse(summary['semantics_verified']);self.assertFalse(summary['adapter_input_approved'])
            self.assertEqual(summary['category_coverage']['unmapped_new'],1)
            self.assertTrue((folder/'world/scene_3d.html').exists())
            self.assertTrue((folder/'typed_graph.svg').exists())
            self.assertIn('G_full_semantic.json',(folder/'index.html').read_text())

    def test_zero_selected_has_no_geometry_call_and_no_false_completion_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,_=make_case(root);out=root/'out';out.mkdir()
            c=contents();c['additions']=[]
            model=Model(content=c)
            self.assertTrue(run(model,case,out))
            self.assertEqual(model.calls,['context','contents'])
            summary=read(out/'case/summary.json')
            self.assertEqual(summary['status'],'no_selected_additions')
            self.assertFalse(summary['geometry_generated'])

    def test_layout_cannot_silently_change_drop_duplicate_or_rename_instances(self):
        f=validate_factors(factors(),[]);c=validate_candidates(contents(),f,[])
        plan=selected_plan(f,execute(f,c,[]));mappings=map_plan(plan,'semantic_lexicon_v1')
        for kind in ('missing','duplicate','rename','category','unknown_visibility','nonfinite'):
            a=response_boxes()
            if kind=='missing':a['objects']=[]
            elif kind=='duplicate':a['objects']*=2
            elif kind=='rename':a['objects'][0]['id']='new404'
            elif kind=='category':a['objects'][0]['category']='sofa'
            elif kind=='unknown_visibility':a['objects'][0]['unseen_reason']='hidden'
            else:a['objects'][0]['center'][0]=float('nan')
            with self.subTest(kind=kind),self.assertRaises(ValueError):boxes_answer(a,plan,mappings)

    def test_camera_axes_and_unknown_support_are_reported_honestly(self):
        items=[dict(id='a',center=[0,2,2],size=[1,1,1]),dict(id='b',center=[0,0,3],size=[1,1,1])]
        relations=[dict(source='a',target='b',kind=k,reason='Assumed') for k in ('above','behind','supported_by','near')]
        results=relation_checks(relations,dict(objects=[]),items)
        self.assertTrue(results[0]['proxy_passed']);self.assertFalse(results[1]['proxy_passed'])
        self.assertIsNone(results[2]['proxy_passed']);self.assertIsNone(results[3]['proxy_passed'])

    def test_semantics_omit_predicted_numeric_depth_and_no_furniture_example_is_injected(self):
        case=dict(hfov=90,context=[dict(id='o',category='sofa',description='visible sofa',
                  visible_bbox_xyxy_norm=[0,0,1,1],center=[123456,0,0],full_aabb_size=[0,0,999999])])
        a=context_prompt(case);b=contents_prompt(case,validate_factors(factors(),['o']))
        self.assertNotIn('123456',a+b);self.assertNotIn('999999',a+b)
        self.assertNotIn('h3',b)
        self.assertNotIn('bookshelf',a+b)
        f=validate_factors(factors(),['o']);c=validate_candidates(contents(),f,['o'])
        q=boxes_prompt(case,selected_plan(f,execute(f,c,['o'])))
        self.assertIn('"unseen_reason":...',q)
        self.assertNotIn('outside the photograph, anywhere around the same camera.',q)

    def test_real_frozen_pilot_provenance_is_validation_and_has_37_coarse_categories(self):
        case=load_pilot()
        self.assertEqual(case['split_verified'],'val');self.assertEqual(len(case['context']),9)
        self.assertEqual(case['supported_coarse_categories'],37)

    def test_invalid_raw_reply_is_preserved_and_fails_without_empty_world_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,_=make_case(root);out=root/'out';out.mkdir()
            bad=contents();bad['additions'][0]['grounds']=[['h404']]
            model=Model(content=bad)
            self.assertFalse(run(model,case,out))
            folder=out/'case'
            self.assertTrue((folder/'contents.txt').exists())
            self.assertFalse((folder/'G_full_semantic.json').exists())
            self.assertEqual(read(folder/'summary.json')['status'],'failed')

    def test_saved_semantics_can_feed_layout_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,observed=make_case(root);case['id']=CASE_ID
            semantic=root/'semantic';semantic.mkdir()
            write(semantic/'config.json',dict(protocol=PROTOCOL,case=case,code_sha256=code_hashes()))
            self.assertTrue(run(Model(),case,semantic,stage='semantic'))
            f,c,provenance=load_semantics(semantic,case)
            self.assertEqual(c['additions'][0]['id'],'new1')
            out=root/'layout';out.mkdir();model=Model()
            self.assertTrue(run(model,case,out,semantic_source=semantic,semantic_source_hashes=provenance))
            self.assertEqual(model.calls,['boxes'])
            self.assertEqual(read(out/CASE_ID/'G_full_semantic.json')['objects'][:len(observed['objects'])],observed['objects'])
            folder=semantic/CASE_ID
            changed=read(folder/'contents.json');changed['additions']=[];write(folder/'contents.json',changed)
            with self.assertRaisesRegex(ValueError,'raw model response'):load_semantics(semantic,case)

    def test_reuse_rejects_valid_but_changed_source_after_preflight_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,_=make_case(root);case['id']=CASE_ID
            source=root/'semantic';source.mkdir()
            write(source/'config.json',dict(protocol=PROTOCOL,case=case,code_sha256=code_hashes()))
            self.assertTrue(run(Model(),case,source,stage='semantic'))
            _,_,bound=load_semantics(source,case)
            generation=source/CASE_ID/'contents.generation.json'
            changed=read(generation);changed['test_change']=True;write(generation,changed)
            out=root/'out';out.mkdir();model=Model()
            self.assertFalse(run(model,case,out,semantic_source=source,semantic_source_hashes=bound))
            self.assertEqual(model.calls,[])
            self.assertIn('changed after preflight',read(out/CASE_ID/'summary.json')['error'])

    def test_audited_v1_failed_validation_output_can_recover_without_rewriting_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,_=make_case(root);case['id']=CASE_ID
            source=root/'v1';source.mkdir()
            model=Model()
            self.assertTrue(run(model,case,source,stage='semantic'))
            legacy_hashes={**code_hashes(),**{
                str(Path(__file__).with_name(n).resolve()):h for n,h in LEGACY_CORE_SHA256.items()}}
            write(source/'config.json',dict(protocol=LEGACY_PROTOCOL,case=case,code_sha256=legacy_hashes))
            request=source/CASE_ID/'contents.request.json';old=read(request)
            old['prompt']=old['prompt'].replace('/supported_by/on_top_of/inside.','/supported_by/inside.')
            write(request,old)
            _,_,bound=load_semantics(source,case)
            out=root/'recovered';out.mkdir()
            write(out/'config.json',dict(protocol=PROTOCOL,semantic_request_protocol=LEGACY_PROTOCOL,
                                        case=case,code_sha256=code_hashes()))
            self.assertTrue(run(None,case,out,stage='semantic',semantic_source=source,semantic_source_hashes=bound))
            self.assertEqual((out/CASE_ID/'contents.request.json').read_bytes(),request.read_bytes())
            self.assertEqual(load_semantics(out,case)[1]['additions'][0]['id'],'new1')
            config=read(source/'config.json');config['code_sha256'][str(Path(__file__).with_name('causal_contract.py').resolve())]='unknown'
            write(source/'config.json',config)
            with self.assertRaisesRegex(ValueError,'code binding'):load_semantics(source,case)


if __name__ == '__main__':
    unittest.main()
