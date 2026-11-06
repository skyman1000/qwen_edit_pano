"""Numerical, schema and full CPU regressions; no real-image quality claim."""
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ..common import read,write
from .scm_count_model import completion_diagnostics,count_distribution,draw,infer,trace,validate
from .scm_mechanisms import TextMechanismScorer,domain_answer,registration_answer
from .scm_world import code_hashes,run
from .scm_world_audit import audit
from .scm_world_layout import instantiate,solve_batches,support_answer
from .conditional_world_geometry import support_plane
from .scene_completion import map_plan
from .test_causal_world_completion import make_case


def fixture_spec():
    return dict(backgrounds=[dict(id='b1',description='social'),dict(id='b2',description='reading')],
        configurations=[dict(id='k1',description='simple'),dict(id='k2',description='storage')],
        object_types=['lamp','shelf'],max_count=2,
        mechanisms=dict(B=[.6,.4],K={'b1':[.8,.2],'b2':[.2,.8]},N={
            'b1':{'k1':{'lamp':[.7,.2,.1],'shelf':[.6,.3,.1]},'k2':{'lamp':[.4,.4,.2],'shelf':[.1,.3,.6]}},
            'b2':{'k1':{'lamp':[.3,.5,.2],'shelf':[.3,.5,.2]},'k2':{'lamp':[.1,.4,.5],'shelf':[.05,.25,.7]}}}))


def beta(a,b):return math.exp(math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b))


def brute(spec,obs):
    """Independent exact integral by enumerating all counts and partitions."""
    total=0.;bg={b['id']:0. for b in spec['backgrounds']};outs={}
    for bi,b in enumerate(bg):
        for ki,k in enumerate(s['id'] for s in spec['configurations']):
            for counts in itertools.product(range(spec['max_count']+1),repeat=len(spec['object_types'])):
                prior=spec['mechanisms']['B'][bi]*spec['mechanisms']['K'][b][ki]
                for t,n in zip(spec['object_types'],counts):prior*=spec['mechanisms']['N'][b][k][t][n]
                ranges=[]
                for t,n in zip(spec['object_types'],counts):ranges.append(range(n-obs.get(t,0)+1))
                for outside in itertools.product(*ranges):
                    coeff=1.;K=0;M=0;unknowninside=0;A=sum(outside)
                    for t,n,a in zip(spec['object_types'],counts,outside):
                        if t in obs:
                            recorded=obs[t];missed=n-recorded-a;K+=recorded;M+=missed
                            coeff*=math.factorial(n)/(math.factorial(recorded)*math.factorial(a)*math.factorial(missed))
                        else:
                            unknowninside+=n-a;coeff*=math.comb(n,a)
                    weight=prior*coeff*beta(K+M+unknowninside+1,A+1)*beta(K+1,M+1)
                    total+=weight;bg[b]+=weight;outs[A]=outs.get(A,0.)+weight
    return total,{b:p/total for b,p in bg.items()},{a:p/total for a,p in outs.items()}


class CountTests(unittest.TestCase):
    def test_completion_diagnostics_match_independent_full_partition_integral(self):
        spec=fixture_spec()
        for obs in ({},{'lamp':1},{'lamp':1,'shelf':1},{'lamp':2}):
            _,_,pmf=brute(spec,obs);post=infer(spec,obs);saved=draw(post,3407)
            d=completion_diagnostics(post)
            self.assertAlmostEqual(d['expected_outside_instances'],sum(n*p for n,p in pmf.items()),places=11)
            self.assertAlmostEqual(d['probability_zero_outside'],pmf.get(0,0.),places=11)
            self.assertAlmostEqual(d['expected_total_instances'],sum(obs.values())+
                d['expected_outside_instances']+d['expected_inside_not_exported_instances'],places=11)
            self.assertEqual(draw(post,3407),saved)  # Diagnostic must not consume or alter sample RNG.

    def test_censored_types_share_view_uncertainty_not_independent_empty_events(self):
        spec=fixture_spec()
        for b in spec['mechanisms']['N'].values():
            for k in b.values():
                k['lamp']=[0.,1.,0.];k['shelf']=[0.,1.,0.]
        d=completion_diagnostics(infer(spec,{}))
        self.assertAlmostEqual(d['expected_outside_instances'],1.,places=11)
        self.assertAlmostEqual(d['probability_zero_outside'],1/3,places=11) # integral F**2, not E[F]**2
        self.assertTrue(all(r['observation_censored'] for r in d['by_type'].values()))

    def test_posterior_matches_independent_exact_count_partition_enumeration(self):
        spec=fixture_spec()
        for obs in ({},{'lamp':1},{'lamp':1,'shelf':1},{'lamp':2}):
            total,bg,_=brute(spec,obs);result=infer(spec,obs)
            self.assertAlmostEqual(math.exp(result['evidence_log_probability']),total,places=11)
            for b in bg:self.assertAlmostEqual(result['background_posterior'][b],bg[b],places=11)

    def test_outside_samples_match_exact_partition_mean_and_preserve_counts(self):
        spec=fixture_spec();_,_,pout=brute(spec,{'lamp':1});result=infer(spec,{'lamp':1})
        values=[]
        for seed in range(6000):
            sample=draw(result,seed);values.append(sample['new_instances'])
            for row in sample['counts']:
                self.assertEqual(row['total'],(row['observed'] or 0)+row['outside']+row['inside_not_exported'])
                self.assertGreaterEqual(row['outside'],0)
        self.assertAlmostEqual(float(np.mean(values)),sum(a*p for a,p in pout.items()),delta=.04)

    def test_unknown_category_is_censored_not_zero_evidence(self):
        spec=fixture_spec();positive=infer(spec,{'lamp':1});zero=infer(spec,{'lamp':1,'shelf':0})
        self.assertGreater(abs(positive['background_posterior']['b2']-zero['background_posterior']['b2']),.01)
        fixed=infer(spec,{'lamp':1},{'B':'b1','K':'k2'})
        self.assertTrue(np.allclose(count_distribution(fixed,'shelf'),spec['mechanisms']['N']['b1']['k2']['shelf']))

    def test_observation_updates_background_and_no_evidence_recovers_prior(self):
        spec=fixture_spec();prior=infer(spec,{});post=infer(spec,{'lamp':2})
        self.assertTrue(np.allclose(list(prior['background_posterior'].values()),spec['mechanisms']['B']))
        self.assertGreater(post['background_posterior']['b2'],prior['background_posterior']['b2'])
        self.assertTrue(np.allclose(count_distribution(post,'lamp')[:2],[0.,0.]))

    def test_do_K_cuts_B_to_K_and_background_prior_is_preserved_without_observation(self):
        spec=fixture_spec();do=infer(spec,{}, {'K':'k2'})
        self.assertAlmostEqual(do['configuration_posterior']['k2'],1.)
        self.assertTrue(np.allclose(list(do['background_posterior'].values()),spec['mechanisms']['B']))
        expected=sum(spec['mechanisms']['B'][bi]*spec['mechanisms']['N'][b['id']]['k2']['shelf'][2] for bi,b in enumerate(spec['backgrounds']))
        self.assertAlmostEqual(count_distribution(do,'shelf')[2],expected)

    def test_finite_schema_zero_evidence_and_unknown_interventions_are_explicit(self):
        spec=fixture_spec();bad=deepcopy(spec);bad['mechanisms']['B']=[1.,1.]
        with self.assertRaises(ValueError):validate(bad,{})
        with self.assertRaises(ValueError):infer(spec,{'lamp':3})
        with self.assertRaises(ValueError):infer(spec,{}, {'N_lamp':'1'})
        for b in spec['mechanisms']['N'].values():
            for k in b.values():k['lamp']=[1.,0.,0.]
        with self.assertRaisesRegex(ValueError,'zero probability'):infer(spec,{'lamp':1})

    def test_seed_replay_and_inputs_are_immutable(self):
        spec=fixture_spec();before=deepcopy(spec);result=infer(spec,{'lamp':1})
        self.assertEqual(draw(result,3407),draw(result,3407));self.assertEqual(spec,before)
        self.assertFalse(trace(result)['joint_metric_posterior_defined'])


class SchemaTests(unittest.TestCase):
    def test_observed_registration_covers_every_ID(self):
        d=domain_answer(dict(backgrounds=['a','b'],configurations=['x','y'],object_types=['lamp']))
        case=dict(context=[dict(id='obs0'),dict(id='obs1')])
        with self.assertRaises(ValueError):registration_answer(dict(instances=[dict(id='obs0',object_type='lamp')],floor_anchor_ids=[]),case,d)
        value=dict(instances=[dict(id=i,object_type='table_lamp') for i in ('obs0','obs1')],floor_anchor_ids=[])
        result=registration_answer(value,case,d)
        self.assertEqual(result['observed_counts'],{'table lamp':2});self.assertIn('table lamp',result['object_types'])

    def test_layout_cannot_reselect_count_or_identity_and_retains_wall_declaration(self):
        sample=dict(seed=1,B='b1',K='k1',counts=[dict(object_type='shelf',outside=1)])
        spec=fixture_spec();case=dict(context=[]);plan=instantiate(sample,case,spec);oid=plan['additions'][0]['id']
        v=dict(objects=[dict(id=oid,placement='Outside camera, attached to wall.',support=dict(kind='wall',target=None))])
        solved=support_answer(v,plan,case)
        self.assertEqual(solved['additions'][0]['support']['kind'],'unspecified')
        self.assertEqual(solved['additions'][0]['support_declaration']['kind'],'wall')
        v['objects'][0]['object_type']='chair'
        with self.assertRaisesRegex(ValueError,'identity'):support_answer(v,plan,case)
        with self.assertRaises(ValueError):support_answer(dict(objects=[]),plan,case)


class SyntheticScorer:
    def __init__(self):self.calls=[]
    def score(self,description,parents,states):
        self.calls.append((description,deepcopy(parents),deepcopy(states)))
        if states[0]['id'].isdigit():prob=[0.]*(len(states)-1)+[1.]
        else:prob=[1/len(states)]*len(states)
        return dict(probabilities=dict(zip([s['id'] for s in states],prob)),rounds=[],synthetic_test=True)


class SyntheticModel:
    def __init__(self,case):self.case=case;self.calls=[]
    def ask(self,image,prompt,path,**options):
        self.calls.append((path.stem,image is None))
        if path.stem=='domain':v=dict(backgrounds=['living','study'],configurations=['open','storage'],object_types=['lamp','shelf','fixture unknown object'])
        elif path.stem=='registration':v=dict(instances=[dict(id=o['id'],object_type='lamp') for o in self.case['context']],floor_anchor_ids=[self.case['context'][0]['id']])
        elif path.stem.startswith('supports'):
            part=prompt.split('\nFIXED SAMPLED PLAN: ')[1].split('\nOBSERVED: ')[0];plan=json.loads(part)
            v=dict(objects=[dict(id=o['id'],placement='Behind original camera.',support=dict(kind='floor',target=None)) for o in plan['additions']])
        else:
            current=json.JSONDecoder().raw_decode(prompt.split('Place ONLY these batch IDs: ')[1])[0]
            v=dict(objects=[dict(id=i,azimuth_deg=180.,range_m=3.+.5*int(i.rsplit('_',1)[1]),size=[.3,.6,.3],reason='Synthetic fixed off-camera placement.') for i in current])
        path.write_text(json.dumps(v));write(path.with_suffix('.json'),v)
        write(path.with_suffix('.request.json'),dict(prompt=prompt,text_only=image is None,**options));return v


class IntegrationTests(unittest.TestCase):
    def test_registration_failure_reports_stage_without_fake_posterior_links(self):
        class BrokenRegistration(SyntheticModel):
            def ask(self,image,prompt,path,**options):
                if path.stem.startswith('registration'):
                    value=dict(instances=[],floor_anchor_ids=[])
                    path.write_text(json.dumps(value));write(path.with_suffix('.json'),value)
                    return value
                return super().ask(image,prompt,path,**options)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,obs=make_case(root);case['image_size']=[64,64]
            out=root/'out';out.mkdir();scorer=SyntheticScorer()
            r=run(BrokenRegistration(case),scorer,case,out,max_count=3,generic_types=3,worlds=1)
            self.assertEqual(r['failed_stage'],'observed_registration')
            self.assertFalse(r['semantic_count_posterior_defined']);self.assertEqual(scorer.calls,[])
            self.assertEqual(read(out/'case/G_obs.json'),obs)
            html=(out/'case/index.html').read_text()
            self.assertIn('href="G_obs.json"',html)
            self.assertNotIn('href="semantic_model.json"',html)
            self.assertNotIn('src="causal_graph.svg"',html)

    def test_empty_outside_draw_preserves_observations_and_visuals(self):
        class EmptyScorer(SyntheticScorer):
            def score(self,description,parents,states):
                if states[0]['id'].isdigit():
                    # Observed lamp count two, all other category counts zero.
                    values=[0.]*len(states);values[2 if 'type lamp.' in description else 0]=1.
                    return dict(probabilities=dict(zip([s['id'] for s in states],values)),rounds=[])
                return super().score(description,parents,states)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,observed=make_case(root);case['image_size']=[64,64]
            out=root/'out';out.mkdir()
            # Seed search uses the same finite model, with no forced nonempty draw.
            spec=fixture_spec()
            spec['object_types']=['lamp','shelf','fixture unknown object'];spec['max_count']=3
            spec['mechanisms']['B']=[.5,.5];spec['mechanisms']['K']={'b1':[.5,.5],'b2':[.5,.5]}
            for b in spec['mechanisms']['N'].values():
                for k in b.values():k.update(lamp=[0.,0.,1.,0.],shelf=[1.,0.,0.,0.],**{'fixture unknown object':[1.,0.,0.,0.]})
            post=infer(spec,{'lamp':2});seed=next(s for s in range(100) if draw(post,1000+s)['new_instances']==0)
            r=run(SyntheticModel(case),EmptyScorer(),case,out,max_count=3,generic_types=3,worlds=1,seed=seed)
            self.assertEqual(r['status'],'posterior_samples_exported_needs_review',r)
            self.assertEqual(read(out/'case/G_full.json')['objects'],observed['objects'])
            self.assertEqual(read(out/'case/hypothesis_01/G_out.json')['objects'],[])

    def test_failed_first_sample_is_not_replaced_by_successful_later_sample(self):
        class FailedFirst(SyntheticModel):
            def ask(self,image,prompt,path,**options):
                v=super().ask(image,prompt,path,**options)
                if path.parent.name=='hypothesis_01' and path.stem.startswith('supports'):
                    v['objects']=[];path.write_text(json.dumps(v));write(path.with_suffix('.json'),v)
                return v
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,_=make_case(root);case['image_size']=[64,64]
            out=root/'out';out.mkdir()
            r=run(FailedFirst(case),SyntheticScorer(),case,out,max_count=3,generic_types=3,worlds=2,seed=0)
            self.assertEqual(r['status'],'partial_failure')
            self.assertEqual(r['hypotheses'][0]['status'],'failed')
            self.assertEqual(r['hypotheses'][0]['failed_stage'],'supports')
            self.assertIn('ValueError',r['hypotheses'][0]['error'])
            self.assertTrue((out/'case/hypothesis_01/sampled_instances.json').exists())
            self.assertFalse((out/'case/G_full.json').exists())

    def test_complete_posterior_sampling_batched_layout_export_visuals_and_cpu_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,observed=make_case(root);case['image_size']=[64,64]
            # Select a deterministic synthetic seed that exercises multiple layout batches.
            specs=fixture_spec();specs['object_types']=['lamp','shelf','fixture unknown object'];specs['max_count']=3
            for b in specs['mechanisms']['N'].values():
                for k in b.values():
                    for t in specs['object_types']:k[t]=[0.,0.,0.,1.]
            specs['mechanisms']['B']=[.5,.5];specs['mechanisms']['K']={'b1':[.5,.5],'b2':[.5,.5]}
            post=infer(specs,{'lamp':len(case['context'])})
            seed=next(s for s in range(100) if draw(post,s+1000)['new_instances']>=3)
            out=root/'out';out.mkdir();model=SyntheticModel(case);scorer=SyntheticScorer()
            record=run(model,scorer,case,out,max_count=3,generic_types=3,worlds=2,seed=seed,batch_size=2)
            self.assertEqual(record['status'],'posterior_samples_exported_needs_review',record)
            full=read(out/'case/G_full.json');sample=read(out/'case/hypothesis_01/posterior_sample.json')
            self.assertEqual(full['objects'][:len(observed['objects'])],observed['objects'])
            self.assertEqual(len(full['objects'])-len(observed['objects']),sample['new_instances'])
            self.assertGreaterEqual(sample['new_instances'],3)
            self.assertGreaterEqual(len(read(out/'case/hypothesis_01/layout_batches.json')),2)
            self.assertTrue(any(not o['category_id']['valid'] for o in full['objects'][len(observed['objects']):]))
            self.assertTrue(model.calls[0][1]);self.assertEqual(model.calls[0][0],'domain')
            self.assertTrue(all(set(parents)<= {'B','K'} for _,parents,_ in scorer.calls))
            args=dict(generic_types=3,max_count=3,worlds=2,seed=seed,max_objects=256)
            write(out/'config.json',dict(case=case,code_sha256=code_hashes(),arguments=args))
            self.assertEqual(audit(out)['status'],'scm_postflight_contract_passed')
            broken=read(out/'case/hypothesis_01/posterior_sample.json');broken['new_instances']+=1
            write(out/'case/hypothesis_01/posterior_sample.json',broken)
            with self.assertRaises(ValueError):audit(out)

    def test_actual_local_processor_and_tiny_Qwen_text_only_mechanism_forward(self):
        import torch
        from types import SimpleNamespace
        from transformers import AutoProcessor,Qwen3VLConfig,Qwen3VLForConditionalGeneration
        from .qwen_upstream import DEFAULT_MODEL
        processor=AutoProcessor.from_pretrained(str(DEFAULT_MODEL),local_files_only=True)
        config=Qwen3VLConfig(text_config=dict(vocab_size=152064,hidden_size=32,intermediate_size=64,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=4096,
            rope_scaling={'rope_type':'default','mrope_section':[2,3,3]}),
            vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,out_hidden_size=32,
            patch_size=16,spatial_merge_size=2,temporal_patch_size=2,num_position_embeddings=256,deepstack_visual_indexes=[]))
        tiny=Qwen3VLForConditionalGeneration(config).eval()
        scorer=TextMechanismScorer(SimpleNamespace(model=tiny,processor=processor,torch=torch))
        r=scorer.score('synthetic API test',{'B':'one hypothetical background'},[dict(id='0',description='none'),dict(id='1',description='one')])
        self.assertAlmostEqual(sum(r['probabilities'].values()),1.)
        self.assertFalse(r['image_used']);self.assertEqual(len(r['rounds']),2)


class LayoutTests(unittest.TestCase):
    def setup(self,root):
        case,observed=make_case(root);case['image_size']=[64,64]
        sample=dict(seed=1,B='b1',K='k1',counts=[dict(object_type='cabinet',outside=1),dict(object_type='lamp',outside=1)])
        plan=instantiate(sample,case,fixture_spec());a,b=[o['id'] for o in plan['additions']]
        rows=dict(objects=[dict(id=a,placement='Behind camera.',support=dict(kind='floor',target=None)),
                          dict(id=b,placement='On new cabinet.',support=dict(kind='object',target=a))])
        plan=support_answer(rows,plan,case)
        plane=support_plane(observed,[case['context'][0]['id']])
        return case,observed,plan,plane

    def test_support_parent_is_compiled_across_geometry_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,obs,plan,plane=self.setup(root);dest=root/'layout';dest.mkdir()
            class Model:
                def ask(self,image,prompt,path,**options):
                    ids=json.JSONDecoder().raw_decode(prompt.split('Place ONLY these batch IDs: ')[1])[0]
                    value=dict(objects=[dict(id=i,azimuth_deg=180.,range_m=3.,size=[1.,1.,1.],reason='test') for i in ids])
                    path.write_text(json.dumps(value));return value
            items,review=solve_batches(Model(),None,case,obs,plan,map_plan(plan,'semantic_lexicon_v1'),plane,dest,dict(seed=1),1)
            self.assertEqual(len(items),2);self.assertTrue(review['geometry_contract_passed'])
            self.assertAlmostEqual(items[1]['center'][1]-items[0]['center'][1],1.)

    def test_invalid_resize_replan_keeps_first_complete_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,obs,plan,plane=self.setup(root);dest=root/'layout';dest.mkdir()
            plan['additions']=plan['additions'][:1];plan['relations']=[];plan['placement_dependency_order']=plan['placement_dependency_order'][:1]
            class Model:
                def ask(self,image,prompt,path,**options):
                    ids=json.JSONDecoder().raw_decode(prompt.split('Place ONLY these batch IDs: ')[1])[0]
                    size=.1 if '_1' in path.stem else 1.
                    value=dict(objects=[dict(id=i,azimuth_deg=0.,range_m=3.,size=[size]*3,reason='test') for i in ids])
                    path.write_text(json.dumps(value));return value
            items,review=solve_batches(Model(),None,case,obs,plan,map_plan(plan,'semantic_lexicon_v1'),plane,dest,dict(seed=1),1)
            self.assertFalse(review['geometry_contract_passed']);self.assertEqual(items[0]['size'],[1.,1.,1.])


if __name__=='__main__':unittest.main()
