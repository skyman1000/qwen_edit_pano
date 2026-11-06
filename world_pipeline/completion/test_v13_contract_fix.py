"""CPU regression for failures observed in the actual closed-category v1 run."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ..common import ROOT,read,write
from . import v13_contract_fix as fix
from .test_v13_category_contract import reply, MechanismProvider
from . import test_organized_local_world as fixtures


def raw_reply():
    raw=reply()
    for _,o in fix.objects(raw):o.pop('attachment',None)
    raw['alternatives'][0]['groups'][0]['new_objects'].append(dict(key='lamp',category_id=28,
        variant='ceiling light',reason='Possible illumination.',placement='Mounted overhead.'))
    return raw


def fields(raw):
    rows=[]
    for ai,o in fix.objects(raw):
        kind,host = ('top','cabinet') if o['key']=='mug' else ('ceiling',None) if o['key']=='lamp' else ('floor',None)
        row=dict(alternative=ai,key=o['key'],attachment=dict(kind=kind,host=host),
            affordances=['top'] if o['key']=='cabinet' else [])
        if fix.generic_variant(o):row['variant']='ceramic mug'
        rows.append(row)
    return dict(objects=rows)


class ProposalProvider:
    def __init__(self):self.requests=[]
    def ask(self,image,prompt,path,**kwargs):
        stem=Path(path).stem
        value=(dict(hypotheses=[dict(description='Possible storage and lighting.',functions=['storage'],assumptions=['Uncertain.'])])
            if stem=='hypotheses' else fields(raw_reply()) if stem.endswith('_installation') else raw_reply())
        self.requests.append(dict(path=str(path),prompt=prompt,**kwargs))
        Path(path).write_text(json.dumps(value));write(Path(path).with_suffix('.json'),value)
        return value


class FixTests(unittest.TestCase):
    setUp=fixtures.LocalWorldTests.setUp
    source=fixtures.LocalWorldTests.source

    def prepared(self):
        source,sid=self.source();root=self.base/'run'
        fix.prepare(root,source,worlds=1,draws=4)
        fix.prior.local.phase(root,'prepare-scenes')
        comp=root/'scenes'/sid/'completion'
        case=read(comp/'shared/case.json')
        h=dict(id='z0',description='Possible storage.',functions=['storage'],assumptions=['Uncertain.'])
        return root,sid,comp,case,h

    def test_missing_installation_is_never_accepted_as_free(self):
        _,_,_,case,h=self.prepared();raw=raw_reply();before=deepcopy(raw)
        with self.assertRaisesRegex(ValueError,'installation missing'):fix.parse(raw,case,h)
        self.assertEqual(raw,before)
        completed=fix.field_validator(raw,case,h)(fields(raw))
        self.assertEqual(completed['alternatives'][0]['groups'][0]['new_objects'][0]['attachment']['kind'],'floor')
        self.assertEqual([(ai,o['key'],o['category_id'],o['variant']) for ai,o in fix.objects(raw)],
            [(ai,o['key'],o['category_id'],o['variant']) for ai,o in fix.objects(completed)])

    def test_legal_lighting_and_furniture_are_not_rejected_by_legacy_broad_filter(self):
        _,_,_,case,h=self.prepared()
        value=reply();items=value['alternatives'][0]['groups'][0]['new_objects']
        items[0].update(category_id=36,variant='wooden cabinet')
        items.append(dict(key='lamp',category_id=28,variant='ceiling light',reason='Lighting.',placement='Overhead.',attachment=dict(kind='ceiling',host=None)))
        before=set(fix.parser.BROAD)
        programs,_,_=fix.parse(value,case,h)
        self.assertEqual(len(programs[0]['groups'][0]['objects']),3)
        self.assertEqual(fix.parser.BROAD,before)
        self.assertIn('lighting',fix.parser.BROAD)

    def test_fixed_fields_cannot_drop_add_rename_reclassify_or_rewrite_concrete_nouns(self):
        _,_,_,case,h=self.prepared();raw=raw_reply();accept=fix.field_validator(raw,case,h)
        mutations=[]
        v=fields(raw);v['objects'].pop();mutations.append(v)
        v=fields(raw);v['objects'].append(deepcopy(v['objects'][0]));mutations.append(v)
        v=fields(raw);v['objects'][0]['key']='new_name';mutations.append(v)
        v=fields(raw);v['objects'][0]['category_id']=5;mutations.append(v)
        v=fields(raw);v['objects'][0]['variant']='table';mutations.append(v)
        v=fields(raw);v['objects'][0]['alternative']=False;mutations.append(v)
        for v in mutations:
            with self.subTest(value=v),self.assertRaises(ValueError):accept(v)

    def test_generic_noun_may_be_refined_without_changing_coarse_id(self):
        _,_,_,case,h=self.prepared();raw=raw_reply()
        raw['alternatives'][0]['groups'][0]['new_objects'][1]['variant']='objects'
        out=fix.field_validator(raw,case,h)(fields(raw))
        self.assertEqual(list(fix.objects(out))[1][1]['variant'],'ceramic mug')
        self.assertEqual(list(fix.objects(out))[1][1]['category_id'],39)
        bad=fields(raw);bad['objects'][1]['variant']=None
        with self.assertRaises(ValueError):fix.field_validator(raw,case,h)(bad)

    def test_invalid_category_or_host_still_fails(self):
        _,_,_,case,h=self.prepared();raw=raw_reply();bad=deepcopy(raw)
        list(fix.objects(bad))[0][1]['category_id']=999
        with self.assertRaisesRegex(ValueError,'category_id'):fix.raw_validator()(bad)
        value=fields(raw);value['objects'][1]['attachment']['host']='unknown_host'
        with self.assertRaises(ValueError):fix.field_validator(raw,case,h)(value)

    def test_complete_explicit_installation_does_not_make_a_field_call(self):
        _,_,comp,case,h=self.prepared();raw=reply();out=comp/'proposals/qwen';(out/'calls').mkdir(parents=True)
        with patch.object(fix.prior.requests,'_ask',side_effect=AssertionError('Unnecessary model call')):
            result=fix.finalize(raw,case,h,None,read(comp/'run.json'),None,'test',out)
        self.assertTrue(result)
        self.assertFalse(read(out/'calls/test.contract.json')['field_completion_used'])

    def test_v1_and_core_sources_stay_bound(self):
        root,_,_,_,_=self.prepared()
        self.assertEqual(fix.check(root)['backends'],['qwen'])
        with patch.object(fix,'sources',return_value={}):
            with self.assertRaisesRegex(ValueError,'source changed'):fix.check(root)
        with self.assertRaisesRegex(ValueError,'source changed'):fix.prior.check(root)

    def test_actual_three_failed_queries_are_legacy_closed_label_collisions(self):
        suite=read(ROOT/'qwen_edit_pano/outputs/v13_category_optional_20261106_001/suite.json');count=0
        for scene in suite['scenes']:
            comp=ROOT/'qwen_edit_pano/outputs/v13_category_optional_20261106_001/scenes'/scene['id']/'completion'
            summary=read(comp/'proposals/qwen/summary.json');case=read(comp/'shared/case.json')
            hs={h['id']:h for h in read(comp/'proposals/qwen/hypotheses.json')}
            for failure in summary['failed_queries']:
                raw=read(comp/'proposals/qwen/calls'/(failure['stage']+'.json'))
                normalized,_=fix.prior.normalize(raw);h=hs[failure['stage'].split('_')[1]]
                with self.assertRaisesRegex(ValueError,'concrete noun, not lighting'):
                    fix.prior.original.furniture_answer(normalized,case,h)
                with fix.closed_labels():
                    programs=fix.prior.original.furniture_answer(normalized,case,h)
                    self.assertTrue(programs)
                count+=1
        self.assertEqual(count,3)  # Parser diagnosis only; support remains unverified.

    def test_full_original_scm_geometry_renderer_with_floor_host_and_closed_lighting(self):
        root,sid,comp,case,h=self.prepared();proposer=ProposalProvider()
        with fix.runtime(),patch.object(fix.prior,'provider',return_value=proposer):
            self.assertEqual(fix.prior.propose(root)['status'],'complete')
        self.assertEqual(len(proposer.requests),5)  # H, two proposals, two field calls.
        self.assertEqual(read(comp/'proposals/qwen/summary.json')['installation_field_queries'],2)
        with fix.runtime():
            fix.prior.local.phase(root,'pool')
            with patch.object(fix.prior.local.pipeline,'provider',return_value=MechanismProvider(all_present=True)):
                for phase in ('mechanisms','worlds'):
                    self.assertEqual(fix.prior.local.phase(root,phase)['status'],'complete')
            result=fix.prior.report(root)
        self.assertEqual(result['worlds_with_new_objects'],1)
        model=read(comp/'end_to_end/qwen/completion/causal_model.json')
        self.assertEqual(model['model_variant'],'organized_functional_program_scm_v13')
        world=read(comp/'end_to_end/qwen/completion/world_01/semantic_world.json')
        self.assertEqual({o['attachment']['kind'] for o in world['objects']},{'floor','top','ceiling'})
        state=read(comp/'end_to_end/qwen/completion/world_01/G_full.json')
        new=[o for o in state['objects'] if o['local_evidence']!='visible']
        self.assertEqual({o['category_id']['value'] for o in new},{7,28,39})
        self.assertTrue(all(o['category_id']['valid'] for o in new))
        cabinet=next(o for o in new if o['category_id']['value']==7)
        mug=next(o for o in new if o['category_id']['value']==39)
        self.assertAlmostEqual(mug['center_local_m']['value'][1]-mug['size_aabb_local_m']['value'][1]/2,
            cabinet['center_local_m']['value'][1]+cabinet['size_aabb_local_m']['value'][1]/2,places=6)
        views=root/'report'/sid/'world_01/views'
        self.assertTrue((views/'scene_3d.html').exists());self.assertTrue((views/'erp_overlay.png').exists())
        self.assertTrue(any('28:' in r['label'] for r in read(views/'labels.json')))


if __name__=='__main__':unittest.main()
