"""Meaningful CPU contracts; synthetic provider replies are not quality results."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ..common import read, write
from . import v13_category_contract as contract
from . import test_organized_local_world as fixtures
from .test_multiscene import CPUProvider


def reply():
    return dict(alternatives=[dict(description='One plausible storage assembly outside the photographed region.',
        groups=[dict(function='storage', reason='Hidden storage with an ordinary belonging.', observed_refs=[],
            new_objects=[dict(key='cabinet', category_id=7, variant='wooden cabinet',
                reason='One separate storage instance.', placement='In the unseen surrounding room.',
                attachment=dict(kind='floor', host=None), affordances=['top']),
                dict(key='mug', category_id=39, variant='ceramic mug', reason='A drink container.',
                    placement='On this cabinet.', attachment=dict(kind='top', host='cabinet'))])],
        relations=[dict(source='mug', target='cabinet', kind='inside', reason='Uncertain object arrangement.')])])


class ProposalProvider:
    def __init__(self): self.requests=[]
    def ask(self, image, prompt, path, **kwargs):
        value = (dict(hypotheses=[dict(description='An ordinary room with possible hidden storage.',
            functions=['storage'], assumptions=['Surrounding space is not observed.'])])
            if Path(path).stem == 'hypotheses' else reply())
        self.requests.append(dict(path=str(path), prompt=prompt, **kwargs))
        Path(path).write_text(json.dumps(value)); write(Path(path).with_suffix('.json'), value)
        return value


class MechanismProvider(CPUProvider):
    def ask(self,image,prompt,path,**kwargs):
        if Path(path).stem.startswith('recognition_'):
            options=json.loads(prompt.split('\nALLOWED_CHOICES: ')[1])
            value=dict(choice=options[0]['choice'])
            Path(path).write_text(json.dumps(value));write(Path(path).with_suffix('.json'),value)
            return value
        return super().ask(image,prompt,path,**kwargs)


class ContractTests(unittest.TestCase):
    setUp=fixtures.LocalWorldTests.setUp
    source=fixtures.LocalWorldTests.source

    def test_exact_dataset_ids_keep_specific_nouns_and_advisory_information(self):
        value=reply(); before=deepcopy(value); clean,audit=contract.normalize(value)
        self.assertEqual(value,before)
        objects=clean['alternatives'][0]['groups'][0]['new_objects']
        self.assertEqual([(o['category_id'],o['category'],o['variant']) for o in objects],
            [(7,'cabinet','wooden cabinet'),(39,'objects','ceramic mug')])
        self.assertEqual(clean['alternatives'][0]['relations'],[])
        self.assertEqual(audit[0]['relations'],value['alternatives'][0]['relations'])
        self.assertEqual(objects[1]['attachment'],dict(kind='top',host='cabinet'))

    def test_unknown_support_is_optional_without_manufacturing_a_host(self):
        value=reply(); obj=value['alternatives'][0]['groups'][0]['new_objects'][1];del obj['attachment']
        clean,audit=contract.normalize(value)
        self.assertEqual(clean['alternatives'][0]['groups'][0]['new_objects'][1]['attachment'],dict(kind='free',host=None))
        self.assertEqual(audit[0]['groups'][0]['unknown_support'],['mug'])

    def test_unknown_or_noninteger_id_is_rejected_instead_of_mapped_to_misc(self):
        for cid in (-1,0,1,2,17,100,True,'7',7.0,None):
            value=reply(); value['alternatives'][0]['groups'][0]['new_objects'][0]['category_id']=cid
            with self.subTest(cid=cid),self.assertRaisesRegex(ValueError,'category_id must be an integer'):
                contract.normalize(value)

    def test_conflicting_class_or_specific_noun_is_rejected(self):
        value=reply();obj=value['alternatives'][0]['groups'][0]['new_objects'][0]
        obj['category']='table'
        with self.assertRaisesRegex(ValueError,'disagree'):contract.normalize(value)
        del obj['category'];obj['category_id']=5;obj['variant']='sofa'
        with self.assertRaisesRegex(ValueError,'conflicts'):contract.normalize(value)

    def prepared(self):
        source,sid=self.source();root=self.base/'run'
        contract.prepare(root,source,worlds=1,draws=4)
        contract.local.phase(root,'prepare-scenes')
        return root,sid,root/'scenes'/sid/'completion'

    def test_repair_can_fix_support_but_cannot_delete_or_reclassify_inventory(self):
        root,sid,comp=self.prepared();case=read(comp/'shared/case.json')
        h=dict(id='z0',description='Possible storage.',functions=['storage'],assumptions=['Uncertain.'])
        bad=reply();bad['alternatives'][0]['groups'][0]['new_objects'][0]['attachment']=dict(kind='wall',host='ghost')
        accept=contract.validator(case,h,self.base/'audit.json')
        with self.assertRaisesRegex(ValueError,'requires host=null'):accept(bad)
        deleted=reply();deleted['alternatives'][0]['groups'][0]['new_objects'].pop()
        with self.assertRaisesRegex(ValueError,'add/drop'):accept(deleted)
        changed=reply();changed['alternatives'][0]['groups'][0]['new_objects'][0]['category_id']=36
        with self.assertRaisesRegex(ValueError,'change a valid'):accept(changed)
        programs=accept(reply())
        self.assertEqual({o['key'] for p in programs for g in p['groups'] for o in g['objects']},{'cabinet','mug'})
        self.assertTrue(all(not p['relations'] for p in programs))

    def test_empty_hypothesis_is_preserved_without_forced_objects(self):
        value=dict(alternatives=[dict(description='An empty hidden delta.',groups=[],relations=[])])
        clean,audit=contract.normalize(value);self.assertEqual(value,clean)
        self.assertEqual(contract.identity_signature(clean),[[]])

    def test_sources_and_run_interface_are_bound(self):
        root,_,_=self.prepared();self.assertEqual(contract.check(root)['backends'],['qwen'])
        with patch.object(contract,'sources',return_value={}):
            with self.assertRaisesRegex(ValueError,'contract source changed'):contract.check(root)

    def test_original_v13_complete_pipeline_exports_only_dataset_ids_and_full_visuals(self):
        root,sid,comp=self.prepared();proposer=ProposalProvider()
        with patch.object(contract,'provider',return_value=proposer) as factory:
            self.assertEqual(contract.propose(root)['status'],'complete')
        factory.assert_called_once();self.assertEqual(factory.call_args.args[1],'qwen')
        self.assertEqual(len(proposer.requests),3)  # one H plus two original channels
        with patch.object(contract,'provider',side_effect=AssertionError('No resampling cached proposals')):
            contract.propose(root)
        self.assertFalse((comp/'proposals/api').exists())
        contract.local.phase(root,'pool')
        with patch.object(contract.local.pipeline,'provider',return_value=MechanismProvider(all_present=True)):
            for phase in ('mechanisms','worlds'):
                result=contract.local.phase(root,phase)
                self.assertEqual(result['status'],'complete',(phase,result))
        model=read(comp/'end_to_end/qwen/completion/causal_model.json')
        self.assertEqual(model['model_variant'],'organized_functional_program_scm_v13')
        self.assertTrue(model['activation_rows'])
        result=contract.report(root);self.assertEqual(result['worlds_with_new_objects'],1)
        state=read(comp/'end_to_end/qwen/completion/world_01/G_full.json')
        new=[o for o in state['objects'] if o['local_evidence']!='visible']
        self.assertEqual({o['category_id']['value'] for o in new},{7,39})
        self.assertTrue(all(o['category_id']['valid'] for o in new))
        views=root/'report'/sid/'world_01/views'
        labels=read(views/'labels.json')
        self.assertTrue(any('7:cabinet' in r['label'] for r in labels))
        self.assertTrue((views/'scene_3d.html').exists());self.assertTrue((views/'erp_overlay.png').exists())
        self.assertTrue(read(root/'category_contract_report.json')['generated_category_ids_checked'])


if __name__=='__main__':unittest.main()
