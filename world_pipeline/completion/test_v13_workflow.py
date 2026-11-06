"""Restored source identity and isolated proposal-budget experiment contracts."""
from copy import deepcopy
from unittest.mock import patch
import unittest

from ..common import read,write,sha
from . import v13_workflow as workflow,organized_local_world as baseline
from . import test_organized_local_world as fixtures
from .test_compositional import CPUProgramProvider


class WorkflowTests(unittest.TestCase):
    setUp=fixtures.LocalWorldTests.setUp
    source=fixtures.LocalWorldTests.source

    def test_restored_core_matches_pinned_executed_baseline(self):
        result=workflow.verify();self.assertEqual(result['source_files'],118)
        self.assertEqual(result['model'],'organized_functional_program_scm_v13')

    def test_source_drift_is_rejected_without_touching_real_baseline(self):
        p=self.base/'source.py';p.write_text('original');manifest=self.base/'pins.json'
        write(manifest,dict(model='fixture',source_sha256={str(p):sha(p)}))
        self.assertEqual(workflow.verify(manifest)['source_files'],1)
        p.write_text('changed')
        with self.assertRaisesRegex(ValueError,'differs from pinned baseline'):workflow.verify(manifest)

    def test_merge_preserves_instances_hosts_relations_and_separate_empty_round(self):
        h=dict(id='z0',description='A possible room.',functions=['reading'],assumptions=['Uncertain use.'])
        p=dict(hypothesis=deepcopy(h),groups=[dict(objects=[dict(key='table',attachment=dict(kind='floor',host=None)),
            dict(key='book',attachment=dict(kind='top',host='table'))])],relations=[dict(source='book',target='table',kind='near')])
        rounds=[([h],{'f0':p}),([h],{'f0':p})];before=deepcopy(rounds)
        hs,ps=workflow.merge_rounds(rounds)
        self.assertEqual(rounds,before);self.assertEqual(len(hs),2);self.assertEqual(len(ps),2)
        for value in ps.values():
            self.assertEqual(value['groups'],p['groups']);self.assertEqual(value['relations'],p['relations'])
            self.assertEqual(value['proposal_source']['backend'],'qwen')
        hs,ps=workflow.merge_rounds([([],{}),rounds[1]])
        self.assertEqual(len(ps),1);self.assertEqual(len(hs),1)

    def test_two_rounds_use_one_local_client_and_original_pool_then_freeze(self):
        source,sid=self.source();root=self.base/'run'
        workflow.prepare(root,source,worlds=2,draws=4)
        baseline.phase(root,'prepare-scenes')
        calls=[]
        class CountingProvider(CPUProgramProvider):
            def ask(self,image,prompt,path,**kwargs):
                calls.append((str(path),kwargs.get('seed')))
                return super().ask(image,prompt,path,**kwargs)
        client=CountingProvider()
        with patch.object(workflow,'provider',return_value=client) as factory:
            result=workflow.propose(root)
        self.assertIn(result['status'],('complete','complete_with_warnings'))
        factory.assert_called_once();self.assertEqual(factory.call_args.args[1],'qwen')
        comp=root/'scenes'/sid/'completion'
        primary=read(comp/'proposal_rounds/round_00/programs.json')
        secondary=read(comp/'proposal_rounds/round_01/proposals/qwen/programs.json')
        merged=read(comp/'proposals/qwen/programs.json')
        self.assertEqual(len(merged),len(primary)+len(secondary))
        hseeds=[seed for path,seed in calls if path.endswith('/hypotheses.txt')]
        self.assertEqual(len(hseeds),2);self.assertNotEqual(hseeds[0],hseeds[1])
        baseline.phase(root,'pool');domain=read(comp/'shared/domain.json')
        bindings=domain['base_model']['proposal_bindings']
        self.assertLess(len(set(bindings.values())),len(bindings),'Identical proposals from two rounds must use original semantic deduplication')
        self.assertEqual(set(domain['proposal_provenance']),{'qwen'})
        self.assertFalse((comp/'proposals/api').exists())
        self.assertFalse((comp/'proposal_rounds/round_01/proposals/api').exists())
        self.assertEqual(read(comp/'run.json')['completion_model'],'organized')
        with patch.object(workflow,'provider',side_effect=AssertionError('Completed rounds must not be retried')):
            workflow.propose(root)
        metadata=read(root/'v13_proposal_budget.json');metadata['rounds']=3;write(root/'v13_proposal_budget.json',metadata)
        with self.assertRaisesRegex(ValueError,'budget/source changed'):workflow.check(root)


if __name__=='__main__':unittest.main()
