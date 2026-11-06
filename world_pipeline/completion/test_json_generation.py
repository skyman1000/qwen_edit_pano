"""Replay the real registration failure through the generation wrapper on CPU."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ..common import read
from .geometry import parse_json
from .json_generation import complete_array_prefix, repeated_object_tail
from .qwen_upstream import DEFAULT_MODEL, Qwen
from .scm_mechanisms import domain_answer, registration_answer


FIXTURE=Path(__file__).with_name('testdata')/'scm_registration_false_loop.txt'


class GuardTests(unittest.TestCase):
    def test_actual_failed_prefix_contains_distinct_valid_cushions(self):
        raw=FIXTURE.read_text();entries=complete_array_prefix(raw)
        self.assertEqual([o['id'] for o in entries[-3:]],
                         ['observed_0002','observed_0003','observed_0004'])
        old_signatures=[{k:v for k,v in o.items() if k not in ('id','track_id')} for o in entries[-3:]]
        self.assertEqual(old_signatures,[{'object_type':'cushion'}]*3)
        self.assertFalse(repeated_object_tail(raw))
        with self.assertRaises(ValueError):parse_json(raw)  # Never recover missing records from a prefix.

    def test_identity_is_preserved_for_both_id_fields(self):
        for field in ('id','track_id'):
            rows=[{field:str(i),'object_type':'chair'} for i in range(12)]
            self.assertFalse(repeated_object_tail(json.dumps(rows)))
            self.assertTrue(repeated_object_tail(json.dumps([rows[0]]*3)))

    def test_exact_duplicate_records_still_stop_regardless_of_key_order(self):
        self.assertTrue(repeated_object_tail('[{"id":"x","v":1},{"v":1,"id":"x"},{"id":"x","v":1},'))
        self.assertTrue(repeated_object_tail(json.dumps([dict(category='bed',box=[1,2,3,4])]*3)))
        self.assertFalse(repeated_object_tail('[{"id":"x"},{"id":"x"},'))


class WrapperReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoProcessor
        cls.torch=torch
        cls.processor=AutoProcessor.from_pretrained(str(DEFAULT_MODEL),local_files_only=True)

    def wrapper(self,raw):
        torch=self.torch;processor=self.processor
        class ReplayModel:
            device='cpu'
            generation_config=SimpleNamespace(eos_token_id=processor.tokenizer.eos_token_id)
            def generate(self,input_ids,stopping_criteria,**kwargs):
                answer=processor.tokenizer.encode(raw,add_special_tokens=False)+[self.generation_config.eos_token_id]
                result=input_ids.clone()
                for token in answer:
                    result=torch.cat((result,torch.tensor([[token]],dtype=result.dtype)),dim=1)
                    if bool(stopping_criteria(result,None).all()):break
                return result
        q=Qwen.__new__(Qwen);q.torch=torch;q.processor=processor;q.model=ReplayModel()
        return q

    def test_all_nine_instances_survive_the_real_160_token_failure_point(self):
        # The continuation is synthetic; the failed prefix is the untouched real reply.
        names=['couch']*2+['cushion']*4+['tray table','floor lamp','ottoman']
        rows=[dict(id=f'observed_{i:04d}',object_type=t) for i,t in enumerate(names)]
        value=dict(instances=rows,floor_anchor_ids=['observed_0000','observed_0001','observed_0007','observed_0008'])
        raw=json.dumps(value,indent=2);failed=FIXTURE.read_text()
        self.assertTrue(raw.startswith(failed))
        self.assertEqual(len(self.processor.tokenizer.encode(failed,add_special_tokens=False)),160)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'registration.txt'
            result=self.wrapper(raw).ask(None,'Synthetic replay of registration.',path,
                profile='official',sample=True,max_tokens=8192,guard_repetition=True)
            self.assertEqual(result,value)
            meta=read(path.with_suffix('.generation.json'))
            self.assertEqual(meta['finish_reason'],'eos');self.assertGreater(meta['generated_tokens'],160)
            self.assertEqual(read(path.with_suffix('.request.json'))['guard_repetition']['comparison'],
                             'complete_object_including_identity')
            case=dict(context=[dict(id=o['id']) for o in rows]);before=deepcopy(case)
            domain=domain_answer(dict(backgrounds=['a','b'],configurations=['x','y'],object_types=['couch','lamp']))
            registration=registration_answer(result,case,domain)
            self.assertEqual(registration['observed_counts']['cushion'],4)
            self.assertEqual(sum(registration['observed_counts'].values()),9)
            self.assertEqual(case,before)

    def test_true_loop_is_rejected_and_raw_and_reason_preserved(self):
        raw=json.dumps(dict(instances=[dict(id='same',object_type='cushion')]*30))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'loop.txt'
            with self.assertRaisesRegex(ValueError,'Repeated complete JSON'):
                self.wrapper(raw).ask(None,'Synthetic exact duplicate loop.',path,
                    profile='official',sample=True,max_tokens=8192,guard_repetition=True)
            self.assertTrue(path.is_file());self.assertFalse(path.with_suffix('.json').exists())
            self.assertEqual(read(path.with_suffix('.generation.json'))['finish_reason'],'repeated_objects')


if __name__=='__main__':unittest.main()
