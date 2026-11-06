"""CPU checks for prompt separation, reuse provenance, and safe partial resume."""
import copy
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ..build_polished_world import atomic_json
from . import caption_context_probe as probe
from .state_evaluate import expected_record, jobs


class CaptionProbeTests(unittest.TestCase):
    def fixture(self,source):
        old_case=dict(id='val1',split='val',seed=123,source=dict(view='existing-view'))
        old=dict(cases=[old_case],arms=dict(pano_only=None,C6000='/C6000'),prompt='Expand this photo.')
        case=dict(**old_case,full_GT_caption='A room with a table.',
            prompts=dict(generic=old['prompt'],GTcaption=probe.caption_prompt(old['prompt'],'A room with a table.')))
        protocol=dict(cases=[case],arms=dict(pano_generic=None,pano_GTcaption=None,
            C6000_generic='/C6000',C6000_GTcaption='/C6000'),generic_reuse_source=str(source),source_protocol=old)
        execution=dict(device='CPU test fixture, no generation',device_uuid='one')
        return old,protocol,execution

    def test_prompt_variants_keep_instruction_state_and_seed(self):
        old,p,_=self.fixture(Path('/unused'))
        records=[probe.record(p,c,a) for c,a in jobs(p)]
        self.assertEqual(len(records),4)
        self.assertEqual({r['seed'] for r in records},{123})
        self.assertEqual([r['checkpoint'] for r in records],[None,None,'/C6000','/C6000'])
        self.assertEqual([r['full_GT_caption_enters_forward'] for r in records],[False,True,False,True])
        self.assertEqual(records[0]['prompt'],records[2]['prompt'])
        self.assertEqual(records[1]['prompt'],records[3]['prompt'])
        self.assertTrue(records[1]['prompt'].startswith(old['prompt']+'\nScene description: '))
        with self.assertRaises(ValueError):probe.caption_prompt('instruction',' ')

    def test_generic_reuse_requires_identical_runtime_and_correct_arm(self):
        with TemporaryDirectory() as directory:
            source=Path(directory);old,p,execution=self.fixture(source);case=p['cases'][0]
            atomic_json(source/'EXECUTION.json',execution);(source/'val1').mkdir()
            for arm in ('pano_only','C6000'):
                path=source/'val1'/(arm+'.png');path.write_bytes(b'fixture, not a generated image')
                atomic_json(path.with_suffix('.json'),dict(**expected_record(old,old['cases'][0],arm),
                    execution=execution,branch_calls=probe.CALLS))
            self.assertEqual(probe.reusable_generic(p,case,'pano_generic',execution).name,'pano_only.png')
            self.assertEqual(probe.reusable_generic(p,case,'C6000_generic',execution).name,'C6000.png')
            self.assertIsNone(probe.reusable_generic(p,case,'pano_GTcaption',execution))
            self.assertIsNone(probe.reusable_generic(p,case,'C6000_GTcaption',execution))
            self.assertIsNone(probe.reusable_generic(p,case,'pano_generic',{**execution,'device_uuid':'other'}))
            changed=copy.deepcopy(p);changed['cases'][0]['seed']+=1
            with self.assertRaises(ValueError):probe.reusable_generic(changed,changed['cases'][0],'pano_generic',execution)
            changed=copy.deepcopy(p);changed['cases'][0]['prompts']['generic']='Changed instruction'
            with self.assertRaises(ValueError):probe.reusable_generic(changed,changed['cases'][0],'pano_generic',execution)

    def test_partial_resume_checks_actual_prompt_and_oracle_flag(self):
        with TemporaryDirectory() as directory:
            output=Path(directory);_,p,execution=self.fixture(output/'unused-source')
            atomic_json(output/'PROTOCOL.json',p);atomic_json(output/'EXECUTION.json',execution)
            self.assertEqual(len(probe.pending(output,p)),4)
            case=p['cases'][0];arm='C6000_GTcaption';folder=output/case['id'];folder.mkdir()
            actual=dict(**probe.record(p,case,arm),execution=execution,branch_calls=probe.CALLS)
            atomic_json(folder/(arm+'.json'),actual)
            self.assertEqual(len(probe.pending(output,p)),4)  # PNG still missing.
            (folder/(arm+'.png')).write_bytes(b'fixture, not a generated image')
            self.assertEqual(len(probe.pending(output,p)),3)
            for key,value in [('prompt','Wrong text'),('full_GT_caption_enters_forward',False)]:
                changed={**actual,key:value};atomic_json(folder/(arm+'.json'),changed)
                with self.assertRaises(ValueError):probe.pending(output,p)
            atomic_json(folder/(arm+'.json'),actual)
            changed=copy.deepcopy(p);changed['cases'][0]['prompts']['GTcaption']='Different caption'
            with self.assertRaises(ValueError):probe.pending(output,changed)

    def test_preflight_does_not_call_cuda_or_create_output(self):
        with TemporaryDirectory() as directory:
            output=Path(directory)/'not-created';_,p,_=self.fixture(Path('/unused-source'))
            with patch.object(probe,'prepare',return_value=({}, {}, None, {}, p)), \
                    patch.object(sys,'argv',['probe','--preflight','--output',str(output)]), \
                    patch('torch.cuda.init') as cuda:
                probe.main();cuda.assert_not_called()
            self.assertFalse(output.exists())

    def test_summary_keeps_last_oracle_column_and_counts_all_24_images(self):
        # Synthetic CPU images only: catch the former fixed four-column canvas.
        from PIL import Image
        from .state_evaluate import summarize
        with TemporaryDirectory() as directory:
            output=Path(directory);_,p,execution=self.fixture(output/'unused-source')
            original=p['cases'][0]
            p['cases']=[dict(**{k:v for k,v in original.items() if k not in ('id','seed')},
                             id=f'val{i}',seed=i) for i in range(6)]
            p.update(height=64,width=128)
            atomic_json(output/'PROTOCOL.json',p);atomic_json(output/'EXECUTION.json',execution)
            for case in p['cases']:
                folder=output/case['id'];folder.mkdir()
                Image.new('RGB',(64,64),'white').save(folder/'local.png')
                for arm in p['arms']:
                    color=(230,10,10) if arm=='C6000_GTcaption' else (20,40,60)
                    Image.new('RGB',(128,64),color).save(folder/(arm+'.png'))
                    atomic_json(folder/(arm+'.json'),dict(**probe.record(p,case,arm),
                        execution=execution,branch_calls=probe.CALLS))
            with patch(__package__+'.state_evaluate.target_image',side_effect=lambda row:Image.new('RGB',(128,64),'white')), \
                    patch(__package__+'.state_evaluate.select_objects',return_value=[]):
                summarize(output,{}, {c['id']:dict(id=c['id'],objects=[]) for c in p['cases']},p)
            from ..build_polished_world import read
            self.assertEqual(read(output/'COMPLETE.json')['images'],24)
            with Image.open(output/'val0/comparison.jpg') as image:
                self.assertEqual(image.width,5*384)
                red,green,blue=image.getpixel((4*384+192,100))
                self.assertGreater(red,200);self.assertLess(green,30);self.assertLess(blue,30)


if __name__=='__main__':unittest.main()
