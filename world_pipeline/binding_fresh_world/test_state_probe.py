"""CPU state-edit integrity, permutation control, resume and gallery checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from ..build_polished_world import FRAME
from ..object_world.data import tensorize
from ..id_binding_world.runtime import WorldBranch
from .state_probe import variants,validate_variants,expected,pending_jobs,review,changes,VERSION,SID


def example():
    def obj(i,c,center,size):return dict(instance_id=i,category_id=c,center_local_m=center,
        size_aabb_local_m=size,center_valid=True,size_valid=True)
    return dict(id=SID,frame=FRAME,objects=[obj('360',22,[-2.,.2,-1.],[1.4,.9,.15]),
        obj('328',3,[-1.,-1.,-3.],[.6,1.,.8]),obj('1',3,[0.,-.7,2.],[.8,1.,.8])])


NAMES={'22':'tv_monitor','3':'chair','6':'picture','21':'mirror','14':'plant'}


class StateProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_variants_are_single_attribute_offscreen_and_do_not_mutate_data(self):
        row=example();saved=copy.deepcopy(row);cases=variants(row)
        validate_variants(row,cases,NAMES,256);self.assertEqual(row,saved)
        self.assertEqual(len(cases),7);states={n:r for n,_,r in cases}
        for name in ('tv_to_picture','tv_to_mirror','chair_to_plant'):
            changes=[(a,b) for a,b in zip(row['objects'],states[name]['objects']) if a!=b]
            self.assertEqual(len(changes),1)
            a,b=changes[0]
            self.assertEqual({k:v for k,v in a.items() if k!='category_id'},
                {k:v for k,v in b.items() if k!='category_id'})
        a=states['move_tv']['objects'][0];b=row['objects'][0]
        self.assertEqual(a['center_local_m'],[b['center_local_m'][0]+1.,*b['center_local_m'][1:]])
        self.assertEqual(a['size_aabb_local_m'],b['size_aabb_local_m'])
        self.assertEqual([o['instance_id'] for o in states['remove_tv']['objects']],['328','1'])
        self.assertEqual([o['instance_id'] for o in states['permuted']['objects']],['1','328','360'])
        invalid=copy.deepcopy(cases);invalid[-1][2]['objects'][2]['category_id']=14
        with self.assertRaises(ValueError):validate_variants(row,invalid,NAMES,256)

    def test_permuted_condition_is_the_same_object_set_for_original_forward(self):
        states={n:r for n,_,r in variants(example())}
        torch.manual_seed(4);world=WorldBranch(32,(0,2),16,binding_block=2)
        world.branches['2'].gate.data.fill_(.2)
        hidden=torch.randn(1,9,32);results=[];encodings=[]
        with torch.no_grad():
            for name in ('correct','permuted'):
                state=tensorize(states[name]);tokens,mask=world.encoder(*state);encodings.append(tokens)
                result,_=world.branches['2'](hidden,tokens,mask,8);results.append(result)
        order=torch.tensor([0,3,2,1])
        torch.testing.assert_close(encodings[0],encodings[1][:,order],rtol=0,atol=0)
        torch.testing.assert_close(results[0],results[1],rtol=1e-3,atol=1e-3)

    def test_fourteen_job_resume_and_cpu_review(self):
        row=example();cases=variants(row);states={n:r for n,_,r in cases}
        protocol=dict(version=VERSION,id=SID,seed=17,step=10000,
            models={'A10000':'/fake/A','fresh_binding10000':'/fake/B'},height=256,width=512,
            variants=[dict(name=n,description=desc,changes=changes(row,r)) for n,desc,r in cases],images=14)
        execution={'device':'CPU fake images only'}
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);self.assertEqual(len(pending_jobs(out,protocol)),14)
            (out/'PROTOCOL.json').write_text(json.dumps(protocol));(out/'EXECUTION.json').write_text(json.dumps(execution))
            for i,(model,name) in enumerate(pending_jobs(out,protocol)):
                directory=out/model;directory.mkdir(exist_ok=True)
                pixels=np.full((256,512,3),i,dtype=np.uint8);Image.fromarray(pixels).save(directory/(name+'.png'))
                meta=dict(**expected(protocol,model,name),execution=execution,
                    branch_calls=dict(policy='both',cond=28,uncond=28,disabled_uncond=0))
                (directory/(name+'.json')).write_text(json.dumps(meta))
            self.assertFalse(pending_jobs(out,protocol))
            with patch('qwen_edit_pano.world_pipeline.binding_fresh_world.state_probe.target_image',
                    return_value=Image.new('RGB',(512,256))):
                review(out,dict(height=256,names=NAMES),row,states,protocol)
            self.assertTrue((out/'review.html').is_file())
            self.assertEqual(json.loads((out/'COMPLETE.json').read_text())['images'],14)
            metrics=json.loads((out/'CHANGE_METRICS.json').read_text())['rows']
            self.assertEqual(len(metrics),14)
            self.assertTrue(all(r['full_RGB_change']==0 for r in metrics if r['variant']=='correct'))
            for r in metrics:
                for k,v in r.items():
                    if isinstance(v,float):self.assertTrue(np.isfinite(v))
            self.assertIn('clipPath',(out/'review.html').read_text())
            with self.assertRaises(ValueError):pending_jobs(out,dict(protocol,seed=18))


if __name__=='__main__':unittest.main()
