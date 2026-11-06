"""CPU checks of the diagnostic intervention, not tests of generation quality."""
import unittest
from pathlib import Path
import tempfile
import torch
from ..build_polished_world import atomic_json,read
from ..oracle_memory import training_autocast
from ..spatial_world.test_spatial import state
from .test_grounded import GroundedTests
from .value_probe import mean_class_values,replace_values,report,MODES


class ValueProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_average_preserves_each_geometry_and_is_category_order_independent(self):
        model,world,kw=GroundedTests().model();s=state();categories=[3,14,21,22]
        with torch.no_grad(),training_autocast('cpu'):
            values=mean_class_values(world,s,categories);reverse=mean_class_values(world,s,list(reversed(categories)))
            explicit={key:[] for key in world.branches}
            for cid in categories:
                c=s[0].clone();c[:,1:]=cid;tokens,_=world.encoder(c,s[1],s[2])
                for key,branch in world.branches.items():explicit[key].append(branch.v(branch.world_norm(tokens.float())).float())
            for key in values:
                expected=torch.stack(explicit[key]).mean(0).to(values[key].dtype);expected[:,0]=0
                torch.testing.assert_close(values[key],expected,rtol=0,atol=0)
                torch.testing.assert_close(values[key],reverse[key],rtol=.01,atol=.002)
                self.assertEqual(values[key][:,0].count_nonzero(),0)
            # A single real category reproduces that category's actual V,
            # rather than taking a scene average that would erase geometry.
            one=mean_class_values(world,s,[categories[0]])
            for key in one:torch.testing.assert_close(one[key][:,1:],explicit[key][0][:,1:].to(one[key].dtype),rtol=0,atol=0)

    def test_actual_forward_ablation_keeps_keys_weights_routes_and_removes_hooks(self):
        model,world,kw=GroundedTests().model();before={k:v.clone() for k,v in world.state_dict().items()}
        for branch in world.branches.values():branch.gate.data.fill_(.2)
        before={k:v.clone() for k,v in world.state_dict().items()};keys=[];handles=[]
        for branch in world.branches.values():handles.append(branch.k.register_forward_hook(lambda m,a,o:keys.append(o.detach().clone())))
        with torch.no_grad(),training_autocast('cpu'):
            values=mean_class_values(world,state(),[3,14,21,22]);world.set_world(state());route=world.current[2].clone()
            correct=model(**kw)[0];normal_keys=keys[:];keys.clear()
            with replace_values(world,values):altered=model(**kw)[0]
            for a,b in zip(normal_keys,keys):torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(route,world.current[2],rtol=0,atol=0)
            self.assertGreater(float((correct.float()-altered.float()).abs().max()),0)
            repeated=model(**kw)[0];torch.testing.assert_close(correct,repeated,rtol=0,atol=0)
        for handle in handles:handle.remove()
        for key,value in world.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
        self.assertTrue(all(not branch.v._forward_hooks for branch in world.branches.values()))

    def test_report_matches_controlled_gaps_and_rejects_changed_identity_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);expected=[]
            protocol=dict(checkpoints={'A':'a','B':'b'},indices=[499],cases=[dict(id='train',split='train',latent_seed=0),dict(id='val',split='val',latent_seed=1)])
            for group in protocol['checkpoints']:
                for case in protocol['cases']:
                    for mode in MODES:
                        file=output/'records'/group/case['id']/'499'/(mode+'.json');expected.append(file);file.parent.mkdir(parents=True,exist_ok=True)
                        atomic_json(file,dict(group=group,id=case['id'],split=case['split'],latent_seed=case['latent_seed'],index=499,sigma=.5,mode=mode,
                            total=5. if mode=='mean_class_value' else 4.,object_region=2.,selected_object_errors=[3. if mode=='mean_class_value' else 2.],
                            relative_prediction_change=.1 if mode in ('mean_class_value','world_off') else 0.))
            report(output,expected,protocol);summary=read(output/'SUMMARY.json')
            self.assertEqual(len(summary['paired']),4)
            self.assertTrue(all(r['selected_error_gap_pct']==50. for r in summary['paired']))
            file=expected[0];original=read(file);atomic_json(file,{**original,'latent_seed':999})
            with self.assertRaises(ValueError):report(output,expected,protocol)
            atomic_json(file,{**original,'id':'different'})
            with self.assertRaises(ValueError):report(output,expected,protocol)


if __name__=='__main__':unittest.main()
