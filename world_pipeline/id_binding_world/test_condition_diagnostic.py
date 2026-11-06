"""CPU regressions: controlled counterfactuals, real hooks, RNG and reports."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .condition_diagnostic_data import condition_variant,case_support,history_ids,MODES,checkpoint_exposures
from .condition_diagnostic import fixed_noise,metrics
from .condition_diagnostic_probe import ActualAttentionProbe
from .runtime import WorldBranch
from .. import oracle_test
from ..oracle_memory import training_autocast


class ConditionDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def tensors(self):
        category=torch.tensor([[0,3,3,7,11,14]])
        geometry=torch.arange(48,dtype=torch.float32).reshape(1,6,8)/10;geometry[:,:,6:]=1
        mask=torch.ones(1,6,dtype=torch.bool)
        return category,geometry,mask

    def test_wrong_categories_preserve_histogram_geometry_global_and_source(self):
        data=self.tensors();before=[t.clone() for t in data]
        variant,meta=condition_variant(data,'wrong_category')
        self.assertEqual(sorted(variant[0][0].tolist()),sorted(data[0][0].tolist()))
        self.assertEqual(int(variant[0][0,0]),0);self.assertGreater(meta['changed_objects'],0)
        for a,b in zip(variant[1:],data[1:]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(before,data):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_wrong_positions_only_change_known_centers(self):
        data=self.tensors();variant,meta=condition_variant(data,'wrong_position')
        torch.testing.assert_close(variant[0],data[0]);torch.testing.assert_close(variant[1][:,0],data[1][:,0])
        torch.testing.assert_close(variant[1][...,3:],data[1][...,3:])
        self.assertEqual(sorted(map(tuple,variant[1][0,1:,:3].tolist())),sorted(map(tuple,data[1][0,1:,:3].tolist())))
        self.assertGreater(meta['mean_center_displacement_m'],0)
        self.assertIsNone(condition_variant(data,'world_off')[0])

    def test_fixed_noise_is_independent_of_global_rng_and_repeats_exactly(self):
        posterior=dict(mean=torch.zeros(1,4,1,4,8),std=torch.full((1,4,1,4,8),.2))
        mean=torch.zeros(1,4,1,1,1);std=torch.ones_like(mean)
        state=torch.get_rng_state().clone();a=fixed_noise(posterior,mean,std,83)
        torch.testing.assert_close(state,torch.get_rng_state(),rtol=0,atol=0)
        torch.randn(20);b=fixed_noise(posterior,mean,std,83)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertFalse(torch.equal(a[0],fixed_noise(posterior,mean,std,84)[0]))

    def test_projection_probe_reads_real_outputs_once_and_never_changes_forward(self):
        model=oracle_test.model(False).bfloat16();world=WorldBranch(32,(0,2),16,binding_block=2).eval().requires_grad_(False)
        for branch in world.branches.values():branch.gate.data.fill_(.1)
        world.attach(model);kwargs=oracle_test.inputs()
        for k in ('hidden_states','encoder_hidden_states'):kwargs[k]=kwargs[k].bfloat16()
        spec=dict(queries=[0,2,7,10,24,38],pairs=[(0,1),(1,1),(2,2),(3,2)],
            objects=[dict(instance_id='1',category_id=3,slot=1),dict(instance_id='2',category_id=3,slot=2)])
        with torch.inference_mode(),training_autocast('cpu'):
            world.set_world(self.tensors());expected=model(**kwargs)[0].clone()
            probe=ActualAttentionProbe(world);probe.reset(spec)
            calls=[];handle=world.branches['0'].q.register_forward_hook(lambda m,a,o:calls.append(1))
            actual=model(**kwargs)[0]
        torch.testing.assert_close(expected,actual,rtol=0,atol=0);self.assertEqual(len(calls),1)
        self.assertEqual(set(probe.records),{'0','2'});self.assertFalse(probe.pending)
        for value in probe.records.values():self.assertTrue(np.isfinite(value['gt_instance_probability_over_uniform']))
        handle.remove();probe.close()

    def test_queries_and_instance_weights_handle_padded_target_grid(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'regions.npz';instance=np.ones((8,16),np.int32);instance[:,8:]=2
            np.savez(p,instance=instance,object_loss_valid=np.ones((8,16),bool))
            row=dict(objects=[dict(instance_id='1',category_id=3),dict(instance_id='2',category_id=7)],
                     object_region_supervision=dict(path=str(p)))
            case=dict(objects=[dict(instance_id=str(i),name='object',outside_local_fraction=.9) for i in (1,2)])
            weights,spec=case_support(row,case,64,1)
            torch.testing.assert_close(weights.sum((-2,-1)),torch.ones(2))
            self.assertTrue(all(1<=q%10<=8 and q<40 for q in spec['queries']))
            self.assertEqual({s for q,s in spec['pairs']},{1,2})

    def test_history_does_not_exclude_entire_pack_or_confuse_panorama_exposure(self):
        old='X7HyMhZNoso_'+'a'*32;new='X7HyMhZNoso_'+'b'*32
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'qwen_edit_pano/outputs/oracle_probe';p.mkdir(parents=True)
            (p/'run_config.json').write_text(json.dumps(dict(ids=[old],config=dict(pack_plan=dict(sources=[dict(id=new)])))))
            nested=p/'benchmark/generation/correct'/old;nested.mkdir(parents=True)
            b=root/'qwen_edit_pano/outputs/paired_baseline';b.mkdir()
            (b/'run_config.json').write_text(json.dumps(dict(ids=[new])))
            world,panorama=history_ids(root)
            self.assertEqual(set(world),{old});self.assertEqual(set(panorama),{new})
            self.assertTrue(any('benchmark/generation/correct' in p for p in world[old]))

    def test_metric_yaw_is_paired_and_does_not_change_global_python_rng(self):
        import random
        p=torch.randn(1,4,1,8,16);n=torch.randn_like(p);z=torch.randn_like(p)
        weights=torch.ones(8,16);iw=torch.ones(2,8,16)/(8*16)
        state=random.getstate();a=metrics(p,n-z,weights,iw,n,z,dict(region_weight=.25),17)
        self.assertEqual(state,random.getstate())
        random.random();b=metrics(p,n-z,weights,iw,n,z,dict(region_weight=.25),17)
        self.assertEqual(a,b)

    def test_checkpoint_exposures_follow_resume_and_discard_abandoned_tails(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);first=root/'first';second=root/'second'
            a=first/'checkpoint-step000002';b=second/'checkpoint-step000004'
            a.mkdir(parents=True);b.mkdir(parents=True)
            (a/'COMPLETE.json').write_text(json.dumps(dict(step=2)))
            (b/'COMPLETE.json').write_text(json.dumps(dict(step=4)))
            (first/'run_plan.json').write_text(json.dumps(dict(resume=None)))
            (second/'run_plan.json').write_text(json.dumps(dict(resume=str(a))))
            for directory,rr in [(first,[(1,'A'),(2,'B'),(3,'discarded')]),(second,[(3,'A'),(4,'C'),(5,'discarded')])]:
                (directory/'train_log.jsonl').write_text(''.join(json.dumps(dict(step=i,sample_ids=[sid]))+'\n' for i,sid in rr))
            self.assertEqual(dict(checkpoint_exposures(b)),dict(A=2,B=1,C=1))
            (second/'train_log.jsonl').write_text(json.dumps(dict(step=4,sample_ids=['C']))+'\n')
            with self.assertRaisesRegex(ValueError,'incomplete'):checkpoint_exposures(b)

    def test_report_computes_known_paired_gaps_and_requires_every_record(self):
        from .condition_diagnostic_report import report
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cases=[dict(id=s,split=s,scene_id=s,latent_seed=1,objects=[{}],prior_world_diagnostic=False) for s in ('train','val')]
            protocol=dict(version='test',steps=[1000,10000],schedule=[dict(index=500,sigma=.5)],modes=list(MODES),
                          panel=dict(cases=cases,train_count=1,val_count=1))
            for step in protocol['steps']:
                for c in cases:
                    for mode in MODES:
                        directory=root/'records'/f'{step:06d}'/c['id']/'0500';directory.mkdir(parents=True,exist_ok=True)
                        delta=.2 if mode.startswith('wrong') else .3 if mode=='world_off' else 0.
                        att={key:dict(global_attention_mass=.2,gt_instance_probability_over_uniform=2.,gt_instance_top1_fraction=.4,
                            head_mean_entropy_normalized=.9,object_value_common_ratio=.8,update_query_variation_ratio=.1) for key in ('20','40','59')}
                        row=dict(step=step,id=c['id'],split=c['split'],latent_seed=1,mode=mode,scheduler_index=500,sigma=.5,
                            total=1.+delta,object_region=2.+delta,relative_prediction_change=0.,selected_object_errors=[2.+delta],attention=att)
                        (directory/(mode+'.json')).write_text(json.dumps(row))
            report(root,protocol);summary=json.loads((root/'SUMMARY.json').read_text())
            r=next(r for r in summary['paired_summary'] if r['mode']=='wrong_category')
            self.assertAlmostEqual(r['relative_total_gap_pct'],20.)
            self.assertTrue((root/'report.html').is_file());self.assertTrue((root/'DIAGNOSTIC.png').is_file())
            (root/'records/010000/val/0500/world_off.json').unlink()
            with self.assertRaisesRegex(ValueError,'incomplete'):report(root,protocol)


if __name__=='__main__':unittest.main()
