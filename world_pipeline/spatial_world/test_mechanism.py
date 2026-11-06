"""Actual tiny-Edit equivalence and readout controls, CPU only."""
import unittest
import tempfile
import json
from pathlib import Path
import torch
from .. import oracle_test
from ..oracle_memory import training_autocast
from .runtime import WorldBranch
from .mechanism_runtime import MechanismWorld
from .test_spatial import state


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def models(self):
        model=oracle_test.model(False).bfloat16();torch.manual_seed(7)
        world=WorldBranch(32,(0,2),16,64,1)
        for branch in world.branches.values():branch.gate.data.fill_(.05);branch.local.gate.data.fill_(.002)
        world.eval().requires_grad_(False);world.attach(model)
        other=oracle_test.model(False).bfloat16();probe=MechanismWorld(32,(0,2),16,64,1)
        probe.load_state_dict(world.state_dict(),strict=True);probe.eval().requires_grad_(False);probe.attach(other)
        kw=oracle_test.inputs()
        for k in ('hidden_states','encoder_hidden_states'):kw[k]=kw[k].bfloat16()
        return model,world,other,probe,kw

    def test_native_measured_and_off_match_actual_v1(self):
        model,world,other,probe,kw=self.models()
        with torch.inference_mode(),training_autocast('cpu'):
            world.set_world(state());reference=model(**kw)[0]
            for mode in ('native_v1','measured_v1'):
                probe.set_measure_mode(mode);probe.set_world(state());value=other(**kw)[0]
                torch.testing.assert_close(reference,value,rtol=0,atol=0)
            self.assertEqual(set(probe.measurements()),{'0','2'})
            world.enabled=False;world.set_world(state());expected=model(**kw)[0]
            probe.set_measure_mode('local_off');probe.set_world(state());actual=other(**kw)[0]
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)
            self.assertFalse(probe.measurements())

    def test_single_cast_zero_gate_is_legacy_exact_and_state_is_unmodified(self):
        model,world,other,probe,kw=self.models()
        for b in world.branches.values():b.local.gate.data.zero_()
        probe.load_state_dict(world.state_dict(),strict=True)
        before={k:v.clone() for k,v in probe.state_dict().items()}
        with torch.inference_mode(),training_autocast('cpu'):
            world.set_world(state());reference=model(**kw)[0]
            probe.set_measure_mode('single_cast');probe.set_world(state());actual=other(**kw)[0]
        torch.testing.assert_close(actual,reference,rtol=0,atol=0)
        for k,v in before.items():torch.testing.assert_close(v,probe.state_dict()[k],rtol=0,atol=0)
        for value in probe.measurements().values():
            self.assertEqual(value['actual_delta_rms'],0)
            self.assertEqual(value['changed_element_fraction'],0)

    def test_single_cast_readout_finite_and_reference_suffix_unchanged(self):
        _,_,_,probe,_=self.models();branch=probe.branches['0']
        hidden=torch.randn(1,44,32).bfloat16();tokens=torch.randn(1,4,16)
        routing=torch.zeros(1,40,4);routing[:,10:20,1]=1;mask=torch.ones(1,4,dtype=torch.bool)
        branch.measure_mode='single_cast'
        with torch.inference_mode(),training_autocast('cpu'):output,_=branch(hidden,tokens,mask,40,routing)
        torch.testing.assert_close(output[:,40:],hidden[:,40:],rtol=0,atol=0)
        self.assertEqual(branch.measurement['hidden_dtype'],'torch.bfloat16')
        for k,v in branch.measurement.items():
            if isinstance(v,float):self.assertTrue(torch.isfinite(torch.tensor(v)))

    def test_gradient_execution_is_rejected(self):
        _,_,_,probe,_=self.models();branch=probe.branches['0']
        with self.assertRaises(RuntimeError):branch(torch.randn(1,5,32),torch.randn(1,2,16),torch.ones(1,2,dtype=torch.bool),4,None)

    def test_summary_requires_all_records_and_exact_native_reproduction(self):
        from .mechanism import summarize,record_path,MODES
        with tempfile.TemporaryDirectory() as d:
            output=Path(d);cases=[dict(id='train-case',split='train',latent_seed=1),dict(id='val-case',split='val',latent_seed=2)]
            protocol=dict(version='cpu_contract',checkpoint='temporary_fixture_only',panel=dict(cases=cases),schedule=[dict(index=200,sigma=.8)])
            for case in cases:
                for mode in MODES:
                    path=record_path(output,case['id'],200,mode);path.parent.mkdir(parents=True,exist_ok=True)
                    m=dict(hidden_dtype='torch.bfloat16',nominal_relative_rms=.001,actual_relative_rms=.002,
                        actual_to_nominal_rms=2.,changed_element_fraction=.1,changed_supported_element_fraction=.2,supported_element_fraction=.5)
                    row=dict(**case,mode=mode,scheduler_index=200,sigma=.8,total=1.,object_region=1.,relative_prediction_change=0.,
                        native_prediction_exact=True,measurements={k:m for k in ('20','40','59')} if mode in ('measured_v1','single_cast') else {})
                    path.write_text(json.dumps(row))
            summarize(output,protocol)
            self.assertEqual(json.loads((output/'COMPLETE.json').read_text())['records'],8)
            self.assertTrue((output/'report.html').is_file())
            path=record_path(output,'val-case',200,'measured_v1');row=json.loads(path.read_text());row['native_prediction_exact']=False;path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):summarize(output,protocol)


if __name__=='__main__':unittest.main()
