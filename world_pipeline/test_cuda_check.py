"""CPU-only tests: preserve CUDA's real failure and Slurm's visibility mask."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from .cuda_check import require_cuda


def fake_torch(cuda_version='12.4'):
    return SimpleNamespace(__version__='test',version=SimpleNamespace(cuda=cuda_version),
        cuda=SimpleNamespace(init=mock.Mock(),set_device=mock.Mock(),synchronize=mock.Mock(),
            device_count=mock.Mock(return_value=1),get_device_name=mock.Mock(return_value='allocated GPU')),
        empty=mock.Mock(return_value=object()))


class CudaCheckTests(unittest.TestCase):
    def test_success_preserves_assigned_visibility(self):
        import os
        fake=fake_torch()
        with tempfile.TemporaryDirectory() as d,mock.patch.dict('sys.modules',{'torch':fake}),\
                mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-assigned-uuid'}),\
                mock.patch('subprocess.run') as smi,contextlib.redirect_stdout(io.StringIO()):
            result=require_cuda('test',d)
            self.assertTrue(result['ok']);self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'GPU-assigned-uuid')
            self.assertEqual(result, json.loads((Path(d)/'CUDA_STATUS.json').read_text()))
            smi.assert_not_called()

    def test_driver_error_survives_even_when_nvidia_smi_sees_gpus(self):
        fake=fake_torch();original=RuntimeError('CUDA error 802: system not yet initialized')
        fake.cuda.init.side_effect=original
        smi=SimpleNamespace(returncode=0,stdout='GPU 0: L40',stderr='')
        with tempfile.TemporaryDirectory() as d,mock.patch.dict('sys.modules',{'torch':fake}),\
                mock.patch('subprocess.run',return_value=smi),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'CUDA error 802') as caught:require_cuda('test',d)
            self.assertIs(caught.exception.__cause__,original)
            report=json.loads((Path(d)/'CUDA_STATUS.json').read_text())
            self.assertFalse(report['ok']);self.assertEqual(report['error'],str(original))
            self.assertEqual(report['nvidia_smi']['returncode'],0)
            fake.empty.assert_not_called()

    def test_cpu_build_and_missing_nvidia_smi_are_explicit(self):
        fake=fake_torch(None)
        with mock.patch.dict('sys.modules',{'torch':fake}),\
                mock.patch('subprocess.run',side_effect=FileNotFoundError('nvidia-smi unavailable')),\
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'no CUDA support'):require_cuda('test')
            fake.cuda.init.assert_not_called()


if __name__=='__main__':unittest.main()
