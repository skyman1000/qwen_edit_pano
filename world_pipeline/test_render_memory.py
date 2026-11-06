"""CPU-only checks of placement decisions and preservation of VAE precision."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from .render_memory import configure_pipeline_memory,GIB


class Pipeline:
    def __init__(self):
        self.vae_dtype='float32';self.transformer_dtype='bfloat16';self.device='cpu';self.offloaded=False
    def to(self,device):
        self.device=device;return self
    def enable_model_cpu_offload(self):self.offloaded=True


class MemoryTests(unittest.TestCase):
    def check(self,free,total):
        pipe=Pipeline();fake=SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda device:(free*GIB,total*GIB)))
        with tempfile.TemporaryDirectory() as d,mock.patch.dict('sys.modules',{'torch':fake}),contextlib.redirect_stdout(io.StringIO()):
            report=configure_pipeline_memory(pipe,d)
            self.assertEqual(report,json.loads((Path(d)/'MEMORY_RUNTIME.json').read_text()))
        self.assertEqual(pipe.vae_dtype,'float32');self.assertEqual(pipe.transformer_dtype,'bfloat16')
        return pipe,report

    def test_H200_with_headroom_releases_CPU_weights_to_GPU(self):
        pipe,r=self.check(130,141)
        self.assertEqual(r['mode'],'cuda_resident');self.assertEqual(pipe.device,'cuda:0');self.assertFalse(pipe.offloaded)

    def test_48G_keeps_existing_offload(self):
        pipe,r=self.check(43,45)
        self.assertEqual(r['mode'],'model_cpu_offload');self.assertTrue(pipe.offloaded);self.assertEqual(pipe.device,'cpu')

    def test_large_device_with_low_free_memory_is_not_assumed_available(self):
        pipe,r=self.check(70,141)
        self.assertEqual(r['mode'],'model_cpu_offload');self.assertTrue(pipe.offloaded)


if __name__=='__main__':unittest.main()
