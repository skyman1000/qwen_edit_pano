"""Evaluation-only placement policy; preserve weights and inference precision."""
import json
import os
from pathlib import Path
import resource

GIB = 1024**3
# Plenty of room for the BF16 transformer/text encoder, FP32 VAE and activations.
# 48/80GB devices retain the previously used model CPU offload strategy.
RESIDENT_MINIMUM_FREE_BYTES = 96*GIB


def configure_pipeline_memory(pipe, output):
    import torch
    free,total = torch.cuda.mem_get_info(0)
    if free >= RESIDENT_MINIMUM_FREE_BYTES:
        # No dtype argument: in particular, keep the existing VAE in FP32.
        pipe.to('cuda:0')
        mode = 'cuda_resident'
    else:
        pipe.enable_model_cpu_offload()
        mode = 'model_cpu_offload'
    report = dict(mode=mode,free_gpu_gib_before_placement=free/GIB,total_gpu_gib=total/GIB,
                  resident_minimum_free_gib=RESIDENT_MINIMUM_FREE_BYTES/GIB,
                  slurm_mem_per_node=os.environ.get('SLURM_MEM_PER_NODE'),
                  low_cpu_mem_usage=True,precision_changed=False)
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    (directory/'MEMORY_RUNTIME.json').write_text(json.dumps(report,indent=2)+'\n')
    print('RENDER_MEMORY '+json.dumps(report),flush=True)
    return report


def peak_process_cpu_gib():
    """Linux process high-water RSS, not total Slurm cgroup usage/file cache."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2)
