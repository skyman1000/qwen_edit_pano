"""User-run CUDA initialization check; never rewrite Slurm device visibility."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


ENV_KEYS = ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_JOB_NODELIST',
            'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES',
            'CUDA_DEVICE_ORDER', 'NVIDIA_VISIBLE_DEVICES')


def require_cuda(stage='evaluation', output=None):
    import torch
    report = dict(stage=stage, hostname=socket.gethostname(), python=sys.executable,
                  torch_version=str(torch.__version__), torch_cuda_build=torch.version.cuda,
                  environment={k:os.environ[k] for k in ENV_KEYS if k in os.environ})
    try:
        # is_available() can suppress the driver's initialization exception.
        # Keep that original exception instead of blaming GPU allocation.
        if torch.version.cuda is None:
            raise RuntimeError('This PyTorch build has no CUDA support')
        torch.cuda.init()
        torch.cuda.set_device(0)
        probe = torch.empty(1, device='cuda:0')
        torch.cuda.synchronize(0)
        del probe
        report.update(ok=True, visible_device_count=torch.cuda.device_count(),
                      device0=torch.cuda.get_device_name(0))
    except Exception as exc:
        report.update(ok=False, error_type=type(exc).__name__, error=str(exc))
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=10)
            report['nvidia_smi'] = dict(returncode=result.returncode,
                stdout=result.stdout[-4000:], stderr=result.stderr[-4000:])
        except (OSError, subprocess.TimeoutExpired) as smi:
            report['nvidia_smi'] = dict(error=str(smi))
        _save(report, output)
        raise RuntimeError(
            'CUDA initialization failed inside the allocated job. '
            'Check CUDA_STATUS.json / the CUDA_CHECK line for the original error. '
            'GPU allocation alone does not establish driver/device accessibility. '
            'Do not manually override CUDA_VISIBLE_DEVICES to bypass Slurm. '
            f'Original error: {type(exc).__name__}: {exc}'
        ) from exc
    _save(report, output)
    return report


def _save(report, output):
    print('CUDA_CHECK ' + json.dumps(report, ensure_ascii=False), flush=True)
    if output is not None:
        directory = Path(output); directory.mkdir(parents=True, exist_ok=True)
        (directory/'CUDA_STATUS.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    require_cuda('manual_cuda_check')
