"""Read-only audit; no torch imports, model loading, or GPU execution."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from .common import ROOT
from .profiles import REPO_PANORAMA


def main():
    root = ROOT / 'qwen_edit_pano'
    for path in root.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    for path in (root/'scripts').glob('*.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    baseline = json.loads((root/'audit/baseline_training_config.json').read_text())
    for key, value in REPO_PANORAMA.items():
        assert baseline[key] == value, (key, baseline[key], value)
    for path, digest in json.loads((root/'audit/baseline_sources.json').read_text()).items():
        assert hashlib.sha256((ROOT/'qwen_pano'/path).read_bytes()).hexdigest() == digest, path
    assert hashlib.sha256(Path(baseline['panorama_manifest']).read_bytes()).hexdigest() == baseline['panorama_sha256']
    completed = json.loads((root/'audit/baseline_COMPLETE.json').read_text())
    assert completed == dict(completed_epochs=25, optimizer_step=64750)
    print('PASS: Python syntax, shell syntax, original sources unchanged, official recipe and manifest identity')
    print('Full model load / forward / gradients / checkpoint / inference: NOT RUN')


if __name__ == '__main__':
    main()
