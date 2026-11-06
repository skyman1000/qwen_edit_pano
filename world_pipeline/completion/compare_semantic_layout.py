"""Replay frozen numerical-layout requests using another local Qwen3-VL checkpoint.

This is a capability comparison, not a claimed layout fix. The baseline's RGB,
G_obs, semantic plan, prompts and sampling settings are held fixed. GPU inference
and downloading weights must be initiated manually. No existing runner is changed.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from PIL import Image

from ..common import new_output, read, sha, write
from .audit_semantic_layout import audit, report
from .qwen_upstream import Qwen
from .semantic_layout import run


def frozen_cases(baseline, only_ids=None):
    config = read(baseline/'run_config.json')
    cases = deepcopy(config['cases'])
    if only_ids:
        if set(only_ids)-{c['id'] for c in cases}:
            raise ValueError('Unknown baseline case ID')
        cases = [c for c in cases if c['id'] in only_ids]
    if not cases:
        raise ValueError('No baseline cases')
    for case in cases:
        folder = baseline/case['id']
        request = read(folder/'layout.request.json')
        expected = dict(prompt=case['prompt'], seed=config['seed'], sample=True,
            max_tokens=config['max_tokens'], profile='official', response_format='json',
            generation=dict(do_sample=True, temperature=.7, top_p=.8, top_k=20, repetition_penalty=1.),
            presence_penalty=1.5, guard_repetition=dict(repeated_complete_objects=3, check_every=32))
        if request != expected:
            raise ValueError('Baseline prompt or sampling differs from this comparison protocol')
        if read(folder/'G_obs.json') != case['observed'] or read(folder/'semantic_plan.json') != case['plan']:
            raise ValueError('Baseline plan or observed state changed')
        for p, digest in case['source_sha256'].items():
            if sha(Path(p)) != digest:
                raise ValueError(f'Frozen source changed: {p}')
        with Image.open(case['image']) as handle:
            original = handle.convert('RGB')
        original.thumbnail((768,768))
        with Image.open(folder/'local.png') as handle:
            saved = handle.convert('RGB')
        if original.size != saved.size or original.tobytes() != saved.tobytes():
            raise ValueError('Baseline resized RGB differs from its original source')
        # Bind the already resized RGB actually seen by the baseline, not a new resize.
        case['image'] = str((folder/'local.png').resolve())
        for p in (baseline/'run_config.json', folder/'local.png', folder/'layout.request.json'):
            case['source_sha256'][str(p.resolve())] = sha(p)
    return config, cases


def checkpoint_metadata(model):
    config = read(model/'config.json')
    if config.get('model_type') != 'qwen3_vl':
        raise ValueError('This comparison requires a Qwen3-VL checkpoint')
    index = model/'model.safetensors.index.json'
    names = sorted(set(read(index)['weight_map'].values())) if index.exists() else ['model.safetensors']
    if any(not (model/p).is_file() for p in names):
        raise ValueError('Checkpoint shards missing; download manually before running')
    if not (model/'preprocessor_config.json').is_file():
        raise ValueError('Missing image processor configuration')
    return dict(config_sha256=sha(model/'config.json'), shards=len(names),
                weight_bytes=sum((model/p).stat().st_size for p in names))


class AutoQwen(Qwen):
    """Same inference method and precision; Accelerate handles GPU/CPU placement."""
    def __init__(self, model, gpu_memory, cpu_memory):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        if not torch.cuda.is_available():
            raise ValueError('Run inference manually inside a GPU allocation')
        self.processor = AutoProcessor.from_pretrained(str(model), local_files_only=True)
        memory = {i: gpu_memory for i in range(torch.cuda.device_count())}
        memory['cpu'] = cpu_memory
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model), local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map='auto', max_memory=memory).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only-id', action='append')
    parser.add_argument('--gpu-memory', default='40GiB', help='Weight allocation per visible GPU; leave activation headroom')
    parser.add_argument('--cpu-memory', default='80GiB')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    baseline = args.baseline.resolve(); model_path = args.model.resolve()
    config, cases = frozen_cases(baseline, args.only_id)
    metadata = checkpoint_metadata(model_path)
    comparison = dict(protocol=config['protocol'], baseline=str(baseline),
        baseline_config_sha256=sha(baseline/'run_config.json'), cases=cases, model=str(model_path),
        model_metadata=metadata, seed=config['seed'], max_tokens=config['max_tokens'],
        device_map='auto', gpu_memory=args.gpu_memory, cpu_memory=args.cpu_memory,
        code_sha256={p.name:sha(p) for p in (Path(__file__), Path(__file__).with_name('qwen_upstream.py'),
                                           Path(__file__).with_name('semantic_layout.py'))})
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed', cases=len(cases), checkpoint=metadata,
            model_loaded=False, prompts_unchanged=True, observed_state_unchanged=True)))
        return
    output = new_output(args.output)
    write(output/'run_config.json', comparison)
    model = AutoQwen(model_path, args.gpu_memory, args.cpu_memory)
    write(output/'device_map.json', {k:str(v) for k,v in model.model.hf_device_map.items()})
    passed = run(model, cases, output, config['seed'], config['max_tokens'])
    result = audit(output)
    write(output/'audit.json', result)
    (output/'audit.md').write_text(report(result))
    if not passed:
        raise SystemExit('Some cases failed; raw outputs retained')


if __name__ == '__main__':
    main()
