"""CPU Gloo parity on the real tiny checkpointed Edit path; no GPU/models downloads."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .. import oracle_test
from ..oracle_memory import ActivationStorage, training_autocast
from ..scaled_world.runtime import EpochSampler
from .parallel import micro_indices, seed_microbatch, average_gradients, check_replicas
from .restart import latest
from .runtime import WorldBranch, object_region_loss


def fixture():
    model = oracle_test.model(True)
    torch.manual_seed(29)
    world = WorldBranch(32, (0, 2), 16, binding_block=2)
    for branch in world.branches.values():
        branch.gate.data.fill_(.1)
    world.attach(model)
    return model, world, torch.optim.AdamW(world.parameters(), lr=1e-4, eps=1e-6, weight_decay=1e-5)


def backward(model, world, index, denominator):
    from diffusers import QwenImageEditPlusPipeline as Pipeline
    from qwen_edit_pano.losses import panorama_loss
    from qwen_pano.profiles import REPO_PANORAMA
    kwargs = oracle_test.inputs()
    seed_microbatch(11, index)
    kwargs['hidden_states'] = kwargs['hidden_states'] + .1 * torch.randn_like(kwargs['hidden_states'])
    graph = oracle_test.tensorize(oracle_test.graph(), 3)
    storage = ActivationStorage(enabled=True, device_type='cpu')
    with storage.context():
        with training_autocast('cpu'):
            world.set_world(graph)
            out = model(**kwargs)[0][:, :32]
            pred = Pipeline._unpack_latents(out, 64, 128, 8)
            target = torch.randn_like(pred)
            noise = torch.randn_like(pred)
            args = SimpleNamespace(**REPO_PANORAMA)
            global_loss, _ = panorama_loss(pred, target, torch.ones(1, 1, 8, 16, dtype=torch.bool),
                torch.ones(1, dtype=torch.bool), torch.ones(1), args.geometry_start_epoch, args,
                noise=noise, clean=noise-target)
            loss = global_loss + .25 * object_region_loss(pred, target, torch.ones(8, 16))
        world.clear_context()
        (loss / denominator).backward()
    assert all(p.grad is None for p in model.parameters())


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=120))
    model, world, optimizer = fixture()
    reference = torch.load(Path(directory) / 'reference.pt', weights_only=True)
    max_grad_error = 0.
    max_parameter_error = 0.
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        for index in micro_indices(step, 4, rank, 2):
            backward(model, world, index, 2)
        average_gradients(world)
        for name, parameter in world.named_parameters():
            expected = reference[step]['gradients'][name]
            torch.testing.assert_close(parameter.grad, expected, rtol=1e-4, atol=3e-7)
            max_grad_error = max(max_grad_error, float((parameter.grad-expected).abs().max()))
        torch.nn.utils.clip_grad_norm_(world.parameters(), 1.)
        optimizer.step()
        check_replicas(world)
        for name, value in world.state_dict().items():
            torch.testing.assert_close(value, reference[step]['state'][name], rtol=1e-5, atol=3e-6)
            max_parameter_error = max(max_parameter_error, float((value-reference[step]['state'][name]).abs().max()))
        if step == 1:
            if rank == 0:
                torch.save(dict(world=world.state_dict(), optimizer=optimizer.state_dict()), Path(directory) / 'resume.pt')
            dist.barrier()
            restored = torch.load(Path(directory) / 'resume.pt', weights_only=True)
            # Rebuild the frozen base/hooks and both optimizer states as on restart.
            model, world, optimizer = fixture()
            world.load_state_dict(restored['world'], strict=True)
            optimizer.load_state_dict(restored['optimizer'])
    assert all(int(state['step']) == 3 for state in optimizer.state.values())
    (Path(directory) / f'rank{rank}.json').write_text(json.dumps(dict(
        max_gradient_abs_error=max_grad_error, max_parameter_abs_error=max_parameter_error,
        Adam_step=3, replicas_identical=True, resumed_after_step=2)))
    dist.destroy_process_group()


class ParallelTests(unittest.TestCase):
    def test_disjoint_global_batch_and_epoch_boundary(self):
        sampler = EpochSampler(7, 0)
        for step in (0, 1, 1000, 9999):
            serial = list(micro_indices(step, 4))
            two = list(micro_indices(step, 4, 0, 2)) + list(micro_indices(step, 4, 1, 2))
            self.assertEqual(serial, two)
            self.assertEqual([sampler.index(i) for i in serial], [sampler.index(i) for i in two])
        with self.assertRaises(ValueError):
            micro_indices(0, 3, 0, 2)

    def test_ignore_incomplete_or_mismatched_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(group='A', binding_weight=0.)
            def make(path, step, committed=True):
                path.mkdir(parents=True)
                for name in ('world.safetensors', 'optimizer.pt'):
                    (path / name).write_bytes(b'fixture')
                (path / 'config.json').write_text(json.dumps(config))
                if committed:
                    (path / 'COMPLETE.json').write_text(json.dumps(dict(step=step)))
            initial = root / 'initial/checkpoint-step001000'
            make(initial, 1000)
            first = root / 'run-1/checkpoint-step001123'
            make(first, 1123)
            make(root / 'run-2/checkpoint-step001500', 1500, committed=False)
            self.assertEqual(latest(root, initial), (first, 1123))
            (first / 'config.json').write_text(json.dumps(dict(group='B', binding_weight=.01)))
            with self.assertRaises(ValueError):
                latest(root, initial)

    def test_two_process_gradients_and_adam_resume_match_serial(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='world-parallel-') as directory:
            model, world, optimizer = fixture()
            reference = []
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                for index in micro_indices(step, 4):
                    backward(model, world, index, 4)
                grads = {name: p.grad.clone() for name, p in world.named_parameters()}
                torch.nn.utils.clip_grad_norm_(world.parameters(), 1.)
                optimizer.step()
                reference.append(dict(gradients=grads, state=copy.deepcopy(world.state_dict())))
            torch.save(reference, Path(directory) / 'reference.pt')
            mp.spawn(worker, args=('file://' + str(Path(directory) / 'rendezvous'), directory), nprocs=2, join=True)
            reports = [json.loads((Path(directory) / f'rank{rank}.json').read_text()) for rank in (0, 1)]
            print(json.dumps(dict(CPU_Gloo_parity=reports, GPU=False)), flush=True)


if __name__ == '__main__':
    unittest.main()
