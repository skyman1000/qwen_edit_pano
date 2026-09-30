"""Runtime assertions used only by the explicit GPU smoke test."""
import torch
from safetensors.torch import load_file
from peft import get_peft_model_state_dict


def parameter_report(transformer, vae):
    trainable = [(n, p) for n, p in transformer.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n, p in trainable):
        raise AssertionError('Only nonempty LoRA parameters may be trainable')
    if any(p.requires_grad for p in vae.parameters()):
        raise AssertionError('VAE must be frozen')
    return dict(trainable=sum(p.numel() for n, p in trainable),
                frozen_transformer=sum(p.numel() for p in transformer.parameters() if not p.requires_grad),
                frozen_vae=sum(p.numel() for p in vae.parameters()),
                trainable_names=[n for n, p in trainable],
                base_frozen=True, vae_frozen=True)


def gradient_report(transformer, vae):
    missing, nonzero = [], 0
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                missing.append(name)
            else:
                if not torch.isfinite(param.grad).all():
                    raise AssertionError(f'Nonfinite gradient: {name}')
                nonzero += int(torch.count_nonzero(param.grad) > 0)
        elif param.grad is not None:
            raise AssertionError(f'Frozen base gradient: {name}')
    if missing or not nonzero or any(p.grad is not None for p in vae.parameters()):
        raise AssertionError(f'Gradient check failed: missing={missing}, nonzero={nonzero}')
    # Gaussian LoRA initialization gives zero A-gradients initially; B must be nonzero.
    return dict(all_lora_gradients_present=True, nonzero_gradient_tensors=nonzero,
                frozen_gradients_absent=True)


def update_probe(params):
    for param in params:
        if param.grad is not None and torch.count_nonzero(param.grad):
            return param, param.detach().cpu().clone()
    raise AssertionError('No LoRA update candidate')


def checkpoint_roundtrip(accelerator, transformer, checkpoint):
    model = accelerator.unwrap_model(transformer)
    state = load_file(str(checkpoint / 'adapter_resume.safetensors'))
    # Corrupt one adapter tensor in memory, then verify accelerator restores it.
    with torch.no_grad():
        next(p for p in model.parameters() if p.requires_grad).add_(1)
    accelerator.load_state(str(checkpoint))
    restored = get_peft_model_state_dict(model)
    if set(state) != set(restored) or any(not torch.equal(value, restored[key].cpu()) for key, value in state.items()):
        raise AssertionError('Checkpoint adapter roundtrip differs')
