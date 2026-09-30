"""Optional CPU test of real Diffusers Edit modulation + padding, no model download.

This reduced random transformer is an interface test, not the full-model GPU validation.
Run: python -m qwen_edit_pano.test_interface
"""
import copy
import torch
from diffusers import QwenImageTransformer2DModel
from .circular import install_circular_padding, pad_columns, crop_columns
from .common import require_versions


def main():
    require_versions()
    torch.manual_seed(0)
    base = QwenImageTransformer2DModel(patch_size=2, in_channels=16, out_channels=4,
        num_layers=1, attention_head_dim=16, num_attention_heads=2,
        joint_attention_dim=32, axes_dims_rope=(4, 6, 6), zero_cond_t=True)
    temporary, persistent = copy.deepcopy(base), copy.deepcopy(base)
    install_circular_padding(temporary, 1)
    install_circular_padding(persistent, 1, mode='persistent')
    h, w = 2, 4
    x = torch.randn(2, h*w, 16)
    kwargs = dict(timestep=torch.tensor([.3, .7]), encoder_hidden_states=torch.randn(2, 5, 32),
                  encoder_hidden_states_mask=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool),
                  img_shapes=[[(1,h,w)]]*2, return_dict=False)
    out = temporary(hidden_states=x, **kwargs)[0]
    padded = persistent(hidden_states=pad_columns(x,h,w,1), **kwargs)[0]
    torch.testing.assert_close(out, crop_columns(padded,h,w,1))
    assert out.shape == x.shape and torch.isfinite(out).all()
    # The no-reference zero_cond_t path is equivalent to selecting only t
    # modulation; compare with the original padding on an otherwise equal T2I model.
    from qwen_pano.circular import install_circular_padding as old_padding
    reference = copy.deepcopy(base)
    reference.zero_cond_t = False
    for block in reference.transformer_blocks:
        block.zero_cond_t = False
    old_padding(reference, 1)
    expected = reference(hidden_states=x, **kwargs)[0]
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)
    temporary.enable_gradient_checkpointing()
    out = temporary(hidden_states=x, **kwargs)[0]
    out.square().mean().backward()
    assert temporary.img_in.weight.grad is not None
    assert torch.isfinite(temporary.img_in.weight.grad).all()
    print('PASS: native zero_cond_t, batch=2, masked text, temporary/persistent padding, checkpointed backward')


if __name__ == '__main__':
    main()
