"""CPU tests using a tiny actual Diffusers transformer, never loads model weights."""
import copy
import tempfile
from pathlib import Path
import torch
from diffusers import QwenImageTransformer2DModel, QwenImageEditPlusPipeline
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from .common import require_versions
from .paired_circular import install_circular_padding
from .circular import pad_columns, crop_columns


def main():
    require_versions()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    base = QwenImageTransformer2DModel(patch_size=2,in_channels=16,out_channels=4,
              num_layers=1,attention_head_dim=16,num_attention_heads=2,
              joint_attention_dim=32,axes_dims_rope=(4,6,6),zero_cond_t=True)
    h,w,rh,rw = 4,8,2,2
    x,ref = torch.randn(2,h*w,16), torch.randn(2,rh*rw,16)
    kw = dict(timestep=torch.tensor([.3,.7]),encoder_hidden_states=torch.randn(2,5,32),
              encoder_hidden_states_mask=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool),
              img_shapes=[[(1,h,w),(1,rh,rw)]]*2,return_dict=False)
    temporary,persistent = copy.deepcopy(base),copy.deepcopy(base)
    install_circular_padding(temporary,1)
    install_circular_padding(persistent,1,'persistent')
    seen = {}
    def capture_rope(m,args,out):
        seen['rope'] = out
    handle = temporary.pos_embed.register_forward_hook(capture_rope)
    out = temporary(hidden_states=torch.cat([x,ref],1),**kw)[0]
    handle.remove()
    expected_rope = base.pos_embed(kw['img_shapes'],max_txt_seq_len=5,device=x.device)
    torch.testing.assert_close(seen['rope'][0][h*(w+2):],expected_rope[0][h*w:])
    torch.testing.assert_close(seen['rope'][1],expected_rope[1])
    padded = persistent(hidden_states=torch.cat([pad_columns(x,h,w,1),ref],1),**kw)[0]
    restored = torch.cat([crop_columns(padded[:,:h*(w+2)],h,w,1),padded[:,h*(w+2):]],1)
    torch.testing.assert_close(out,restored)
    changed = temporary(hidden_states=torch.cat([x,ref+3],1),**kw)[0]
    assert not torch.allclose(out[:,:h*w],changed[:,:h*w]), 'Target ignores reference'
    assert out.shape == (2,h*w+rh*rw,16) and torch.isfinite(out).all()
    temporary.requires_grad_(False)
    temporary.add_adapter(LoraConfig(r=2,lora_alpha=2,lora_dropout=0.,init_lora_weights='gaussian',
                           target_modules=['attn.to_q','attn.to_k','attn.to_v','attn.to_out.0']))
    temporary.enable_gradient_checkpointing()
    params = [p for p in temporary.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params,lr=5e-5)
    target = torch.randn(2,h*w,16)
    loss = (temporary(hidden_states=torch.cat([x,ref],1),**kw)[0][:,:h*w]-target).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert any(torch.count_nonzero(p.grad) for p in params)
    assert all(p.grad is None for p in temporary.parameters() if not p.requires_grad)
    before = [p.detach().clone() for p in params]
    opt.step()
    assert any(not torch.equal(a,b) for a,b in zip(before,params))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'adapter.safetensors'
        state = {k:v.detach().clone().contiguous() for k,v in get_peft_model_state_dict(temporary).items()}
        save_file(state,str(path))
        with torch.no_grad():
            params[0].add_(1)
        result = set_peft_model_state_dict(temporary,load_file(str(path)))
        assert not result.unexpected_keys
        restored = get_peft_model_state_dict(temporary)
        assert all(torch.equal(v,restored[k]) for k,v in state.items())
    # The actual panorama loss, including the third-epoch geometry branch.
    from types import SimpleNamespace
    from qwen_pano.profiles import REPO_PANORAMA
    from .losses import panorama_loss
    args = SimpleNamespace(**REPO_PANORAMA)
    z,n = torch.randn(1,4,1,16,32),torch.randn(1,4,1,16,32)
    prediction = torch.randn_like(z,requires_grad=True)
    mask = torch.ones(1,1,16,32)
    for epoch in (0,2):
        torch.manual_seed(0)
        loss,log = panorama_loss(prediction,n-z,mask,torch.ones(1,dtype=torch.bool),torch.ones(1),epoch,args,noise=n,clean=z)
        assert torch.isfinite(loss) and {'cube','yaw_resampled'} <= log.keys()
        loss.backward()
    print('PASS: target/reference boundaries, reference and text RoPE unchanged, temporary/persistent agreement, reference influence, checkpointed backward, frozen base, LoRA update/reload, cube/yaw loss branches')


if __name__ == '__main__':
    main()
