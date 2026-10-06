"""CPU tests on a tiny real Edit transformer. No downloaded model weights or GPU."""
import copy
import tempfile
import unittest
from pathlib import Path
import torch
from diffusers import QwenImageTransformer2DModel
from safetensors.torch import save_file,load_file
from .common import field,SCHEMA,FRAME
from .oracle_model import tensorize,WorldBranch,TargetAttention
from .oracle_memory import training_autocast,ActivationStorage
from qwen_edit_pano.paired_circular import install_circular_padding
from qwen_edit_pano.circular import pad_columns,crop_columns


def graph():
    def obj(i,center,visible):
        return dict(track_id=str(i),category_id=field(3+i),center_local_m=field(center),
            size_aabb_local_m=field([1.,2.,3.]),bbox2d_visible_xyxy_norm=field(None),
            in_primary_room=field(None),local_evidence='visible' if visible else 'unobserved')
    return dict(schema_version=SCHEMA,frame=FRAME,scope='full',objects=[obj(0,[0.,0.,0.],True),obj(1,[1.,2.,3.],False)])


def model(checkpointed=True,mode='temporary'):
    torch.manual_seed(4)
    t=QwenImageTransformer2DModel(patch_size=2,in_channels=16,out_channels=4,num_layers=3,
        attention_head_dim=16,num_attention_heads=2,joint_attention_dim=32,axes_dims_rope=(4,6,6),zero_cond_t=True)
    from peft import LoraConfig
    t.add_adapter(LoraConfig(r=2,lora_alpha=2,lora_dropout=.05,
        target_modules=['attn.to_q','attn.to_k','attn.to_v','attn.to_out.0']))
    with torch.no_grad():
        for name,param in t.named_parameters():
            if 'lora_B' in name:param.normal_(std=.02)
    t.requires_grad_(False).eval()
    install_circular_padding(t,1,mode)
    if checkpointed:t.enable_gradient_checkpointing()
    return t


def inputs(mode='temporary'):
    torch.manual_seed(17)
    x=torch.randn(1,32,16);r=torch.randn(1,4,16)
    if mode=='persistent':x=pad_columns(x,4,8,1)
    return dict(hidden_states=torch.cat([x,r],1),timestep=torch.tensor([.5]),
        encoder_hidden_states=torch.randn(1,5,32),encoder_hidden_states_mask=torch.ones(1,5,dtype=torch.bool),
        img_shapes=[[(1,4,8),(1,2,2)]],return_dict=False)


class OracleTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_bf16_smoke_then_checkpoint_backward_and_offload(self):
        results=[]
        for checkpointed,offload in ((False,False),(True,False),(True,True)):
            m=model(checkpointed).to(torch.bfloat16)
            torch.manual_seed(23)
            w=WorldBranch(32,(0,2),16)
            for branch in w.branches.values():branch.gate.data.fill_(.1)
            w.attach(m)
            kw=inputs()
            for key in ('hidden_states','encoder_hidden_states'):kw[key]=kw[key].bfloat16()
            storage=ActivationStorage(enabled=offload,device_type='cpu')
            with storage.context():
                with training_autocast('cpu'):
                    # Deliberately stress the old shared no-grad/autograd scope.
                    with torch.no_grad():
                        w.set_world(None);m(**kw)
                        w.set_world(tensorize(graph(),3));m(**kw)
                    w.set_world(tensorize(graph(),3));y=m(**kw)[0]
                w.set_world(None)
                y.float().square().mean().backward()
            self.assertTrue(all(p.grad is None for p in m.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in w.encoder.parameters()))
            if offload:self.assertGreater(storage.count,0)
            results.append((y.detach(),{k:p.grad.clone() for k,p in w.named_parameters() if p.grad is not None}))
        for actual,gradients in results[1:]:
            torch.testing.assert_close(actual,results[0][0],rtol=0,atol=0)
            self.assertEqual(gradients.keys(),results[0][1].keys())
            for key in gradients:torch.testing.assert_close(gradients[key],results[0][1][key],rtol=0,atol=0)

    def test_activation_storage_preserves_weights_and_dtype(self):
        storage=ActivationStorage(device_type='cpu')
        frozen=torch.nn.Parameter(torch.randn(3,4),requires_grad=False)
        trainable=torch.nn.Parameter(torch.randn(3,4))
        self.assertIs(storage.pack(frozen),frozen)
        self.assertIs(storage.pack(trainable),trainable)
        view=frozen.t();self.assertIs(storage.pack(view),view)
        activation=(trainable*2).bfloat16()
        saved=storage.pack(activation)
        self.assertEqual(saved[1].dtype,activation.dtype)
        self.assertFalse(saved[1].requires_grad)
        torch.testing.assert_close(storage.unpack(saved),activation,rtol=0,atol=0)

    def test_missing_zero_constant_and_limit(self):
        g=graph();a=tensorize(g,3)
        self.assertEqual(float(a[1][0,1,6]),1.)  # true zero center is known
        g['objects'][0]['center_local_m']=field(None)
        b=tensorize(g,3)
        self.assertEqual(float(b[1][0,1,6]),0.)
        self.assertEqual(int(tensorize(g,3,'observed')[2].sum()),2)
        self.assertEqual(int(tensorize(g,3,'constant')[2].sum()),1)
        with self.assertRaises(ValueError):tensorize(g,1)
        g['objects']=[]
        self.assertEqual(int(tensorize(g,3)[2].sum()),1)

    def test_target_only_and_reference_exact(self):
        torch.manual_seed(0)
        b=TargetAttention(32,16);b.gate.data.fill_(.2)
        h=torch.randn(1,44,32);tokens=torch.randn(1,4,16);mask=torch.tensor([[1,1,0,0]],dtype=torch.bool)
        y=b(h,tokens,mask,40)
        self.assertTrue(torch.equal(y[:,40:],h[:,40:]))
        self.assertFalse(torch.equal(y[:,:40],h[:,:40]))
        tokens[:,2:]+=100
        torch.testing.assert_close(y,b(h,tokens,mask,40))

    def test_zero_gate_baseline_and_persistent(self):
        temp=model();kwargs=inputs()
        original=temp(**kwargs)[0]
        w=WorldBranch(32,(0,2),16);w.attach(temp);w.set_world(tensorize(graph(),3))
        torch.testing.assert_close(original,temp(**kwargs)[0],rtol=0,atol=0)
        for b in w.branches.values():b.gate.data.fill_(.1)
        changed=temp(**kwargs)[0]
        persistent=model(mode='persistent');v=WorldBranch(32,(0,2),16);v.load_state_dict(w.state_dict())
        v.attach(persistent);v.set_world(tensorize(graph(),3))
        padded=persistent(**inputs('persistent'))[0]
        restored=torch.cat([crop_columns(padded[:,:40],4,8,1),padded[:,40:]],1)
        torch.testing.assert_close(changed,restored)
        self.assertFalse(torch.allclose(changed[:,:32],original[:,:32]))

    def test_checkpoint_backward_explicit_world_and_freeze(self):
        models=[model(False),model(True)]
        torch.manual_seed(8)
        first=WorldBranch(32,(0,2),16)
        for b in first.branches.values():b.gate.data.fill_(.1)
        second=copy.deepcopy(first)
        gradients=[]
        for m,w in zip(models,[first,second]):
            w.attach(m);w.set_world(tensorize(graph(),3))
            y=m(**inputs())[0]
            # Mutate runtime context BEFORE recomputation; captured tokens must win.
            w.set_world(tensorize(graph(),3,'constant'))
            w.set_world(None)
            y.square().mean().backward()
            self.assertTrue(all(p.grad is None for p in m.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in w.encoder.parameters()))
            gradients.append({k:p.grad.clone() for k,p in w.named_parameters() if p.grad is not None})
        self.assertEqual(gradients[0].keys(),gradients[1].keys())
        for k in gradients[0]:torch.testing.assert_close(gradients[0][k],gradients[1][k])

    def test_gate_opens_encoder_then_roundtrip_and_permutation(self):
        m=model();w=WorldBranch(32,(0,2),16);w.attach(m)
        opt=torch.optim.AdamW(w.parameters(),lr=.01)
        for step in range(2):
            opt.zero_grad();w.set_world(tensorize(graph(),3));m(**inputs())[0].square().mean().backward()
            if step==0:
                self.assertTrue(any(b.gate.grad.abs()>0 for b in w.branches.values()))
            else:
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in w.encoder.parameters()))
            opt.step()
        w.set_world(tensorize(graph(),3));expected=m(**inputs())[0]
        g=graph();g['objects'].reverse()
        for o in g['objects']:o['track_id']='new_'+o['track_id']
        w.set_world(tensorize(g,3));torch.testing.assert_close(expected,m(**inputs())[0],atol=1e-6,rtol=1e-5)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'world.safetensors';save_file(w.state_dict(),str(path))
            m2=model();w2=WorldBranch(32,(0,2),16);w2.load_state_dict(load_file(str(path)));w2.attach(m2)
            w2.set_world(tensorize(graph(),3));torch.testing.assert_close(expected,m2(**inputs())[0])
        w.set_world(None)
        fresh=model();torch.testing.assert_close(m(**inputs())[0],fresh(**inputs())[0],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
