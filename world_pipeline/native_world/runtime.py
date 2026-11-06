"""Decoupled layout attention using the frozen model's actual projections.

The side attention uses BOTH Q and K before RoPE: layout objects have no text
sequence positions. Native joint-attention RoPE/masks/sequence lengths are never
changed. Metric camera geometry is encoded in each object, without a GT mask,
projected support, hard routing, or an ERP attention bias in the forward path.
"""
import inspect
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..fourier_world.runtime import GeometryFeatures
from ..oracle_memory import training_autocast
from ..binding_refine_world.supervision import binding_terms


class LowRank(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x.float()))


class Encoder(nn.Module):
    def __init__(self, semantic_base, rank=32, geometry_width=256):
        super().__init__()
        if semantic_base.ndim != 2 or semantic_base.shape[0] != 42 or not torch.isfinite(semantic_base).all():
            raise ValueError('Expected 42 finite native-projected category vectors')
        width = semantic_base.shape[1]
        self.register_buffer('semantic_base', semantic_base.detach().float().clone())
        self.semantic = LowRank(width, rank)
        self.geometry = nn.Sequential(GeometryFeatures(), nn.Linear(392, geometry_width),
                                      nn.SiLU(), nn.Linear(geometry_width, width))
        # At initialization, actual objects start at their frozen name features.
        nn.init.zeros_(self.geometry[-1].weight)
        nn.init.zeros_(self.geometry[-1].bias)
        self.empty = nn.Parameter(semantic_base[39:40].detach().float()[None].clone())

    def forward(self, category, geometry, mask):
        if category.shape != mask.shape or geometry.shape != (*category.shape, 8):
            raise ValueError('Invalid category/geometry/mask shapes')
        semantic = self.semantic_base[category]
        tokens = semantic + self.semantic(semantic) + self.geometry(geometry)
        return torch.cat((self.empty.expand(tokens.shape[0], -1, -1), tokens[:, 1:]), 1), mask


class LayoutAttention(nn.Module):
    def __init__(self, width, rank=16):
        super().__init__()
        # Zero deltas retain the exact pretrained K/V maps at initialization.
        self.key_delta = LowRank(width, rank)
        self.value_delta = LowRank(width, rank)
        self.gate = nn.Parameter(torch.zeros(()))
        self.diagnostics = {}

    def forward(self, attn, image, objects, mask, boundary, labels=None):
        if not 0 < boundary < image.shape[1] or image.shape[0] != objects.shape[0]:
            raise ValueError('Invalid target/reference boundary or batch')
        # image is the ACTUAL native img_norm1 + timestep-modulated input.
        q = attn.to_q(image[:, :boundary])
        k = attn.add_k_proj(objects) + self.key_delta(objects)
        v = attn.add_v_proj(objects) + self.value_delta(objects)
        def heads(x):
            return x.unflatten(-1, (attn.heads, -1))
        q, k, v = map(heads, (q, k, v))
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_added_k is not None:
            k = attn.norm_added_k(k)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :], dropout_p=0.)
        value = value.transpose(1, 2).flatten(2)
        # Reuse the actual native output map INCLUDING the frozen Panorama LoRA.
        # Its bias already occurs in the original attention; don't add it twice.
        projected = attn.to_out[0](value)
        if attn.to_out[0].bias is not None:
            projected = projected - attn.to_out[0].bias.to(projected.dtype)
        for layer in attn.to_out[1:]:
            projected = layer(projected)
        delta = torch.tanh(self.gate) * projected.float()
        aux = torch.stack(binding_terms(q, k, mask, labels)) if labels is not None else delta.new_zeros(2)
        with torch.no_grad():
            self.diagnostics = dict(gate=float(torch.tanh(self.gate)),
                projected_rms=float(projected.float().square().mean().sqrt()),
                gated_attention_delta_rms=float(delta.square().mean().sqrt()))
        return delta, aux


class Processor:
    """Leave the base pass untouched; add layout output only to target queries."""
    def __init__(self, original, branch):
        self.original = original
        self.branch = branch  # Plain Python object, NOT registered in the base.

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 encoder_hidden_states_mask=None, attention_mask=None,
                 image_rotary_emb=None, native_layout_context=None, native_layout_losses=None):
        image, text = self.original(attn, hidden_states, encoder_hidden_states,
            encoder_hidden_states_mask, attention_mask, image_rotary_emb)
        if native_layout_context is None:
            return image, text
        objects, mask, boundary, labels = native_layout_context
        delta, aux = self.branch(attn, hidden_states, objects, mask, boundary, labels)
        native_layout_losses.append(aux)
        changed = (image[:, :boundary].float() + delta).to(image.dtype)
        # Reference and text receive exactly their original direct outputs.
        return torch.cat((changed, image[:, boundary:]), 1), text


class WorldBranch(nn.Module):
    def __init__(self, semantic_base, blocks=(20, 40, 59), binding_block=40,
                 rank=16, connector_rank=32, geometry_width=256):
        super().__init__()
        if binding_block not in blocks or len(set(blocks)) != len(blocks):
            raise ValueError('Invalid injection/binding blocks')
        self.encoder = Encoder(semantic_base, connector_rank, geometry_width)
        self.branches = nn.ModuleDict({str(i): LayoutAttention(semantic_base.shape[1], rank) for i in blocks})
        self.binding_block = binding_block
        self.current = None
        self.boundary = None
        self.supervision = None
        self._losses = []

    @classmethod
    def from_model(cls, model, features, **kwargs):
        if features.shape != (42, model.config.joint_attention_dim):
            raise ValueError('Class feature dimension differs from the native text input')
        device = next(model.parameters()).device
        with torch.no_grad(), training_autocast(device.type):
            # Existing raw Qwen name vectors, NOT unit-normalized 256d vectors.
            # This is a lexical anchor through the real frozen input projection;
            # it is NOT a claim that pooled names equal full contextual prompts.
            semantic = model.txt_in(model.txt_norm(features.to(device, dtype=model.txt_in.weight.dtype))).float()
        return cls(semantic, **kwargs).to(device)

    def set_world(self, tensors):
        if tensors is None:
            self.current = None
        else:
            with training_autocast(tensors[0].device.type):
                self.current = self.encoder(*tensors)

    def set_supervision(self, labels):
        self.supervision = labels

    def clear_context(self):
        self.current = None
        self.supervision = None
        self._losses = []

    def take_binding_loss(self):
        if len(self._losses) != 1:
            raise ValueError(f'Expected exactly one collected binding loss, got {len(self._losses)}')
        value = self._losses[0]
        self._losses = []
        return value

    def diagnostics(self):
        return {k: b.diagnostics for k, b in self.branches.items()}

    def attach(self, transformer):
        if getattr(transformer, '_oracle_world_attached', False):
            raise ValueError('A World branch is already attached')
        if any(p.requires_grad for p in transformer.parameters()):
            raise ValueError('Freeze the entire base and Panorama LoRA before attaching World')
        transformer._oracle_world_attached = True
        def before(module, args, kwargs):
            shapes = kwargs['img_shapes']
            if any(s != shapes[0] for s in shapes) or len(shapes[0]) < 2:
                raise ValueError('Expected target/reference shapes')
            self.boundary = math.prod(shapes[0][0])
            if kwargs['hidden_states'].shape[1] != sum(math.prod(s) for s in shapes[0]):
                raise ValueError('Target/reference boundary mismatch')
            self._losses = []
        # Installed AFTER existing circular pre-hook, sees the padded grid.
        transformer.register_forward_pre_hook(before, with_kwargs=True)
        selected = {}
        for key, branch in self.branches.items():
            block = transformer.transformer_blocks[int(key)]
            original = block.forward
            signature = inspect.signature(original)
            if type(block.attn.processor).__name__ != 'QwenDoubleStreamAttnProcessor2_0':
                raise ValueError('Unsupported native attention processor')
            block.attn.set_processor(Processor(block.attn.processor, branch))
            def context(_key=key):
                state = None if self.current is None else (*self.current, self.boundary)
                labels = self.supervision if int(_key) == self.binding_block and state is not None else None
                return state, labels
            def compute(args, kwargs, state, labels, _original=original, _block=block, _signature=signature):
                if state is None:
                    text, hidden = _original(*args, **kwargs)
                    return text, hidden, hidden.new_zeros(2, dtype=torch.float32)
                bound = _signature.bind(*args, **kwargs)
                tokens, mask, boundary = state
                temb = bound.arguments['temb']
                if _block.zero_cond_t:
                    temb = torch.chunk(temb, 2, dim=0)[0]
                # Match this block's frozen text normalization/time modulation.
                mod1 = _block.txt_mod(temb).chunk(2, dim=-1)[0]
                objects, _ = _block._modulate(_block.txt_norm1(tokens.to(temb.dtype)), mod1)
                losses = []  # Local to THIS invocation, also on checkpoint replay.
                joint = dict(bound.arguments.get('joint_attention_kwargs') or {})
                joint.update(native_layout_context=(objects, mask, boundary, labels), native_layout_losses=losses)
                bound.arguments['joint_attention_kwargs'] = joint
                text, hidden = _original(*bound.args, **bound.kwargs)
                if len(losses) != 1:
                    raise ValueError('Native attention did not consume exactly one layout context')
                return text, hidden, losses[0]
            def wrapped(*args, _compute=compute, _context=context, **kwargs):
                state, labels = _context()
                text, hidden, aux = _compute(args, kwargs, state, labels)
                if labels is not None:
                    self._losses.append(aux)
                return text, hidden
            block.forward = wrapped
            selected[block] = (compute, context)
        native_checkpoint = getattr(transformer, '_gradient_checkpointing_func', None)
        if native_checkpoint is not None:
            def with_world(function, *inputs, **kwargs):
                if function not in selected:
                    return native_checkpoint(function, *inputs, **kwargs)
                compute, context = selected[function]
                state, labels = context()
                # Capture immutable per-microbatch state, never mutable self.current.
                def run(*packed):
                    return compute(packed, kwargs, state, labels)
                text, hidden, aux = checkpoint(run, *inputs, use_reentrant=False)
                if labels is not None:
                    self._losses.append(aux)
                return text, hidden
            transformer._gradient_checkpointing_func = with_world
