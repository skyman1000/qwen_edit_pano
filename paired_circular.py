"""Target-only circular padding with unchanged native Edit reference RoPE."""
import math
import torch
from .circular import pad_columns, crop_columns


def shape_info(shapes):
    if not shapes or any(len(s) < 2 for s in shapes):
        raise ValueError('Paired Edit requires ERP plus reference img_shapes')
    first = shapes[0]
    if any(s != first for s in shapes):
        raise ValueError('Variable shapes in batch are unsupported')
    f, h, w = first[0]
    if f != 1 or w != 2*h:
        raise ValueError('Expected unpadded 2:1 ERP target')
    return h, w, sum(math.prod(s) for s in first[1:])


def install_circular_padding(transformer, columns, mode='temporary'):
    if mode not in ('temporary', 'persistent') or columns < 0:
        raise ValueError('Invalid padding mode/columns')
    if getattr(transformer, '_pano_padding_installed', False):
        raise ValueError('Padding already installed')
    if not transformer.zero_cond_t:
        raise ValueError('Expected Edit zero_cond_t model')
    if not columns:
        return

    def before(module, args, kwargs):
        if args:
            raise ValueError('Use transformer keyword arguments')
        h, w, reference_count = shape_info(kwargs['img_shapes'])
        if columns > w:
            raise ValueError('Too many padding columns')
        x = kwargs['hidden_states']
        target_count = h * (w + 2*columns) if mode == 'persistent' else h*w
        if x.shape[1] != target_count + reference_count:
            raise ValueError('Incorrect target/reference token boundary')
        kw = dict(kwargs)
        if mode == 'temporary':
            kw['hidden_states'] = torch.cat([pad_columns(x[:, :target_count], h, w, columns), x[:, target_count:]], 1)
        kw['img_shapes'] = [[(1,h,w+2*columns), *sample[1:]] for sample in kwargs['img_shapes']]
        return args, kw

    def rope_before(module, args, kwargs):
        if not args:
            raise ValueError('Audited Diffusers calls pos_embed with positional shapes')
        original = [[(1,s[0][1],s[0][2]-2*columns), *s[1:]] for s in args[0]]
        shape_info(original)
        return (original, *args[1:]), kwargs

    def rope_after(module, args, kwargs, output):
        h, w, _ = shape_info(args[0])
        image_rope, text_rope = output
        image_rope = torch.cat([pad_columns(image_rope[..., :h*w, :],h,w,columns), image_rope[..., h*w:, :]], -2)
        return image_rope, text_rope

    def after(module, args, kwargs, output):
        if mode == 'persistent':
            return output
        _, h, padded_w = kwargs['img_shapes'][0][0]
        w, boundary = padded_w-2*columns, h*padded_w
        x = output[0] if isinstance(output, tuple) else output.sample
        x = torch.cat([crop_columns(x[:, :boundary],h,w,columns), x[:, boundary:]], 1)
        if isinstance(output, tuple):
            return (x, *output[1:])
        output.sample = x
        return output

    transformer.register_forward_pre_hook(before, with_kwargs=True)
    transformer.pos_embed.register_forward_pre_hook(rope_before, with_kwargs=True)
    transformer.pos_embed.register_forward_hook(rope_after, with_kwargs=True)
    transformer.register_forward_hook(after, with_kwargs=True)
    transformer._pano_padding_installed = True
