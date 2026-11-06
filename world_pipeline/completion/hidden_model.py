"""Task-specific use of official M3DLayout AR backbone and distribution heads."""
import copy
import sys

import numpy as np
import torch
import torch.nn.functional as F

from ..common import field, validate_graph
from .hidden_data import GEOMETRY_BINS, normalize, denormalize, objects, prefix


def build(repo, device, config=None):
    import yaml
    sys.path.insert(0, str(repo.resolve()))
    from scene_synthesis.networks.autoregressive import build_network
    if config is None:
        config = yaml.safe_load((repo / 'config/m3dlayout_autoregressive.yaml').read_text())
        config = {k: config[k] for k in ('network', 'feature_extractor')}
        config['network']['text_condition'] = False
    from .hidden_data import CATEGORIES
    model = build_network(7 + len(CATEGORIES) + 2, len(CATEGORIES) + 2,
                          config, device=device)[0]
    # AABBs have no yaw target. The retained API angle is always zero.
    model.hidden2output.angle_layer.requires_grad_(False)
    return model, config


def initialize(model, weights):
    source = torch.load(weights, map_location='cpu', weights_only=True)
    target = model.state_dict()
    selected = {k: v for k, v in source.items() if k in target and v.shape == target[k].shape}
    model.load_state_dict(selected, strict=False)
    return dict(path=str(weights.resolve()), loaded_tensors=len(selected),
                fresh_keys=sorted(set(target) - set(selected)),
                omitted_keys=sorted(set(source) - set(selected)))


def encode(items, meta):
    categories = meta['categories']
    labels = np.zeros((len(items), len(categories) + 2), dtype=np.float32)
    for i, obj in enumerate(items):
        labels[i, categories.index(obj['category'])] = 1
    return dict(class_labels=torch.from_numpy(labels),
        translations=torch.tensor(normalize([o['center'] for o in items], meta['bounds']['center'])
                                  if items else np.empty((0, 3)), dtype=torch.float32),
        sizes=torch.tensor(normalize([o['size'] for o in items], meta['bounds']['size'])
                           if items else np.empty((0, 3)), dtype=torch.float32),
        angles=torch.zeros(len(items), 1))


def batch(examples, meta, device):
    contexts, targets = [], []
    for scene, k in examples:
        ctx, target = prefix(scene, k)
        contexts.append(encode(ctx, meta))
        if target is None:
            t = {key: torch.zeros(1, value.shape[-1]) for key, value in contexts[-1].items()}
            t['class_labels'][0, -1] = 1
        else:
            t = encode([target], meta)
        targets.append(t)
    lengths = [len(c['class_labels']) for c in contexts]
    result = {}
    for key in contexts[0]:
        result[key] = torch.stack([F.pad(c[key], (0, 0, 0, max(lengths) - n))
                                   for c, n in zip(contexts, lengths)])
        result[key + '_tr'] = torch.stack([t[key] for t in targets])
    result['lengths'] = torch.tensor(lengths, dtype=torch.long)
    # Constant interface tensor; this is NOT inferred room geometry.
    result['room_layout'] = torch.ones(len(examples), 1, 64, 64)
    return {k: v.to(device) for k, v in result.items()}


def losses(pred, sample):
    from scene_synthesis.networks.autoregressive.losses import cross_entropy_loss, dmll
    label = cross_entropy_loss(pred.class_labels, sample['class_labels_tr'])
    position, size = 0, 0
    for i, axis in enumerate('xyz'):
        position = position + dmll(getattr(pred, f'translations_{axis}'),
                                  sample['translations_tr'][..., i:i+1], num_classes=GEOMETRY_BINS)
        size = size + dmll(getattr(pred, f'sizes_{axis}'),
                          sample['sizes_tr'][..., i:i+1], num_classes=GEOMETRY_BINS)
    return label, position, size


def objective(pred, sample):
    label, position, size = losses(pred, sample)
    real = 1 - sample['class_labels_tr'][..., -1]
    # END carries no physical box. Average over examples, including END.
    return label.mean() + ((position + size) * real).mean()


def sample_dmll(parameters):
    """Use exp(log_scale), matching the official DMLL training likelihood."""
    logits, means, log_scales = parameters.chunk(3, -1)
    index = torch.multinomial(logits.reshape(-1, logits.shape[-1]).softmax(-1), 1)
    index = index.reshape(*logits.shape[:-1], 1)
    mean = means.gather(-1, index)
    scale = log_scales.clamp_min(-7).gather(-1, index).exp()
    u = torch.rand_like(mean).clamp(torch.finfo(mean.dtype).eps, 1 - torch.finfo(mean.dtype).eps)
    return (mean + scale * (u.log() - torch.log1p(-u))).clamp(-1, 1)


def sample_geometry(head, features, label):
    cf = torch.cat([features, head.fc_class_labels(label)], -1)
    center = torch.cat([sample_dmll(getattr(head, f'centroid_layer_{a}')(cf)) for a in 'xyz'], -1)
    position = [getattr(head, f'pe_trans_{a}')(center[..., i:i+1]) for i, a in enumerate('xyz')]
    angle = torch.zeros_like(center[..., :1])
    sf = torch.cat([cf, *position, head.pe_angle_z(angle)], -1)
    size = torch.cat([sample_dmll(getattr(head, f'size_layer_{a}')(sf)) for a in 'xyz'], -1)
    return center, size, angle


@torch.no_grad()
def generate(model, observed_graph, meta, seed, device):
    if observed_graph['scope'] != 'observed':
        raise ValueError('Input must be G_obs, never GT G_full')
    original = objects(observed_graph)
    if len(original) > meta['max_objects']:
        raise ValueError('Observed objects exceed Adapter capacity')
    torch.manual_seed(seed)
    model.eval()
    encoded = encode(original, meta)
    boxes = model.start_symbol(device)
    for key in boxes:
        boxes[key] = torch.cat([boxes[key], encoded[key][None].to(device)], dim=1)
    room = torch.ones(1, 1, 64, 64, device=device)
    generated, ended = [], False
    # One extra class query permits END at exactly max_objects.
    for _ in range(meta['max_objects'] - len(original) + 1):
        features = model._encode(boxes, room)
        head = model.hidden2output
        logits = head.class_layer(features).clone()
        logits[..., -2] = -torch.inf  # START is context syntax, never an object.
        index = torch.multinomial(logits.softmax(-1).reshape(1, -1), 1).item()
        if index == len(meta['categories']) + 1:
            ended = True
            break
        if len(original) + len(generated) == meta['max_objects']:
            break
        label = F.one_hot(torch.tensor([[index]], device=device), logits.shape[-1]).float()
        center, size, angle = sample_geometry(head, features, label)
        generated.append(dict(category=meta['categories'][index],
            center=denormalize(center[0, 0].cpu().numpy(), meta['bounds']['center']).tolist(),
            size=denormalize(size[0, 0].cpu().numpy(), meta['bounds']['size']).tolist()))
        for key, value in dict(class_labels=label, translations=center, sizes=size, angles=angle).items():
            boxes[key] = torch.cat([boxes[key], value], dim=1)
    graph = copy.deepcopy(observed_graph)
    graph.update(scope='full', stage='prediction', ready_for_training=False)
    ids = {o['track_id'] for o in graph['objects']}
    for index, obj in enumerate(generated):
        track_id = f'completion_{seed}_{index}'
        while track_id in ids:
            track_id += '_'
        ids.add(track_id)
        graph['objects'].append(dict(track_id=track_id,
            category_id=field(obj['category'], source='predicted', method=meta['version']),
            center_local_m=field(obj['center'], source='predicted', method=meta['version']),
            size_aabb_local_m=field(obj['size'], source='predicted', method=meta['version']),
            bbox2d_visible_xyxy_norm=field(None, reason='not_observed'),
            in_primary_room=field(None, reason='room_not_predicted'), local_evidence='unobserved'))
    validate_graph(graph)
    return graph, dict(observed=len(original), generated=len(generated), ended=ended,
                       capacity_reached=not ended, seed=seed)
