"""Single-query Qwen offscreen hypotheses; reuse observed states, no layout model."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..common import read, rows, write
from .geometry import SIGNS
from .hybrid import outside_view
from .qwen_upstream import Qwen, DEFAULT_MODEL, load_image, camera_text
from .qwen_upstream_core import VOCAB, compact_observations, assemble


def prompt(observed, camera, text, limit):
    return (
        'Infer a sparse plausible continuation OUTSIDE this photographed field of view. '
        'Look at the image and existing observed instances together. Preserve all observed instances. '
        'A partially visible object is ALREADY observed: its unseen base or continuation is NOT a new instance. '
        'Propose only distinct additional objects with a plausible scene function, not decorations for visible empty surfaces. '
        'Do not assume every room is symmetric. Do not add a second nightstand if both are already visible. '
        'This is a hypothetical world, not a claim to recover the unique real world. '
        f'Output zero to {limit} additions; fewer is fine. '
        'Return JSON {"hidden_objects": [{"category": ..., "reason": ..., '
        '"center_local_m": [x,y,z], "size_aabb_local_m": [sx,sy,sz]}]}. '
        'reason briefly explains the visible cue and why this is a NEW offscreen instance. '
        'Use approximate metres consistent with the observed boxes: camera origin, x right, y UP, z forward. '
        'Sizes are FULL camera-axis AABB edge lengths, all positive. '
        'The WHOLE new box must be outside the supplied field of view. '
        'Negative z means behind the CAMERA, not behind furniture or beyond a doorway ahead. '
        'Choose a plausible offscreen placement without changing observed objects; never output rotation. '
        'Allowed categories: '+VOCAB+'. Camera: '+camera+
        '\nObserved instances: '+json.dumps(compact_observations(observed))+
        '\nOptional user requirements: '+text)


def convert(observed, answer, capacity, limit, hfov, image_size):
    accepted, audit = [], []
    items = answer['hidden_objects']
    if not isinstance(items, list) or len(items) > limit:
        raise ValueError('Expected hidden_objects list within requested budget')
    for item in items:
        try:
            # Reuse the existing category, shape and geometry conversion.
            assemble(observed, {'hidden_objects': [item]}, capacity, limit)
            corners = np.asarray(item['center_local_m']) + SIGNS*np.asarray(item['size_aabb_local_m'])/2
            reason = None if outside_view(corners, hfov, image_size) else 'not_wholly_offscreen'
        except (ValueError, KeyError, TypeError) as exc:
            reason = str(exc)
        audit.append(dict(proposal=item, accepted=reason is None, reason=reason))
        if reason is None:
            accepted.append(item)
    hidden, full = assemble(observed, {'hidden_objects': accepted}, capacity, limit)
    for obj, item in zip(hidden['objects'], accepted):
        obj['hypothesis_scope'] = 'outside_input_fov'
        obj['proposal'] = item
    full['objects'] = observed['objects'] + hidden['objects']
    return hidden, full, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--observer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    a = p.parse_args()
    capacity = read(a.bundle/'bundle.json')['max_objects']
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output/'config.json', dict(conditions=['pred'], seeds=[0], observer=str(a.observer),
                                     protocol='qwen_direct_offscreen_v1'))
    model = Qwen(a.model)
    failures = []
    for entry in rows(a.bundle/'inputs.jsonl'):
        d = a.output/'pred'/entry['id']/'seed0'
        d.mkdir(parents=True, exist_ok=True)
        try:
            observed = read(a.observer/entry['id']/'G_obs.json')
            image = load_image(entry)
            limit = min(3, capacity-len(observed['objects']))
            answer = model.ask(image, prompt(observed, camera_text(image, entry), entry.get('scene_text', ''), limit),
                               d/'proposal.txt', profile='official', max_tokens=2048) if limit > 0 else {'hidden_objects': []}
            hidden, full, audit = convert(observed, answer, capacity, limit, entry['hfov_deg'], image.size)
            for name, value in [('G_obs', observed), ('G_hidden', hidden), ('G_full', full), ('sampling_audit', audit)]:
                write(d/(name+'.json'), value)
            write(d/'status.json', dict(status='completed', observed=len(observed['objects']),
                  proposed=len(audit), generated=len(hidden['objects']),
                  outcome='candidate_generated' if hidden['objects'] else 'no_accepted_hypothesis'))
            print(entry['id'], f"{len(hidden['objects'])}/{len(audit)} accepted", flush=True)
        except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
            write(d/'status.json', dict(status='failed', error=str(exc)))
            failures.append(entry['id']); print(entry['id'], str(exc), flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} samples failed; raw outputs retained')


if __name__ == '__main__':
    main()
