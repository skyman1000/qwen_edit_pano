"""Manual, no-training Local RGB -> observed instances -> hidden world candidates."""
import argparse
from collections import defaultdict
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import zipfile

from PIL import Image, ImageDraw

from ..common import ROOT, read, rows, sha, write, write_rows, validate_graph
from .geometry import graph, parse_json, observer_categories
from .qwen_upstream_core import (VOCAB, assemble, hidden_prompt,
                                 observed_object, grounded_instances, observed_without_3d)

DEFAULT_MODEL = Path('/data-nfs/gpu1-2/u13529658780/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
INDOOR = set('abcdefghijklmnorstuvw') | {'B', 'C', 'D', 'S'}


def prepare(a):
    a.bundle.mkdir(parents=True, exist_ok=False)
    grouped, excluded = defaultdict(list), []
    for entry in rows(a.oracle/'samples.jsonl'):
        if entry['split'] != 'val':
            continue
        full = read(entry['graph'])
        if full['room']['type']['value'] not in INDOOR:
            excluded.append(dict(id=entry['id'], reason='non_indoor_or_unknown_room_code'))
            continue
        grouped[entry['paired']['scene_id']].append(entry)
    inputs, references, counts, vocabularies = [], [], {}, set()
    oracle = read(a.oracle/'BUNDLE.json')
    for building, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda r: r['id'])
        random.Random(a.selection_seed).shuffle(candidates)
        selected = candidates[:a.per_building]
        counts[building] = len(selected)
        for row in selected:
            sid = row['id']
            directory = a.bundle/sid
            directory.mkdir()
            local = row['paired']['local_condition']
            with zipfile.ZipFile(local['archive']) as archive:
                image = Image.open(io.BytesIO(archive.read(local['member']))).convert('RGB')
            image.save(directory/'local.png')
            full = validate_graph(read(row['graph']))
            vocabularies.add(full['vocabulary_sha256'])
            obs = graph(sid, 'val', 'observed',
                        [o for o in full['objects'] if o['local_evidence'] == 'visible'],
                        full['vocabulary_sha256'])
            hidden = graph(sid, 'val', 'hidden',
                           [o for o in full['objects'] if o['local_evidence'] == 'unobserved'],
                           full['vocabulary_sha256'])
            write(directory/'GT_obs.json', obs)
            write(directory/'GT_hidden.json', hidden)
            # Skybox face2 is a 90-degree face; no GT room pose/depth is sent to Qwen.
            inputs.append(dict(id=sid, split='val', building=building,
                               local_rgb=str((directory/'local.png').resolve()),
                               hfov_deg=90., scene_text=''))
            references.append(dict(id=sid, building=building,
                                   gt_obs=str((directory/'GT_obs.json').resolve()),
                                   gt_hidden=str((directory/'GT_hidden.json').resolve())))
    if not inputs:
        raise ValueError('No indoor validation samples in Oracle manifest')
    if len(vocabularies) != 1:
        raise ValueError('Selected GT graphs do not share one category vocabulary')
    write_rows(a.bundle/'inputs.jsonl', inputs)
    write_rows(a.bundle/'references.jsonl', references)
    write(a.bundle/'bundle.json', dict(oracle=str(a.oracle.resolve()), per_building=a.per_building,
          selection_seed=a.selection_seed, buildings=counts, samples=len(inputs),
          max_objects=oracle['max_objects'], vocabulary_sha256=next(iter(vocabularies)),
          exclusions=excluded))
    print(f'Prepared {len(inputs)} images: {counts}\nInputs: {a.bundle / "inputs.jsonl"}')


class Qwen:
    def __init__(self, model):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(model), local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model), local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map={'': 0}).eval()

    def ask(self, image, prompt, path, seed=0, sample=False, max_tokens=4096,
            profile='completion', response_format='json', guard_repetition=False, examples=None):
        # Same command can resume; never reuse a response to a changed question.
        options = (dict(do_sample=True, temperature=.7, top_p=.8, top_k=20, repetition_penalty=1.)
                   if profile in ('observer', 'official') else
                   dict(do_sample=sample, temperature=.7 if sample else 1., top_p=.9 if sample else 1., top_k=50))
        settings = dict(prompt=prompt, seed=seed, sample=sample, max_tokens=max_tokens,
                        profile=profile, generation=options, response_format=response_format,
                        presence_penalty=1.5 if profile == 'official' else 0.)
        if image is None:
            settings['text_only'] = True
        if examples is not None:
            if (not isinstance(examples, list) or len(examples) % 2 or
                    any(not isinstance(m, dict) or m.get('role') != ('user' if i % 2 == 0 else 'assistant')
                        or not isinstance(m.get('content'), str) for i, m in enumerate(examples))):
                raise ValueError('Few-shot examples must be alternating text user/assistant messages')
            settings['examples'] = deepcopy(examples)
        if guard_repetition:
            settings['guard_repetition'] = dict(repeated_complete_objects=3, check_every=32,
                comparison='complete_object_including_identity')
        config_path = path.with_suffix('.request.json')
        parsed_path = path.with_suffix('.json')
        if parsed_path.exists() and read(config_path) == settings:
            return read(parsed_path)
        # A failed new request must not leave an old parsed answer under its new settings.
        parsed_path.unlink(missing_ok=True)
        path.with_suffix('.generation.json').unlink(missing_ok=True)
        write(config_path, settings)
        messages = deepcopy(examples) if examples is not None else []
        content = ([{'type': 'image'}] if image is not None else [])
        content.append({'type': 'text', 'text': prompt})
        messages.append({'role': 'user', 'content': content})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = dict(text=[text], return_tensors='pt')
        if image is not None:
            inputs['images'] = [image]
        encoded = self.processor(**inputs).to(self.model.device)
        encoded.pop('token_type_ids', None)
        self.torch.manual_seed(seed)
        if profile == 'official':
            from transformers import LogitsProcessorList
            options['logits_processor'] = LogitsProcessorList([
                GeneratedPresencePenalty(encoded.input_ids.shape[1], 1.5)])
        guard = None
        if guard_repetition:
            from transformers import StoppingCriteriaList
            from .json_generation import RepeatedJSONObjectStop
            guard = RepeatedJSONObjectStop(self.processor, encoded.input_ids.shape[1])
            options['stopping_criteria'] = StoppingCriteriaList([guard])
        with self.torch.inference_mode():
            tokens = self.model.generate(**encoded, max_new_tokens=max_tokens, **options)
        tokens = tokens[:, encoded.input_ids.shape[1]:]
        raw = self.processor.batch_decode(tokens, skip_special_tokens=True)[0]
        path.write_text(raw)
        eos = getattr(getattr(self.model, 'generation_config', None), 'eos_token_id', None)
        eos = eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos])
        ended = bool(tokens.shape[1] and int(tokens[0, -1]) in eos)
        finish = ('repeated_objects' if guard is not None and guard.triggered else
                  'eos' if ended else 'length' if tokens.shape[1] >= max_tokens else 'stop')
        write(path.with_suffix('.generation.json'), dict(generated_tokens=tokens.shape[1],
              max_tokens=max_tokens, finish_reason=finish, ended_with_eos=ended))
        if finish == 'repeated_objects':
            raise ValueError(f'Repeated complete JSON objects; generation stopped and raw answer preserved: {path}')
        if finish == 'length':
            raise ValueError(f'Unfinished response reached token limit: {path}; inspect for repetition before changing the budget')
        try:
            value = raw.strip() if response_format == 'text' else parse_json(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSON saved at {path}: {exc}') from exc
        write(parsed_path, value)
        return value


class GeneratedPresencePenalty:
    """HF generate equivalent of presence penalty, applied once per generated token ID."""
    def __init__(self, prompt_length, penalty):
        self.prompt_length, self.penalty = prompt_length, penalty

    def __call__(self, input_ids, scores):
        scores = scores.clone()
        for i in range(input_ids.shape[0]):
            generated = input_ids[i, self.prompt_length:].unique()
            scores[i, generated] -= self.penalty
        return scores


def start_run(a):
    a.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config['inputs_sha256'] = sha(a.bundle/'inputs.jsonl')
    config['bundle_sha256'] = sha(a.bundle/'bundle.json')
    config['protocol'] = 'category_2d_instance_3d_v3' if a.action == 'observe' else 'hidden_only_v1'
    if getattr(a, 'simple', False):
        config['protocol'] = 'official_query_plan_geometry_v1'
    config_path = a.output/'config.json'
    if config_path.exists() and read(config_path) != config:
        raise ValueError('Output belongs to different arguments; select another --output')
    write(config_path, config)
    return Qwen(a.model)


def load_image(entry):
    image = Image.open(entry['local_rgb']).convert('RGB')
    image.thumbnail((768, 768))
    return image


def camera_text(image, entry):
    import math
    w, h = image.size
    focal = w/(2*math.tan(math.radians(entry['hfov_deg'])/2))
    return (f'Image size {w}x{h}; pinhole camera fx=fy={focal:.3f}, '
            f'cx={w/2:.3f}, cy={h/2:.3f} pixels. ')


def observe_query(model, image, prompt, path, validate, max_tokens, profile='observer'):
    """One retry for a malformed response; never invent missing coordinates."""
    errors = []
    for attempt in range(2):
        target = path if attempt == 0 else path.with_name(path.stem+'.retry.txt')
        try:
            response = model.ask(image, prompt, target, seed=attempt,
                                 max_tokens=max_tokens, profile=profile)
            return validate(response)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))
    raise ValueError('; '.join(errors))


def observe(a):
    model = start_run(a)
    metadata = read(a.bundle/'bundle.json')
    failures = []
    for entry in rows(a.bundle/'inputs.jsonl'):
        d = a.output/entry['id']; d.mkdir(exist_ok=True)
        if (d/'status.json').exists() and read(d/'status.json')['status'] in ('completed', 'partial'):
            continue
        image = load_image(entry)
        try:
            if getattr(a, 'reuse_instances', None):
                source = a.reuse_instances/entry['id']
                instances = read(source/'instances.normalized.json')
                response = dict(description=(source/'description.txt').read_text().strip())
                grounding_audit = read(source/'grounding_audit.json')
                issues = [issue for issue in read(source/'status.json').get('issues', []) if issue['stage']=='2d']
                write(d/'reused_instances.json', dict(source=str(source.resolve()),
                      sha256=sha(source/'instances.normalized.json')))
            else:
                response = model.ask(image,
                    'List the distinct object CATEGORIES actually visible in this image. '
                    'Include partially visible objects, but do not infer hidden objects. '
                    'Choose only from: '+VOCAB+'. Each category once. Exclude people, walls, floors '
                    'and ceilings. Use generic classes only for objects without a specific matching class. '
                    'Also describe visible evidence in ONE short sentence (at most 35 words). '
                    'Do NOT enumerate instances or bounding boxes here. Return only compact JSON: '
                    '{"description":"A partial view of a room.","categories":["chair","table"]}.',
                    d/'categories.txt', max_tokens=min(a.max_new_tokens, 768), profile='observer')
                names, _, changes, skipped = observer_categories(response)
                write(d/'categories.normalized.json', dict(categories=names, changes=changes, skipped=skipped))
                if skipped:
                    raise ValueError(f'Unmapped category responses: {skipped}; no guessed class IDs used')
                instances, grounding_audit, issues = [], {}, []
                for category in names:
                    query = (
                        f'Locate up to 8 clearly visible {category} instances in this image. '
                        'Give one tight visible bounding box per instance, xyxy normalized to 0..1000. '
                        'Do not extrapolate or repeat objects. Stop when the visible instances are exhausted. '
                        'Return only a JSON list of objects with bbox_2d, no prose. '
                        'Format example for TWO boxes: '
                        '[{"bbox_2d":[100,200,300,400]},{"bbox_2d":[500,600,700,800]}]. '
                        'Use image measurements, not the example numbers. Return [] if none are visible.')
                    try:
                        found, audit = observe_query(model, image, query, d/f'ground_{category}.txt',
                            lambda result: grounded_instances(category, result, offset=len(instances), limit=8),
                            max_tokens=min(a.max_new_tokens, 768))
                        instances.extend(found); grounding_audit[category] = audit
                    except ValueError as exc:
                        issues.append(dict(stage='2d', category=category, error=str(exc)))
                if names and not instances:
                    raise ValueError('No localized instances; inspect category grounding responses')
            write(d/'grounding_audit.json', grounding_audit)
            if len(instances) > metadata['max_objects']:
                raise ValueError('Observed instances exceed Adapter capacity')
            write(d/'instances.normalized.json', instances)
            (d/'description.txt').write_text(str(response['description'])+'\n')
            overlay = image.copy(); draw = ImageDraw.Draw(overlay)
            for inst in instances:
                b = inst['bbox_2d']; w, h = image.size
                box = [b[0]*w/1000, b[1]*h/1000, b[2]*w/1000, b[3]*h/1000]
                draw.rectangle(box, outline='red', width=2)
                draw.text((box[0], box[1]), inst['id']+' '+inst['category'], fill='red')
            overlay.save(d/'instances.png')
            objects, boxes = [], []
            for inst in instances:
                target_image = image
                if getattr(a, 'mark_target', False):
                    target_image = image.copy()
                    w, h = image.size
                    b = inst['bbox_2d']
                    ImageDraw.Draw(target_image).rectangle(
                        [b[0]*w/1000, b[1]*h/1000, b[2]*w/1000, b[3]*h/1000],
                        outline='red', width=3)
                query = (camera_text(image, entry)+'Locate only the visible '+inst['category']+
                    ' instance described as '+json.dumps(inst['description'])+
                    ', in 2D xyxy box '+json.dumps(inst['bbox_2d'])+' (normalized 0..1000). '
                    'Use the whole image camera, NOT a cropped camera. Estimate the full object 3D box. '
                    'Camera coordinates in METRES: x right, y DOWN, z forward; camera origin is zero. '
                    'Return full OBB side lengths, NOT half extents. Last three values are rotations '
                    'about x, then y, then z, each divided by pi (1 means 180 degrees). '
                    'Return ONE JSON object with exactly 9 numeric values. Format example: '
                    '{"bbox_3d":[0.2,0.1,2.5,0.6,0.9,0.5,0.0,0.0,0.0]}. '
                    'Use estimated measurements, not these example values. No trailing commas.')
                if getattr(a, 'simple', False):
                    from .simple_world_prompts import native_3d_prompt
                    query = native_3d_prompt(inst, camera_text(image, entry), getattr(a, 'mark_target', False))
                try:
                    obj, box = observe_query(model, target_image, query, d/(inst['id']+'.txt'),
                        lambda result: observed_object(inst, result), max_tokens=min(a.max_new_tokens, 512),
                        profile='official' if getattr(a, 'simple', False) else 'observer')
                    objects.append(obj); boxes.append(box)
                except ValueError as exc:
                    objects.append(observed_without_3d(inst, 'qwen_3d_response_unavailable'))
                    issues.append(dict(stage='3d', instance_id=inst['id'], error=str(exc)))
            state = graph(entry['id'], entry['split'], 'observed', objects, metadata['vocabulary_sha256'])
            validate_graph(state)
            write(d/'G_obs.json', state)
            write(d/'boxes.json', boxes)
            write(d/'status.json', dict(status='partial' if issues else 'completed', objects=len(objects), issues=issues,
                  valid_3d=len(boxes),
                  categories_at_capacity=[k for k, v in grounding_audit.items() if v['at_capacity']],
                  exact_duplicate_boxes=sum(len(v['exact_duplicates']) for v in grounding_audit.values())))
            print(f'{entry["id"]}: {len(objects)} observed, {len(boxes)} valid 3D, {len(issues)} unresolved queries', flush=True)
        except (ValueError, KeyError, TypeError) as exc:
            failures.append(entry['id'])
            write(d/'status.json', dict(status='failed', error=str(exc)))
            print(f'{entry["id"]}: FAILED: {exc}', flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} Observer samples failed; inspect status.json/raw text. No empty worlds substituted.')


def complete(a):
    model = start_run(a)
    metadata = read(a.bundle/'bundle.json')
    references = {r['id']: r for r in rows(a.bundle/'references.jsonl')} if 'gt' in a.conditions else {}
    failures = []
    for entry in rows(a.bundle/'inputs.jsonl'):
        sid = entry['id']; od = a.observer/sid
        image = load_image(entry)
        # Same image-only description for both conditions, never a full-scene GT caption.
        desc = (od/'description.txt').read_text() if (od/'description.txt').exists() else ''
        for condition in a.conditions:
            for seed in a.seeds:
                d = a.output/condition/sid/f'seed{seed}'
                d.mkdir(parents=True, exist_ok=True)
                try:
                    if condition == 'pred':
                        if read(od/'status.json')['status'] not in ('completed', 'partial'):
                            raise ValueError('Observer failed; no predicted context available')
                        observed = read(od/'G_obs.json')
                    else:
                        observed = read(references[sid]['gt_obs'])
                    capacity = min(a.max_hidden, metadata['max_objects']-len(observed['objects']))
                    if capacity < 0:
                        raise ValueError('Observed context exceeds Adapter capacity')
                    if getattr(a, 'simple', False):
                        from .simple_world_prompts import semantic_plan_prompt, plan_geometry_prompt
                        if (d/'status.json').exists() and read(d/'status.json')['status']=='completed':
                            if read(d/'G_obs.json') != observed:
                                raise ValueError('Observed input changed; select a new output')
                            continue
                        if capacity:
                            plan = model.ask(image,
                                semantic_plan_prompt(observed, desc, entry.get('scene_text', ''), capacity, camera_text(image, entry)),
                                d/'hidden_plan.txt', seed=seed, profile='official', response_format='text',
                                max_tokens=min(a.max_new_tokens, 1024))
                            answer = model.ask(image, plan_geometry_prompt(observed, plan, capacity, camera_text(image, entry)),
                                d/'proposal.txt', seed=seed, profile='official', max_tokens=a.max_new_tokens)
                        else:
                            answer = {'hidden_objects': []}
                    else:
                        prompt = hidden_prompt(observed, desc, entry.get('scene_text', ''), capacity)
                        if (d/'status.json').exists() and read(d/'status.json')['status'] == 'completed':
                            # Observations can finish between resumed jobs; never reuse a different context.
                            old_prompt = read(d/'proposal.request.json')['prompt'] if capacity else prompt
                            if read(d/'G_obs.json') != observed or old_prompt != prompt:
                                raise ValueError('Source observations/description changed; use a new output directory')
                            continue
                        answer = (model.ask(image, prompt, d/'proposal.txt', seed=seed, sample=True,
                                            max_tokens=a.max_new_tokens) if capacity else {'hidden_objects': []})
                    hidden, full = assemble(observed, answer, metadata['max_objects'], a.max_hidden)
                    write(d/'G_obs.json', observed)
                    write(d/'G_hidden.json', hidden)
                    write(d/'G_full.json', full)
                    write(d/'status.json', dict(status='completed', observed=len(observed['objects']),
                          generated=len(hidden['objects']), capacity=capacity,
                          outcome='candidate_generated' if hidden['objects'] else 'no_hidden_proposal',
                          at_capacity=len(hidden['objects']) == capacity, condition=condition, seed=seed))
                    print(f'{condition} {sid} seed{seed}: +{len(hidden["objects"])} objects', flush=True)
                except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
                    failures.append(str(d))
                    write(d/'status.json', dict(status='failed', error=str(exc)))
                    print(f'{d}: FAILED: {exc}', flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} proposals failed; inspect status.json. Run review to report failures too.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--oracle', type=Path, default=ROOT/'qwen_edit_pano/data/oracle_supplement_epoch006_v1')
    prep.add_argument('--bundle', type=Path, required=True)
    prep.add_argument('--per-building', type=int, default=3)
    prep.add_argument('--selection-seed', type=int, default=0)
    for action in ('observe', 'complete'):
        sp = sub.add_parser(action)
        sp.add_argument('--bundle', type=Path, required=True)
        sp.add_argument('--output', type=Path, required=True)
        sp.add_argument('--model', type=Path, default=DEFAULT_MODEL)
        sp.add_argument('--max-new-tokens', type=int, default=8192)
        sp.add_argument('--simple', action='store_true', help='Official-style 3D query and hypothetical plan then geometry')
        if action == 'observe':
            sp.add_argument('--mark-target', action='store_true', help='Mark the target on the full image instead of writing 2D coordinates; use with --simple')
            sp.add_argument('--reuse-instances', type=Path, help='Reuse 2D instances ONLY; always re-estimate 3D')
        if action == 'complete':
            sp.add_argument('--observer', type=Path, required=True)
            sp.add_argument('--conditions', nargs='+', choices=['gt', 'pred'], default=['gt', 'pred'])
            sp.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
            sp.add_argument('--max-hidden', type=int, default=64)
    a = p.parse_args()
    {'prepare': prepare, 'observe': observe, 'complete': complete}[a.action](a)


if __name__ == '__main__':
    main()
