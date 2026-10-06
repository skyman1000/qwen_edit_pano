"""Qwen3-VL Pass A: Local RGB only, offline model loading, 3D remains unknown.

Use a SEPARATE Observer environment. This module never reads GT graph files.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path

from PIL import Image, ImageDraw

from .common import (SCHEMA, FRAME, ELIGIBLE, field, finite_vector, validate_graph,
                     read, rows, sha, write, new_output)

INPUT_KEYS = {'id', 'split', 'local_rgb', 'local_sha256', 'width', 'height', 'hfov_degrees'}


def input_rows(path):
    entries = rows(path)
    if not entries:
        raise ValueError('Empty observer input list')
    for row in entries:
        if set(row) != INPUT_KEYS or row['split'] not in ('train', 'val'):
            raise ValueError('Observer accepts only the sanitized Local RGB input manifest, no GT/captions')
        if Path(row['id']).name != row['id'] or row['id'] in ('.', '..'):
            raise ValueError('Unsafe ID')
        if not Path(row['local_rgb']).is_absolute() or sha(row['local_rgb']) != row['local_sha256']:
            raise ValueError('Local image path/hash mismatch')
        with Image.open(row['local_rgb']) as image:
            if image.size != (row['width'], row['height']):
                raise ValueError('Local image dimensions differ')
    return entries


def prompt(vocabulary, bbox_scale=1, version='compact_v2'):
    if version == 'baseline_v1':
        return ('Report only objects actually visible in this image. Do not infer hidden or off-camera objects. '
            'Use the following fixed category ID vocabulary: '+json.dumps(vocabulary, ensure_ascii=False)+'. '
            'Do not report wall, floor, or ceiling as objects. Do not report an entire room as an object. '
            'Use category_id=null if a visible object cannot be classified. '
            'For each instance give the tight bounding box of its VISIBLE image region, not its imagined full extent. '
            f'Coordinates are [left,top,right,bottom], normalized to [0,{bbox_scale}], with top-left image origin. '
            'Include distinct visible instances separately; avoid duplicates. '
            'room_type is a short English candidate label or null. Do not estimate 3D, distances, or hidden contents. '
            'Return ONLY JSON in this exact structure: '
            '{"room_type":null,"objects":[{"category_id":3,"bbox2d":['+
            ','.join(str(int(x*bbox_scale)) if bbox_scale == 1000 else str(x) for x in (.1,.2,.3,.4))+']}]}.' )
    if version != 'compact_v2':
        raise ValueError('Unknown prompt version')
    return ('Report only objects actually visible in this image. Do not infer hidden or off-camera objects. '
            'Use the following fixed category ID vocabulary: '+json.dumps(vocabulary, ensure_ascii=False)+'. '
            'Do not report wall, floor, or ceiling as objects. Do not report an entire room as an object. '
            'Use category_id=null if a visible object cannot be classified. '
            'For each instance give the tight bounding box of its VISIBLE image region, not its imagined full extent. '
            f'Coordinates are [left,top,right,bottom] on a [0,{bbox_scale}] relative coordinate scale, '
            f'with top-left image origin; full image width and height both equal {bbox_scale}. '
            'Include distinct visible instances separately; avoid duplicates. '
            'room_type is a short English candidate label or null. Do not estimate 3D, distances, or hidden contents. '
            'Return compact JSON without markdown or explanation. Every object MUST contain both category_id '
            'and four bbox2d coordinates. Never emit an incomplete object. Finish the list and close the JSON '
            'when all distinct visible instances have been listed; do not repeat entries to continue the answer. '
            'Use this exact structure: '+json.dumps({'room_type':None,'objects':[
                {'category_id':3,'bbox2d':[bbox_scale*x for x in (0.1,0.2,0.3,0.4)]}]})+'.')


def parse_prediction(text, row, vocabulary_hash, bbox_scale=1):
    if bbox_scale not in (1, 1000):
        raise ValueError('Unsupported explicit bbox scale')
    value = text.strip()
    if value.startswith('```') and value.endswith('```'):
        value = '\n'.join(value.splitlines()[1:-1])
    def invalid_constant(s):
        raise ValueError(f'Non-finite JSON constant: {s}')
    def unique_keys(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = item
        return result
    data = json.loads(value, parse_constant=invalid_constant, object_pairs_hook=unique_keys)
    if not isinstance(data, dict) or set(data) != {'room_type', 'objects'} or not isinstance(data['objects'], list):
        raise ValueError('Unexpected Observer output shape')
    if data['room_type'] is not None and not isinstance(data['room_type'], str):
        raise ValueError('room_type must be string or null')
    objects = []
    for i, obj in enumerate(data['objects']):
        if not isinstance(obj, dict) or set(obj) != {'category_id', 'bbox2d'}:
            raise ValueError('Unexpected object fields')
        cat, box = obj['category_id'], obj['bbox2d']
        if cat is not None and (type(cat) is not int or cat not in ELIGIBLE):
            raise ValueError('Category outside vocabulary')
        if not finite_vector(box, 4) or not (0 <= box[0] < box[2] <= bbox_scale and 0 <= box[1] < box[3] <= bbox_scale):
            raise ValueError('Invalid normalized bbox; no silent clipping or coordinate guessing')
        box = [v / bbox_scale for v in box]
        objects.append(dict(track_id=f'pred_{i}',
            category_id=field(cat, 'observer', 'qwen3vl_pass_a', 'unclassified_visible_object'),
            center_local_m=field(None, reason='Pass_A_does_not_estimate_metric_3D'),
            size_aabb_local_m=field(None, reason='Pass_A_does_not_estimate_metric_3D'),
            bbox2d_visible_xyxy_norm=field(box, 'observer', 'qwen3vl_pass_a_visible_box'),
            local_evidence='visible', in_primary_room=field(None, reason='not_observable_from_Pass_A')))
    return validate_graph(dict(schema_version=SCHEMA, id=row['id'], split=row['split'], stage='observer',
        scope='observed', frame=FRAME, vocabulary_sha256=vocabulary_hash,
        room=dict(type=field(None, reason='free_text_room_candidate_not_mapped_to_Matterport_code'),
                  extent_aabb_local=field(None, reason='not_observed'),
                  floor_plane_local=field(None, reason='not_estimated'),
                  ceiling_plane_local=field(None, reason='not_estimated')),
        room_type_candidate=field(data['room_type'], 'observer', 'free_text_candidate_not_GT_code', 'uncertain'),
        objects=objects, relations=[], ready_for_training=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--vocabulary', type=Path, required=True)
    p.add_argument('--model', required=True, help='Existing local snapshot directory or cached Hub ID; NEVER downloaded here')
    p.add_argument('--output', type=Path)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--image-size', type=int, default=768)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--bbox-scale', type=int, choices=(1, 1000), default=1,
                   help='Explicit model output coordinate scale; stored graph always uses [0,1]')
    p.add_argument('--repetition-penalty', type=float, default=1.0,
                   help='Experimental repetition control; recorded for comparison, 1 disables it')
    p.add_argument('--prompt-version', choices=('baseline_v1', 'compact_v2'), default='compact_v2')
    p.add_argument('--decoding', choices=('greedy', 'checkpoint'), default='greedy',
                   help='checkpoint uses sampling settings from the local model generation_config.json')
    p.add_argument('--preflight', action='store_true', help='Check inputs/environment/cached snapshot; do not load model')
    args = p.parse_args()
    if args.limit < 0 or args.image_size < 128 or args.max_new_tokens < 1:
        p.error('Invalid limits')
    import math
    if not math.isfinite(args.repetition_penalty) or args.repetition_penalty < 1:
        p.error('repetition-penalty must be finite and >= 1')
    if not args.preflight and args.output is None:
        p.error('--output required')
    if importlib.metadata.version('transformers') != '4.57.1':
        raise RuntimeError('Use separate Observer environment with transformers==4.57.1; do not upgrade Edit environment')
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from huggingface_hub import snapshot_download
    entries = input_rows(args.inputs)
    if args.limit:
        entries = entries[:args.limit]
    vocabulary = read(args.vocabulary)
    if set(map(int, vocabulary['categories'])) != ELIGIBLE:
        raise ValueError('Unexpected vocabulary')
    snapshot = Path(args.model).expanduser()
    if not snapshot.is_dir():
        snapshot = Path(snapshot_download(args.model, local_files_only=True))
    snapshot = snapshot.resolve()
    if read(snapshot/'config.json')['model_type'] != 'qwen3_vl':
        raise ValueError('Expected dense Qwen3-VL model, e.g. 8B-Instruct')
    if args.preflight:
        print(f'PREFLIGHT PASSED: {len(entries)} Local images; snapshot={snapshot}; no model loaded.')
        return
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Observer inference requires a separately allocated bf16 CUDA GPU')
    torch.manual_seed(args.seed)
    output = new_output(args.output)
    def memory_report(stage):
        free, total = torch.cuda.mem_get_info(0)
        info = dict(stage=stage, device=torch.cuda.get_device_name(0),
                    free_gib=free / 2**30, total_gib=total / 2**30,
                    allocated_gib=torch.cuda.memory_allocated(0) / 2**30,
                    reserved_gib=torch.cuda.memory_reserved(0) / 2**30,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(0) / 2**30)
        print(info, flush=True)
        with (output/'memory.jsonl').open('a') as handle:
            handle.write(json.dumps(info) + '\n')
        return info

    memory_report('before_processor_and_model_load')
    instruction = prompt(vocabulary['categories'], args.bbox_scale, args.prompt_version)
    if args.decoding == 'checkpoint':
        checkpoint_generation = read(snapshot/'generation_config.json')
        generation_options = {k:checkpoint_generation[k] for k in ('do_sample','temperature','top_p','top_k')}
    else:
        generation_options = dict(do_sample=False, temperature=1.0, top_p=1.0, top_k=50)
    generation_options['repetition_penalty'] = args.repetition_penalty
    write(output/'run_config.json', dict(model_snapshot=str(snapshot), model_config_sha256=sha(snapshot/'config.json'),
        inputs_sha256=sha(args.inputs), vocabulary_sha256=sha(args.vocabulary),
        category_mapping_sha256=vocabulary['category_mapping_sha256'],
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        prompt=instruction, selected_ids=[r['id'] for r in entries],
        seed=args.seed, do_sample=generation_options['do_sample'], max_new_tokens=args.max_new_tokens, image_size=args.image_size,
        bbox_scale=args.bbox_scale, repetition_penalty=args.repetition_penalty,
        prompt_version=args.prompt_version, decoding=args.decoding, generation_options=generation_options,
        generation_config_sha256=sha(snapshot/'generation_config.json'),
        attention='sdpa', dtype='bfloat16',
        versions={n:importlib.metadata.version(n) for n in ('torch','transformers','accelerate','Pillow')},
        stage='Pass_A_only_no_metric_3D', local_only=True))
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(str(snapshot), local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'':0}).eval()
    except torch.OutOfMemoryError as exc:
        write(output/'OOM.json', dict(stage='model_load', error=str(exc),
                                     memory=memory_report('model_load_oom')))
        raise
    parameter_types = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        parameter_types[key] = parameter_types.get(key, 0) + parameter.numel()
    write(output/'model_runtime.json', dict(parameter_numel_by_dtype=parameter_types,
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        memory=memory_report('model_loaded')))
    print(f'Actual parameter dtypes (element counts): {parameter_types}', flush=True)
    success, failed = 0, 0
    for row in entries:
        # Reset per image so comparing decoding modes/subsets uses the same seed.
        torch.manual_seed(args.seed)
        directory = output/row['id']
        directory.mkdir()
        with Image.open(row['local_rgb']) as source:
            image = source.convert('RGB')
        image.thumbnail((args.image_size, args.image_size), Image.Resampling.BICUBIC)
        messages = [{'role':'user', 'content':[{'type':'image'}, {'type':'text','text':instruction}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            inputs = processor(text=[text], images=[image], return_tensors='pt')
            write(directory/'input_runtime.json', {
                key:dict(shape=list(value.shape), dtype=str(value.dtype))
                for key,value in inputs.items() if torch.is_tensor(value)})
            inputs = inputs.to(model.device)
            inputs.pop('token_type_ids', None)
            memory_report(f'{row["id"]}:before_generate')
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, **generation_options)
        except torch.OutOfMemoryError as exc:
            write(output/'OOM.json', dict(stage='input_or_generate', id=row['id'], error=str(exc),
                                         memory=memory_report('generation_oom')))
            raise
        tokens = ids[:, inputs.input_ids.shape[1]:]
        answer = processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        (directory/'raw.txt').write_text(answer)
        write(directory/'generation.json', dict(new_tokens=int(tokens.shape[1]),
            reached_token_limit=tokens.shape[1] >= args.max_new_tokens))
        try:
            if tokens.shape[1] >= args.max_new_tokens:
                raise ValueError('Output reached token limit; do not treat a possibly truncated object list as complete')
            graph = parse_prediction(answer, row, vocabulary['category_mapping_sha256'], args.bbox_scale)
            write(directory/'G_obs.json', graph)
            draw = ImageDraw.Draw(image)
            for obj in graph['objects']:
                x1,y1,x2,y2 = obj['bbox2d_visible_xyxy_norm']['value']
                draw.rectangle((x1*image.width,y1*image.height,x2*image.width,y2*image.height), outline='red', width=2)
                draw.text((x1*image.width,y1*image.height), f'{obj["track_id"]}: {obj["category_id"]["value"]}', fill='yellow')
            image.save(directory/'boxes.jpg')
            write(directory/'status.json',dict(status='parsed', id=row['id']))
            success += 1
        except (ValueError, KeyError, TypeError) as exc:
            write(directory/'status.json',dict(status='invalid_prediction', id=row['id'], error=str(exc)))
            failed += 1
        print(f'{row["id"]}: parsed={success} invalid={failed}', flush=True)
        del inputs, ids, tokens
    write(output/'COMPLETE.json', dict(status='INFERENCE_FINISHED_NOT_ACCURACY_CERTIFICATION', parsed=success, invalid=failed))


if __name__ == '__main__':
    main()
