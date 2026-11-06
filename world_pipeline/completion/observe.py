"""Local RGB -> Qwen3-VL category queries -> visible 3D boxes. Run manually."""
import argparse
import json
import os
from pathlib import Path

from .geometry import (CANONICAL_CATEGORIES, graph, parse_json, qwen_corners,
                       world_object, write, observer_categories, observer_boxes)
from ..common import ROOT, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--image', type=Path)
    inputs.add_argument('--inputs', type=Path, help='Existing observer_inputs.jsonl')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--image-size', type=int, default=768)
    p.add_argument('--max-new-tokens', type=int, default=2048)
    p.add_argument('--max-per-category', type=int, default=8, help='Small-test instance cap per category')
    p.add_argument('--decoding', choices=['checkpoint', 'greedy'], default='checkpoint')
    p.add_argument('--angle-unit', choices=['normalized', 'radians', 'degrees'], default='normalized')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mapping', type=Path, default=ROOT/'benchmark_assets/Matterport3D_metadata/official/category_mapping.tsv')
    args = p.parse_args()

    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from huggingface_hub import snapshot_download

    snapshot = Path(args.model).expanduser()
    if not snapshot.is_dir():
        snapshot = Path(snapshot_download(args.model, local_files_only=True))
    args.model = str(snapshot.resolve())

    if args.image:
        entries = [dict(id=args.image.stem, split='val', local_rgb=str(args.image.resolve()))]
    else:
        entries = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if args.limit:
        entries = entries[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output/'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map={'': 0}).eval()
    generation_options = {} if args.decoding == 'checkpoint' else dict(
        do_sample=False, temperature=1., top_p=1., top_k=50)
    write(args.output/'generation_config.json', model.generation_config.to_dict())
    vocabulary_hash = sha(args.mapping)

    def ask(image, prompt, path):
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = processor(text=[text], images=[image], return_tensors='pt').to(model.device)
        encoded.pop('token_type_ids', None)
        torch.manual_seed(args.seed)
        with torch.inference_mode():
            tokens = model.generate(**encoded, max_new_tokens=args.max_new_tokens,
                                    **generation_options)
        new_tokens = tokens[:, encoded.input_ids.shape[1]:]
        raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        path.write_text(raw)
        path.with_suffix('.prompt.txt').write_text(prompt)
        status = dict(new_tokens=new_tokens.shape[1],
                      reached_token_limit=new_tokens.shape[1] >= args.max_new_tokens)
        try:
            result = parse_json(raw)
        except json.JSONDecodeError as exc:
            write(path.with_suffix('.status.json'), dict(status, parsed=False, error=str(exc)))
            print(f'{path.name}: incomplete/invalid JSON; saved raw response, skipping query.', flush=True)
            return None
        write(path.with_suffix('.status.json'), dict(status, parsed=True))
        return result

    for entry in entries:
        directory = args.output/entry['id']; directory.mkdir(parents=True, exist_ok=True)
        image = Image.open(entry['local_rgb']).convert('RGB')
        image.thumbnail((args.image_size, args.image_size))
        image.save(directory/'local.jpg')
        categories = ask(image,
            'Identify the room type and distinct object categories actually visible in this image. '
            'Do not infer hidden objects. Do not list walls, floors or ceilings. '
            'Choose category names from: '+', '.join(CANONICAL_CATEGORIES.values())+'. '
            'Use each category once. Use generic furniture, seating, objects or misc only '
            'for instances without a more specific matching category; do not list the same instance twice. '
            'Return only JSON: {"room_type":"living room","categories":["sofa","table"]}.',
            directory/'categories.txt')
        if categories is None:
            write(directory/'status.json', dict(status='failed_categories'))
            continue
        try:
            names, room_type, changes, skipped = observer_categories(categories)
        except ValueError as exc:
            write(directory/'status.json', dict(status='failed_categories', error=str(exc)))
            continue
        write(directory/'categories.normalized.json', dict(
            room_type=room_type, categories=names, changes=changes, skipped=skipped))
        boxes, objects = [], []
        seen = set()
        for category in names:
            results = ask(image,
                f'Locate all visible {category} instances in the provided image. '
                'Output their positions and full object dimensions using 3D bounding boxes. '
                f'Return at most {args.max_per_category} distinct instances. '
                'If there are more, select the most clearly visible instances. '
                'Do not repeat boxes or extrapolate a sequence of positions. '
                'Do not add hidden or off-camera objects. Return one compact JSON array without markdown, '
                'and close the array immediately after the last instance: '
                '[{"bbox_3d":[x_center,y_center,z_center,x_size,y_size,z_size,roll,pitch,yaw],'
                f'"label":"{category}"}}]. Use an empty array if none are visible.',
                directory/f'boxes_{category}.txt')
            if results is None:
                skipped.append(dict(category=category, reason='invalid_json', raw_file=f'boxes_{category}.txt'))
                continue
            valid_boxes, rejected = observer_boxes(results, category, args.max_per_category, seen)
            skipped.extend(rejected)
            for bbox in valid_boxes:
                center, corners = qwen_corners(bbox, args.angle_unit)
                track_id = f'obs_{len(boxes):04d}'
                boxes.append(dict(track_id=track_id, category=category, bbox_3d=bbox,
                                  center_local=center.tolist(), corners_local=corners.tolist(), observed=True))
                objects.append(world_object(track_id, category, center, corners, 'observer', True))
        write(directory/'observed.json', dict(id=entry['id'], split=entry['split'],
            image=str(Path(entry['local_rgb']).resolve()), room_type=room_type,
            angle_unit=args.angle_unit, boxes=boxes, skipped=skipped))
        write(directory/'G_obs.json', graph(entry['id'], entry['split'], 'observed', objects, vocabulary_hash))
        write(directory/'status.json', dict(status='empty' if not boxes else ('partial' if skipped else 'completed'),
                                           objects=len(boxes), skipped=skipped))
        print(f"{entry['id']}: {len(boxes)} visible 3D boxes -> {directory}", flush=True)


if __name__ == '__main__':
    main()
