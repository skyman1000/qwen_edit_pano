"""Local RGB -> short English observation caption; no GT panorama input."""
import argparse
from pathlib import Path

from ..common import read, rows, write

PROMPT = (
    'Describe the visible indoor scene in 3 to 5 short English sentences, at most 90 words. '
    'Mention the likely room function, the main visible furniture, and clearly visible spatial '
    'relationships. Describe this as a partial view, not a complete room inventory. '
    'Do not invent objects outside the image or claim that unobserved objects do not exist. '
    'Avoid exact metric measurements, camera pose estimates, decorative prose, lists, and JSON. '
    'Return only the description.'
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    entries = rows(a.inputs)
    if not entries:
        raise ValueError(f'No input images in {a.inputs}; inspect preparation audit.json before description generation.')
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    a.output.mkdir(parents=True, exist_ok=False)
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0}).eval()
    write(a.output/'config.json', dict(model=str(a.model), inputs=str(a.inputs), prompt=PROMPT,
                                      decoding='greedy', max_new_tokens=384))
    for entry in entries:
        image = Image.open(entry['local_rgb']).convert('RGB')
        image.thumbnail((768, 768))
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = processor(text=[prompt], images=[image], return_tensors='pt').to(model.device)
        encoded.pop('token_type_ids', None)
        with torch.inference_mode():
            result = model.generate(**encoded, max_new_tokens=384, do_sample=False,
                                    temperature=1., top_p=1., top_k=50)
        tokens = result[:, encoded.input_ids.shape[1]:]
        text = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        folder = a.output/entry['id']; folder.mkdir()
        (folder/'description.txt').write_text(text+'\n')
        write(folder/'status.json', dict(id=entry['id'], local_rgb=entry['local_rgb'],
            new_tokens=tokens.shape[1], reached_token_limit=tokens.shape[1] >= 384))
        print(f"{entry['id']}: {text}", flush=True)


if __name__ == '__main__':
    main()
