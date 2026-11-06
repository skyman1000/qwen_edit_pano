"""User-run GPU cache of Local+caption embeddings, sharded across one/two GPUs."""
import argparse
import json
import os
from pathlib import Path
import time

import torch
from safetensors.torch import save_file

from . import data as d
from ..object_world.data import identity


def shard(rows,rank,size):
    if size not in (1,2) or not 0<=rank<size:raise ValueError('Use one/two cache workers')
    return rows[rank::size]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=d.OUTPUT);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    plan,allrows,_,prepared,entries=d.inputs(a.root)
    rows=d.smoke_rows(allrows) if a.smoke else [r for r in allrows if r['split']=='train']
    size=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local_rank=int(os.environ.get('LOCAL_RANK','0'))
    missing=[]
    for row in shard(rows,rank,size):
        if Path(row['caption_cache']).is_file():d.check_caption_cache(row,entries[row['id']],prepared)
        else:missing.append(row)
    if not missing:print(dict(cache_rank=rank,missing=0),flush=True);return
    if not torch.cuda.is_available() or torch.cuda.device_count()<size:raise ValueError('Allocate one/two visible CUDA GPU(s) matching --gpus')
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():raise ValueError('BF16 CUDA GPU(s) required')
    device=torch.device('cuda',local_rank)
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions, CONDITION_IMAGE_SIZE
    from qwen_edit_pano.paired_data import local_image
    # VAE/DiT are not loaded: references and posterior caches are already valid.
    pipe=QwenImageEditPlusPipeline.from_pretrained(plan['base_snapshot'],transformer=None,vae=None,
        torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
    pipe.text_encoder.requires_grad_(False).eval().to(device)
    for i,row in enumerate(missing,1):
        tick=time.monotonic();image=local_image(identity(row))
        width,height=calculate_dimensions(CONDITION_IMAGE_SIZE,image.width/image.height)
        image=pipe.image_processor.resize(image,height,width)
        with torch.inference_mode():emb,mask=pipe.encode_prompt(prompt=row['caption_prompt'],image=[image],device=device)
        if mask is None:mask=torch.ones(emb.shape[:2],device=device,dtype=torch.bool)
        if not torch.isfinite(emb).all():raise FloatingPointError('Nonfinite caption embedding')
        path=Path(row['caption_cache']);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.safetensors')
        save_file(dict(embeddings=emb[0].cpu().contiguous(),mask=mask[0].bool().cpu().contiguous()),str(tmp),
            metadata={'contract':json.dumps(d.cache_contract(row,entries[row['id']],prepared),sort_keys=True)})
        tmp.replace(path);d.check_caption_cache(row,entries[row['id']],prepared)
        print(dict(stage='Local_caption_cache',rank=rank,completed=i,total_pending=len(missing),id=row['id'],
            text_tokens=emb.shape[1],seconds=time.monotonic()-tick),flush=True)


if __name__=='__main__':main()
