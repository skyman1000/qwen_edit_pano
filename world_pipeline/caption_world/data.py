"""CPU manifests and prompt-aware caches; retain original images and splits."""
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from ..build_polished_world import ROOT, read, atomic_json
from ..object_world.data import DEFAULT_PACK, load_pack, identity
from ..object_world.cache import semantic_phrases, check_condition, check_posterior

VERSION='caption_fresh_world_nobinding_v1'
OUTPUT=ROOT/'qwen_edit_pano/outputs/object_world_caption_fresh_nobinding_v1'


def stat(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns)


def prompt(instruction,caption):
    if not isinstance(caption,str) or not caption.strip():raise ValueError('Empty panorama caption')
    return instruction+'\nScene description: '+caption.strip()


def scene_caption(caption):
    """Keep raw scene semantics; discard uncalibrated ERP image directions.

    The polished image is yaw-aligned to Local, so its original textual
    left/right/unspecified image-center statements cannot be treated as camera
    coordinates. Preserve physical relations, e.g. 'in the center of the room'.
    Raw captions remain verbatim in CAPTIONS.jsonl; only conditioning is cleaned.
    """
    text=caption.strip()
    text=re.sub(r'\b(?:on|along|to|at)\s+the\s+(?:left\s+and\s+right|right\s+and\s+left)\s+sides?\b','on both sides',text,flags=re.I)
    text=re.sub(r'\b(?:left|right)(?:[- ]hand)?\s+(wall|door|window)\b',r'\1',text,flags=re.I)
    text=re.sub(r'\b(?:(?:on|to|at|along|toward|towards|from|in)\s+)?(?:the\s+)?(?:(?:far|extreme)\s+)?(?:left|right)(?:[- ]hand)?(?:\s+(?:side|edge))?(?:\s+of\s+(?:the\s+)?(?:image|picture|photo|panorama|frame))?\b','',text,flags=re.I)
    text=re.sub(r'\b(?:in|at|near)\s+(?:the\s+)?(?:center|centre|middle)(?:\s+of\s+(?:the\s+)?(?:image|picture|photo|panorama|frame))?\b(?!\s+of\b)','',text,flags=re.I)
    text=re.sub(r'\s+([,.])',r'\1',text);text=re.sub(r' +',' ',text).strip()
    return text


def training_prompt(instruction,caption):return prompt(instruction,scene_caption(caption))


def base_inputs():
    plan,rows=load_pack(DEFAULT_PACK)
    if (plan['train_count'],plan['val_count'])!=(7390,923):raise ValueError('Expected reviewed 7390/923 data')
    base=Path(plan['base_checkpoint']);pano=read(base/'pano_config.json')
    if any(pano[k]!=plan[k] for k in ('base_snapshot','height','padding_columns','prompt')):
        raise ValueError('Panorama base does not match source data')
    expected=dict(policy=plan['semantic_policy'],names=plan['names'],phrases=semantic_phrases(plan['names']),
        base_snapshot=plan['base_snapshot'],pooling='mean of name tokens only; padding/prefix excluded')
    if read(DEFAULT_PACK/'class_features.json')!=expected:raise ValueError('Category semantic source changed')
    features=load_file(str(DEFAULT_PACK/'class_features.safetensors'),device='cpu')['features']
    if features.shape!=(42,3584) or not torch.isfinite(features).all():raise ValueError('Invalid frozen category features')
    if any(features[int(key)].norm()==0 for key in plan['names']):raise ValueError('Missing class semantics')
    return plan,rows,features


def source_contract(plan):
    base=Path(plan['base_checkpoint'])
    return dict(version=VERSION,pack=str(DEFAULT_PACK),manifest=stat(DEFAULT_PACK/'samples.jsonl'),
        pack_plan=stat(DEFAULT_PACK/'PLAN.json'),base_snapshot=plan['base_snapshot'],base_checkpoint=str(base),
        panorama_weights=stat(base/'adapter_resume.safetensors'),panorama_config=stat(base/'pano_config.json'),
        class_features=stat(DEFAULT_PACK/'class_features.safetensors'),height=plan['height'],padding_columns=plan['padding_columns'],
        instruction=plan['prompt'],caption_source='matching polished Arrow caption, retained verbatim',
        caption_dropout=0.,task='Local + camera-local objects + supplied full-scene description -> aligned polished ERP',
        frame='local_view_x_right_y_image_up_z_forward_metres',hashes_computed=False)


def immutable_text(path,text):
    if path.exists():
        if path.read_text()!=text:raise ValueError('Prepared source changed; choose a new experiment root: '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(text);temp.replace(path)


def prepare(root):
    """Read Arrow text only; no target image decode, models, GPU, or downloads."""
    import pyarrow as pa
    root=Path(root);plan,rows,_=base_inputs();groups={};captions={}
    for row in rows:
        ptr=row['image'];groups.setdefault(ptr['arrow_file'],{}).setdefault(ptr['batch_index'],{})[ptr['row_index']]=row['id']
    for file,batches in groups.items():
        with pa.memory_map(file,'r') as source:
            for bi,batch in enumerate(pa.ipc.open_stream(source)):
                if bi not in batches:continue
                column=batch.column(batch.schema.get_field_index('caption'))
                for ri,sid in batches[bi].items():captions[sid]=column[ri].as_py()
    if len(captions)!=len(rows):raise ValueError('Missing paired polished caption')
    entries=[]
    for row in rows:
        item=dict(id=row['id'],split=row['split'],source=identity(row),prompt=prompt(plan['prompt'],captions[row['id']]),
            caption=captions[row['id']],reference_cache=stat(row['condition_cache']) if row['split']=='train' else None)
        entries.append(item)
    root.mkdir(parents=True,exist_ok=True)
    immutable_text(root/'CAPTIONS.jsonl',''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in entries))
    contract=dict(**source_contract(plan),caption_index=stat(root/'CAPTIONS.jsonl'))
    immutable_text(root/'DATA_PLAN.json',json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
    train=[r for r in rows if r['split']=='train']
    lengths=[len(captions[r['id']].split()) for r in train]
    result=dict(train=len(train),val=plan['val_count'],caption_words_min=min(lengths),caption_words_max=max(lengths),
        train_caption_caches=sum((root/'conditions'/(r['id']+'.safetensors')).is_file() for r in train),
        binding=False,fresh_world=True,world_checkpoint_loaded=False,optimizer_checkpoint_loaded=False,
        CPU_only=True,hashes_computed=False)
    atomic_json(root/'PREPARED.json',result);print(result,flush=True)
    return result


def inputs(root):
    root=Path(root);plan,rows,features=base_inputs();prepared=read(root/'DATA_PLAN.json')
    if prepared!={**source_contract(plan),'caption_index':stat(root/'CAPTIONS.jsonl')}:
        raise ValueError('Caption experiment source/configuration changed')
    entries={r['id']:r for r in (json.loads(line) for line in (root/'CAPTIONS.jsonl').open())}
    if set(entries)!={r['id'] for r in rows}:raise ValueError('Caption/source ID mismatch')
    for row in rows:
        entry=entries[row['id']]
        if entry['source']!=identity(row) or entry['split']!=row['split'] or entry['prompt']!=prompt(plan['prompt'],entry['caption']):
            raise ValueError('Caption/source pairing changed')
        if row['split']=='train' and entry['reference_cache']!=stat(row['condition_cache']):
            raise ValueError('Reference cache source changed: '+row['id'])
        row['caption_prompt']=training_prompt(plan['prompt'],entry['caption']);row['caption_cache']=str(root/'conditions'/(row['id']+'.safetensors'))
    return plan,rows,features,prepared,entries


def smoke_rows(rows):
    train=[r for r in rows if r['split']=='train'];found={}
    for row in (max(train,key=lambda r:len(r['objects'])),max(train,key=lambda r:len(r['caption_prompt'])),train[0]):
        found[row['id']]=row
    return list(found.values())


def cache_contract(row,entry,prepared):
    return dict(version=VERSION,id=row['id'],base_snapshot=prepared['base_snapshot'],prompt=row['caption_prompt'],
        source=entry['source'],reference_cache=entry['reference_cache'],
        encoding='native Qwen Edit encode_prompt(prompt, single Local image), BF16; no target RGB')


def check_caption_cache(row,entry,prepared):
    with safe_open(row['caption_cache'],framework='pt',device='cpu') as f:
        if f.metadata()!={'contract':json.dumps(cache_contract(row,entry,prepared),sort_keys=True)}:
            raise ValueError('Caption cache prompt/source mismatch: '+row['id'])
        if set(f.keys())!={'embeddings','mask'}:raise ValueError('Unexpected caption cache keys')
        shape=f.get_slice('embeddings').get_shape()
        if len(shape)!=2 or shape[1]!=3584 or f.get_slice('mask').get_shape()!=[shape[0]]:
            raise ValueError('Invalid image+caption embedding dimensions')


def check(root,smoke=False):
    plan,allrows,_,prepared,entries=inputs(root)
    rows=smoke_rows(allrows) if smoke else [r for r in allrows if r['split']=='train']
    missing=[]
    for index,row in enumerate(rows,1):
        check_condition(row['condition_cache']);check_posterior(row['posterior_cache'],plan['height'])
        with safe_open(row['region_cache'],framework='pt',device='cpu') as f:
            if f.get_slice('weights').get_shape()!=[plan['height']//8,plan['height']//4]:raise ValueError('Region size mismatch')
        if Path(row['caption_cache']).is_file():check_caption_cache(row,entries[row['id']],prepared)
        else:missing.append(row['id'])
        if index%1000==0:print(dict(stage='CPU_cache_headers',checked=index,total=len(rows),missing_caption_caches=len(missing)),flush=True)
    print(dict(stage='caption_cache_check',samples=len(rows),missing_caption_caches=len(missing),smoke=smoke,hashes_computed=False),flush=True)
    return missing


def load_condition(row,device):
    """New prompt embeddings; original image-only reference. Never old text."""
    values=load_file(row['caption_cache'],device=str(device))
    with safe_open(row['condition_cache'],framework='pt',device='cpu') as f:
        values['reference']=f.get_tensor('reference').to(device)
        values['ref_shape']=f.get_tensor('ref_shape').to(device)
    return values


def configuration(root,plan,rows,prepared,smoke=False):
    from qwen_pano.profiles import REPO_PANORAMA
    loss_keys=('lambda_cube','lambda_yaw','lambda_seam','perspective_weight','schedule','geometry_start_epoch','geometry_backend','mask_reduction')
    return dict(version=VERSION,architecture_version='polished_single_face_semantic_grounded_world_v2',
        data=prepared,train_ids=[r['id'] for r in rows if r['split']=='train'],smoke=bool(smoke),
        seed=0,blocks=[20,40,59],width=256,binding_block=40,binding_weight=0.,region_weight=.25,
        pano_lora_scale=.5,lr=1e-4,warmup_steps=0 if smoke else 100,betas=[.9,.999],adam_eps=1e-8,weight_decay=1e-5,
        clip_norm=1.,accumulation=4,save_every=100,maximum_steps=10000,
        initialization='fresh Encoder/Adapter and fresh FP32 Adam; zero residual gates; no C/A/smoke weights inherited',
        category_encoding='frozen Qwen category-name features + fresh separate geometry/fusion projections',
        loss='flow + 0.5 cube + 0.5 yaw + 0.25 region; no binding labels/loss',
        panorama_loss={k:REPO_PANORAMA[k] for k in loss_keys},panorama_geometry_enabled_from_step=0,
        prompt_policy='original expansion instruction + per-image polished scene caption; uncalibrated image-direction phrases removed; present on every training sample',
        caption_normalization='scene_caption_direction_neutral_v1; raw caption retained, no LLM rewrite or invented attributes',
        lr_policy='constant full LR for two diagnostic smoke steps' if smoke else '100-step linear warmup then constant; independent of requested stop step',
        base_trainable=0,caption_dropout=0.,automatic_evaluation=False,hashes_computed=False)
