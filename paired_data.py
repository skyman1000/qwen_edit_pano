"""Paired face2 data; full captions never enter model conditioning."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageOps
from .common import ROOT, check_ids, read_jsonl, sha256

CONDITIONING = 'edit2511_native_local_rgb_face2_v1'
DEFAULT_PROMPT = ROOT / 'qwen_edit_pano/prompts/local_to_erp_simple.txt'


def local_image(row, flip=False):
    c = row['local_condition']
    with zipfile.ZipFile(c['archive']) as z:
        raw = z.read(c['member'])
    if c.get('sha256') and hashlib.sha256(raw).hexdigest() != c['sha256']:
        raise ValueError('Local image changed: ' + row['id'])
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert('RGB')
    if image.size != (1024, 1024):
        raise ValueError('Expected original 1024x1024 face2: ' + row['id'])
    return ImageOps.mirror(image) if flip else image


def target_image(row):
    from qwen_pano.cached_data import ArrowImages
    p = row['image']
    if isinstance(p, dict):
        stat = Path(p['arrow_file']).stat()
        if stat.st_size != p['size_bytes'] or stat.st_mtime_ns != p['mtime_ns']:
            raise ValueError('Indexed Arrow file changed')
        image = ArrowImages().open(p)
    else:
        image = Image.open(p)
    with image:
        if image.width != 2 * image.height:
            raise ValueError('Target must be a 2:1 ERP')
        result = image.convert('RGB')
    if row.get('target_alignment'):
        from .pair_alignment import apply_alignment
        result = apply_alignment(result,row['target_alignment'])
    return result


def cache_key(row, flip):
    return hashlib.sha256((row['id'] + ':' + row['local_condition']['sha256'] + ':' + str(int(flip))).encode()).hexdigest()


def paired_rows(manifest, split=None):
    rows = read_jsonl(manifest)
    check_ids(rows)
    for row in rows:
        if split and row['split'] != split:
            raise ValueError('Unexpected split: ' + row['id'])
        if not row.get('target_alignment'):
            raise ValueError('Missing target orientation calibration; regenerate with prepare_pairs')
        c = row['local_condition']
        expected = [row['scene_id'], 'matterport_skybox_images', row['source_view_id'] + '_skybox2_sami.jpg']
        if [s for s in c['member'].split('/') if s] != expected or c['face_index'] != 2 or not c.get('sha256'):
            raise ValueError('Run prepare_pairs to validate source identity: ' + row['id'])
    return rows


class PairedDataset:
    def __init__(self, rows, height, cache, augment=True, seed=0):
        import torch
        self.rows, self.height, self.cache = rows, height, Path(cache)
        self.augment, self.seed = augment, seed
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)

    def __getitem__(self, index):
        import numpy as np
        import torch
        from safetensors.torch import load_file
        row = self.rows[index]
        # Deterministic per sample/epoch, independent of worker restarts.
        digest = hashlib.sha256(f'{self.seed}:{int(self.epoch)}:{row["id"]}'.encode()).digest()
        flip = self.augment and bool(digest[0] & 1)
        image = target_image(row).resize((self.height*2, self.height), Image.Resampling.BICUBIC)
        if flip:
            image = ImageOps.mirror(image)
        pixels = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float() / 127.5 - 1
        tensors = load_file(str(self.cache / (cache_key(row, flip) + '.safetensors')))
        return dict(pixels=pixels, mask=torch.ones(1, self.height//8, self.height//4, dtype=torch.bool),
                    embeddings=tensors['embeddings'], text_mask=tensors['mask'], reference=tensors['reference'],
                    ref_shape=tensors['ref_shape'].tolist(), id=row['id'])


def paired_collate(items):
    import torch
    if len(items) != 1:
        raise ValueError('Paired entry currently preserves batch_size=1')
    x = items[0]
    return dict(pixels=x['pixels'][None], mask=x['mask'][None], embeddings=x['embeddings'][None],
                text_mask=x['text_mask'][None], reference=x['reference'][None], ref_shape=x['ref_shape'], ids=[x['id']])
