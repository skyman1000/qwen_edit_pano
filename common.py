"""Manifest and provenance helpers; importing this module never loads a model."""
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ROOT / "benchmark_assets/mp3d_stitched1092/prompts.jsonl"
DEFAULT_MODEL = "Qwen/Qwen-Image-Edit-2511"
ERP_PREFIX = "A seamless 360-degree equirectangular panorama, full spherical view, 2:1 projection. "


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def caption_text(value):
    """Released mixed loader chooses the longest candidate caption."""
    if isinstance(value, list):
        if not value or any(not isinstance(x, str) for x in value):
            raise ValueError("caption list must contain strings")
        value = max(value, key=len)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("caption must be a nonempty string or list of strings")
    return value


def check_ids(rows):
    ids = [r["id"] for r in rows]
    if any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids):
        raise ValueError("IDs must contain only letters, numbers, underscore or hyphen")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs")


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def sample_seed(base_seed, sample_id, mode):
    # Exactly the same derivation as DiT360/inference_benchmark.py.
    if mode == "shared":
        return base_seed
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63)


def resolve_image(manifest, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(manifest).resolve().parent / path).resolve()


def require_versions():
    # Hooks/private packing APIs below are audited against this release.
    version = importlib.metadata.version("diffusers")
    if version != "0.37.0":
        raise RuntimeError(f"Use a separate Qwen environment with diffusers==0.37.0; found {version}")


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "diffusers", "transformers", "accelerate", "peft", "safetensors")}


def model_snapshot(model, revision=None, local_files_only=False):
    if Path(model).is_dir():
        if revision is not None:
            raise ValueError("--revision is only for Hub model IDs, not local directories")
        snapshot = str(Path(model).resolve())
    else:
        from huggingface_hub import snapshot_download
        snapshot = snapshot_download(model, revision=revision, local_files_only=local_files_only,
                                     cache_dir=ROOT / "qwen_edit_pano/cache/hub")
    index = json.loads((Path(snapshot) / "model_index.json").read_text())
    if index.get("_class_name") != "QwenImageEditPlusPipeline":
        raise ValueError("Expected the official QwenImageEditPlusPipeline Edit-2511 snapshot")
    transformer_config = json.loads((Path(snapshot) / "transformer/config.json").read_text())
    if (not transformer_config.get("zero_cond_t") or transformer_config.get("guidance_embeds")
            or transformer_config.get("in_channels") != 64):
        raise ValueError("Expected non-distilled Edit-2511 transformer: zero_cond_t=True, in_channels=64")
    return snapshot


CONDITIONING = "edit2511_text_only_native_t2i_template_no_reference_v1"


def require_local_output(path, category):
    root = (ROOT / "qwen_edit_pano" / category).resolve()
    if not Path(path).resolve().is_relative_to(root):
        raise ValueError(f"This entry writes only below {root}")


def source_hashes():
    root = Path(__file__).parent
    paths = list(root.glob("*.py"))
    shared = ROOT / "qwen_pano"
    paths += [shared / (name + ".py") for name in
              ("common", "data", "cached_data", "profiles", "losses", "geometry", "inference_state")]
    paths += list((shared / "vendor").rglob("*.py"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}
