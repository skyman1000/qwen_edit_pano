from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(r"D:\proiects\CauPano")
SOURCE_ROOT = PROJECT_ROOT / "src"
if not PROJECT_ROOT.is_dir():
    raise FileNotFoundError(f"CauPano project not found: {PROJECT_ROOT}")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


# 建模接口
from panogen_ft.adapters import (  # noqa: E402,F401
    CircularConv2d,
    GeometryAdapter,
    ReferenceAdapter,
    SphericalAdapter,
    SphericalCausalAdapter,
    StructureReadoutHeads,
)
from panogen_ft.caupano_model import (  # noqa: E402,F401
    CausalAdapterController,
    activate_causal_adapter,
    build_training_module,
    causal_adapter_inputs,
    explicit_flowmatch_loss,
    flowmatch_clean_from_model_call,
    load_adapter_checkpoint,
    load_inference_adapter,
    offline_environment,
    rotated_stochastic_state,
)

# 数据、视角和相机条件
from panogen_ft.dataset import (  # noqa: E402,F401
    BalancedEpochSampler,
    DATA_CATEGORY_WEIGHTS,
    DiffSynthPanoDataset,
    PanoPairDataset,
    completion_prompt,
    data_category,
    sampled_camera_condition,
)
from panogen_ft.manifest import (  # noqa: E402,F401
    MANIFEST_VERSION,
    OOD_DATASETS,
    VIEWS_PER_PANORAMA,
    atomic_write_jsonl,
    make_view_specs,
    prepare_records,
    split_for_record,
    summarize_duplicates,
)
from panogen_ft.pano_ops import (  # noqa: E402,F401
    camera_condition_vector,
    erp_array_to_perspective,
    erp_pixel_rays,
    erp_to_cubemap,
    erp_to_perspective,
    perspective_to_erp_splat,
    pinhole_intrinsics,
    rotation_matrix,
    source_center_panorama,
)

# 损失、推理与验收
from panogen_ft.losses import (  # noqa: E402,F401
    LOSS_WEIGHTS,
    average_loss_components,
    cubemap_reconstruction_losses,
    flowmatch_x0,
    latitude_cosine_weights,
    loss_bundle,
    normal_cosine_loss,
    roll_intervention_loss,
    scale_invariant_depth_loss,
    seam_loss,
    spherical_mean,
    torch_erp_face,
    torch_erp_to_cubemap,
)
from panogen_ft.external_inference import (  # noqa: E402,F401
    ExternalCausalPreprocessor,
    Stage3ExternalInferenceEngine,
    camera_spec_from_estimate,
    hard_preserve_reference,
    normalize_external_reference,
)
from panogen_ft.metrics import (  # noqa: E402,F401
    completion_acceptance_verdict,
    cubemap_edge_consistency,
    evaluate_generated,
    polar_cubemap_metrics,
    rotation_equivariance_error,
    seam_gradient_error,
)


CAUSAL_SOURCE_FILES = (
    "src/panogen_ft/manifest.py",
    "src/panogen_ft/dataset.py",
    "src/panogen_ft/pano_ops.py",
    "src/panogen_ft/adapters.py",
    "src/panogen_ft/losses.py",
    "src/panogen_ft/caupano_model.py",
    "src/panogen_ft/external_inference.py",
    "src/panogen_ft/metrics.py",
    "src/panogen_ft/validation.py",
    "scripts/merge_caupano_sources.py",
    "scripts/prepare_panorama_all.py",
    "scripts/validate_caupano_v3.py",
    "scripts/profile_h200_memory.py",
    "scripts/train_qwen_edit_pano_lora.py",
    "scripts/infer_caupano_external.py",
    "scripts/validate_caupano_stage.py",
    "scripts/select_probe_initialization.py",
    "scripts/summarize_distribution_metrics.py",
)


SCRIPT_COMMANDS = {
    "merge-data": "scripts/merge_caupano_sources.py",
    "prepare": "scripts/prepare_panorama_all.py",
    "validate-data": "scripts/validate_caupano_v3.py",
    "profile": "scripts/profile_h200_memory.py",
    "train": "scripts/train_qwen_edit_pano_lora.py",
    "infer": "scripts/infer_caupano_external.py",
    "validate-model": "scripts/validate_caupano_stage.py",
    "select-probe": "scripts/select_probe_initialization.py",
    "distribution-metrics": "scripts/summarize_distribution_metrics.py",
}


def run_project_script(relative_path: str, arguments: list[str]) -> None:
    script = PROJECT_ROOT / relative_path
    if not script.is_file():
        raise FileNotFoundError(script)
    old_argv = sys.argv
    try:
        sys.argv = [str(script), *arguments]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def print_sources() -> None:
    print("CauPano causal source map:")
    for relative_path in CAUSAL_SOURCE_FILES:
        path = PROJECT_ROOT / relative_path
        state = "OK" if path.is_file() else "MISSING"
        print(f"  [{state}] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("sources", "test", *SCRIPT_COMMANDS),
        help="要执行的因果流水线阶段",
    )
    args, forwarded = parser.parse_known_args()
    if args.command == "sources":
        print_sources()
        return
    if args.command == "test":
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, check=True)
        return
    run_project_script(SCRIPT_COMMANDS[args.command], forwarded)


if __name__ == "__main__":
    main()

if False:  # pragma: no cover - embedded source snapshots

    import hashlib
    import json
    import math
    import os
    import tempfile
    from collections import defaultdict
    from dataclasses import dataclass
    from functools import lru_cache
    from pathlib import Path
    from typing import Any, Iterable, Iterator
    
    import numpy as np
    from PIL import Image
    
    from .pano_ops import pinhole_intrinsics, relative_or_absolute, rotation_matrix
    
    
    MANIFEST_NAME = "panorama-all.jsonl"
    MANIFEST_VERSION = 3
    VIEW_SEED = 3407
    VIEW_SIZE = 768
    VIEWS_PER_PANORAMA = 8
    OOD_DATASETS = frozenset({"zind", "holicity"})
    
    
    def require_master_manifest(path: str | Path) -> Path:
        path = Path(path)
        if path.name != MANIFEST_NAME:
            raise ValueError(f"CauPano only accepts {MANIFEST_NAME}; got {path}")
        return path
    
    
    def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
        path = require_master_manifest(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{line_no}")
                yield value
    
    
    def stable_uint64(*parts: Any) -> int:
        text = "\x1f".join(str(part) for part in parts).encode("utf-8")
        return int.from_bytes(hashlib.sha256(text).digest()[:8], "big", signed=False)
    
    
    def _rng(record_id: str, view_index: int, seed: int) -> np.random.Generator:
        return np.random.Generator(np.random.Philox(stable_uint64(record_id, view_index, seed)))
    
    
    def make_view_specs(
        record_id: str,
        *,
        seed: int = VIEW_SEED,
        view_size: int = VIEW_SIZE,
    ) -> list[dict[str, Any]]:
        aspect_ratios = ((1.0, 0.40), (4.0 / 3.0, 0.30), (16.0 / 9.0, 0.30))
        aspect_values = np.asarray([item[0] for item in aspect_ratios], dtype=np.float64)
        aspect_probabilities = np.asarray([item[1] for item in aspect_ratios], dtype=np.float64)
        specs: list[dict[str, Any]] = []
        for view_index in range(VIEWS_PER_PANORAMA):
            rng = _rng(record_id, view_index, seed)
            yaw_deg = -180.0 + view_index * 45.0 + float(rng.uniform(4.0, 41.0))
            if view_index < 4:
                pitch_deg = float(rng.uniform(-15.0, 15.0))
            elif view_index < 6:
                sign = -1.0 if view_index == 4 else 1.0
                pitch_deg = sign * float(rng.uniform(15.0, 45.0))
            else:
                sign = -1.0 if view_index == 6 else 1.0
                pitch_deg = sign * float(rng.uniform(45.0, 75.0))
            hfov_deg = float(rng.triangular(50.0, 85.0, 120.0))
            if float(rng.random()) < 0.80:
                roll_deg = float(rng.uniform(-5.0, 5.0))
            else:
                roll_deg = float(rng.choice((-1.0, 1.0)) * rng.uniform(5.0, 30.0))
            aspect = float(rng.choice(aspect_values, p=aspect_probabilities))
            width = int(view_size)
            height = max(32, int(round(view_size / aspect / 32.0)) * 32)
            k = pinhole_intrinsics(width, height, hfov_deg)
            r = rotation_matrix(math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg))
            specs.append(
                {
                    "view_id": f"v{view_index}",
                    "projection": "pinhole",
                    "width": width,
                    "height": height,
                    "yaw_deg": yaw_deg,
                    "pitch_deg": pitch_deg,
                    "roll_deg": roll_deg,
                    "hfov_deg": hfov_deg,
                    "intrinsics": k.round(8).tolist(),
                    "camera_to_world": r.round(8).tolist(),
                    "seed": int(stable_uint64(record_id, view_index, seed)),
                }
            )
        return specs
    
    
    def image_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
        return digest.hexdigest()
    
    
    @lru_cache(maxsize=8)
    def _dct_basis(size: int) -> np.ndarray:
        indices = np.arange(size, dtype=np.float32)
        basis = np.cos(np.pi / size * (indices[None, :] + 0.5) * indices[:, None])
        basis[0] *= 1.0 / np.sqrt(2.0)
        return basis * np.sqrt(2.0 / size)
    
    
    def perceptual_hash(path: str | Path, hash_size: int = 16, highfreq_factor: int = 4) -> int:
        """DCT perceptual hash without an imagehash dependency."""
        size = hash_size * highfreq_factor
        with Image.open(path) as image:
            gray = np.asarray(image.convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)
        # Orthonormal 2-D DCT-II implemented with NumPy so manifest preparation
        # does not require the GPU teacher environment.
        basis = _dct_basis(size)
        dct = basis @ gray @ basis.T
        low = dct[:hash_size, :hash_size]
        median = float(np.median(low[1:]))
        bits = (low > median).reshape(-1)
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return value
    
    
    def hamming_distance(a: int, b: int) -> int:
        return int((int(a) ^ int(b)).bit_count())
    
    
    class UnionFind:
        def __init__(self, size: int):
            self.parent = list(range(size))
            self.rank = [0] * size
    
        def find(self, value: int) -> int:
            while self.parent[value] != value:
                self.parent[value] = self.parent[self.parent[value]]
                value = self.parent[value]
            return value
    
        def union(self, a: int, b: int) -> None:
            ra, rb = self.find(a), self.find(b)
            if ra == rb:
                return
            if self.rank[ra] < self.rank[rb]:
                ra, rb = rb, ra
            self.parent[rb] = ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1
    
    
    def _phash_candidates(hashes: list[int], bands: int = 8) -> Iterator[tuple[int, int]]:
        """Yield near-hash candidates using multi-index buckets, not O(N^2)."""
        total_bits = 256
        band_bits = total_bits // bands
        mask = (1 << band_bits) - 1
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        seen: set[tuple[int, int]] = set()
        for index, value in enumerate(hashes):
            for band in range(bands):
                key = (band, (value >> (band * band_bits)) & mask)
                for other in buckets[key]:
                    pair = (other, index)
                    if pair not in seen:
                        seen.add(pair)
                        yield pair
                buckets[key].append(index)
    
    
    def _load_embedding_index(path: str | Path | None) -> dict[str, np.ndarray]:
        if not path:
            return {}
        payload = np.load(path, allow_pickle=False)
        ids = payload["ids"]
        features = payload["features"]
        if features.ndim != 3 or features.shape[1] != 4:
            raise ValueError("Embedding index must contain features shaped [N,4,D]")
        features = features.astype(np.float32)
        features /= np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-8)
        return {str(record_id): feature for record_id, feature in zip(ids.tolist(), features)}
    
    
    def cyclic_embedding_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Maximum mean cosine similarity over four cyclic yaw alignments."""
        if a.shape != b.shape or a.ndim != 2 or a.shape[0] != 4:
            raise ValueError("Expected matching [4,D] tangent-view embeddings")
        return max(float(np.mean(np.sum(a * np.roll(b, shift, axis=0), axis=-1))) for shift in range(4))
    
    
    @dataclass
    class PreparedRecord:
        record: dict[str, Any]
        sha256: str | None
        phash: int | None
    
    
    def _inspect_record(record: dict[str, Any], image_root: Path, strict_images: bool) -> PreparedRecord:
        item = dict(record)
        record_id = str(item.get("id") or "").strip()
        if not record_id:
            raise ValueError("Every panorama-all record must have a non-empty id")
        target_path = relative_or_absolute(str(item["target_path"]), image_root)
        quality = dict(item.get("quality") or {})
        sha: str | None = None
        phash_value: int | None = None
        if target_path.is_file():
            with Image.open(target_path) as image:
                width, height = image.size
            if height <= 0 or abs(width / height - 2.0) > 0.03:
                quality["valid_erp"] = False
                if strict_images:
                    raise ValueError(f"Not a 2:1 ERP image: {target_path} ({width}x{height})")
            else:
                quality["valid_erp"] = True
            item["width"], item["height"] = int(width), int(height)
            item["native_resolution"] = [int(width), int(height)]
            sha = image_sha256(target_path)
            phash_value = perceptual_hash(target_path)
            quality["image_exists"] = True
            quality["sha256"] = sha
            quality["phash256"] = f"{phash_value:064x}"
        else:
            quality["image_exists"] = False
            if strict_images:
                raise FileNotFoundError(target_path)
            item.setdefault("width", None)
            item.setdefault("height", None)
            item.setdefault("native_resolution", [item.get("width"), item.get("height")])
        item.setdefault("source_dataset", "unknown")
        item.setdefault("source_scene_id", str(item.get("split_group") or record_id))
        item.setdefault("scene_type", "indoor")
        item.setdefault("task", "completion")
        item.setdefault("prompt_scope", "target" if str(item.get("prompt") or "").strip() else "generic")
        item["quality_flags"] = sorted(
            str(name) for name, value in quality.items() if isinstance(value, bool) and not value
        )
        item["quality"] = quality
        item["camera"] = {
            "target_projection": "equirectangular",
            "target_convention": "source_centered",
            "view_specs": make_view_specs(record_id),
        }
        item.setdefault(
            "causal_targets",
            {
                "version": "v1",
                "shard": None,
                "key": record_id,
                "valid": {"depth": False, "normal": False, "semantic": False, "layout": False, "camera": False},
            },
        )
        item["manifest_version"] = MANIFEST_VERSION
        return PreparedRecord(item, sha, phash_value)
    
    
    def _assign_groups(
        items: list[PreparedRecord],
        embedding_index: dict[str, np.ndarray],
        phash_threshold: int,
        embedding_threshold: float,
        show_progress: bool = False,
    ) -> None:
        def progress(iterable: Iterable[Any], *, desc: str, total: int | None = None) -> Iterable[Any]:
            if not show_progress:
                return iterable
            from tqdm.auto import tqdm
    
            return tqdm(iterable, desc=desc, total=total, unit="item", mininterval=2.0, dynamic_ncols=True)
    
        union_find = UnionFind(len(items))
        exact: dict[str, int] = {}
        source_scenes: dict[tuple[str, str], int] = {}
        phash_indices: list[int] = []
        phashes: list[int] = []
        for index, item in enumerate(progress(items, desc="Indexing image hashes", total=len(items))):
            scene_key = (
                str(item.record.get("source_dataset") or "unknown").lower(),
                str(item.record.get("source_scene_id") or item.record["id"]),
            )
            if scene_key in source_scenes:
                union_find.union(index, source_scenes[scene_key])
            else:
                source_scenes[scene_key] = index
            if item.sha256:
                if item.sha256 in exact:
                    union_find.union(index, exact[item.sha256])
                else:
                    exact[item.sha256] = index
            if item.phash is not None:
                phash_indices.append(index)
                phashes.append(item.phash)
        for local_a, local_b in progress(_phash_candidates(phashes), desc="Comparing pHash candidates"):
            if hamming_distance(phashes[local_a], phashes[local_b]) <= phash_threshold:
                union_find.union(phash_indices[local_a], phash_indices[local_b])
    
        # Optional DAP/DINOv3 tangent embeddings. FAISS keeps 150k matching tractable.
        embedding_rows = [(index, embedding_index.get(str(item.record["id"]))) for index, item in enumerate(items)]
        embedding_rows = [(index, value) for index, value in embedding_rows if value is not None]
        if embedding_rows:
            if show_progress:
                print(f"Matching {len(embedding_rows)} embedding rows with FAISS", flush=True)
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError("faiss is required when --embedding-index is supplied") from exc
            flat = np.stack([value.reshape(-1) for _, value in embedding_rows]).astype(np.float32)
            flat /= np.maximum(np.linalg.norm(flat, axis=-1, keepdims=True), 1e-8)
            index = faiss.IndexFlatIP(flat.shape[1])
            index.add(flat)
            _, neighbours = index.search(flat, min(20, len(flat)))
            for row, candidates in enumerate(neighbours):
                original_a, feature_a = embedding_rows[row]
                for candidate in candidates[1:]:
                    original_b, feature_b = embedding_rows[int(candidate)]
                    if cyclic_embedding_similarity(feature_a, feature_b) >= embedding_threshold:
                        union_find.union(original_a, original_b)
    
        members: dict[int, list[str]] = defaultdict(list)
        for index, item in enumerate(progress(items, desc="Assigning group IDs", total=len(items))):
            members[union_find.find(index)].append(str(item.record["id"]))
        group_names = {
            root: "g_" + hashlib.sha256("\x1f".join(sorted(ids)).encode("utf-8")).hexdigest()[:16]
            for root, ids in members.items()
        }
        for index, item in enumerate(progress(items, desc="Writing group IDs", total=len(items))):
            item.record["group_id"] = group_names[union_find.find(index)]
    
    
    def split_for_record(
        record_id: str,
        scene_type: str,
        seed: int = VIEW_SEED,
        source_dataset: str = "",
    ) -> str:
        """Deterministically split a duplicate/scene group, reserving named OOD sources."""
        del scene_type  # Outdoor scenes are part of training; OOD is dataset-level.
        if str(source_dataset).strip().lower() in OOD_DATASETS:
            return "test_ood"
        value = stable_uint64(record_id, seed, "group-split-v3") / float(2**64)
        if value < 0.90:
            return "train"
        if value < 0.95:
            return "val"
        return "test_id"
    
    
    def summarize_duplicates(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Summarize exact and perceptual duplicate groups without affecting splits."""
        record_list = list(records)
    
        def counts_for(key_counts: dict[str, int]) -> dict[str, int]:
            duplicate_sizes = [count for count in key_counts.values() if count > 1]
            return {
                "groups": len(duplicate_sizes),
                "participating_records": sum(duplicate_sizes),
                "redundant_records": sum(count - 1 for count in duplicate_sizes),
            }
    
        sha_counts: dict[str, int] = defaultdict(int)
        group_counts: dict[str, int] = defaultdict(int)
        group_splits: dict[str, set[str]] = defaultdict(set)
        for record in record_list:
            sha256 = str((record.get("quality") or {}).get("sha256") or "").strip()
            if sha256:
                sha_counts[sha256] += 1
            group_id = str(record.get("group_id") or "").strip()
            if group_id:
                group_counts[group_id] += 1
                group_splits[group_id].add(str(record.get("split") or ""))
    
        return {
            "exact": counts_for(sha_counts),
            "exact_or_near": counts_for(group_counts),
            "cross_split_groups": sum(
                group_counts[group_id] > 1 and len(splits) > 1 for group_id, splits in group_splits.items()
            ),
        }
    
    
    def prepare_records(
        records: Iterable[dict[str, Any]],
        *,
        image_root: str | Path,
        strict_images: bool = False,
        embedding_index_path: str | Path | None = None,
        phash_threshold: int = 6,
        embedding_threshold: float = 0.985,
        show_progress: bool = False,
    ) -> list[dict[str, Any]]:
        image_root = Path(image_root)
        record_list = list(records)
        record_iterator: Iterable[dict[str, Any]] = record_list
        if show_progress:
            from tqdm.auto import tqdm
    
            record_iterator = tqdm(
                record_list,
                desc="Inspecting panoramas",
                total=len(record_list),
                unit="image",
                mininterval=2.0,
                dynamic_ncols=True,
            )
        items = [_inspect_record(record, image_root, strict_images) for record in record_iterator]
        ids = [str(item.record["id"]) for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("panorama-all.jsonl contains duplicate id values")
        if show_progress:
            print("Image inspection complete; grouping exact and near duplicates", flush=True)
        _assign_groups(
            items,
            _load_embedding_index(embedding_index_path),
            phash_threshold,
            embedding_threshold,
            show_progress=show_progress,
        )
        if show_progress:
            print("Duplicate grouping complete; assigning dataset splits by group ID", flush=True)
        grouped: dict[str, list[PreparedRecord]] = defaultdict(list)
        for item in items:
            grouped[str(item.record["group_id"])].append(item)
        for group_id, members in grouped.items():
            sources = {str(member.record.get("source_dataset") or "").lower() for member in members}
            split = "test_ood" if sources & OOD_DATASETS else split_for_record(group_id, "", seed=VIEW_SEED)
            for member in members:
                member.record["split"] = split
        return [item.record for item in items]
    
    
    def atomic_write_jsonl(records: Iterable[dict[str, Any]], destination: str | Path) -> None:
        destination = require_master_manifest(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    
    import random
    import math
    from collections import defaultdict
    from pathlib import Path
    from typing import Any, Iterator, Mapping
    
    import numpy as np
    import torch
    from PIL import Image, ImageEnhance
    from torch.utils.data import Dataset, Sampler
    
    from .causal_shards import CausalShardReader
    from .manifest import MANIFEST_VERSION, iter_jsonl, require_master_manifest, stable_uint64
    from .pano_ops import (
        camera_condition_vector,
        erp_to_perspective,
        fit_equirectangular,
        load_rgb,
        pinhole_intrinsics,
        relative_or_absolute,
        rotation_matrix,
        source_center_panorama,
    )
    
    
    DEFAULT_PROMPT = (
        "Expand the reference view into a photorealistic, source-centered 360-degree equirectangular panorama "
        "with consistent geometry and a seamless longitude boundary."
    )
    
    DATA_CATEGORY_WEIGHTS = {
        "insta360": 0.25,
        "real_indoor": 0.25,
        "real_outdoor": 0.30,
        "rendered": 0.15,
        "hdri": 0.05,
    }
    
    
    def data_category(record: Mapping[str, Any]) -> str:
        explicit = str(record.get("data_category") or "").strip().lower()
        if explicit:
            return explicit
        source = str(record.get("source_dataset") or "unknown").lower()
        if "insta" in source or "dap" in source:
            return "insta360"
        if any(name in source for name in ("structured", "hm3d", "replica", "gibson", "3dfront")):
            return "rendered"
        if any(name in source for name in ("polyhaven", "laval", "hdri", "hdr")):
            return "hdri"
        if str(record.get("scene_type") or "indoor").lower() == "outdoor":
            return "real_outdoor"
        return "real_indoor"
    
    
    class BalancedEpochSampler(Sampler[int]):
        """Deterministic category-balanced sampling with replacement per epoch."""
    
        def __init__(
            self,
            dataset: "PanoPairDataset",
            epoch_size: int,
            category_weights: Mapping[str, float] | None = None,
            seed: int = 3407,
            require_all_categories: bool = True,
            max_source_fraction: float = 0.25,
        ) -> None:
            self.dataset = dataset
            self.epoch_size = int(epoch_size)
            self.weights = {str(k): float(v) for k, v in (category_weights or DATA_CATEGORY_WEIGHTS).items()}
            self.seed = int(seed)
            self.require_all_categories = bool(require_all_categories)
            self.max_source_fraction = float(max_source_fraction)
            if self.epoch_size <= 0 or not self.weights or any(value < 0 for value in self.weights.values()):
                raise ValueError("BalancedEpochSampler requires a positive epoch size and non-negative weights")
            self.indices: dict[str, list[int]] = defaultdict(list)
            self.source_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
            for index, record in enumerate(dataset.records):
                category = data_category(record)
                source = str(record.get("source_dataset") or "unknown").strip().lower()
                self.indices[category].append(index)
                self.source_indices[(category, source)].append(index)
            missing = sorted(name for name, value in self.weights.items() if value > 0 and not self.indices.get(name))
            if missing and self.require_all_categories:
                raise ValueError(f"Balanced training manifest lacks required categories: {missing}")
            self.weights = {name: value for name, value in self.weights.items() if value > 0 and self.indices.get(name)}
            total = sum(self.weights.values())
            self.weights = {name: value / total for name, value in self.weights.items()}
            if not (0.0 < self.max_source_fraction <= 1.0):
                raise ValueError("max_source_fraction must be in (0,1]")
    
        def __len__(self) -> int:
            return self.epoch_size
    
        def __iter__(self) -> Iterator[int]:
            rng = random.Random(stable_uint64(self.seed, self.dataset.epoch, "balanced-epoch"))
            names = sorted(self.weights)
            counts = {name: int(math.floor(self.epoch_size * self.weights[name])) for name in names}
            for name in sorted(names, key=lambda item: self.weights[item], reverse=True)[: self.epoch_size - sum(counts.values())]:
                counts[name] += 1
            sampled: list[int] = []
            source_counts: dict[str, int] = defaultdict(int)
            source_cap = int(math.floor(self.epoch_size * self.max_source_fraction))
            for name in names:
                sources = sorted(source for category, source in self.source_indices if category == name)
                rng.shuffle(sources)
                for draw in range(counts[name]):
                    eligible = [source for source in sources if source_counts[source] < source_cap]
                    if not eligible:
                        raise ValueError(
                            f"Cannot draw category {name!r} while limiting every source_dataset to "
                            f"{self.max_source_fraction:.0%}; add another source or lower its category weight"
                        )
                    source = min(eligible, key=lambda item: (source_counts[item], item))
                    sampled.append(rng.choice(self.source_indices[(name, source)]))
                    source_counts[source] += 1
            rng.shuffle(sampled)
            return iter(sampled)
    
    
    def prompt_from_record(record: dict[str, Any], default: str = DEFAULT_PROMPT) -> str:
        prompt = str(record.get("prompt") or "").strip()
        return prompt or default
    
    
    def completion_prompt(record: dict[str, Any], view_index: int, epoch: int) -> tuple[str, str]:
        """Apply the 50/35/15 generic/reference/target prompt policy."""
        value = stable_uint64(record.get("id"), view_index, epoch, "prompt-scope") / float(2**64)
        if value < 0.50:
            return DEFAULT_PROMPT, "generic"
        reference_prompts = record.get("reference_prompts") or record.get("view_captions") or {}
        if isinstance(reference_prompts, list):
            reference_prompt = str(reference_prompts[view_index]) if view_index < len(reference_prompts) else ""
        elif isinstance(reference_prompts, dict):
            reference_prompt = str(reference_prompts.get(f"v{view_index}") or "")
        else:
            reference_prompt = ""
        if value < 0.85:
            prompt = DEFAULT_PROMPT if not reference_prompt.strip() else f"{DEFAULT_PROMPT} Reference content: {reference_prompt.strip()}"
            return prompt, "reference" if reference_prompt.strip() else "generic"
        return f"{DEFAULT_PROMPT} Target description: {prompt_from_record(record)}", "target"
    
    
    def sampled_camera_condition(record_id: str, spec: dict[str, Any], epoch: int, view_index: int) -> dict[str, Any]:
        """Sample exact/noisy/unknown camera conditioning at 50/35/15 percent."""
        rng = random.Random(stable_uint64(record_id, epoch, view_index, "camera-condition"))
        value = rng.random()
        conditioned = dict(spec)
        conditioned["yaw_deg"] = 0.0
        if value < 0.50:
            mode = "exact"
            quality = {"camera_valid": True, "fov_valid": True, "gravity_valid": True, "fov_std_deg": 0.0, "gravity_std_deg": 0.0}
        elif value < 0.85:
            mode = "noisy"
            conditioned["hfov_deg"] = float(min(140.0, max(30.0, float(spec["hfov_deg"]) + rng.gauss(0.0, 5.0))))
            conditioned["pitch_deg"] = float(min(80.0, max(-80.0, float(spec["pitch_deg"]) + rng.gauss(0.0, 3.0))))
            conditioned["roll_deg"] = float(min(45.0, max(-45.0, float(spec.get("roll_deg", 0.0)) + rng.gauss(0.0, 2.0))))
            conditioned["intrinsics"] = pinhole_intrinsics(
                int(spec["width"]), int(spec["height"]), conditioned["hfov_deg"]
            ).round(8).tolist()
            conditioned["camera_to_world"] = rotation_matrix(
                0.0, math.radians(conditioned["pitch_deg"]), math.radians(conditioned["roll_deg"])
            ).round(8).tolist()
            quality = {"camera_valid": True, "fov_valid": True, "gravity_valid": True, "fov_std_deg": 5.0, "gravity_std_deg": 3.0}
        else:
            mode = "unknown"
            conditioned.update(hfov_deg=90.0, pitch_deg=0.0, roll_deg=0.0)
            quality = {"camera_valid": False, "fov_valid": False, "gravity_valid": False, "fov_std_deg": 90.0, "gravity_std_deg": 90.0}
        return {
            "mode": mode,
            "spec": conditioned,
            "quality": quality,
            "vector": camera_condition_vector(conditioned, quality, mode=mode),
        }
    
    
    def image_to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()
    
    
    def _appearance_intervention(image: Image.Image, seed: int) -> Image.Image:
        rng = random.Random(seed)
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.80, 1.20))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.75, 1.25))
        return image
    
    
    class PanoPairDataset(Dataset):
        """Render reference NFoVs online; target causal maps are supervision only."""
    
        def __init__(
            self,
            manifest_path: str | Path,
            image_root: str | Path = ".",
            target_width: int = 1024,
            target_height: int = 512,
            split: str = "train",
            min_native_width: int = 0,
            caption_dropout_probability: float = 0.0,
            factor_dropout_probability: float = 0.15,
            appearance_intervention_probability: float = 0.0,
            causal_root: str | Path | None = None,
            require_causal: bool = False,
            use_causal: bool = True,
            return_intervention_pair: bool = False,
        ):
            self.manifest_path = require_master_manifest(manifest_path)
            self.image_root = self.manifest_path.parent if str(image_root) in {"", "."} else Path(image_root)
            self.records = [
                record
                for record in iter_jsonl(self.manifest_path)
                if record.get("split") == split
                and int(record.get("width") or 0) >= int(min_native_width)
                and (record.get("quality") or {}).get("image_exists", True) is not False
                and (record.get("quality") or {}).get("valid_erp", True) is not False
            ]
            incompatible = [
                str(record.get("id"))
                for record in self.records
                if int(record.get("manifest_version", 0)) != MANIFEST_VERSION
            ]
            if incompatible:
                raise ValueError(
                    f"CauPano repair requires manifest_version={MANIFEST_VERSION}; "
                    f"re-run scripts/prepare_panorama_all.py (first IDs: {incompatible[:3]})"
                )
            self.target_width, self.target_height = int(target_width), int(target_height)
            if self.target_width != self.target_height * 2:
                raise ValueError("ERP output resolution must be exactly 2:1")
            self.caption_dropout_probability = float(caption_dropout_probability)
            self.factor_dropout_probability = float(factor_dropout_probability)
            self.appearance_intervention_probability = float(appearance_intervention_probability)
            self.causal_root = Path(causal_root) if causal_root else self.image_root
            self.require_causal = bool(require_causal)
            self.use_causal = bool(use_causal)
            self.return_intervention_pair = bool(return_intervention_pair)
            self.epoch = 0
            self._shards: CausalShardReader | None = None
            if not self.records:
                raise ValueError(f"No records for split={split!r}, min_native_width={min_native_width} in {self.manifest_path}")
    
        def __len__(self) -> int:
            return len(self.records)
    
        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)
    
        def _view_index(self, record: dict[str, Any]) -> int:
            specs = record.get("camera", {}).get("view_specs") or []
            if len(specs) != 8:
                raise ValueError(f"{record.get('id')} must contain exactly eight deterministic view_specs")
            return stable_uint64(record["id"], self.epoch, "online-view") % len(specs)
    
        def _causal(self, record: dict[str, Any], view_index: int) -> dict[str, Any]:
            if not self.use_causal:
                return {"predicted": {}, "supervision": {}, "valid": {name: False for name in ("depth", "normal", "semantic", "layout", "camera")}}
            target = record.get("causal_targets") or {}
            shard, key = target.get("shard"), target.get("key")
            if not shard:
                if self.require_causal:
                    raise FileNotFoundError(f"Missing causal shard for {record['id']}")
                return {"predicted": {}, "supervision": {}, "valid": {name: False for name in ("depth", "normal", "semantic", "layout", "camera")}}
            if self._shards is None:
                self._shards = CausalShardReader(self.causal_root)
            arrays, metadata = self._shards.read(shard, key)
            # Generator inputs contain only predictions made from the reference view.
            predicted = {
                name: arrays[name]
                for name in (
                    f"view_{view_index}_depth",
                    f"view_{view_index}_depth_confidence",
                    f"view_{view_index}_normal",
                    f"view_{view_index}_semantic",
                    f"view_{view_index}_structure",
                    f"view_{view_index}_camera",
                    f"view_{view_index}_intrinsics",
                )
                if name in arrays
            }
            # Full ERP labels are explicitly namespaced as supervision and must not
            # be passed into the Qwen forward input builder.
            supervision_names = {
                "erp_depth",
                "erp_range_mask",
                "erp_normal",
                "erp_structure",
                "layout_ceiling_boundary",
                "layout_floor_boundary",
            }
            supervision = {name: value for name, value in arrays.items() if name in supervision_names}
            valid = dict(metadata.get("valid") or target.get("valid") or {})
            for factor in ("depth", "normal", "camera"):
                if self.factor_dropout_probability > 0 and random.random() < self.factor_dropout_probability:
                    valid[factor] = False
            return {"predicted": predicted, "supervision": supervision, "valid": valid, "quality": metadata.get("quality", {})}
    
        def sample_pil(self, idx: int, view_index_override: int | None = None) -> dict[str, Any]:
            record = self.records[idx]
            source = load_rgb(relative_or_absolute(record["target_path"], self.image_root))
            view_index = int(self._view_index(record) if view_index_override is None else view_index_override)
            spec = record["camera"]["view_specs"][view_index]
            reference = erp_to_perspective(
                source,
                spec["yaw_deg"],
                spec["pitch_deg"],
                spec["hfov_deg"],
                int(spec["width"]),
                int(spec["height"]),
                float(spec.get("roll_deg", 0.0)),
            )
            if self.appearance_intervention_probability > 0 and random.random() < self.appearance_intervention_probability:
                reference = _appearance_intervention(reference, stable_uint64(record["id"], self.epoch, "appearance"))
            target = fit_equirectangular(source_center_panorama(source, spec["yaw_deg"]), self.target_width, self.target_height)
            prompt, prompt_scope = completion_prompt(record, view_index, self.epoch)
            if self.caption_dropout_probability > 0 and random.random() < self.caption_dropout_probability:
                prompt, prompt_scope = DEFAULT_PROMPT, "generic"
            camera_condition = sampled_camera_condition(str(record["id"]), spec, self.epoch, view_index)
            return {
                "image": target,
                "edit_image": reference,
                "prompt": prompt,
                "prompt_scope": prompt_scope,
                "task": str(record.get("task") or "completion"),
                "negative_prompt": "",
                "record": record,
                "view_index": view_index,
                "view_spec": spec,
                "camera_condition": camera_condition,
                "reference_valid_mask": np.ones((reference.height, reference.width), dtype=bool),
                "causal": self._causal(record, view_index),
            }
    
        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = self.sample_pil(idx)
            intervention = None
            if self.return_intervention_pair:
                specs = item["record"]["camera"]["view_specs"]
                intervention = self.sample_pil(idx, (item["view_index"] + 1) % len(specs))
            return {
                **item,
                "target": image_to_tensor(item["image"]),
                "condition": image_to_tensor(item["edit_image"]),
                "target_path": str(item["record"]["target_path"]),
                "intervention": intervention,
            }
    
    
    class DiffSynthPanoDataset(PanoPairDataset):
        """PIL interface expected by DiffSynth's Qwen-Image training pipeline."""
    
        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = self.sample_pil(idx)
            if self.return_intervention_pair:
                specs = item["record"]["camera"]["view_specs"]
                item["intervention"] = self.sample_pil(idx, (item["view_index"] + 1) % len(specs))
            # Match the official Qwen-Image-Edit training keys exactly.
            return {"image": item["image"], "edit_image": item["edit_image"], "prompt": item["prompt"], "metadata": item}
    
    
    def collate_pano_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "target": torch.stack([item["target"] for item in items]),
            "condition": torch.stack([item["condition"] for item in items]),
            "prompt": [item["prompt"] for item in items],
            "record": [item["record"] for item in items],
            "view_spec": [item["view_spec"] for item in items],
            "causal": [item["causal"] for item in items],
        }
    # END SOURCE: src/panogen_ft/dataset.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: src/panogen_ft/pano_ops.py
    # ------------------------------------------------------------------------------------------
    """Panorama image geometry helpers.
    
    The training code uses equirectangular panoramas as the canonical target
    format. These helpers intentionally avoid project-specific dependencies so the
    fine-tuning package can run on a clean GPU server.
    """
    
    # from __future__ import annotations
    
    import math
    from pathlib import Path
    from typing import Iterable
    
    try:
        import cv2
    except ImportError:  # Manifest-only CPU environments do not need OpenCV.
        cv2 = None
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps
    
    
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".exr", ".hdr"}
    
    
    def _require_cv2():
        if cv2 is None:
            raise RuntimeError("OpenCV is required for ERP projection; install opencv-python-headless")
        return cv2
    
    
    def is_image_path(path: str | Path) -> bool:
        return Path(path).suffix.lower() in IMAGE_EXTENSIONS
    
    
    def is_equirectangular(width: int, height: int, tolerance: float = 0.03) -> bool:
        if height <= 0:
            return False
        return abs((width / height) - 2.0) <= tolerance
    
    
    def list_images(root: str | Path) -> list[Path]:
        root = Path(root)
        if root.is_file() and is_image_path(root):
            return [root]
        return sorted(path for path in root.rglob("*") if path.is_file() and is_image_path(path))
    
    
    def load_rgb(path: str | Path) -> Image.Image:
        return Image.open(path).convert("RGB")
    
    
    def fit_equirectangular(image: Image.Image, width: int, height: int) -> Image.Image:
        """Resize/crop an image to the target 2:1 panorama canvas."""
        if not is_equirectangular(width, height, tolerance=0.001):
            raise ValueError(f"Target panorama size must be 2:1, got {width}x{height}")
        return ImageOps.fit(image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS)
    
    
    def make_blank_condition(width: int, height: int, color: int = 127) -> Image.Image:
        return Image.new("RGB", (width, height), (color, color, color))
    
    
    def make_low_frequency_condition(image: Image.Image, width: int, height: int, blur_radius: int = 24) -> Image.Image:
        image = fit_equirectangular(image, width, height)
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    
    def rotation_matrix(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
        """Return camera-to-world rotation using yaw(Y), pitch(X), roll(Z).
    
        Angles are in radians.  The convention is shared by manifest generation,
        online viewport rendering, and the camera intervention tests.
        """
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
    
        ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
        rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return ry @ rx @ rz
    
    
    def erp_to_perspective(
        panorama: Image.Image,
        yaw_deg: float,
        pitch_deg: float,
        fov_deg: float,
        out_width: int,
        out_height: int,
        roll_deg: float = 0.0,
    ) -> Image.Image:
        """Render a perspective viewport from an equirectangular panorama."""
        cv = _require_cv2()
        pano = np.asarray(panorama.convert("RGB"))
        pano_h, pano_w = pano.shape[:2]
    
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        hfov = math.radians(fov_deg)
        vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (out_height / max(1, out_width)))
    
        xs = np.linspace(-math.tan(hfov / 2.0), math.tan(hfov / 2.0), out_width, dtype=np.float64)
        ys = np.linspace(math.tan(vfov / 2.0), -math.tan(vfov / 2.0), out_height, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.ones_like(xx)
    
        dirs = np.stack([xx, yy, zz], axis=-1)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs = dirs @ rotation_matrix(yaw, pitch, math.radians(roll_deg)).T
    
        lon = np.arctan2(dirs[..., 0], dirs[..., 2])
        lat = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0))
        map_x = ((lon / (2.0 * math.pi) + 0.5) * pano_w).astype(np.float32)
        map_y = ((0.5 - lat / math.pi) * pano_h).astype(np.float32)
        map_x = np.mod(map_x, pano_w).astype(np.float32)
        map_y = np.clip(map_y, 0, pano_h - 1).astype(np.float32)
    
        viewport = cv.remap(pano, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP)
        return Image.fromarray(viewport)
    
    
    def erp_array_to_perspective(
        panorama: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        fov_deg: float,
        out_width: int,
        out_height: int,
        interpolation: int = 1,
        roll_deg: float = 0.0,
    ) -> np.ndarray:
        """Project a numeric ERP array without quantising it through PIL."""
        pano = np.asarray(panorama)
        pano_h, pano_w = pano.shape[:2]
        hfov = math.radians(float(fov_deg))
        vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * out_height / max(1, out_width))
        xs = np.linspace(-math.tan(hfov / 2.0), math.tan(hfov / 2.0), out_width, dtype=np.float64)
        ys = np.linspace(math.tan(vfov / 2.0), -math.tan(vfov / 2.0), out_height, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
        rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-12)
        rays = rays @ rotation_matrix(
            math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg)
        ).T
        lon = np.arctan2(rays[..., 0], rays[..., 2])
        lat = np.arcsin(np.clip(rays[..., 1], -1.0, 1.0))
        map_x = np.mod((lon / (2.0 * math.pi) + 0.5) * pano_w, pano_w).astype(np.float32)
        map_y = np.clip((0.5 - lat / math.pi) * pano_h, 0, pano_h - 1).astype(np.float32)
        cv = _require_cv2()
        return cv.remap(pano, map_x, map_y, interpolation=interpolation, borderMode=cv.BORDER_WRAP)
    
    
    def erp_face(
        panorama: np.ndarray,
        face: str,
        face_size: int,
    ) -> np.ndarray:
        """Sample one cubemap face from an equirectangular panorama."""
        cv = _require_cv2()
        h, w = panorama.shape[:2]
        uv = np.linspace(-1.0, 1.0, face_size, dtype=np.float64)
        xx, yy = np.meshgrid(uv, -uv)
    
        if face == "front":
            dirs = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
        elif face == "right":
            dirs = np.stack([np.ones_like(xx), yy, -xx], axis=-1)
        elif face == "back":
            dirs = np.stack([-xx, yy, -np.ones_like(xx)], axis=-1)
        elif face == "left":
            dirs = np.stack([-np.ones_like(xx), yy, xx], axis=-1)
        elif face == "up":
            dirs = np.stack([xx, np.ones_like(xx), -yy], axis=-1)
        elif face == "down":
            dirs = np.stack([xx, -np.ones_like(xx), yy], axis=-1)
        else:
            raise ValueError(f"Unknown cubemap face: {face}")
    
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        lon = np.arctan2(dirs[..., 0], dirs[..., 2])
        lat = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0))
        map_x = ((lon / (2.0 * math.pi) + 0.5) * w).astype(np.float32)
        map_y = ((0.5 - lat / math.pi) * h).astype(np.float32)
        return cv.remap(panorama, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP)
    
    
    def erp_to_cubemap(image: Image.Image | np.ndarray, face_size: int = 512) -> dict[str, np.ndarray]:
        pano = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
        return {name: erp_face(pano, name, face_size) for name in ("front", "right", "back", "left", "up", "down")}
    
    
    def horizontal_roll_image(image: Image.Image, pixels: int) -> Image.Image:
        arr = np.asarray(image.convert("RGB"))
        return Image.fromarray(np.roll(arr, shift=int(pixels), axis=1))
    
    
    def source_centered_roll_pixels(yaw_deg: float, panorama_width: int) -> int:
        """Roll an ERP so the given camera yaw appears at the image centre."""
        return int(round(-float(yaw_deg) * int(panorama_width) / 360.0))
    
    
    def source_center_panorama(image: Image.Image, yaw_deg: float) -> Image.Image:
        """Create the canonical target used by image-to-panorama training."""
        return horizontal_roll_image(image, source_centered_roll_pixels(yaw_deg, image.width))
    
    
    def pinhole_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
        """Return a 3x3 centred pinhole intrinsic matrix."""
        hfov = math.radians(float(hfov_deg))
        fx = 0.5 * float(width) / math.tan(hfov / 2.0)
        fy = fx
        return np.array(
            [[fx, 0.0, (float(width) - 1.0) / 2.0], [0.0, fy, (float(height) - 1.0) / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    
    
    def erp_pixel_rays(width: int, height: int) -> np.ndarray:
        """Return unit camera rays for every pixel of a 2:1 ERP image."""
        x = (np.arange(width, dtype=np.float64) + 0.5) / float(width)
        y = (np.arange(height, dtype=np.float64) + 0.5) / float(height)
        lon = (x - 0.5) * (2.0 * math.pi)
        lat = (0.5 - y) * math.pi
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        cos_lat = np.cos(lat_grid)
        return np.stack(
            [np.sin(lon_grid) * cos_lat, np.sin(lat_grid), np.cos(lon_grid) * cos_lat],
            axis=-1,
        ).astype(np.float32)
    
    
    def pinhole_pixel_rays(width: int, height: int, hfov_deg: float) -> np.ndarray:
        """Return unit camera rays for a centred pinhole image."""
        k = pinhole_intrinsics(width, height, hfov_deg)
        xs, ys = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        rays = np.stack([(xs - k[0, 2]) / k[0, 0], (k[1, 2] - ys) / k[1, 1], np.ones_like(xs)], axis=-1)
        rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-12)
        return rays.astype(np.float32)
    
    
    def camera_condition_vector(
        spec: dict,
        quality: dict | None = None,
        *,
        mode: str = "auto",
    ) -> np.ndarray:
        """Encode camera geometry identically for training and deployment.
    
        Unknown cameras use zero numeric features and explicit zero validity bits;
        this is intentionally different from a valid zero-degree camera.
        """
        mode = str(mode).lower()
        if mode not in {"auto", "override", "exact", "noisy", "unknown"}:
            raise ValueError(f"Unsupported camera mode: {mode}")
        if mode == "unknown":
            return np.zeros(10, dtype=np.float32)
        quality = dict(quality or {})
        pitch = math.radians(float(spec.get("pitch_deg", 0.0)))
        roll = math.radians(float(spec.get("roll_deg", 0.0)))
        hfov = math.radians(float(np.clip(float(spec.get("hfov_deg", 90.0)), 1.0, 179.0)))
        width = max(1.0, float(spec.get("width", 1)))
        height = max(1.0, float(spec.get("height", 1)))
        manual = mode in {"override", "exact"}
        camera_valid = bool(quality.get("camera_valid", manual or mode == "noisy"))
        fov_valid = bool(quality.get("fov_valid", camera_valid))
        gravity_valid = bool(quality.get("gravity_valid", camera_valid))
        fov_std = 0.0 if manual else float(np.clip(float(quality.get("fov_std_deg", 0.0)), 0.0, 90.0))
        gravity_std = 0.0 if manual else float(
            np.clip(float(quality.get("gravity_std_deg", 0.0)), 0.0, 90.0)
        )
        return np.asarray(
            [
                math.sin(pitch),
                math.cos(pitch),
                math.sin(roll),
                math.cos(roll),
                float(np.clip(math.log(max(math.tan(hfov / 2.0), 1e-6)), -3.0, 3.0)),
                float(np.clip(math.log(width / height), -3.0, 3.0)),
                min(fov_std / 30.0, 3.0),
                min(gravity_std / 30.0, 3.0),
                float(fov_valid),
                float(gravity_valid),
            ],
            dtype=np.float32,
        )
    
    
    def depth_to_normals(depth: np.ndarray, rays: np.ndarray, wrap_x: bool) -> np.ndarray:
        """Derive geometry-consistent normals from depth and camera rays.
    
        Invalid depth values produce zero normals.  ERP longitude uses circular
        neighbours while pinhole images use replicated border neighbours.
        """
        depth = np.asarray(depth, dtype=np.float32)
        rays = np.asarray(rays, dtype=np.float32)
        if depth.ndim != 2 or rays.shape != (*depth.shape, 3):
            raise ValueError(f"Expected depth HxW and rays HxWx3, got {depth.shape} and {rays.shape}")
        points = rays * depth[..., None]
        if wrap_x:
            dx = np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1)
        else:
            padded = np.pad(points, ((0, 0), (1, 1), (0, 0)), mode="edge")
            dx = padded[:, 2:] - padded[:, :-2]
        padded_y = np.pad(points, ((1, 1), (0, 0), (0, 0)), mode="edge")
        dy = padded_y[:-2] - padded_y[2:]
        normals = np.cross(dx, dy)
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.maximum(norm, 1e-8)
        valid = np.isfinite(depth) & (depth > 0) & (norm[..., 0] > 1e-8)
        normals[~valid] = 0.0
        # Make normals face the camera consistently.
        flip = np.sum(normals * rays, axis=-1) > 0
        normals[flip] *= -1.0
        return normals.astype(np.float32)
    
    
    def perspective_to_erp_splat(
        values: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        hfov_deg: float,
        erp_width: int,
        erp_height: int,
        roll_deg: float = 0.0,
        valid_mask: np.ndarray | None = None,
        interpolation: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inverse-warp an NFoV array into ERP without forward-splat holes.
    
        Each ERP direction is transformed into the pinhole camera, projected into
        the input image, and bilinearly sampled.  Yaw, pitch, roll, aspect ratio,
        and an optional letterbox-validity mask share one geometry convention.
        """
        cv = _require_cv2()
        value = np.asarray(values)
        if value.ndim not in {2, 3}:
            raise ValueError(f"Expected HxW or HxWxC values, got {value.shape}")
        h, w = value.shape[:2]
        world = erp_pixel_rays(int(erp_width), int(erp_height)).astype(np.float64)
        camera_to_world = rotation_matrix(
            math.radians(float(yaw_deg)),
            math.radians(float(pitch_deg)),
            math.radians(float(roll_deg)),
        )
        camera = world @ camera_to_world
        z = camera[..., 2]
        k = pinhole_intrinsics(w, h, hfov_deg)
        safe_z = np.where(z > 1e-8, z, 1.0)
        map_x = (k[0, 0] * camera[..., 0] / safe_z + k[0, 2]).astype(np.float32)
        map_y = (k[1, 2] - k[1, 1] * camera[..., 1] / safe_z).astype(np.float32)
        visible = (z > 1e-8) & (map_x >= 0.0) & (map_x <= w - 1.0) & (map_y >= 0.0) & (map_y <= h - 1.0)
        if valid_mask is not None:
            mask = np.asarray(valid_mask)
            if mask.shape != (h, w):
                raise ValueError(f"valid_mask must be {(h, w)}, got {mask.shape}")
            sampled_mask = cv.remap(
                mask.astype(np.uint8), map_x, map_y, interpolation=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT
            )
            visible &= sampled_mask > 0
        if interpolation is None:
            interpolation = cv.INTER_NEAREST if np.issubdtype(value.dtype, np.integer) else cv.INTER_LINEAR
        sampled = cv.remap(value, map_x, map_y, interpolation=interpolation, borderMode=cv.BORDER_CONSTANT)
        sampled = np.asarray(sampled)
        sampled[~visible] = 0
        return sampled.astype(value.dtype, copy=False), visible
    
    
    def normalize_prompt(prompt: str | None, default: str = "") -> str:
        prompt = (prompt or "").strip()
        return prompt if prompt else default
    
    
    def relative_or_absolute(path: str | Path, root: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else Path(root) / path
    
    
    def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
        import json
    
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    
    def read_jsonl(path: str | Path) -> list[dict]:
        import json
    
        records: list[dict] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    # END SOURCE: src/panogen_ft/pano_ops.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: src/panogen_ft/adapters.py
    # ------------------------------------------------------------------------------------------
    """Trainable camera, local-structure and seam-aware spherical adapters."""
    
    # from __future__ import annotations
    
    import math
    from typing import Mapping
    
    import torch
    from torch import nn
    from torch.nn import functional as F
    
    
    class CircularConv2d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
            super().__init__()
            self.pad = kernel_size // 2
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(self.pad, 0))
    
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            value = F.pad(value, (self.pad, self.pad, 0, 0), mode="circular")
            return self.conv(value)
    
    
    class ReferenceAdapter(nn.Module):
        """Encode the source-centered partial ERP RGB canvas and visibility."""
    
        def __init__(self, feature_dim: int = 256):
            super().__init__()
            self.network = nn.Sequential(
                CircularConv2d(4, feature_dim // 2),
                nn.SiLU(),
                CircularConv2d(feature_dim // 2, feature_dim),
                nn.SiLU(),
            )
    
        def forward(self, reference_rgb_mask: torch.Tensor) -> torch.Tensor:
            return self.network(reference_rgb_mask)
    
    
    class GeometryAdapter(nn.Module):
        def __init__(self, feature_dim: int = 256, camera_dim: int = 10):
            super().__init__()
            self.rays = nn.Sequential(CircularConv2d(4, feature_dim // 2), nn.SiLU(), CircularConv2d(feature_dim // 2, feature_dim))
            self.parameters_mlp = nn.Sequential(nn.Linear(camera_dim, feature_dim), nn.SiLU(), nn.Linear(feature_dim, feature_dim))
            branch = feature_dim // 2
            self.depth = nn.Sequential(CircularConv2d(2, branch), nn.SiLU(), CircularConv2d(branch, branch))
            self.normal = nn.Sequential(CircularConv2d(3, branch), nn.SiLU(), CircularConv2d(branch, branch))
            self.unknown_depth = nn.Parameter(torch.zeros(1, branch, 1, 1))
            self.unknown_normal = nn.Parameter(torch.zeros(1, branch, 1, 1))
            self.merge = nn.Sequential(CircularConv2d(branch * 2, feature_dim), nn.SiLU())
    
        def forward(
            self,
            rays_and_confidence: torch.Tensor,
            camera: torch.Tensor,
            factors: Mapping[str, torch.Tensor],
            drop_mask: Mapping[str, bool] | None = None,
        ) -> torch.Tensor:
            drop_mask = drop_mask or {}
            camera_feature = self.parameters_mlp(camera)[:, :, None, None]
            if drop_mask.get("depth"):
                depth_feature = self.unknown_depth.expand(camera.shape[0], -1, rays_and_confidence.shape[-2], rays_and_confidence.shape[-1])
            else:
                depth_feature = self.depth(factors["depth"])
            if drop_mask.get("normal"):
                normal_feature = self.unknown_normal.expand(camera.shape[0], -1, rays_and_confidence.shape[-2], rays_and_confidence.shape[-1])
            else:
                normal_feature = self.normal(factors["normal"])
            geometry = self.merge(torch.cat([depth_feature, normal_feature], dim=1))
            return self.rays(rays_and_confidence) + camera_feature + geometry
    
    
    class SphericalAdapter(nn.Module):
        def __init__(self, feature_dim: int = 256, output_dim: int = 3072, heads: int = 8, injection_layers: int = 4):
            super().__init__()
            self.fuse = CircularConv2d(feature_dim * 2, feature_dim)
            self.seam_attention = nn.MultiheadAttention(feature_dim, heads, batch_first=True)
            self.outputs = nn.ModuleList(
                [nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, output_dim)) for _ in range(injection_layers)]
            )
            self.gates = nn.Parameter(torch.zeros(injection_layers))
            for output in self.outputs:
                nn.init.zeros_(output[-1].weight)
                nn.init.zeros_(output[-1].bias)
    
        @staticmethod
        def positional_encoding(height: int, width: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            lon = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width * (2 * math.pi) - math.pi
            lat = (0.5 - (torch.arange(height, device=device, dtype=dtype) + 0.5) / height) * math.pi
            lon, lat = torch.meshgrid(lon, lat, indexing="xy")
            base = torch.stack([torch.sin(lon), torch.cos(lon), torch.sin(lat), torch.cos(lat)], dim=-1)
            return base.repeat(1, 1, math.ceil(dim / 4))[..., :dim]
    
        def forward(
            self, geometry: torch.Tensor, reference: torch.Tensor, output_size: tuple[int, int]
        ) -> tuple[torch.Tensor, ...]:
            height, width = output_size
            value = self.fuse(torch.cat([geometry, reference], dim=1))
            value = F.interpolate(value, size=(height, width), mode="bilinear", align_corners=False)
            tokens = value.flatten(2).transpose(1, 2)
            position = self.positional_encoding(height, width, tokens.shape[-1], tokens.device, tokens.dtype).reshape(1, height * width, -1)
            tokens = tokens + position
            # Cross-seam attention is limited to paired boundary columns.
            boundary_index = torch.cat([torch.arange(0, height * width, width, device=value.device), torch.arange(width - 1, height * width, width, device=value.device)])
            boundary = tokens[:, boundary_index]
            boundary, _ = self.seam_attention(boundary, boundary, boundary, need_weights=False)
            tokens = tokens.clone()
            tokens[:, boundary_index] = tokens[:, boundary_index] + boundary
            # Both projection and gate parameters start at zero, while the
            # residual path remains learnable on step one (a zero multiplier on a
            # zero projection would permanently deadlock both parameters).
            return tuple((1.0 + torch.tanh(gate)) * output(tokens) for gate, output in zip(self.gates, self.outputs))
    
    
    class SphericalCausalAdapter(nn.Module):
        def __init__(self, feature_dim: int = 256, output_dim: int = 3072, camera_dim: int = 10):
            super().__init__()
            self.reference_adapter = ReferenceAdapter(feature_dim)
            self.geometry_adapter = GeometryAdapter(feature_dim, camera_dim)
            self.spherical = SphericalAdapter(feature_dim, output_dim)
    
        def forward(
            self,
            rays_and_confidence: torch.Tensor,
            camera_parameters: torch.Tensor,
            factors: Mapping[str, torch.Tensor],
            output_size: tuple[int, int],
            drop_mask: Mapping[str, bool] | None = None,
        ) -> tuple[torch.Tensor, ...]:
            reference = factors.get("reference_rgb")
            if reference is None:
                raise KeyError("SphericalCausalAdapter requires factors['reference_rgb']")
            return self.spherical(
                self.geometry_adapter(rays_and_confidence, camera_parameters, factors, drop_mask),
                self.reference_adapter(reference),
                output_size,
            )
    
    
    class StructureReadoutHeads(nn.Module):
        """Reliable differentiable geometry readouts used for pseudo-label supervision."""
    
        def __init__(self, latent_channels: int = 16, hidden: int = 128, semantic_classes: int = 8):
            super().__init__()
            self.trunk = nn.Sequential(CircularConv2d(latent_channels, hidden), nn.SiLU(), CircularConv2d(hidden, hidden), nn.SiLU())
            self.depth = CircularConv2d(hidden, 1)
            self.normal = CircularConv2d(hidden, 3)
            del semantic_classes
    
        def forward(self, clean_latent: torch.Tensor) -> dict[str, torch.Tensor]:
            feature = self.trunk(clean_latent)
            return {
                "depth": F.softplus(self.depth(feature)) + 1e-4,
                "normal": F.normalize(self.normal(feature), dim=1),
            }
    # END SOURCE: src/panogen_ft/adapters.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: src/panogen_ft/losses.py
    # ------------------------------------------------------------------------------------------
    """FlowMatch and spherical causal losses with non-tautological interventions."""
    
    # from __future__ import annotations
    
    from typing import Mapping
    
    import torch
    import torch.nn.functional as F
    
    
    LOSS_WEIGHTS = {
        "flow": 1.00,
        "visible_reference": 0.30,
        "roll_intervention": 0.20,
        "seam": 0.05,
        "tangent_perceptual": 0.10,
        "polar_perceptual": 0.15,
        "cubemap_edge": 0.05,
        "depth": 0.05,
        "normal": 0.03,
    }
    
    LOSS_COMPONENTS = tuple(LOSS_WEIGHTS)
    
    
    def compose_weighted_losses(
        losses: Mapping[str, torch.Tensor],
        weights: Mapping[str, float] = LOSS_WEIGHTS,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the exact weighted objective and its per-component contributions."""
    
        if not losses:
            raise ValueError("At least one loss component is required")
        missing = set(losses) - set(weights)
        if missing:
            raise KeyError(f"No registered loss weight for {sorted(missing)}")
        first = next(iter(losses.values()))
        weighted = {name: value * float(weights[name]) for name, value in losses.items()}
        total = sum(weighted.values(), first.new_zeros(()))
        return total, weighted
    
    
    def average_loss_components(
        first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Average shared branch losses without halving one-branch auxiliaries."""
    
        if not first or not second:
            raise ValueError("Both intervention branches must contain at least the flow loss")
        averaged: dict[str, torch.Tensor] = {}
        for name in first.keys() | second.keys():
            if name in first and name in second:
                averaged[name] = 0.5 * (first[name] + second[name])
            else:
                averaged[name] = first[name] if name in first else second[name]
        return averaged
    
    
    def loss_bundle(
        sample_components: list[Mapping[str, torch.Tensor]],
        weights: Mapping[str, float] = LOSS_WEIGHTS,
    ) -> dict[str, torch.Tensor]:
        """Build a differentiable total plus logging sums/counts for one local microbatch."""
    
        if not sample_components:
            raise ValueError("A training microbatch cannot be empty")
        totals = [compose_weighted_losses(components, weights)[0] for components in sample_components]
        total_sum = torch.stack(totals).sum()
        bundle: dict[str, torch.Tensor] = {
            "loss_total": total_sum / len(totals),
            "sum_loss_total": total_sum,
            "count_loss_total": total_sum.new_tensor(float(len(totals))),
        }
        for name in weights:
            values = [components[name] for components in sample_components if name in components]
            if values:
                component_sum = torch.stack(values).sum()
                bundle[f"sum_loss_{name}"] = component_sum
                bundle[f"count_loss_{name}"] = component_sum.new_tensor(float(len(values)))
                bundle[f"sum_weighted_loss_{name}"] = component_sum * float(weights[name])
        return bundle
    
    
    def flowmatch_x0(x_t: torch.Tensor, velocity: torch.Tensor, timestep: torch.Tensor | float) -> torch.Tensor:
        """Recover a clean latent for the DiffSynth FlowMatch convention."""
        t = torch.as_tensor(timestep, device=x_t.device, dtype=x_t.dtype)
        while t.ndim < x_t.ndim:
            t = t.unsqueeze(-1)
        return x_t - t * velocity
    
    
    def latitude_cosine_weights(height: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        latitude = (0.5 - (torch.arange(height, device=device, dtype=dtype) + 0.5) / height) * torch.pi
        weight = 0.25 + 0.75 * torch.cos(latitude).clamp_min(0.0)
        return weight / weight.mean()
    
    
    def spherical_mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if value.ndim < 3:
            return value.mean()
        weights = latitude_cosine_weights(value.shape[-2], device=value.device, dtype=value.dtype)
        weights = weights.view(*([1] * (value.ndim - 2)), value.shape[-2], 1)
        if mask is not None:
            weights = weights * mask.to(value.dtype)
        return (value * weights).sum() / weights.expand_as(value).sum().clamp_min(1e-8)
    
    
    def seam_loss(images: torch.Tensor, seam_width: int = 8) -> torch.Tensor:
        """Match the periodic first derivative without forcing whole strips equal."""
        if images.ndim != 4:
            raise ValueError("Expected BCHW tensor")
        del seam_width
        seam_gradient = images[..., :1] - images[..., -1:]
        neighbour_gradient = 0.5 * ((images[..., 1:2] - images[..., :1]) + (images[..., -1:] - images[..., -2:-1]))
        return F.smooth_l1_loss(seam_gradient, neighbour_gradient)
    
    
    def _cubemap_directions(face: str, face_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coordinate = (torch.arange(face_size, device=device, dtype=dtype) + 0.5) / face_size * 2.0 - 1.0
        yy, xx = torch.meshgrid(-coordinate, coordinate, indexing="ij")
        one = torch.ones_like(xx)
        if face == "front":
            directions = torch.stack((xx, yy, one), -1)
        elif face == "right":
            directions = torch.stack((one, yy, -xx), -1)
        elif face == "back":
            directions = torch.stack((-xx, yy, -one), -1)
        elif face == "left":
            directions = torch.stack((-one, yy, xx), -1)
        elif face == "up":
            directions = torch.stack((xx, one, -yy), -1)
        elif face == "down":
            directions = torch.stack((xx, -one, yy), -1)
        else:
            raise ValueError(f"Unknown cubemap face: {face}")
        return F.normalize(directions, dim=-1)
    
    
    def torch_erp_face(images: torch.Tensor, face: str, face_size: int = 256) -> torch.Tensor:
        """Differentiably sample one cubemap face from BCHW ERP tensors."""
        if images.ndim != 4:
            raise ValueError("Expected BCHW ERP tensor")
        _, _, height, width = images.shape
        directions = _cubemap_directions(face, int(face_size), images.device, images.dtype)
        longitude = torch.atan2(directions[..., 0], directions[..., 2])
        latitude = torch.asin(directions[..., 1].clamp(-1.0, 1.0))
        source_x = torch.remainder((longitude / (2.0 * torch.pi) + 0.5) * width - 0.5, width)
        source_y = ((0.5 - latitude / torch.pi) * height - 0.5).clamp(0.0, height - 1.0)
        padded = F.pad(images, (1, 1, 0, 0), mode="circular")
        grid_x = 2.0 * (source_x + 1.5) / (width + 2.0) - 1.0
        grid_y = 2.0 * (source_y + 0.5) / height - 1.0
        grid = torch.stack((grid_x, grid_y), -1).unsqueeze(0).expand(images.shape[0], -1, -1, -1)
        return F.grid_sample(padded, grid, mode="bilinear", padding_mode="border", align_corners=False)
    
    
    def torch_erp_to_cubemap(images: torch.Tensor, face_size: int = 256) -> dict[str, torch.Tensor]:
        return {
            face: torch_erp_face(images, face, face_size)
            for face in ("front", "right", "back", "left", "up", "down")
        }
    
    
    def cubemap_reconstruction_losses(
        prediction: torch.Tensor,
        target: torch.Tensor,
        perceptual: torch.nn.Module,
        face_size: int = 256,
        edge_width: int = 4,
    ) -> dict[str, torch.Tensor]:
        """Compute side/polar perceptual losses and target-supervised face edges."""
        predicted_faces = torch_erp_to_cubemap(prediction, face_size)
        with torch.no_grad():
            target_faces = torch_erp_to_cubemap(target, face_size)
        side_names = ("front", "right", "back", "left")
        side_prediction = torch.cat([predicted_faces[name] for name in side_names], dim=0)
        side_target = torch.cat([target_faces[name] for name in side_names], dim=0)
        polar_prediction = torch.cat([predicted_faces[name] for name in ("up", "down")], dim=0)
        polar_target = torch.cat([target_faces[name] for name in ("up", "down")], dim=0)
        tangent = perceptual(side_prediction.float(), side_target.float()).mean()
        polar = perceptual(polar_prediction.float(), polar_target.float()).mean()
        del edge_width
        front, right = predicted_faces["front"], predicted_faces["right"]
        back, left = predicted_faces["back"], predicted_faces["left"]
        up, down = predicted_faces["up"], predicted_faces["down"]
        # Every physical cube edge appears twice. Orient each pair along the same
        # world-space direction before measuring continuity.
        pairs = (
            (front[..., :, -1], right[..., :, 0]),
            (right[..., :, -1], back[..., :, 0]),
            (back[..., :, -1], left[..., :, 0]),
            (left[..., :, -1], front[..., :, 0]),
            (front[..., 0, :], up[..., -1, :]),
            (back[..., 0, :], torch.flip(up[..., 0, :], dims=(-1,))),
            (right[..., 0, :], torch.flip(up[..., :, -1], dims=(-1,))),
            (left[..., 0, :], up[..., :, 0]),
            (front[..., -1, :], down[..., 0, :]),
            (back[..., -1, :], torch.flip(down[..., -1, :], dims=(-1,))),
            (right[..., -1, :], down[..., :, -1]),
            (left[..., -1, :], torch.flip(down[..., :, 0], dims=(-1,))),
        )
        edge_errors = [torch.abs(first - second).mean() for first, second in pairs]
        return {
            "tangent_perceptual": tangent,
            "polar_perceptual": polar,
            "cubemap_edge": torch.stack(edge_errors).mean(),
        }
    
    
    def roll_intervention_loss(prediction_a: torch.Tensor, prediction_b: torch.Tensor, b_to_a_pixels: int) -> torch.Tensor:
        """Align two independently predicted interventions before comparison."""
        if prediction_a.shape != prediction_b.shape:
            raise ValueError("Roll-intervention predictions must have the same shape")
        aligned_b = torch.roll(prediction_b, shifts=int(b_to_a_pixels), dims=-1)
        return spherical_mean(torch.abs(prediction_a - aligned_b))
    
    
    def visible_reference_loss(prediction: torch.Tensor, target: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
        return spherical_mean(torch.abs(prediction - target), visibility)
    
    
    def scale_invariant_depth_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.bool() & torch.isfinite(target) & (target > 0) & torch.isfinite(prediction) & (prediction > 0)
        if not bool(valid.any()):
            return prediction.new_zeros(())
        residual = torch.log(prediction[valid]) - torch.log(target[valid])
        residual = residual - residual.mean()
        return torch.sqrt(torch.mean(residual.square()) + 1e-8)
    
    
    def normal_cosine_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        error = 1.0 - F.cosine_similarity(prediction, target, dim=1).clamp(-1, 1)
        return spherical_mean(error[:, None], mask)
    
    
    def semantic_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        error = F.cross_entropy(logits, target.long(), reduction="none")[:, None]
        return spherical_mean(error, mask)
    
    
    def layout_boundary_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        error = F.binary_cross_entropy_with_logits(prediction, target.to(prediction.dtype), reduction="none")
        return spherical_mean(error, mask)
    
    
    def weighted_causal_objective(losses: Mapping[str, torch.Tensor], weights: Mapping[str, float] = LOSS_WEIGHTS) -> torch.Tensor:
        return compose_weighted_losses(losses, weights)[0]
    
    
    def should_decode_structure(global_step: int, interval: int = 4) -> bool:
        return int(global_step) % int(interval) == 0
    # END SOURCE: src/panogen_ft/losses.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: src/panogen_ft/caupano_model.py
    # ------------------------------------------------------------------------------------------
    """Shared CauPano DiffSynth training and causal-inference runtime."""
    
    # from __future__ import annotations
    
    import os
    import subprocess
    import sys
    from pathlib import Path
    from typing import Any
    
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    
    from .adapters import SphericalCausalAdapter, StructureReadoutHeads
    from .losses import (
        LOSS_WEIGHTS,
        average_loss_components,
        cubemap_reconstruction_losses,
        flowmatch_x0,
        loss_bundle,
        normal_cosine_loss,
        roll_intervention_loss,
        scale_invariant_depth_loss,
        seam_loss,
        spherical_mean,
    )
    from .pano_ops import erp_pixel_rays, perspective_to_erp_splat, source_centered_roll_pixels
    from .pano_ops import camera_condition_vector
    
    
    DIFFSYNTH_REVISION = "6477e11e4a81a5521d0f24972bcabf80838f32da"
    LORA_TARGETS = (
        "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,"
        "img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
    )
    
    
    def offline_environment(diffsynth_root: str | Path) -> None:
        root = Path(diffsynth_root)
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            DIFFUSERS_OFFLINE="1",
            DIFFSYNTH_SKIP_DOWNLOAD="True",
            TOKENIZERS_PARALLELISM="false",
        )
        try:
            head = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"DiffSynth is not a readable Git checkout: {root}") from exc
        if head != DIFFSYNTH_REVISION:
            raise RuntimeError(f"DiffSynth must be checkout {DIFFSYNTH_REVISION}; got {head}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    
    
    def local_model_configs(model_dir: str | Path) -> tuple[list[Any], Any, Any]:
        from diffsynth.core import ModelConfig
    
        root = Path(model_dir)
        patterns = (
            "transformer/diffusion_pytorch_model*.safetensors",
            "text_encoder/model*.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        )
        files: list[list[str]] = []
        for pattern in patterns:
            matches = sorted(str(path) for path in root.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"Missing Qwen model component: {root / pattern}")
            files.append(matches)
        tokenizer, processor = root / "tokenizer", root / "processor"
        if not tokenizer.is_dir() or not processor.is_dir():
            raise FileNotFoundError("The local Qwen-Image-Edit-2511 snapshot must include tokenizer/ and processor/")
        components = [ModelConfig(path=value if len(value) > 1 else value[0]) for value in files]
        return components, ModelConfig(path=str(tokenizer)), ModelConfig(path=str(processor))
    
    
    def _resize_array(value: np.ndarray, size: tuple[int, int], mode: str, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(value).copy(), device=device)
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)[None]
        tensor = tensor.float()
        return F.interpolate(tensor, size=size, mode=mode, align_corners=False if mode != "nearest" else None)
    
    
    def flowmatch_clean_from_model_call(
        latents: torch.Tensor,
        velocity: torch.Tensor,
        timestep: torch.Tensor | float,
        num_train_timesteps: int,
    ) -> torch.Tensor:
        """Recover x0 from the exact latent/velocity pair used by DiffSynth's loss."""
    
        if latents.shape != velocity.shape:
            raise RuntimeError(
                f"Pinned DiffSynth FlowMatch shape changed: latents={tuple(latents.shape)}, velocity={tuple(velocity.shape)}"
            )
        if int(num_train_timesteps) <= 0:
            raise ValueError("num_train_timesteps must be positive")
        sigma = torch.as_tensor(timestep, device=latents.device, dtype=latents.dtype) / float(num_train_timesteps)
        return flowmatch_x0(latents, velocity, sigma)
    
    
    def flowmatch_loss_with_clean(
        pipe: Any,
        loss_fn: Any,
        inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run pinned DiffSynth FlowMatch loss and capture its real model_fn tensors.
    
        QwenImagePipeline.model_fn calls the DiT submodules directly rather than
        invoking dit.forward(), so hooks on the outer DiT never run. Wrapping the
        model_fn keeps the upstream loss implementation unchanged while exposing
        the noisy latent, timestep, and predicted velocity used by that loss.
        """
    
        original_model_fn = pipe.model_fn
        captured: dict[str, torch.Tensor] = {}
    
        def capturing_model_fn(*args: Any, **kwargs: Any) -> torch.Tensor:
            output = original_model_fn(*args, **kwargs)
            latents = kwargs.get("latents")
            timestep = kwargs.get("timestep")
            if not isinstance(latents, torch.Tensor) or not isinstance(timestep, torch.Tensor):
                raise RuntimeError("Pinned DiffSynth FlowMatch model_fn no longer receives tensor latents/timestep")
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("Pinned DiffSynth FlowMatch model_fn no longer returns a tensor velocity")
            captured.update(latents=latents, timestep=timestep, velocity=output)
            return output
    
        pipe.model_fn = capturing_model_fn
        try:
            loss = loss_fn(pipe, **inputs)
        finally:
            pipe.model_fn = original_model_fn
    
        required = {"latents", "timestep", "velocity"}
        if required - captured.keys():
            raise RuntimeError("Unable to capture latents/timestep/velocity from pinned DiffSynth FlowMatch model_fn")
        clean = flowmatch_clean_from_model_call(
            captured["latents"],
            captured["velocity"],
            captured["timestep"],
            int(pipe.scheduler.num_train_timesteps),
        )
        return loss, clean
    
    
    def explicit_flowmatch_loss(
        pipe: Any,
        inputs: dict[str, Any],
        *,
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """FlowMatch SFT with caller-controlled stochastic state.
    
        This mirrors the pinned DiffSynth loss while exposing a reusable timestep
        and canonical noise tensor for rotation-equivariant intervention pairs.
        """
        max_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
        min_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
        if timestep is None:
            timestep_id = torch.randint(min_boundary, max_boundary, (1,))
            timestep = pipe.scheduler.timesteps[timestep_id]
        timestep = timestep.to(dtype=pipe.torch_dtype, device=pipe.device)
        input_latents = inputs["input_latents"]
        if noise is None:
            noise = torch.randn_like(input_latents) * float(inputs.get("noise_scale", 1.0))
        else:
            noise = noise.to(device=input_latents.device, dtype=input_latents.dtype)
        if noise.shape != input_latents.shape:
            raise RuntimeError(f"Shared noise/latent mismatch: {tuple(noise.shape)} vs {tuple(input_latents.shape)}")
        model_inputs = dict(inputs)
        latents = pipe.scheduler.add_noise(input_latents, noise, timestep)
        target = pipe.scheduler.training_target(input_latents, noise, timestep)
        model_inputs["latents"] = latents
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        velocity = pipe.model_fn(**models, **model_inputs, timestep=timestep)
        loss = F.mse_loss(velocity.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        clean = flowmatch_clean_from_model_call(
            latents, velocity, timestep, int(pipe.scheduler.num_train_timesteps)
        )
        return loss, clean, velocity, {"timestep": timestep.detach(), "noise": noise.detach()}
    
    
    def rotated_stochastic_state(
        stochastic: dict[str, torch.Tensor], b_to_a_pixels: int
    ) -> dict[str, torch.Tensor]:
        """Create branch-B noise whose inverse yaw rotation exactly equals branch A."""
        timestep, noise = stochastic.get("timestep"), stochastic.get("noise")
        if not isinstance(timestep, torch.Tensor) or not isinstance(noise, torch.Tensor):
            raise TypeError("A paired stochastic state requires tensor timestep and noise")
        return {
            "timestep": timestep,
            "noise": torch.roll(noise, shifts=-int(b_to_a_pixels), dims=-1),
        }
    
    
    def roll_spatial_supervision(supervision: dict[str, Any], yaw_deg: float) -> dict[str, np.ndarray]:
        """Center spatial ERP/layout labels without treating geometry vectors as images."""
    
        labels: dict[str, np.ndarray] = {}
        for name, value in supervision.items():
            array = np.asarray(value)
            if name.startswith("erp_"):
                if array.ndim not in {2, 3}:
                    raise RuntimeError(f"Invalid spatial teacher shape for {name}: {array.shape}")
                shift = source_centered_roll_pixels(yaw_deg, int(array.shape[1]))
                labels[name] = np.roll(array, shift, axis=1)
            elif name in {"layout_ceiling_boundary", "layout_floor_boundary"}:
                if array.ndim != 1:
                    raise RuntimeError(f"Invalid uLayout boundary shape for {name}: {array.shape}")
                shift = source_centered_roll_pixels(yaw_deg, int(array.shape[0]))
                labels[name] = np.roll(array, shift, axis=0)
            else:
                # For example, layout_room_polygon is an [N,2] geometry array. It
                # is quality metadata, not a raster supervision target.
                labels[name] = array
        return labels
    
    
    def rasterize_layout_target(
        ceiling_boundary: np.ndarray,
        floor_boundary: np.ndarray,
        size: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Build ceiling-line, floor-line, and wall-band targets from uLayout angles."""
    
        height, width = map(int, size)
        ceiling = np.asarray(ceiling_boundary, dtype=np.float32)
        floor = np.asarray(floor_boundary, dtype=np.float32)
        if ceiling.ndim != 1 or floor.ndim != 1 or ceiling.size < 2 or floor.size < 2:
            raise RuntimeError(f"Invalid uLayout boundaries: ceiling={ceiling.shape}, floor={floor.shape}")
    
        def resample(values: np.ndarray) -> torch.Tensor:
            tensor = torch.from_numpy(values.copy()).to(device=device, dtype=torch.float32).view(1, 1, -1)
            return F.interpolate(tensor, size=width, mode="linear", align_corners=False)[0, 0]
    
        ceiling = resample(ceiling)
        floor = resample(floor)
        ceiling_row = ((ceiling / torch.pi + 0.5) * height).round().long().clamp(0, height - 1)
        floor_row = ((floor / torch.pi + 0.5) * height).round().long().clamp(0, height - 1)
        rows = torch.arange(height, device=device)[:, None]
        ceiling_line = (rows - ceiling_row[None]).abs().le(1)
        floor_line = (rows - floor_row[None]).abs().le(1)
        upper = torch.minimum(ceiling_row, floor_row)[None]
        lower = torch.maximum(ceiling_row, floor_row)[None]
        wall_band = (rows >= upper) & (rows <= lower)
        return torch.stack((ceiling_line, floor_line, wall_band), dim=0).unsqueeze(0).float()
    
    
    class CausalAdapterController:
        """Inject one sample at img_in and three Qwen transformer depths."""
    
        def __init__(self, dit: torch.nn.Module, block_indices: tuple[int, ...] = (14, 29, 44)):
            self.tokens: tuple[torch.Tensor, ...] | None = None
            self.injection_count = 0
            self._handles = [dit.img_in.register_forward_hook(self._make_injector(0, "img_in"))]
            self._callback_blocks: list[torch.nn.Module] = []
            checkpoint_safe_callbacks = self._patch_qwen_checkpoint_dispatch()
            blocks = getattr(dit, "transformer_blocks", None)
            if blocks is not None:
                for site, block_index in enumerate(block_indices, start=1):
                    if block_index >= len(blocks):
                        raise RuntimeError(
                            f"Qwen transformer has {len(blocks)} blocks; cannot inject block {block_index}"
                        )
                    block = blocks[block_index]
                    injector = self._make_injector(site, f"transformer_blocks.{block_index}")
                    if checkpoint_safe_callbacks:
                        setattr(
                            block,
                            "_caupano_post_checkpoint",
                            lambda output, block=block, injector=injector: injector(block, (), output),
                        )
                        self._callback_blocks.append(block)
                    else:
                        self._handles.append(block.register_forward_hook(injector))
            self._site_count = 1 + len(self._callback_blocks) + len(self._handles) - 1
    
        @staticmethod
        def _patch_qwen_checkpoint_dispatch() -> bool:
            """Place block residuals outside torch.checkpoint recomputation."""
            try:
                from diffsynth.pipelines import qwen_image
            except ImportError:
                return False
            dispatch = qwen_image.gradient_checkpoint_forward
            if getattr(dispatch, "_caupano_patched", False):
                return True
    
            def checkpoint_then_adapter(model: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
                output = dispatch(model, *args, **kwargs)
                callback = getattr(model, "_caupano_post_checkpoint", None)
                return output if callback is None else callback(output)
    
            checkpoint_then_adapter._caupano_patched = True  # type: ignore[attr-defined]
            qwen_image.gradient_checkpoint_forward = checkpoint_then_adapter
            return True
    
        @property
        def active(self) -> bool:
            return self.tokens is not None
    
        @property
        def expected_injections(self) -> int:
            return self._site_count
    
        def activate(self, tokens: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]) -> None:
            if self.tokens is not None:
                raise RuntimeError("Causal adapter tokens leaked from the previous sample")
            values = (tokens,) if isinstance(tokens, torch.Tensor) else tuple(tokens)
            if len(values) != self.expected_injections:
                raise RuntimeError(
                    f"Adapter emitted {len(values)} residuals for {self.expected_injections} injection sites"
                )
            self.tokens = values
            self.injection_count = 0
    
        def clear(self) -> None:
            self.tokens = None
    
        def close(self) -> None:
            self.clear()
            for handle in self._handles:
                handle.remove()
            for block in self._callback_blocks:
                if hasattr(block, "_caupano_post_checkpoint"):
                    delattr(block, "_caupano_post_checkpoint")
    
        @staticmethod
        def _add_residual(output: torch.Tensor, residual: torch.Tensor, site: str) -> torch.Tensor:
            residual = residual.to(device=output.device, dtype=output.dtype)
            if output.ndim == 3:
                if (
                    output.shape[0] != residual.shape[0]
                    or output.shape[1] < residual.shape[1]
                    or output.shape[2] != residual.shape[2]
                ):
                    raise RuntimeError(f"Adapter/{site} shape mismatch: {tuple(residual.shape)} vs {tuple(output.shape)}")
                return torch.cat(
                    [output[:, : residual.shape[1]] + residual, output[:, residual.shape[1] :]], dim=1
                )
            if output.ndim == 2 and residual.shape[0] == 1:
                flat = residual[0]
                if output.shape[0] < flat.shape[0] or output.shape[1] != flat.shape[1]:
                    raise RuntimeError(f"Adapter/{site} shape mismatch: {tuple(flat.shape)} vs {tuple(output.shape)}")
                return torch.cat([output[: flat.shape[0]] + flat, output[flat.shape[0] :]], dim=0)
            raise RuntimeError(f"Unsupported {site} output: {tuple(output.shape)}")
    
        def _make_injector(self, token_index: int, site: str):
            def inject(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> Any:
                del module, args
                if self.tokens is None:
                    return output
                self.injection_count += 1
                residual = self.tokens[token_index]
                if isinstance(output, tuple):
                    if len(output) < 2 or not isinstance(output[1], torch.Tensor):
                        raise RuntimeError(f"Unsupported tuple output at {site}")
                    return (output[0], self._add_residual(output[1], residual, site), *output[2:])
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError(f"Unsupported non-tensor output at {site}: {type(output)!r}")
                return self._add_residual(output, residual, site)
    
            return inject
    
    
    def causal_adapter_inputs(
        metadata: dict[str, Any],
        patch_size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        camera_teacher_forcing: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, bool], torch.Tensor]:
        # Kept in the signature so old launchers fail neither at import nor resume;
        # v3 deliberately removes the old real/predicted linear interpolation.
        del camera_teacher_forcing
        causal, true_spec = metadata["causal"], metadata["view_spec"]
        predicted = causal.get("predicted") or {}
        valid = causal.get("valid") or {}
        index = int(metadata["view_index"])
        height, width = patch_size
    
        camera_condition = metadata.get("camera_condition") or {}
        mode = str(camera_condition.get("mode") or "exact")
        condition_spec = dict(camera_condition.get("spec") or true_spec)
        if mode == "unknown":
            condition_spec.update(pitch_deg=0.0, roll_deg=0.0, hfov_deg=90.0)
        condition_spec["yaw_deg"] = 0.0
    
        reference = np.asarray(metadata["edit_image"].convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        reference_mask = np.asarray(
            metadata.get("reference_valid_mask", np.ones(reference.shape[:2], dtype=bool)), dtype=bool
        )
    
        def project(value: np.ndarray, interpolation: int | None = None) -> tuple[np.ndarray, np.ndarray]:
            return perspective_to_erp_splat(
                value,
                0.0,
                float(condition_spec.get("pitch_deg", 0.0)),
                float(condition_spec.get("hfov_deg", 90.0)),
                width,
                height,
                roll_deg=float(condition_spec.get("roll_deg", 0.0)),
                valid_mask=reference_mask,
                interpolation=interpolation,
            )
    
        reference_erp, visible = project(reference)
        reference_erp[~visible] = 0.0
        depth_name, confidence_name = f"view_{index}_depth", f"view_{index}_depth_confidence"
        normal_name = f"view_{index}_normal"
        if depth_name in predicted and bool(valid.get("depth", False)):
            depth, depth_visible = project(np.asarray(predicted[depth_name], dtype=np.float32))
            confidence_source = np.asarray(
                predicted.get(confidence_name, np.ones_like(predicted[depth_name])), dtype=np.float32
            )
            confidence, _ = project(confidence_source)
            depth = np.squeeze(depth)
            confidence = np.squeeze(confidence)
            depth = np.log(np.maximum(depth, 1e-4))
            observed = depth_visible & np.isfinite(depth)
            if observed.any():
                depth[observed] = (depth[observed] - np.median(depth[observed])) / (
                    np.std(depth[observed]) + 1e-6
                )
            depth[~observed] = 0.0
            confidence[~observed] = 0.0
        else:
            depth = np.zeros((height, width), dtype=np.float32)
            confidence = np.zeros_like(depth)
        if normal_name in predicted and bool(valid.get("normal", False)):
            normal, normal_visible = project(np.asarray(predicted[normal_name], dtype=np.float32))
            normal = np.asarray(normal, dtype=np.float32)
            normal[~normal_visible] = 0.0
        else:
            normal = np.zeros((height, width, 3), dtype=np.float32)
    
        rays = erp_pixel_rays(width, height).transpose(2, 0, 1)
        factors = {
            "reference_rgb": torch.from_numpy(
                np.concatenate([reference_erp.transpose(2, 0, 1), visible[None]], axis=0).copy()
            ).to(device=device, dtype=dtype)[None],
            "depth": torch.from_numpy(np.stack([depth, confidence])).to(device=device, dtype=dtype)[None],
            "normal": torch.from_numpy(normal.transpose(2, 0, 1).copy()).to(device=device, dtype=dtype)[None],
        }
        rays_conf = torch.from_numpy(np.concatenate([rays, visible[None]], axis=0).copy()).to(device=device, dtype=dtype)[None]
        vector = camera_condition.get("vector")
        if vector is None:
            vector = camera_condition_vector(
                condition_spec, camera_condition.get("quality") or causal.get("quality") or {}, mode=mode
            )
        camera = torch.as_tensor(np.asarray(vector, dtype=np.float32), device=device, dtype=dtype)[None]
        if camera.shape != (1, 10):
            raise RuntimeError(f"Camera conditioning must have 10 elements, got {tuple(camera.shape)}")
        drop = {factor: not bool(valid.get(factor, False)) for factor in ("depth", "normal")}
        visible_tensor = torch.from_numpy(visible.copy()).to(device=device)[None, None]
        return rays_conf, camera, factors, drop, visible_tensor
    
    
    def activate_causal_adapter(
        adapter: SphericalCausalAdapter,
        controller: CausalAdapterController,
        metadata: dict[str, Any],
        patch_size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        camera_teacher_forcing: float = 1.0,
    ) -> torch.Tensor:
        rays, camera, factors, drop, visible = causal_adapter_inputs(
            metadata, patch_size, device, dtype, camera_teacher_forcing
        )
        controller.activate(adapter(rays, camera, factors, patch_size, drop))
        return visible
    
    
    def load_adapter_checkpoint(module: torch.nn.Module, path: str | Path, device: torch.device | str) -> None:
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        state = load_file(str(checkpoint), device=str(device))
        missing, unexpected = module.load_state_dict(state, strict=False)
        if unexpected or not any(name.startswith("causal_adapter.") for name in state):
            raise RuntimeError(f"Invalid adapter checkpoint: unexpected={unexpected}, missing={missing[:8]}")
    
    
    def load_inference_adapter(
        path: str | Path, device: torch.device | str, dtype: torch.dtype = torch.bfloat16
    ) -> SphericalCausalAdapter:
        state = load_file(str(path), device=str(device))
        prefix = "causal_adapter."
        causal = {name.removeprefix(prefix): value for name, value in state.items() if name.startswith(prefix)}
        if not causal:
            raise RuntimeError(f"No causal_adapter weights in {path}")
        adapter = SphericalCausalAdapter(feature_dim=256, output_dim=3072, camera_dim=10).to(device=device, dtype=dtype)
        missing, unexpected = adapter.load_state_dict(causal, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Causal adapter mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
        adapter.eval()
        return adapter
    
    
    def build_training_module(
        model_dir: str | Path,
        device: torch.device,
        stage: str,
        lora_checkpoint: str | None,
        training_config: dict[str, Any] | None = None,
    ):
        from diffsynth.diffusion import DiffusionTrainingModule
        from diffsynth.pipelines.qwen_image import QwenImagePipeline
    
        model_configs, tokenizer_config, processor_config = local_model_configs(model_dir)
        training_config = training_config or {}
    
        class CauPanoTrainingModule(DiffusionTrainingModule):
            def __init__(self) -> None:
                super().__init__()
                self.pipe = QwenImagePipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device=device,
                    model_configs=model_configs,
                    tokenizer_config=tokenizer_config,
                    processor_config=processor_config,
                )
                self.pipe = self.split_pipeline_units("sft", self.pipe, None, "dit")
                self.switch_pipe_to_training_mode(
                    self.pipe, None, "dit", LORA_TARGETS, 64, lora_checkpoint, None, None, task="sft"
                )
                self.stage = stage
                self.global_step = 0
                self.total_steps = 1
                self.force_structure = False
                self.validation_mode = False
                self.gradient_checkpointing = bool(training_config.get("gradient_checkpointing", False))
                self.cube_face_size = int(training_config.get("cube_face_size", 256))
                self.loss_ramp_start = int(training_config.get("loss_ramp_start", 2000))
                self.loss_ramp_end = int(training_config.get("loss_ramp_end", 5000))
                self.loss_weights = dict(LOSS_WEIGHTS)
                self.loss_weights.update(training_config.get("loss_weights") or {})
                self._lpips: torch.nn.Module | None = None
                self.adapter_controller: CausalAdapterController | None = None
                if stage in {"causal", "highres"}:
                    self.causal_adapter = SphericalCausalAdapter(feature_dim=256, output_dim=3072, camera_dim=10).to(
                        device=device, dtype=torch.bfloat16
                    )
                    self.structure_heads = StructureReadoutHeads().to(device=device, dtype=torch.bfloat16)
                    self.adapter_controller = CausalAdapterController(self.pipe.dit)
                self._configure_lora_dropout(0.05)
    
            def _configure_lora_dropout(self, probability: float) -> None:
                trainable = [(name, parameter) for name, parameter in self.pipe.dit.named_parameters() if parameter.requires_grad]
                if not trainable:
                    raise RuntimeError("DiffSynth did not create any trainable LoRA weights")
                for name, module in self.pipe.dit.named_modules():
                    if "lora" in name.lower() and isinstance(module, torch.nn.Dropout):
                        module.p = probability
    
            def _adapter_context(self, metadata: dict[str, Any], patch_size: tuple[int, int]) -> torch.Tensor:
                assert self.adapter_controller is not None
                return activate_causal_adapter(
                    self.causal_adapter,
                    self.adapter_controller,
                    metadata,
                    patch_size,
                    device,
                    torch.bfloat16,
                )
    
            def _active_loss_weights(self) -> dict[str, float]:
                if self.validation_mode:
                    ramp = 1.0
                elif self.global_step <= self.loss_ramp_start:
                    ramp = 0.0
                elif self.global_step >= self.loss_ramp_end:
                    ramp = 1.0
                else:
                    ramp = (self.global_step - self.loss_ramp_start) / max(
                        1, self.loss_ramp_end - self.loss_ramp_start
                    )
                initially_active = {"flow", "visible_reference", "seam"}
                return {
                    name: float(weight) * (1.0 if name in initially_active else ramp)
                    for name, weight in self.loss_weights.items()
                }
    
            def _perceptual_model(self) -> torch.nn.Module:
                if self._lpips is None:
                    try:
                        import lpips
                    except ImportError as exc:
                        raise RuntimeError("CauPano cube losses require lpips>=0.1.4") from exc
                    self._lpips = lpips.LPIPS(net="alex").eval().to(device=device)
                    self._lpips.requires_grad_(False)
                return self._lpips
    
            def _decode_cube_losses(
                self, clean: torch.Tensor, target_clean: torch.Tensor
            ) -> dict[str, torch.Tensor]:
                if not hasattr(self.pipe, "vae"):
                    raise RuntimeError("Pinned Qwen pipeline no longer exposes the VAE needed by cube losses")
                self.pipe.load_models_to_device(["vae"])
                prediction = self.pipe.vae.decode(clean)
                with torch.no_grad():
                    target = self.pipe.vae.decode(target_clean)
                if prediction.shape[1] != 3 or target.shape != prediction.shape:
                    raise RuntimeError(
                        f"Unexpected decoded Qwen VAE shapes: prediction={tuple(prediction.shape)}, target={tuple(target.shape)}"
                    )
                return cubemap_reconstruction_losses(
                    prediction.clamp(-1, 1),
                    target.clamp(-1, 1),
                    self._perceptual_model(),
                    face_size=self.cube_face_size,
                )
    
            def _structure_losses(
                self,
                clean: torch.Tensor,
                target_clean: torch.Tensor,
                metadata: dict[str, Any],
                visible: torch.Tensor,
                expensive: bool,
            ) -> dict[str, torch.Tensor]:
                causal = metadata["causal"]
                supervision, valid = causal.get("supervision") or {}, causal.get("valid") or {}
                spec = metadata["view_spec"]
                size = clean.shape[-2:]
                losses: dict[str, torch.Tensor] = {}
                losses["seam"] = seam_loss(clean, seam_width=1)
                if target_clean.shape != clean.shape:
                    raise RuntimeError(f"Target/x0 latent shape mismatch: {tuple(target_clean.shape)} vs {tuple(clean.shape)}")
                visible_latent = F.interpolate(visible.float(), size=size, mode="nearest")
                losses["visible_reference"] = spherical_mean(torch.abs(clean - target_clean), visible_latent)
                if expensive:
                    labels = roll_spatial_supervision(supervision, float(spec["yaw_deg"]))
                    output = self.structure_heads(clean)
                    visible_bool = visible_latent.bool()
                    if valid.get("depth") and "erp_depth" in labels:
                        target = _resize_array(labels["erp_depth"], size, "bilinear", device)
                        mask = _resize_array(
                            labels.get("erp_range_mask", np.ones_like(labels["erp_depth"])),
                            size,
                            "nearest",
                            device,
                        ).bool()
                        losses["depth"] = scale_invariant_depth_loss(output["depth"], target, mask)
                    if valid.get("normal") and "erp_normal" in labels:
                        target = _resize_array(labels["erp_normal"], size, "bilinear", device)
                        losses["normal"] = normal_cosine_loss(
                            output["normal"], F.normalize(target, dim=1), torch.ones_like(visible_bool)
                        )
                    losses.update(self._decode_cube_losses(clean, target_clean))
                return losses
    
            def _forward_one(
                self,
                data: dict[str, Any],
                structure: bool,
                stochastic: dict[str, torch.Tensor] | None = None,
            ) -> tuple[
                dict[str, torch.Tensor],
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor,
                dict[str, torch.Tensor],
            ]:
                metadata = data["metadata"]
                patch_size = (data["image"].height // 16, data["image"].width // 16)
                visible = None
                try:
                    if self.stage in {"causal", "highres"}:
                        visible = self._adapter_context(metadata, patch_size)
                    inputs_shared = {
                        "input_image": data["image"],
                        "edit_image": data["edit_image"],
                        "height": data["image"].height,
                        "width": data["image"].width,
                        "cfg_scale": 1,
                        "rand_device": self.pipe.device,
                        "use_gradient_checkpointing": self.gradient_checkpointing and not self.validation_mode,
                        "use_gradient_checkpointing_offload": False,
                        "edit_image_auto_resize": True,
                        "zero_cond_t": True,
                    }
                    inputs_posi, inputs_nega = {"prompt": data["prompt"]}, {"negative_prompt": ""}
                    inputs = self.transfer_data_to_device(
                        (inputs_shared, inputs_posi, inputs_nega), self.pipe.device, self.pipe.torch_dtype
                    )
                    for unit in self.pipe.units:
                        inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
                    flow, clean, velocity, stochastic_state = explicit_flowmatch_loss(
                        self.pipe,
                        {**inputs[0], **inputs[1]},
                        timestep=None if stochastic is None else stochastic["timestep"],
                        noise=None if stochastic is None else stochastic["noise"],
                    )
                    if self.stage in {"causal", "highres"}:
                        if (
                            self.adapter_controller is None
                            or self.adapter_controller.injection_count != self.adapter_controller.expected_injections
                        ):
                            count = 0 if self.adapter_controller is None else self.adapter_controller.injection_count
                            expected = 0 if self.adapter_controller is None else self.adapter_controller.expected_injections
                            raise RuntimeError(
                                f"Causal adapter injection count must be {expected} per sample; got {count}"
                            )
                    components = {"flow": flow}
                    if clean is not None and visible is not None:
                        target_clean = inputs[0].get("input_latents")
                        if not isinstance(target_clean, torch.Tensor):
                            raise RuntimeError("Pinned FlowMatch units no longer expose clean input_latents")
                        components.update(
                            self._structure_losses(clean, target_clean, metadata, visible, expensive=structure)
                        )
                    return components, clean, visible, velocity, stochastic_state
                finally:
                    if self.adapter_controller is not None:
                        self.adapter_controller.clear()
    
            def forward(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
                sample_components: list[dict[str, torch.Tensor]] = []
                for sample_index, data in enumerate(batch):
                    # Decode exactly one sample per rank every four optimizer steps.
                    calculate_structure = self.force_structure or (
                        self.global_step > self.loss_ramp_start
                        and self.global_step % 4 == 0
                        and sample_index == 0
                    )
                    components, clean_a, _, velocity_a, stochastic = self._forward_one(
                        data, calculate_structure
                    )
                    intervention = data["metadata"].get("intervention")
                    if intervention is not None and clean_a is not None:
                        second = {
                            "image": intervention["image"],
                            "edit_image": intervention["edit_image"],
                            "prompt": intervention["prompt"],
                            "metadata": intervention,
                        }
                        yaw_a = data["metadata"]["view_spec"]["yaw_deg"]
                        yaw_b = intervention["view_spec"]["yaw_deg"]
                        align = round((yaw_b - yaw_a) / 360.0 * clean_a.shape[-1])
                        shared = rotated_stochastic_state(stochastic, align)
                        second_components, clean_b, _, velocity_b, _ = self._forward_one(
                            second, False, shared
                        )
                        components = average_loss_components(components, second_components)
                        components["roll_intervention"] = 0.5 * (
                            roll_intervention_loss(clean_a, clean_b, align)
                            + roll_intervention_loss(velocity_a, velocity_b, align)
                        )
                    sample_components.append(components)
    
                return loss_bundle(sample_components, weights=self._active_loss_weights())
    
        return CauPanoTrainingModule()
    # END SOURCE: src/panogen_ft/caupano_model.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: src/panogen_ft/external_inference.py
    # ------------------------------------------------------------------------------------------
    """External-reference causal preprocessing and Stage 3 inference.
    
    Training-time ERP teachers (DAP and uLayout) supervise complete target
    panoramas and are intentionally not fabricated for a single external NFoV.
    Deployment uses only the reference-view predictions consumed by the causal
    adapter: reference RGB/mask, DA3 depth, depth-derived normals, and GeoCalib
    camera parameters.
    """
    
    # from __future__ import annotations
    
    import gc
    import hashlib
    import json
    import math
    import os
    import re
    import threading
    from pathlib import Path
    from typing import Any
    
    import numpy as np
    import cv2
    import torch
    from PIL import Image, ImageOps
    
    from .caupano_model import CausalAdapterController, activate_causal_adapter, load_inference_adapter, offline_environment
    from .metrics import validate_panorama
    from .pano_ops import (
        camera_condition_vector,
        depth_to_normals,
        perspective_to_erp_splat,
        pinhole_intrinsics,
        pinhole_pixel_rays,
    )
    from .qwen_edit_wrapper import load_pipeline_lora, load_qwen_edit_pipeline, run_qwen_edit_inference
    from .teachers import DA3Teacher, GeoCalibTeacher, force_offline
    
    
    EXTERNAL_CONDITION_VERSION = "external-reference-v2"
    REFERENCE_SIZE = 768
    
    
    def safe_external_id(value: str) -> str:
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
        return value or "external"
    
    
    def normalize_external_reference(
        image: Image.Image, size: int = REFERENCE_SIZE, *, return_mask: bool = False
    ) -> Image.Image | tuple[Image.Image, np.ndarray]:
        """Preserve aspect ratio, resize the long edge, and pad only to a multiple of 32."""
    
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.width < 32 or image.height < 32:
            raise ValueError(f"Reference image is too small: {image.width}x{image.height}")
        scale = float(size) / max(image.width, image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        canvas_width = int(math.ceil(resized.width / 32.0) * 32)
        canvas_height = int(math.ceil(resized.height / 32.0) * 32)
        left = (canvas_width - resized.width) // 2
        top = (canvas_height - resized.height) // 2
        canvas = Image.new("RGB", (canvas_width, canvas_height))
        canvas.paste(resized, (left, top))
        mask = np.zeros((canvas_height, canvas_width), dtype=bool)
        mask[top : top + resized.height, left : left + resized.width] = True
        return (canvas, mask) if return_mask else canvas
    
    
    def _finite_float(value: Any, default: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float(default)
        return result if math.isfinite(result) else float(default)
    
    
    def _clamp(value: float, lower: float, upper: float) -> float:
        return float(min(max(float(value), lower), upper))
    
    
    def camera_spec_from_estimate(
        estimate: dict[str, Any],
        width: int,
        height: int,
        *,
        hfov_deg: float | None = None,
        pitch_deg: float | None = None,
        roll_deg: float | None = None,
        camera_mode: str = "auto",
    ) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
        if camera_mode not in {"auto", "override", "unknown"}:
            raise ValueError(f"Unsupported camera mode: {camera_mode}")
        estimated_vfov = _finite_float(estimate.get("vfov_deg"), 90.0)
        estimated_hfov = math.degrees(
            2.0 * math.atan(math.tan(math.radians(_clamp(estimated_vfov, 20.0, 150.0)) / 2.0) * width / height)
        )
        if camera_mode == "unknown":
            hfov, pitch, roll = 90.0, 0.0, 0.0
        elif camera_mode == "override":
            hfov = _clamp(90.0 if hfov_deg is None else hfov_deg, 30.0, 140.0)
            pitch = _clamp(0.0 if pitch_deg is None else pitch_deg, -75.0, 75.0)
            roll = _clamp(0.0 if roll_deg is None else roll_deg, -45.0, 45.0)
        else:
            hfov = _clamp(hfov_deg if hfov_deg is not None else estimated_hfov, 30.0, 140.0)
            pitch = _clamp(
                pitch_deg if pitch_deg is not None else _finite_float(estimate.get("pitch_deg"), 0.0), -75.0, 75.0
            )
            roll = _clamp(
                roll_deg if roll_deg is not None else _finite_float(estimate.get("roll_deg"), 0.0), -45.0, 45.0
            )
        fov_std = max(0.0, _finite_float(estimate.get("fov_std_deg"), 90.0))
        gravity_std = max(0.0, _finite_float(estimate.get("gravity_std_deg"), 90.0))
        k = pinhole_intrinsics(width, height, hfov).astype(np.float32)
        spec = {
            "view_id": "external-v0",
            "projection": "pinhole",
            "width": int(width),
            "height": int(height),
            "yaw_deg": 0.0,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "hfov_deg": hfov,
            "intrinsics": k.tolist(),
        }
        fov_valid = camera_mode == "override" or (camera_mode == "auto" and fov_std <= 10.0)
        gravity_valid = camera_mode == "override" or (camera_mode == "auto" and gravity_std <= 7.0)
        quality = {
            "estimated_vfov_deg": estimated_vfov,
            "used_hfov_deg": hfov,
            "used_pitch_deg": pitch,
            "used_roll_deg": roll,
            "fov_std_deg": fov_std,
            "gravity_std_deg": gravity_std,
            "fov_valid": bool(fov_valid),
            "gravity_valid": bool(gravity_valid),
            "camera_valid": bool(fov_valid and gravity_valid),
            "camera_mode": camera_mode,
            "camera_overrides": {
                "hfov": hfov_deg is not None,
                "pitch": pitch_deg is not None,
                "roll": roll_deg is not None,
            },
        }
        if camera_mode == "override":
            quality.update(fov_std_deg=0.0, gravity_std_deg=0.0)
        encoding_mode = camera_mode
        if camera_mode == "auto" and not quality["camera_valid"]:
            encoding_mode = "unknown"
            spec.update(
                hfov_deg=90.0,
                pitch_deg=0.0,
                roll_deg=0.0,
                intrinsics=pinhole_intrinsics(width, height, 90.0).astype(np.float32).tolist(),
            )
        quality["camera_mode_used"] = encoding_mode
        camera = camera_condition_vector(spec, quality, mode=encoding_mode)
        return spec, camera, quality
    
    
    def external_condition_key(
        image: Image.Image,
        hfov_deg: float | None,
        pitch_deg: float | None,
        roll_deg: float | None,
        camera_mode: str = "auto",
    ) -> str:
        digest = hashlib.sha256()
        digest.update(EXTERNAL_CONDITION_VERSION.encode("utf-8"))
        digest.update(f"{image.width}x{image.height}".encode("ascii"))
        digest.update(image.tobytes())
        digest.update(
            json.dumps(
                {
                    "hfov_deg": hfov_deg,
                    "pitch_deg": pitch_deg,
                    "roll_deg": roll_deg,
                    "camera_mode": camera_mode,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()
    
    
    def _condition_paths(cache_root: str | Path, key: str) -> tuple[Path, Path]:
        root = Path(cache_root)
        return root / f"{key}.npz", root / f"{key}.json"
    
    
    def load_cached_external_condition(cache_root: str | Path, key: str) -> dict[str, Any] | None:
        arrays_path, metadata_path = _condition_paths(cache_root, key)
        if not arrays_path.is_file() or not metadata_path.is_file():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if payload.get("version") != EXTERNAL_CONDITION_VERSION or payload.get("key") != key:
                return None
            with np.load(arrays_path, allow_pickle=False) as archive:
                predicted = {name: archive[name] for name in archive.files}
            required = {
                "view_0_depth",
                "view_0_depth_confidence",
                "view_0_normal",
                "view_0_camera",
                "view_0_valid_mask",
            }
            if not required.issubset(predicted):
                return None
            return {
                "record": payload["record"],
                "view_index": 0,
                "view_spec": payload["view_spec"],
                "camera_condition": payload["camera_condition"],
                "reference_valid_mask": predicted["view_0_valid_mask"].astype(bool),
                "causal": {
                    "predicted": predicted,
                    "supervision": {},
                    "valid": payload["valid"],
                    "quality": payload["quality"],
                },
                "condition_cache": {"key": key, "arrays": str(arrays_path), "metadata": str(metadata_path), "hit": True},
            }
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return None
    
    
    def save_cached_external_condition(cache_root: str | Path, key: str, metadata: dict[str, Any]) -> dict[str, str]:
        arrays_path, metadata_path = _condition_paths(cache_root, key)
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        predicted = metadata["causal"]["predicted"]
        arrays_temporary = arrays_path.with_suffix(".npz.tmp")
        with arrays_temporary.open("wb") as handle:
            np.savez_compressed(handle, **predicted)
        os.replace(arrays_temporary, arrays_path)
        payload = {
            "version": EXTERNAL_CONDITION_VERSION,
            "key": key,
            "record": metadata["record"],
            "view_spec": metadata["view_spec"],
            "valid": metadata["causal"]["valid"],
            "quality": metadata["causal"]["quality"],
            "camera_condition": metadata["camera_condition"],
        }
        metadata_temporary = metadata_path.with_suffix(".json.tmp")
        metadata_temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(metadata_temporary, metadata_path)
        return {"key": key, "arrays": str(arrays_path), "metadata": str(metadata_path), "hit": False}
    
    
    class ExternalCausalPreprocessor:
        """Lazily load local reference-view teachers and cache their predictions."""
    
        def __init__(self, models_root: str | Path, device: str | torch.device = "cuda"):
            force_offline()
            self.models_root = Path(models_root)
            self.device = torch.device(device)
            self.da3: DA3Teacher | None = None
            self.geocalib: GeoCalibTeacher | None = None
            self._lock = threading.Lock()
    
        def _ensure_loaded(self) -> None:
            if self.da3 is not None:
                return
            self.da3 = DA3Teacher(self.models_root / "da3metric-large", str(self.device))
            self.geocalib = GeoCalibTeacher(self.models_root / "geocalib/geocalib-pinhole.tar", str(self.device))
            print(f"External causal teachers ready on {self.device}", flush=True)
    
        @torch.inference_mode()
        def process(
            self,
            image: Image.Image,
            *,
            cache_root: str | Path,
            record_id: str = "external",
            hfov_deg: float | None = None,
            pitch_deg: float | None = None,
            roll_deg: float | None = None,
            camera_mode: str = "auto",
        ) -> tuple[Image.Image, dict[str, Any]]:
            reference, reference_mask = normalize_external_reference(image, return_mask=True)
            key = external_condition_key(reference, hfov_deg, pitch_deg, roll_deg, camera_mode)
            cached = load_cached_external_condition(cache_root, key)
            if cached is not None:
                cached["record"] = {"id": safe_external_id(record_id), "external": True}
                cached["edit_image"] = reference
                return reference, cached
            with self._lock:
                cached = load_cached_external_condition(cache_root, key)
                if cached is not None:
                    cached["record"] = {"id": safe_external_id(record_id), "external": True}
                    cached["edit_image"] = reference
                    return reference, cached
                self._ensure_loaded()
                assert self.da3 is not None and self.geocalib is not None
                da3 = self.da3([reference])[0]
                camera_estimate = self.geocalib(reference)
                if not isinstance(camera_estimate, dict):
                    raise TypeError("GeoCalib single-image inference did not return a mapping")
    
                spec, camera, camera_quality = camera_spec_from_estimate(
                    camera_estimate,
                    reference.width,
                    reference.height,
                    hfov_deg=hfov_deg,
                    pitch_deg=pitch_deg,
                    roll_deg=roll_deg,
                    camera_mode=camera_mode,
                )
                depth = np.asarray(da3["depth"], dtype=np.float32).squeeze()
                confidence = np.asarray(da3.get("confidence", np.ones_like(depth)), dtype=np.float32).squeeze()
                if depth.ndim != 2 or confidence.shape != depth.shape:
                    raise ValueError(f"Invalid DA3 output shapes: depth={depth.shape}, confidence={confidence.shape}")
                finite = np.isfinite(depth) & (depth > 0)
                finite_ratio = float(finite.mean())
                confidence_finite_ratio = float(np.isfinite(confidence).mean())
                if finite_ratio < 0.90:
                    raise RuntimeError(f"DA3 depth is invalid for too many reference pixels: finite_ratio={finite_ratio:.4f}")
                depth = np.where(finite, depth, np.nanmedian(depth[finite])).astype(np.float32)
                confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
                normals = depth_to_normals(
                    depth,
                    pinhole_pixel_rays(depth.shape[1], depth.shape[0], spec["hfov_deg"]),
                    wrap_x=False,
                ).astype(np.float16)
                valid = {
                    "depth": bool(finite_ratio >= 0.98 and confidence_finite_ratio >= 0.98),
                    "normal": bool(np.isfinite(normals).mean() >= 0.999),
                    "camera": bool(camera_quality["camera_valid"]),
                }
                predicted = {
                    "view_0_depth": depth,
                    "view_0_depth_confidence": confidence,
                    "view_0_intrinsics": np.asarray(da3.get("intrinsics", spec["intrinsics"]), dtype=np.float32),
                    "view_0_normal": normals,
                    "view_0_camera": camera,
                    "view_0_valid_mask": reference_mask.astype(np.uint8),
                }
                metadata = {
                    "record": {"id": safe_external_id(record_id), "external": True},
                    "view_index": 0,
                    "view_spec": spec,
                    "edit_image": reference,
                    "reference_valid_mask": reference_mask,
                    "camera_condition": {
                        "mode": camera_quality["camera_mode_used"],
                        "spec": spec,
                        "quality": camera_quality,
                        "vector": camera.tolist(),
                    },
                    "causal": {
                        "predicted": predicted,
                        "supervision": {},
                        "valid": valid,
                        "quality": {
                            "depth_finite_ratio": finite_ratio,
                            "confidence_finite_ratio": confidence_finite_ratio,
                            **camera_quality,
                        },
                    },
                }
                metadata["condition_cache"] = save_cached_external_condition(cache_root, key, metadata)
                return reference, metadata
    
        def close(self) -> None:
            self.da3 = None
            self.geocalib = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    
    def hard_preserve_reference(
        generated: Image.Image,
        reference: Image.Image,
        metadata: dict[str, Any],
        feather_pixels: int = 16,
    ) -> Image.Image:
        """Reproject and feather the valid reference into a generated ERP."""
        generated_array = np.asarray(generated.convert("RGB"), dtype=np.float32)
        spec = (metadata.get("camera_condition") or {}).get("spec") or metadata["view_spec"]
        valid_mask = np.asarray(
            metadata.get("reference_valid_mask", np.ones((reference.height, reference.width), dtype=bool)),
            dtype=bool,
        )
        reference_erp, visible = perspective_to_erp_splat(
            np.asarray(reference.convert("RGB"), dtype=np.float32),
            0.0,
            float(spec.get("pitch_deg", 0.0)),
            float(spec.get("hfov_deg", 90.0)),
            generated.width,
            generated.height,
            roll_deg=float(spec.get("roll_deg", 0.0)),
            valid_mask=valid_mask,
        )
        distance = cv2.distanceTransform(visible.astype(np.uint8), cv2.DIST_L2, 5)
        alpha = np.clip(distance / max(1.0, float(feather_pixels)), 0.0, 1.0)
        sigma = max(0.5, float(feather_pixels) / 4.0)
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
        alpha = (alpha * visible.astype(np.float32))[..., None]
        blended = reference_erp * alpha + generated_array * (1.0 - alpha)
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), "RGB")
    
    
    class Stage3ExternalInferenceEngine:
        """Persistent, serialized Stage 3 LoRA + causal-adapter generator."""
    
        def __init__(
            self,
            *,
            model_dir: str | Path,
            diffsynth_root: str | Path,
            lora_path: str | Path,
            adapter_path: str | Path,
            device: str | torch.device = "cuda",
            dtype: torch.dtype = torch.bfloat16,
        ):
            offline_environment(diffsynth_root)
            self.device = torch.device(device)
            self.dtype = dtype
            self.pipe = load_qwen_edit_pipeline(model_dir, torch_dtype=dtype, device=self.device)
            load_pipeline_lora(self.pipe, lora_path)
            self.adapter = load_inference_adapter(adapter_path, self.device, dtype)
            self.controller = CausalAdapterController(self.pipe.dit)
            self._lock = threading.Lock()
            print(f"Stage 3 external inference engine ready on {self.device}", flush=True)
    
        @torch.inference_mode()
        def generate(
            self,
            reference: Image.Image,
            metadata: dict[str, Any],
            *,
            prompt: str,
            width: int = 2048,
            height: int = 1024,
            steps: int = 40,
            seed: int = 3407,
            preserve_reference: str = "soft",
        ) -> tuple[Image.Image, int]:
            if width != 2 * height:
                raise ValueError("Stage 3 output must be a 2:1 equirectangular panorama")
            if preserve_reference not in {"soft", "hard"}:
                raise ValueError(f"Unsupported reference preservation mode: {preserve_reference}")
            with self._lock:
                try:
                    activate_causal_adapter(
                        self.adapter,
                        self.controller,
                        metadata,
                        (height // 16, width // 16),
                        self.device,
                        self.dtype,
                        1.0,
                    )
                    image = run_qwen_edit_inference(
                        self.pipe,
                        prompt=prompt,
                        images=[reference],
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=1.0,
                        true_cfg_scale=1.0,
                        seed=seed,
                    )
                    injections = int(self.controller.injection_count)
                    errors = validate_panorama(image)
                    if errors:
                        raise RuntimeError("; ".join(errors))
                    if injections <= 0:
                        raise RuntimeError("Stage 3 causal adapter hook did not run")
                    image = image.convert("RGB")
                    if preserve_reference == "hard":
                        image = hard_preserve_reference(image, reference, metadata, feather_pixels=16)
                    return image, injections
                finally:
                    self.controller.clear()
    
        def close(self) -> None:
            self.controller.close()
            self.pipe = None
            self.adapter = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    # END SOURCE: src/panogen_ft/external_inference.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: src/panogen_ft/metrics.py
    # ------------------------------------------------------------------------------------------
    """Panorama evaluation metrics."""
    
    # from __future__ import annotations
    
    import json
    import math
    from pathlib import Path
    from typing import Iterable
    
    import numpy as np
    from PIL import Image
    
    from .pano_ops import erp_to_cubemap, erp_to_perspective, load_rgb, perspective_to_erp_splat
    
    
    def _to_float(image: Image.Image | np.ndarray) -> np.ndarray:
        arr = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image).astype(np.float32)
        return arr / 255.0
    
    
    def seam_consistency(image: Image.Image | np.ndarray, width: int = 16) -> float:
        arr = _to_float(image)
        width = max(1, min(width, arr.shape[1] // 8))
        left = arr[:, :width]
        right = arr[:, -width:][:, ::-1]
        return float(1.0 - np.mean(np.abs(left - right)))
    
    
    def seam_gradient_error(image: Image.Image | np.ndarray) -> float:
        """Periodic first-derivative error; lower is better."""
        array = _to_float(image)
        seam = array[:, :1] - array[:, -1:]
        neighbours = 0.5 * ((array[:, 1:2] - array[:, :1]) + (array[:, -1:] - array[:, -2:-1]))
        return float(np.mean(np.abs(seam - neighbours)))
    
    
    def gradient_energy(image: Image.Image | np.ndarray) -> float:
        array = _to_float(image)
        dx = np.diff(array, axis=1)
        dy = np.diff(array, axis=0)
        return float(0.5 * (np.mean(np.abs(dx)) + np.mean(np.abs(dy))))
    
    
    def polar_cubemap_metrics(
        prediction: Image.Image | np.ndarray,
        target: Image.Image | np.ndarray,
        *,
        face_size: int = 256,
        texture_threshold: float = 0.01,
    ) -> dict[str, float | bool]:
        """Measure up/down reconstruction and the locked polar-collapse rule."""
        predicted_faces = erp_to_cubemap(prediction, face_size=face_size)
        target_faces = erp_to_cubemap(target, face_size=face_size)
        psnr_values: list[float] = []
        ratios: list[float] = []
        collapsed: list[bool] = []
        for face in ("up", "down"):
            prediction_face, target_face = predicted_faces[face], target_faces[face]
            psnr_values.append(psnr_image(prediction_face, target_face))
            target_energy = gradient_energy(target_face)
            ratio = gradient_energy(prediction_face) / max(target_energy, 1e-8)
            ratios.append(float(ratio))
            collapsed.append(bool(target_energy >= texture_threshold and ratio < 0.25))
        return {
            "polar_psnr": float(np.mean(psnr_values)),
            "polar_gradient_energy_ratio": float(np.mean(ratios)),
            "polar_collapsed": bool(any(collapsed)),
            "up_collapsed": collapsed[0],
            "down_collapsed": collapsed[1],
        }
    
    
    def rotation_equivariance_error(
        prediction_a: Image.Image | np.ndarray,
        prediction_b: Image.Image | np.ndarray,
        b_to_a_pixels: int,
    ) -> float:
        a = _to_float(prediction_a)
        b = _to_float(prediction_b)
        if a.shape != b.shape:
            raise ValueError(f"Equivariance images must match, got {a.shape} and {b.shape}")
        return float(np.mean(np.abs(a - np.roll(b, int(b_to_a_pixels), axis=1))))
    
    
    def cubemap_edge_consistency(image: Image.Image | np.ndarray, face_size: int = 256, edge_width: int = 8) -> float:
        faces = erp_to_cubemap(image, face_size=face_size)
        del edge_width
        front, right = (faces[name].astype(np.float32) / 255.0 for name in ("front", "right"))
        back, left = (faces[name].astype(np.float32) / 255.0 for name in ("back", "left"))
        up, down = (faces[name].astype(np.float32) / 255.0 for name in ("up", "down"))
        pairs = (
            (front[:, -1], right[:, 0]),
            (right[:, -1], back[:, 0]),
            (back[:, -1], left[:, 0]),
            (left[:, -1], front[:, 0]),
            (front[0], up[-1]),
            (back[0], up[0, ::-1]),
            (right[0], up[::-1, -1]),
            (left[0], up[:, 0]),
            (front[-1], down[0]),
            (back[-1], down[-1, ::-1]),
            (right[-1], down[:, -1]),
            (left[-1], down[::-1, 0]),
        )
        diffs = [float(np.mean(np.abs(first - second))) for first, second in pairs]
        return float(1.0 - np.mean(diffs))
    
    
    def l1_image_distance(a: Image.Image | np.ndarray, b: Image.Image | np.ndarray) -> float:
        aa = _to_float(a)
        bb = _to_float(b)
        if aa.shape != bb.shape:
            b_img = Image.fromarray((bb * 255).astype(np.uint8)).resize((aa.shape[1], aa.shape[0]), Image.Resampling.LANCZOS)
            bb = _to_float(b_img)
        return float(np.mean(np.abs(aa - bb)))
    
    
    def psnr_image(a: Image.Image | np.ndarray, b: Image.Image | np.ndarray) -> float:
        aa = _to_float(a)
        bb = _to_float(b)
        if aa.shape != bb.shape:
            b_img = Image.fromarray(np.clip(bb * 255, 0, 255).astype(np.uint8)).resize(
                (aa.shape[1], aa.shape[0]), Image.Resampling.LANCZOS
            )
            bb = _to_float(b_img)
        mse = float(np.mean(np.square(aa - bb)))
        return 100.0 if mse <= 1e-10 else float(-10.0 * np.log10(mse))
    
    
    def masked_image_metrics(
        prediction: Image.Image | np.ndarray,
        target: Image.Image | np.ndarray,
        mask: np.ndarray,
    ) -> dict[str, float]:
        prediction_array, target_array = _to_float(prediction), _to_float(target)
        if prediction_array.shape != target_array.shape:
            raise ValueError("Masked metrics require matching prediction and target shapes")
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != prediction_array.shape[:2] or not valid.any():
            raise ValueError(f"Invalid/non-empty metric mask: {valid.shape}")
        error = prediction_array[valid] - target_array[valid]
        mse = float(np.mean(np.square(error)))
        return {
            "l1": float(np.mean(np.abs(error))),
            "psnr": 100.0 if mse <= 1e-10 else float(-10.0 * np.log10(mse)),
        }
    
    
    def validate_panorama(image: Image.Image | np.ndarray) -> list[str]:
        array = _to_float(image)
        errors: list[str] = []
        if array.ndim != 3 or array.shape[2] != 3:
            errors.append(f"invalid RGB shape: {array.shape}")
        elif array.shape[1] != 2 * array.shape[0]:
            errors.append(f"output is not 2:1: {array.shape[1]}x{array.shape[0]}")
        if not np.isfinite(array).all():
            errors.append("output contains NaN/Inf")
        return errors
    
    
    def evaluate_generated(
        prediction: Image.Image | np.ndarray,
        target: Image.Image | np.ndarray,
        reference: Image.Image | np.ndarray,
        view_spec: dict,
    ) -> dict[str, float | bool]:
        pred_image = prediction if isinstance(prediction, Image.Image) else Image.fromarray(np.asarray(prediction).astype(np.uint8))
        reference_array = np.asarray(reference)
        reprojection = erp_to_perspective(
            pred_image,
            0.0,
            float(view_spec["pitch_deg"]),
            float(view_spec["hfov_deg"]),
            int(reference_array.shape[1]),
            int(reference_array.shape[0]),
            float(view_spec.get("roll_deg", 0.0)),
        )
        result = {
            "seam_consistency": seam_consistency(pred_image),
            "seam_gradient_error": seam_gradient_error(pred_image),
            "cubemap_edge_consistency": cubemap_edge_consistency(pred_image),
            "reference_l1": l1_image_distance(reprojection, reference),
            "reference_psnr": psnr_image(reprojection, reference),
            "target_l1": l1_image_distance(pred_image, target),
            "target_psnr": psnr_image(pred_image, target),
        }
        target_array = np.asarray(target.convert("RGB") if isinstance(target, Image.Image) else target)
        _, visible = perspective_to_erp_splat(
            np.ones(reference_array.shape[:2], dtype=np.float32),
            0.0,
            float(view_spec["pitch_deg"]),
            float(view_spec["hfov_deg"]),
            target_array.shape[1],
            target_array.shape[0],
            roll_deg=float(view_spec.get("roll_deg", 0.0)),
        )
        visible_metrics = masked_image_metrics(pred_image, target, visible)
        unknown_metrics = masked_image_metrics(pred_image, target, ~visible)
        result.update(
            visible_l1=visible_metrics["l1"],
            visible_psnr=visible_metrics["psnr"],
            unknown_l1=unknown_metrics["l1"],
            unknown_psnr=unknown_metrics["psnr"],
        )
        result.update(polar_cubemap_metrics(pred_image, target))
        return result
    
    
    def paired_metric_summary(baseline: list[dict], candidate: list[dict]) -> dict[str, dict[str, float | int]]:
        base_by_id = {str(item["id"]): item for item in baseline}
        candidate_by_id = {str(item["id"]): item for item in candidate}
        common = sorted(base_by_id.keys() & candidate_by_id.keys())
        result: dict[str, dict[str, float | int]] = {}
        metric_names = (
            "seam_consistency",
            "seam_gradient_error",
            "cubemap_edge_consistency",
            "reference_l1",
            "reference_psnr",
            "target_l1",
            "target_psnr",
            "visible_l1",
            "visible_psnr",
            "unknown_l1",
            "unknown_psnr",
            "polar_psnr",
            "polar_gradient_energy_ratio",
            "polar_collapsed",
        )
        lower_is_better = {
            "seam_gradient_error",
            "reference_l1",
            "target_l1",
            "visible_l1",
            "unknown_l1",
            "polar_collapsed",
        }
        for name in metric_names:
            base = np.asarray([base_by_id[item][name] for item in common], dtype=np.float64)
            cand = np.asarray([candidate_by_id[item][name] for item in common], dtype=np.float64)
            if not len(base):
                continue
            delta = cand - base
            wins = delta < 0 if name in lower_is_better else delta > 0
            result[name] = {
                "count": int(len(base)),
                "baseline_mean": float(base.mean()),
                "baseline_std": float(base.std()),
                "candidate_mean": float(cand.mean()),
                "candidate_std": float(cand.std()),
                "delta_mean": float(delta.mean()),
                "win_rate": float(wins.mean()),
            }
        return result
    
    
    def relative_validation_verdict(
        paired: dict[str, dict[str, float | int]],
        baseline_flow: float | None = None,
        candidate_flow: float | None = None,
        technical_errors: list[str] | None = None,
    ) -> dict[str, object]:
        errors = list(technical_errors or [])
        if errors:
            return {"status": "FAIL", "reasons": errors, "improvements": []}
        warnings: list[str] = []
        improvements: list[str] = []
        if baseline_flow is not None and candidate_flow is not None and baseline_flow > 0:
            relative = (candidate_flow - baseline_flow) / baseline_flow
            if relative > 0.02:
                warnings.append(f"validation flow loss regressed by {relative:.2%} (>2%)")
            elif candidate_flow < baseline_flow:
                improvements.append("validation flow loss improved")
        checks = (
            ("reference_psnr", -0.5, "reference PSNR dropped by more than 0.5 dB", True),
            ("seam_consistency", -0.01, "seam consistency dropped by more than 0.01", True),
        )
        for name, threshold, message, higher in checks:
            if name not in paired:
                continue
            delta = float(paired[name]["delta_mean"])
            if delta < threshold:
                warnings.append(message)
            elif (delta > 0) == higher and delta != 0:
                improvements.append(f"{name} improved")
        if "target_l1" in paired:
            base = float(paired["target_l1"]["baseline_mean"])
            cand = float(paired["target_l1"]["candidate_mean"])
            relative = (cand - base) / max(base, 1e-12)
            if relative > 0.02:
                warnings.append(f"target L1 regressed by {relative:.2%} (>2%)")
            elif cand < base:
                improvements.append("target L1 improved")
        if not improvements:
            warnings.append("no primary metric showed a measurable improvement")
        return {"status": "WARN" if warnings else "PASS", "reasons": warnings, "improvements": improvements}
    
    
    def completion_acceptance_verdict(
        baseline: dict[str, float],
        candidate: dict[str, float],
        *,
        blind_preferred: int,
        blind_total: int,
    ) -> dict[str, object]:
        """Apply all locked repair gates; PASS requires every criterion."""
        required = {
            "polar_lpips",
            "polar_collapse_rate",
            "unknown_lpips",
            "unknown_dino",
            "visible_psnr",
            "seam_gradient_error",
            "cubemap_edge_error",
            "indoor_polar_collapse_rate",
            "outdoor_polar_collapse_rate",
        }
        missing = sorted((required - baseline.keys()) | (required - candidate.keys()))
        if missing:
            return {"status": "FAIL", "reasons": [f"missing metrics: {missing}"], "checks": {}}
    
        def improvement(name: str) -> float:
            return (baseline[name] - candidate[name]) / max(abs(baseline[name]), 1e-12)
    
        if blind_total <= 0 or not (0 <= blind_preferred <= blind_total):
            raise ValueError("Invalid blind preference counts")
        proportion = blind_preferred / blind_total
        z = 1.959963984540054
        denominator = 1.0 + z * z / blind_total
        centre = proportion + z * z / (2.0 * blind_total)
        radius = z * math.sqrt(proportion * (1.0 - proportion) / blind_total + z * z / (4.0 * blind_total**2))
        wilson_lower = (centre - radius) / denominator
        checks = {
            "polar_lpips_improves_15pct": improvement("polar_lpips") >= 0.15,
            "polar_collapse_below_5pct": candidate["polar_collapse_rate"] < 0.05,
            "polar_collapse_halved": candidate["polar_collapse_rate"] <= 0.5 * baseline["polar_collapse_rate"],
            "unknown_improves_10pct": max(improvement("unknown_lpips"), improvement("unknown_dino")) >= 0.10,
            "visible_psnr_drop_within_0_5db": candidate["visible_psnr"] >= baseline["visible_psnr"] - 0.5,
            "seam_regression_within_5pct": candidate["seam_gradient_error"] <= 1.05 * baseline["seam_gradient_error"],
            "cubemap_edge_improves_3pct": improvement("cubemap_edge_error") >= 0.03,
            "blind_preference_at_least_65pct": proportion >= 0.65,
            "blind_wilson_lower_above_50pct": wilson_lower > 0.50,
            "indoor_collapse_below_8pct": candidate["indoor_polar_collapse_rate"] < 0.08,
            "outdoor_collapse_below_8pct": candidate["outdoor_polar_collapse_rate"] < 0.08,
        }
        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "reasons": [name for name, passed in checks.items() if not passed],
            "checks": checks,
            "blind_preference_rate": proportion,
            "blind_preference_wilson95_lower": wilson_lower,
        }
    
    
    def evaluate_pair(pred_path: str | Path, target_path: str | Path | None = None) -> dict:
        pred = load_rgb(pred_path)
        result = {
            "pred_path": str(pred_path),
            "seam_consistency": seam_consistency(pred),
            "cubemap_edge_consistency": cubemap_edge_consistency(pred),
        }
        if target_path:
            target = load_rgb(target_path)
            result["l1_to_target"] = l1_image_distance(pred, target)
        return result
    
    
    def summarize_metrics(items: Iterable[dict]) -> dict:
        items = list(items)
        summary: dict[str, float | int] = {"count": len(items)}
        if not items:
            return summary
        keys = sorted(k for k in items[0].keys() if k not in {"id", "pred_path", "target_path", "prompt"})
        for key in keys:
            values = [float(item[key]) for item in items if key in item]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
        return summary
    
    
    def write_metrics_report(items: list[dict], out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summarize_metrics(items), "items": items}
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    # END SOURCE: src/panogen_ft/metrics.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: src/panogen_ft/metric_logging.py
    # ------------------------------------------------------------------------------------------
    """Restart-safe scalar logging shared by CauPano training and validation."""
    
    # from __future__ import annotations
    
    import json
    import math
    from pathlib import Path
    from typing import Any, Iterable
    
    import numpy as np
    import torch
    
    from .losses import LOSS_COMPONENTS, LOSS_WEIGHTS
    
    
    class LossAccumulator:
        """Accumulate sums/counts locally, then reduce them exactly across ranks."""
    
        def __init__(self, device: torch.device | str):
            self.device = torch.device(device)
            self.sums = {name: torch.zeros((), dtype=torch.float64, device=self.device) for name in ("total", *LOSS_COMPONENTS)}
            self.counts = {name: torch.zeros((), dtype=torch.float64, device=self.device) for name in ("total", *LOSS_COMPONENTS)}
            self.weighted_sums = {
                name: torch.zeros((), dtype=torch.float64, device=self.device) for name in LOSS_COMPONENTS
            }
            self.extra_sums: dict[str, torch.Tensor] = {}
            self.extra_counts: dict[str, torch.Tensor] = {}
    
        def update(self, bundle: dict[str, torch.Tensor]) -> None:
            for name in ("total", *LOSS_COMPONENTS):
                sum_key, count_key = f"sum_loss_{name}", f"count_loss_{name}"
                if sum_key in bundle:
                    self.sums[name] += bundle[sum_key].detach().to(device=self.device, dtype=torch.float64)
                    self.counts[name] += bundle[count_key].detach().to(device=self.device, dtype=torch.float64)
                    if name in self.weighted_sums:
                        weighted_key = f"sum_weighted_loss_{name}"
                        weighted = bundle.get(weighted_key, bundle[sum_key] * float(LOSS_WEIGHTS[name]))
                        self.weighted_sums[name] += weighted.detach().to(device=self.device, dtype=torch.float64)
    
        def add_scalar(self, name: str, value: float | torch.Tensor, count: float = 1.0) -> None:
            tensor = torch.as_tensor(value, device=self.device, dtype=torch.float64).detach()
            self.extra_sums[name] = self.extra_sums.get(name, torch.zeros_like(tensor)) + tensor * float(count)
            self.extra_counts[name] = self.extra_counts.get(name, torch.zeros_like(tensor)) + float(count)
    
        def snapshot(self, accelerator: Any | None = None) -> dict[str, float | int | None]:
            names = ("total", *LOSS_COMPONENTS)
            packed = torch.stack(
                [
                    *(self.sums[name] for name in names),
                    *(self.counts[name] for name in names),
                    *(self.weighted_sums[name] for name in LOSS_COMPONENTS),
                ]
            )
            if accelerator is not None:
                packed = accelerator.reduce(packed, reduction="sum")
            size = len(names)
            sums, counts = packed[:size], packed[size : 2 * size]
            weighted_sums = packed[2 * size :]
            result: dict[str, float | int | None] = {}
            total_count = float(counts[0].item())
            for index, name in enumerate(names):
                count = int(round(float(counts[index].item())))
                value = float(sums[index].item() / count) if count else None
                result[f"loss_{name}"] = value
                result[f"count_loss_{name}"] = count
                if name in LOSS_WEIGHTS:
                    weighted_sum = float(weighted_sums[index - 1].item())
                    result[f"weighted_loss_{name}"] = weighted_sum / count if count else None
                    result[f"contribution_loss_{name}"] = (
                        weighted_sum / total_count if total_count else 0.0
                    )
            for name in sorted(self.extra_sums):
                pair = torch.stack([self.extra_sums[name], self.extra_counts[name]])
                if accelerator is not None:
                    pair = accelerator.reduce(pair, reduction="sum")
                result[name] = float(pair[0].item() / pair[1].item()) if pair[1].item() else None
            return result
    
        def reset(self) -> None:
            for mapping in (self.sums, self.counts, self.weighted_sums):
                for value in mapping.values():
                    value.zero_()
            self.extra_sums.clear()
            self.extra_counts.clear()
    
    
    def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
    
    
    def read_metric_events(path: str | Path) -> list[dict[str, Any]]:
        """Read events and keep the last copy of a split/step after a resumed run."""
    
        path = Path(path)
        if not path.is_file():
            return []
        deduplicated: dict[tuple[str, int], dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                deduplicated[(str(event.get("split", "train")), int(event["step"]))] = event
        return sorted(deduplicated.values(), key=lambda item: (str(item.get("split", "")), int(item["step"])))
    
    
    class MetricWriter:
        def __init__(self, root: str | Path, enabled: bool = True):
            self.root = Path(root)
            self.enabled = bool(enabled)
            self.jsonl_path = self.root / "metrics.jsonl"
            self.tensorboard_dir = self.root / "tensorboard"
            self.writer = None
            if self.enabled:
                self.root.mkdir(parents=True, exist_ok=True)
                try:
                    from torch.utils.tensorboard import SummaryWriter
                except ImportError as exc:
                    raise RuntimeError("TensorBoard is required; install requirements.txt") from exc
                self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir), purge_step=None)
    
        def write(self, split: str, step: int, values: dict[str, Any], **context: Any) -> None:
            if not self.enabled:
                return
            payload = {"split": split, "step": int(step), **context, **values}
            append_jsonl(self.jsonl_path, payload)
            assert self.writer is not None
            for name, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    self.writer.add_scalar(f"{split}/{name}", float(value), int(step))
            self.writer.flush()
    
        def close(self) -> None:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
    
    
    def write_loss_summary(metrics_root: str | Path) -> None:
        """Write a compact JSON summary and a smoothed loss PNG."""
    
        root = Path(metrics_root)
        events = [event for event in read_metric_events(root / "metrics.jsonl") if event.get("split") in {"train", "val"}]
        summary: dict[str, Any] = {"events": len(events), "last": {}}
        for split in ("train", "val"):
            subset = [event for event in events if event.get("split") == split]
            if subset:
                summary["last"][split] = subset[-1]
        (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        train = [event for event in events if event.get("split") == "train" and event.get("loss_total") is not None]
        if not train:
            return
        try:
            import matplotlib
    
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required to write loss_summary.png") from exc
        steps = np.asarray([event["step"] for event in train], dtype=np.int64)
        values = np.asarray([event["loss_total"] for event in train], dtype=np.float64)
        window = max(1, min(25, len(values) // 10 or 1))
        smooth = np.convolve(values, np.ones(window) / window, mode="valid")
        smooth_steps = steps[window - 1 :]
        figure, axis = plt.subplots(figsize=(12, 6))
        axis.plot(steps, values, alpha=0.25, linewidth=0.8, label="global loss")
        axis.plot(smooth_steps, smooth, linewidth=2.0, label=f"moving average ({window})")
        for event in (item for item in events if item.get("split") == "val" and item.get("loss_total") is not None):
            axis.scatter([event["step"]], [event["loss_total"]], marker="x", color="red", zorder=3)
        axis.set(title="CauPano training and validation loss", xlabel="Optimizer step", ylabel="Loss")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(root / "loss_summary.png", dpi=160)
        plt.close(figure)
    # END SOURCE: src/panogen_ft/metric_logging.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: src/panogen_ft/validation.py
    # ------------------------------------------------------------------------------------------
    """Deterministic held-out validation helpers."""
    
    # from __future__ import annotations
    
    import contextlib
    import random
    from typing import Any, Iterator
    
    import numpy as np
    import torch
    
    from .manifest import stable_uint64
    from .metric_logging import LossAccumulator
    
    
    def fixed_validation_indices(dataset: Any, count: int, seed: int = 3407) -> list[int]:
        ranked = sorted(
            range(len(dataset.records)),
            key=lambda index: stable_uint64(str(dataset.records[index]["id"]), int(seed), "fixed-validation"),
        )
        return ranked[: min(int(count), len(ranked))]
    
    
    def validation_seed(record_id: str, seed: int = 3407) -> int:
        return int(stable_uint64(str(record_id), int(seed), "validation-noise") % (2**31 - 1))
    
    
    @contextlib.contextmanager
    def preserved_random_state() -> Iterator[None]:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
    
    
    def seed_validation_sample(record_id: str, seed: int = 3407) -> None:
        value = validation_seed(record_id, seed)
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
    
    
    @torch.no_grad()
    def run_validation_loss(
        accelerator: Any,
        model: torch.nn.Module,
        dataloader: Any,
        seed: int = 3407,
    ) -> dict[str, float | int | None]:
        accumulator = LossAccumulator(accelerator.device)
        unwrapped = accelerator.unwrap_model(model)
        was_training = bool(model.training)
        old_force = bool(getattr(unwrapped, "force_structure", False))
        old_validation = bool(getattr(unwrapped, "validation_mode", False))
        old_step = int(getattr(unwrapped, "global_step", 0))
        with preserved_random_state():
            model.eval()
            unwrapped.force_structure = True
            unwrapped.validation_mode = True
            unwrapped.global_step = int(getattr(unwrapped, "total_steps", 1))
            try:
                for batch in dataloader:
                    record_id = str(batch[0]["metadata"]["record"]["id"])
                    seed_validation_sample(record_id, seed)
                    # Validation has no gradients. Calling the unwrapped ZeRO-2 module
                    # avoids DDP collectives, so ranks may own different sample counts
                    # without padding or duplicating validation records.
                    accumulator.update(unwrapped(batch))
            finally:
                unwrapped.force_structure = old_force
                unwrapped.validation_mode = old_validation
                unwrapped.global_step = old_step
                model.train(was_training)
        return accumulator.snapshot(accelerator)
    # END SOURCE: src/panogen_ft/validation.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: src/panogen_ft/training_state.py
    # ------------------------------------------------------------------------------------------
    """Strict, restart-safe checkpoint discovery for long Slurm training jobs."""
    
    # from __future__ import annotations
    
    import copy
    import hashlib
    import json
    import re
    import shutil
    from pathlib import Path
    from typing import Any
    
    
    CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")
    
    
    def accelerate_scheduler_steps(
        optimizer_steps: int,
        warmup_optimizer_steps: int,
        num_processes: int,
        *,
        split_batches: bool = False,
    ) -> tuple[int, int]:
        """Return scheduler horizons expected by Accelerate's scheduler wrapper.
    
        With the default ``split_batches=False``, Accelerate advances the wrapped
        scheduler once per process for every real optimizer update.  The scheduler
        horizon must therefore use the same process scale; otherwise a four-GPU
        run finishes its cosine schedule after only one quarter of the optimizer
        steps.
        """
    
        optimizer_steps = int(optimizer_steps)
        warmup_optimizer_steps = int(warmup_optimizer_steps)
        num_processes = int(num_processes)
        if optimizer_steps < 1:
            raise ValueError("optimizer_steps must be at least 1")
        if warmup_optimizer_steps < 0 or warmup_optimizer_steps > optimizer_steps:
            raise ValueError("warmup_optimizer_steps must be between 0 and optimizer_steps")
        if num_processes < 1:
            raise ValueError("num_processes must be at least 1")
        scale = 1 if split_batches else num_processes
        return warmup_optimizer_steps * scale, optimizer_steps * scale
    
    
    def training_config_hash(config: dict[str, Any]) -> str:
        """Hash training semantics while allowing resume/logging/validation changes."""
    
        normalized = copy.deepcopy(config)
        normalized.pop("logging", None)
        normalized.pop("validation", None)
        normalized.setdefault("training", {})["resume"] = None
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    
    
    def read_trainer_state(checkpoint: str | Path) -> dict[str, Any]:
        checkpoint = Path(checkpoint)
        state_path = checkpoint / "trainer_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Incomplete training checkpoint: {checkpoint}") from exc
        required = {"config_hash", "epoch", "batch_in_epoch", "global_step"}
        if not isinstance(state, dict) or not required <= state.keys():
            raise RuntimeError(f"Invalid trainer state: {state_path}")
        for name in ("epoch", "batch_in_epoch", "global_step"):
            if int(state[name]) < 0:
                raise RuntimeError(f"Invalid {name} in {state_path}: {state[name]}")
            state[name] = int(state[name])
        return state
    
    
    def completed_checkpoints(output: str | Path) -> list[tuple[int, Path, dict[str, Any]]]:
        output = Path(output)
        result: list[tuple[int, Path, dict[str, Any]]] = []
        for path in output.glob("checkpoint-*"):
            match = CHECKPOINT_PATTERN.fullmatch(path.name)
            if not path.is_dir() or match is None:
                continue
            try:
                state = read_trainer_state(path)
            except RuntimeError:
                continue
            step = int(state["global_step"])
            if step != int(match.group(1)):
                raise RuntimeError(f"Checkpoint name/state mismatch: {path} has global_step={step}")
            result.append((step, path, state))
        return sorted(result, key=lambda item: item[0])
    
    
    def resolve_resume_checkpoint(
        output: str | Path,
        requested: str | Path | None,
        config_hash: str,
    ) -> tuple[Path | None, dict[str, Any] | None]:
        if requested is None or str(requested).strip() == "":
            return None, None
        if str(requested).lower() != "auto":
            path = Path(requested)
            state = read_trainer_state(path)
            if state["config_hash"] != config_hash:
                raise RuntimeError(f"Strict resume rejected: configuration changed for {path}")
            return path, state
    
        candidates = completed_checkpoints(output)
        compatible = [item for item in candidates if item[2]["config_hash"] == config_hash]
        if compatible:
            _, path, state = compatible[-1]
            return path, state
        if candidates:
            raise RuntimeError(
                f"Strict auto-resume rejected: {Path(output)} contains checkpoints from a different configuration"
            )
        return None, None
    
    
    def write_trainer_state(checkpoint: str | Path, state: dict[str, Any]) -> None:
        checkpoint = Path(checkpoint)
        destination = checkpoint / "trainer_state.json"
        temporary = checkpoint / "trainer_state.json.tmp"
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
    
    
    def remove_incomplete_checkpoints(output: str | Path) -> list[Path]:
        """Remove exact checkpoint directories that never received valid trainer state."""
    
        output = Path(output)
        completed = {path.resolve() for _, path, _ in completed_checkpoints(output)}
        removed: list[Path] = []
        for path in output.glob("checkpoint-*"):
            if not path.is_dir() or CHECKPOINT_PATTERN.fullmatch(path.name) is None:
                continue
            if path.resolve() in completed:
                continue
            shutil.rmtree(path)
            removed.append(path)
        return removed
    
    
    def prune_checkpoints(output: str | Path, keep: int) -> list[Path]:
        if keep < 1:
            raise ValueError("checkpoints_total_limit must be at least 1")
        candidates = completed_checkpoints(output)
        removed: list[Path] = []
        for _, path, _ in candidates[:-keep]:
            shutil.rmtree(path)
            removed.append(path)
        return removed
    # END SOURCE: src/panogen_ft/training_state.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: scripts/merge_caupano_sources.py
    # ------------------------------------------------------------------------------------------
    # #!/usr/bin/env python3
    """Merge license-confirmed panorama sources into the single master manifest."""
    
    # from __future__ import annotations
    
    import argparse
    import json
    import sys
    from pathlib import Path
    
    import yaml
    
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    
    from panogen_ft.manifest import OOD_DATASETS, atomic_write_jsonl, iter_jsonl, require_master_manifest
    
    
    def read_unrestricted_jsonl(path: str | Path):
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line_number} is not an object")
                    yield value
    
    
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--catalog", required=True, help="YAML source catalog")
        parser.add_argument("--base", default="", help="Optional existing panorama-all.jsonl")
        parser.add_argument("--output", required=True, help="Must be named panorama-all.jsonl")
        parser.add_argument("--write", action="store_true")
        args = parser.parse_args()
        output = require_master_manifest(args.output)
        catalog = yaml.safe_load(Path(args.catalog).read_text(encoding="utf-8")) or {}
        sources = catalog.get("sources") or []
        if not sources:
            raise ValueError("Catalog has no sources")
        records = list(iter_jsonl(args.base)) if args.base else []
        ids = {str(record["id"]) for record in records}
        counts: dict[str, int] = {}
        for source in sources:
            name = str(source["name"]).strip().lower()
            if not bool(source.get("license_confirmed", False)):
                raise PermissionError(f"License must be confirmed before ingesting {name}")
            category = str(source["data_category"])
            scene_type = str(source["scene_type"])
            count = 0
            for raw in read_unrestricted_jsonl(source["manifest"]):
                original_id = str(raw.get("id") or raw.get("panoid") or "").strip()
                if not original_id:
                    raise ValueError(f"{name} contains an item without id/panoid")
                record_id = original_id if original_id.startswith(name + "-") else f"{name}-{original_id}"
                if record_id in ids:
                    raise ValueError(f"Duplicate merged ID: {record_id}")
                ids.add(record_id)
                scene_id = str(
                    raw.get("source_scene_id")
                    or raw.get("scene_id")
                    or raw.get("sequence_id")
                    or raw.get("panoid")
                    or original_id
                )
                record = {
                    **raw,
                    "id": record_id,
                    "source_dataset": name,
                    "source_scene_id": scene_id,
                    "scene_type": scene_type,
                    "data_category": category,
                    "task": "completion",
                    "license": {
                        "name": str(source.get("license_name") or "confirmed by operator"),
                        "url": str(source.get("license_url") or ""),
                        "confirmed": True,
                    },
                }
                if name in OOD_DATASETS:
                    record["dataset_role"] = "dataset_level_ood"
                records.append(record)
                count += 1
            counts[name] = count
        print(json.dumps({"records": len(records), "added_by_source": counts}, ensure_ascii=False, indent=2))
        if not args.write:
            print("Dry run complete; add --write to create the merged master manifest.")
            return
        atomic_write_jsonl(records, output)
        print(f"Wrote {output}")
    
    
    if __name__ == "__main__":
        main()
    # END SOURCE: scripts/merge_caupano_sources.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: scripts/prepare_panorama_all.py
    # ------------------------------------------------------------------------------------------
    # #!/usr/bin/env python3
    """Enrich the one and only CauPano manifest in place, atomically."""
    
    # from __future__ import annotations
    
    import argparse
    import json
    import shutil
    import sys
    from collections import Counter
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    
    from panogen_ft.dataset import data_category
    from panogen_ft.manifest import (
        atomic_write_jsonl,
        iter_jsonl,
        prepare_records,
        require_master_manifest,
        summarize_duplicates,
    )
    
    
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--manifest",
            default="/data-nfs/gpu2/ud202581898/CauPano/data/caupano/panorama-all.jsonl",
        )
        parser.add_argument("--image-root", default="/data-nfs/gpu2/ud202581898/CauPano/data/caupano")
        parser.add_argument("--embedding-index", default=None, help="Optional NPZ with ids and [N,4,D] DAP/DINOv3 features")
        parser.add_argument("--strict-images", action="store_true")
        parser.add_argument("--max-records", type=int, default=None)
        parser.add_argument("--write", action="store_true", help="Without this flag only validate and print a summary")
        parser.add_argument("--no-backup", action="store_true")
        args = parser.parse_args()
    
        manifest = require_master_manifest(args.manifest)
        records = list(iter_jsonl(manifest))
        if args.max_records is not None:
            records = records[: args.max_records]
        print(f"Loaded {len(records)} records from {manifest}", flush=True)
        print("Inspecting every image (dimensions, SHA256, perceptual hash)", flush=True)
        prepared = prepare_records(
            records,
            image_root=args.image_root,
            strict_images=args.strict_images,
            embedding_index_path=args.embedding_index,
            show_progress=True,
        )
        summary = {
            "records": len(prepared),
            "splits": Counter(record["split"] for record in prepared),
            "scenes": Counter(record.get("scene_type", "unknown") for record in prepared),
            "sources": Counter(record.get("source_dataset", "unknown") for record in prepared),
            "categories": Counter(data_category(record) for record in prepared),
            "images_missing": sum(not record["quality"].get("image_exists", False) for record in prepared),
            "duplicates": summarize_duplicates(prepared),
        }
        if summary["duplicates"]["cross_split_groups"] != 0:
            raise RuntimeError(
                f"Refusing manifest v3 with cross-split leakage: {summary['duplicates']['cross_split_groups']} groups"
            )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))
        if not args.write:
            print("Dry run complete. Add --write to atomically replace panorama-all.jsonl.")
            return
        if args.max_records is not None:
            raise ValueError("Refusing to overwrite panorama-all.jsonl when --max-records is set")
        if not args.no_backup:
            backup = manifest.with_suffix(manifest.suffix + ".pre-caupano-v3.bak")
            if not backup.exists():
                shutil.copy2(manifest, backup)
                print(f"Created backup: {backup}", flush=True)
        print(f"Atomically writing {len(prepared)} records", flush=True)
        atomic_write_jsonl(prepared, manifest)
        print(f"Updated {manifest}")
    
    
    if __name__ == "__main__":
        main()
    # END SOURCE: scripts/prepare_panorama_all.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: scripts/validate_caupano_v3.py
    # ------------------------------------------------------------------------------------------
    # #!/usr/bin/env python3
    """Fail-fast validation for the v3 completion manifest and epoch sampler."""
    
    # from __future__ import annotations
    
    import argparse
    import json
    import sys
    from collections import Counter, defaultdict
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    
    from panogen_ft.dataset import BalancedEpochSampler, completion_prompt, data_category
    from panogen_ft.manifest import MANIFEST_VERSION, OOD_DATASETS, iter_jsonl, stable_uint64, summarize_duplicates
    from panogen_ft.pano_ops import relative_or_absolute
    
    
    REQUIRED = {
        "source_dataset",
        "source_scene_id",
        "scene_type",
        "native_resolution",
        "group_id",
        "split",
        "quality_flags",
        "task",
        "prompt_scope",
        "camera",
    }
    
    
    class RecordDataset:
        def __init__(self, records: list[dict]):
            self.records = records
            self.epoch = 0
    
    
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--manifest", required=True)
        parser.add_argument("--image-root", required=True)
        parser.add_argument("--epoch-size", type=int, default=200000)
        parser.add_argument("--check-images", action="store_true")
        parser.add_argument("--skip-balance", action="store_true")
        parser.add_argument("--report", default="")
        args = parser.parse_args()
    
        records = list(iter_jsonl(args.manifest))
        errors: list[str] = []
        source_counts = Counter()
        category_counts = Counter()
        pitch_counts = Counter()
        aspect_counts = Counter()
        group_splits: dict[str, set[str]] = defaultdict(set)
        for line, record in enumerate(records, 1):
            prefix = f"record {line} ({record.get('id', '?')})"
            missing = REQUIRED - set(record)
            if missing:
                errors.append(f"{prefix}: missing {sorted(missing)}")
                continue
            if int(record.get("manifest_version", 0)) != MANIFEST_VERSION:
                errors.append(f"{prefix}: manifest_version is not {MANIFEST_VERSION}")
            if record["task"] != "completion" or record["prompt_scope"] not in {"generic", "reference", "target"}:
                errors.append(f"{prefix}: invalid task/prompt_scope")
            if record["prompt_scope"] == "target" and not str(record.get("prompt") or "").strip():
                errors.append(f"{prefix}: target prompt_scope has an empty caption")
            specs = (record.get("camera") or {}).get("view_specs") or []
            if len(specs) != 8:
                errors.append(f"{prefix}: expected 8 view_specs, got {len(specs)}")
            else:
                sectors = [int((float(spec["yaw_deg"]) + 180.0) // 45.0) for spec in specs]
                if sectors != list(range(8)):
                    errors.append(f"{prefix}: yaw sectors are {sectors}")
                for spec in specs:
                    pitch = abs(float(spec["pitch_deg"]))
                    pitch_counts["near" if pitch <= 15 else "medium" if pitch <= 45 else "high"] += 1
                    ratio = round(float(spec["width"]) / float(spec["height"]), 1)
                    aspect_counts[str(ratio)] += 1
                    if not (50 <= float(spec["hfov_deg"]) <= 120 and abs(float(spec.get("roll_deg", 0))) <= 30):
                        errors.append(f"{prefix}: invalid FOV/roll in {spec.get('view_id')}")
            source = str(record["source_dataset"]).lower()
            if source in OOD_DATASETS and record["split"] != "test_ood":
                errors.append(f"{prefix}: OOD source {source} leaked into {record['split']}")
            group_splits[str(record["group_id"])].add(str(record["split"]))
            source_counts[source] += 1
            category_counts[data_category(record)] += 1
            if args.check_images and not relative_or_absolute(record["target_path"], args.image_root).is_file():
                errors.append(f"{prefix}: target image does not exist")
    
        leaking = sum(len(splits) > 1 for splits in group_splits.values())
        if leaking:
            errors.append(f"{leaking} group_id values cross dataset splits")
        sampled_categories = Counter()
        sampled_sources = Counter()
        sampled_prompts = Counter()
        if not args.skip_balance:
            train_records = [record for record in records if record.get("split") == "train"]
            dataset = RecordDataset(train_records)
            try:
                sampler = BalancedEpochSampler(dataset, args.epoch_size, require_all_categories=True)
                for index in sampler:
                    sampled_categories[data_category(train_records[index])] += 1
                    sampled_sources[str(train_records[index].get("source_dataset") or "unknown").lower()] += 1
                    record = train_records[index]
                    view_index = stable_uint64(record["id"], 0, "online-view") % 8
                    _, scope = completion_prompt(record, view_index, 0)
                    sampled_prompts[scope] += 1
            except ValueError as exc:
                errors.append(str(exc))
            if sampled_sources and max(sampled_sources.values()) > 0.25 * args.epoch_size:
                errors.append("balanced epoch exceeds the 25% single-source cap")
            expected_prompt_ranges = {"generic": (0.47, 0.53), "reference": (0.32, 0.38), "target": (0.12, 0.18)}
            for scope, (lower, upper) in expected_prompt_ranges.items():
                ratio = sampled_prompts[scope] / max(1, sum(sampled_prompts.values()))
                if not lower <= ratio <= upper:
                    errors.append(f"sampled prompt_scope {scope} ratio is {ratio:.2%}, expected {lower:.0%}-{upper:.0%}")
    
        report = {
            "status": "FAIL" if errors else "PASS",
            "records": len(records),
            "errors": errors[:200],
            "duplicates": summarize_duplicates(records),
            "sources": source_counts,
            "categories": category_counts,
            "pitch_buckets": pitch_counts,
            "aspect_buckets": aspect_counts,
            "sampled_categories": sampled_categories,
            "sampled_sources": sampled_sources,
            "sampled_prompts": sampled_prompts,
        }
        rendered = json.dumps(report, ensure_ascii=False, indent=2, default=dict)
        print(rendered)
        if args.report:
            destination = Path(args.report)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered + "\n", encoding="utf-8")
        if errors:
            raise SystemExit(1)
    
    
    if __name__ == "__main__":
        main()
    # END SOURCE: scripts/validate_caupano_v3.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: scripts/profile_h200_memory.py
    # ------------------------------------------------------------------------------------------
    # #!/usr/bin/env python3
    """Run the locked 100-step 4xH200 memory/throughput candidates."""
    
    # from __future__ import annotations
    
    import argparse
    import json
    import subprocess
    import sys
    from pathlib import Path
    
    import yaml
    
    
    CANDIDATES = {
        "1024": ((4, False, 1), (4, True, 1), (2, False, 2)),
        "2048": ((2, False, 2), (2, True, 2), (1, False, 4)),
    }
    
    
    def last_train_metric(path: Path) -> dict:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        train = [event for event in events if event.get("split") == "train"]
        if not train:
            raise RuntimeError(f"No training metrics in {path}")
        return train[-1]
    
    
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--phase", choices=tuple(CANDIDATES), required=True)
        parser.add_argument("--base-config", required=True)
        parser.add_argument("--work-dir", required=True)
        parser.add_argument("--train-script", default=str(Path(__file__).with_name("train_qwen_edit_pano_lora.py")))
        args = parser.parse_args()
        root = Path(args.work_dir)
        root.mkdir(parents=True, exist_ok=True)
        base = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
        results: list[dict] = []
        for batch, checkpointing, accumulation in CANDIDATES[args.phase]:
            name = f"b{batch}-ckpt{int(checkpointing)}-acc{accumulation}"
            output = root / name
            config = json.loads(json.dumps(base))
            config["training"].update(
                output_dir=str(output),
                per_device_batch_size=batch,
                gradient_accumulation_steps=accumulation,
                gradient_checkpointing=checkpointing,
                epochs=max(1, int(config["training"].get("epochs", 1))),
                max_steps=100,
                checkpointing_steps=1000,
                resume=None,
            )
            config.setdefault("logging", {})["interval_steps"] = 10
            config.setdefault("validation", {})["interval_steps"] = 0
            config_path = root / f"{name}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            accelerate = {
                "compute_environment": "LOCAL_MACHINE",
                "debug": False,
                "distributed_type": "DEEPSPEED",
                "downcast_bf16": "no",
                "machine_rank": 0,
                "main_training_function": "main",
                "mixed_precision": "bf16",
                "num_machines": 1,
                "num_processes": 4,
                "rdzv_backend": "static",
                "same_network": True,
                "use_cpu": False,
                "deepspeed_config": {
                    "deepspeed_multinode_launcher": "standard",
                    "gradient_accumulation_steps": accumulation,
                    "gradient_clipping": 1.0,
                    "offload_optimizer_device": "none",
                    "offload_param_device": "none",
                    "zero3_init_flag": False,
                    "zero3_save_16bit_model": False,
                    "zero_stage": 2,
                },
            }
            accelerate_path = root / f"accelerate-{name}.yaml"
            accelerate_path.write_text(yaml.safe_dump(accelerate, sort_keys=False), encoding="utf-8")
            subprocess.run(
                [
                    "accelerate",
                    "launch",
                    "--config_file",
                    str(accelerate_path),
                    args.train_script,
                    "--config",
                    str(config_path),
                ],
                check=True,
            )
            event = last_train_metric(output / "metrics/metrics.jsonl")
            result = {
                "name": name,
                "batch": batch,
                "checkpointing": checkpointing,
                "accumulation": accumulation,
                "cuda_allocated_gib": float(event["cuda_allocated_gib"]),
                "cuda_reserved_gib": float(event["cuda_reserved_gib"]),
                "samples_per_second": float(event["samples_per_second"]),
            }
            result["eligible"] = result["cuda_allocated_gib"] < 120 and result["cuda_reserved_gib"] < 130
            results.append(result)
        eligible = [result for result in results if result["eligible"]]
        if not eligible:
            raise RuntimeError(f"No candidate met the memory limits: {results}")
        selected = max(eligible, key=lambda item: item["samples_per_second"])
        report = {"phase": args.phase, "selected": selected, "candidates": results}
        (root / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    
    
    if __name__ == "__main__":
        main()
    # END SOURCE: scripts/profile_h200_memory.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: scripts/train_qwen_edit_pano_lora.py
    # ------------------------------------------------------------------------------------------
    # #!/usr/bin/env python3
    """Three-stage CauPano training with complete global metrics and fixed validation."""
    
    # from __future__ import annotations
    
    import argparse
    import json
    import math
    import random
    import sys
    import time
    from pathlib import Path
    
    import numpy as np
    import torch
    import yaml
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader, Subset
    from tqdm.auto import tqdm
    from transformers import get_cosine_schedule_with_warmup
    
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    
    from panogen_ft.caupano_model import (
        DIFFSYNTH_REVISION,
        build_training_module,
        load_adapter_checkpoint,
        offline_environment,
    )
    from panogen_ft.dataset import BalancedEpochSampler, DiffSynthPanoDataset
    from panogen_ft.metric_logging import LossAccumulator, MetricWriter, write_loss_summary
    from panogen_ft.training_state import (
        accelerate_scheduler_steps,
        prune_checkpoints,
        remove_incomplete_checkpoints,
        resolve_resume_checkpoint,
        training_config_hash,
        write_trainer_state,
    )
    from panogen_ft.validation import fixed_validation_indices, run_validation_loss
    
    
    def make_dataset(cfg: dict, split: str, validation: bool = False) -> DiffSynthPanoDataset:
        stage = cfg["training"]["stage"]
        return DiffSynthPanoDataset(
            cfg["data"]["manifest"],
            cfg["data"]["image_root"],
            cfg["data"]["width"],
            cfg["data"]["height"],
            split=split,
            min_native_width=cfg["data"].get("min_native_width", 0),
            caption_dropout_probability=0.0 if validation else cfg["data"].get("caption_dropout", 0.0),
            factor_dropout_probability=0.0 if validation else cfg["data"].get("factor_dropout", 0.15),
            appearance_intervention_probability=0.0 if validation else cfg["data"].get("appearance_intervention", 0.0),
            causal_root=cfg["data"].get("causal_root", cfg["data"]["image_root"]),
            require_causal=bool(cfg["data"].get("require_causal", stage in {"causal", "highres"})),
            use_causal=stage in {"causal", "highres"},
            return_intervention_pair=stage in {"causal", "highres"},
        )
    
    
    def save_trainable(accelerator: Accelerator, model: torch.nn.Module, output: Path) -> None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            filtered = {
                name: parameter.detach().cpu().contiguous()
                for name, parameter in unwrapped.named_parameters()
                if parameter.requires_grad
            }
            output.mkdir(parents=True, exist_ok=True)
            counts = {
                "total_parameters": sum(parameter.numel() for parameter in unwrapped.parameters()),
                "trainable_parameters": sum(parameter.numel() for parameter in unwrapped.parameters() if parameter.requires_grad),
                "lora_trainable_parameters": sum(value.numel() for name, value in filtered.items() if name.startswith("pipe.dit.")),
                "adapter_trainable_parameters": sum(
                    value.numel()
                    for name, value in filtered.items()
                    if name.startswith("causal_adapter.") or name.startswith("structure_heads.")
                ),
            }
            metadata = {"diffsynth_revision": DIFFSYNTH_REVISION, **{name: str(value) for name, value in counts.items()}}
            save_file(filtered, output / "trainable.safetensors", metadata=metadata)
            lora = {name.removeprefix("pipe.dit."): value for name, value in filtered.items() if name.startswith("pipe.dit.")}
            adapters = {
                name: value
                for name, value in filtered.items()
                if name.startswith("causal_adapter.") or name.startswith("structure_heads.")
            }
            if not lora:
                raise RuntimeError("No LoRA weights found while exporting")
            save_file(lora, output / "lora.safetensors", metadata=metadata)
            if adapters:
                save_file(adapters, output / "adapters.safetensors", metadata=metadata)
            (output / "parameter_counts.json").write_text(
                json.dumps(counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    
    
    def log_training_window(
        accelerator: Accelerator,
        writer: MetricWriter,
        accumulator: LossAccumulator,
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
        elapsed: float,
        optimizer_steps: int,
    ) -> dict[str, float | int | None]:
        values = accumulator.snapshot(accelerator)
        memory = torch.tensor(
            [
                torch.cuda.memory_allocated(accelerator.device) if torch.cuda.is_available() else 0,
                torch.cuda.memory_reserved(accelerator.device) if torch.cuda.is_available() else 0,
            ],
            dtype=torch.float64,
            device=accelerator.device,
        )
        memory = accelerator.gather(memory[None]).reshape(-1, 2).max(dim=0).values
        values.update(
            lr_lora=float(optimizer.param_groups[0]["lr"]),
            lr_adapter=float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else None,
            step_time_seconds=float(elapsed) / max(int(optimizer_steps), 1),
            samples_per_second=float(values["count_loss_total"] or 0) / max(float(elapsed), 1e-9),
            cuda_allocated_gib=float(memory[0].item() / 2**30),
            cuda_reserved_gib=float(memory[1].item() / 2**30),
            epoch=int(epoch),
            optimizer_step=int(step),
        )
        writer.write("train", step, values)
        accumulator.reset()
        return values
    
    
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--resume", default=None, help="Checkpoint path or 'auto'; overrides training.resume")
        args = parser.parse_args()
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        offline_environment(Path(cfg["model"]["diffsynth_root"]))
        seed = int(cfg["training"].get("seed", 3407))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        accelerator = Accelerator(
            gradient_accumulation_steps=int(cfg["training"]["gradient_accumulation_steps"]),
            mixed_precision="bf16",
            split_batches=False,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        )
        stage = cfg["training"]["stage"]
        output = Path(cfg["training"]["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        config_hash = training_config_hash(cfg)
        resume_request = args.resume if args.resume is not None else cfg["training"].get("resume")
        if str(resume_request).lower() == "auto" and (output / "final/lora.safetensors").is_file():
            accelerator.print(f"Stage {stage} is already complete: {output / 'final/lora.safetensors'}")
            accelerator.end_training()
            return
    
        train_dataset = make_dataset(cfg, "train")
        sampler = None
        if int(cfg["data"].get("epoch_size", 0)) > 0:
            sampler = BalancedEpochSampler(
                train_dataset,
                epoch_size=int(cfg["data"]["epoch_size"]),
                category_weights=cfg["data"].get("category_weights"),
                seed=seed,
                require_all_categories=bool(cfg["data"].get("require_all_categories", True)),
                max_source_fraction=float(cfg["data"].get("max_source_fraction", 0.25)),
            )
        dataloader = DataLoader(
            train_dataset,
            batch_size=int(cfg["training"]["per_device_batch_size"]),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=int(cfg["data"].get("num_workers", 4)),
            collate_fn=lambda items: items,
            # Workers must see the new deterministic view/prompt epoch.
            persistent_workers=False,
        )
        validation_cfg = cfg.get("validation", {})
        val_dataset = make_dataset(cfg, "val", validation=True)
        val_dataset.set_epoch(0)
        val_indices = fixed_validation_indices(val_dataset, int(validation_cfg.get("loss_samples", 32)), seed)
        rank_val_indices = val_indices[accelerator.process_index :: accelerator.num_processes]
        val_loader = DataLoader(
            Subset(val_dataset, rank_val_indices), batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda items: items
        )
    
        lora_checkpoint = cfg["training"].get("lora_checkpoint")
        selected_initialization = None
        if lora_checkpoint == "auto_probe":
            selection_path = Path(cfg["training"]["initialization_selection"])
            selected_initialization = json.loads(selection_path.read_text(encoding="utf-8"))
            lora_checkpoint = selected_initialization.get("lora_checkpoint")
            accelerator.print(
                f"probe-selected initialization={selected_initialization.get('selected')} "
                f"lora_checkpoint={lora_checkpoint}"
            )
        model = build_training_module(
            Path(cfg["model"]["local_dir"]),
            accelerator.device,
            stage,
            lora_checkpoint,
            cfg["training"],
        )
        adapter_checkpoint = cfg["training"].get("adapter_checkpoint")
        if adapter_checkpoint == "auto_probe":
            if selected_initialization is None:
                selection_path = Path(cfg["training"]["initialization_selection"])
                selected_initialization = json.loads(selection_path.read_text(encoding="utf-8"))
            adapter_checkpoint = selected_initialization.get("adapter_checkpoint")
        if adapter_checkpoint:
            load_adapter_checkpoint(model, adapter_checkpoint, accelerator.device)
        adapter_ids = {
            id(parameter)
            for module in (getattr(model, "causal_adapter", None), getattr(model, "structure_heads", None))
            if module
            for parameter in module.parameters()
        }
        lora_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in adapter_ids]
        adapter_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) in adapter_ids]
        if not lora_parameters:
            raise RuntimeError("No trainable LoRA parameters were created")
        if stage in {"causal", "highres"} and not adapter_parameters:
            raise RuntimeError(f"Stage {stage} did not create trainable causal adapter/structure parameters")
        accelerator.print(
            "trainable parameters: "
            f"lora={sum(parameter.numel() for parameter in lora_parameters):,}, "
            f"adapter_and_structure={sum(parameter.numel() for parameter in adapter_parameters):,}"
        )
        groups = [{"params": lora_parameters, "lr": float(cfg["training"].get("lora_lr", 1e-4))}]
        if adapter_parameters:
            groups.append({"params": adapter_parameters, "lr": float(cfg["training"].get("adapter_lr", 5e-5))})
        optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg["training"].get("weight_decay", 0.01)))
        per_rank_batches = math.ceil(len(dataloader) / accelerator.num_processes)
        updates_per_epoch = math.ceil(per_rank_batches / accelerator.gradient_accumulation_steps)
        total_steps = updates_per_epoch * int(cfg["training"].get("epochs", 1))
        if int(cfg["training"].get("max_steps", 0)) > 0:
            total_steps = min(total_steps, int(cfg["training"]["max_steps"]))
        warmup_optimizer_steps = max(1, round(total_steps * 0.03))
        scheduler_warmup_steps, scheduler_total_steps = accelerate_scheduler_steps(
            total_steps,
            warmup_optimizer_steps,
            accelerator.num_processes,
            split_batches=False,
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, scheduler_warmup_steps, scheduler_total_steps)
        model.total_steps = total_steps
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
        prepared_updates_per_epoch = math.ceil(len(dataloader) / accelerator.gradient_accumulation_steps)
        if prepared_updates_per_epoch != updates_per_epoch:
            raise RuntimeError(
                "Distributed DataLoader step calculation changed: "
                f"expected {updates_per_epoch}, prepared {prepared_updates_per_epoch}"
            )
        accelerator.print(
            "training plan: "
            f"records={len(train_dataset)}, validation_records={len(val_indices)}, processes={accelerator.num_processes}, "
            f"per_device_batch={cfg['training']['per_device_batch_size']}, "
            f"gradient_accumulation={accelerator.gradient_accumulation_steps}, optimizer_steps={total_steps}, "
            f"scheduler_warmup_updates={scheduler_warmup_steps}, scheduler_total_updates={scheduler_total_steps}"
        )
    
        start_epoch = start_batch = global_step = 0
        resume, state = resolve_resume_checkpoint(output, resume_request, config_hash)
        if accelerator.is_main_process:
            removed = remove_incomplete_checkpoints(output)
            for path in removed:
                print(f"Removed incomplete checkpoint: {path}", flush=True)
        accelerator.wait_for_everyone()
        if resume is not None and state is not None:
            accelerator.print(f"Resuming from {resume} at global_step={state['global_step']}")
            accelerator.load_state(str(resume), load_module_strict=False)
            start_epoch, start_batch, global_step = state["epoch"], state["batch_in_epoch"], state["global_step"]
    
        logging_cfg = cfg.get("logging", {})
        log_interval = int(logging_cfg.get("interval_steps", 10))
        validation_interval = int(validation_cfg.get("interval_steps", 1000))
        writer = MetricWriter(output / "metrics", enabled=accelerator.is_main_process)
        accumulator = LossAccumulator(accelerator.device)
        progress = tqdm(total=total_steps, initial=global_step, disable=not accelerator.is_local_main_process)
        epochs = int(cfg["training"].get("epochs", 1))
        window_started: float | None = None
        window_steps = 0
        last_validation_step = -1
        try:
            for epoch in range(start_epoch, epochs):
                if global_step >= total_steps:
                    break
                train_dataset.set_epoch(epoch)
                active_loader = accelerator.skip_first_batches(dataloader, start_batch) if epoch == start_epoch and start_batch else dataloader
                for batch_index, batch in enumerate(active_loader, start=start_batch if epoch == start_epoch else 0):
                    if global_step >= total_steps:
                        break
                    if window_started is None:
                        window_started = time.monotonic()
                    grad_norm = None
                    with accelerator.accumulate(model):
                        accelerator.unwrap_model(model).global_step = global_step
                        bundle = model(batch)
                        accumulator.update(bundle)
                        accelerator.backward(bundle["loss_total"])
                        if accelerator.sync_gradients:
                            grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    if not accelerator.sync_gradients:
                        continue
                    global_step += 1
                    window_steps += 1
                    progress.update(1)
                    if grad_norm is not None:
                        accumulator.add_scalar("grad_norm", grad_norm)
                    if accelerator.is_main_process:
                        progress.set_postfix(loss=f"{bundle['loss_total'].detach().item():.4f}")
                    if global_step % log_interval == 0 or global_step == total_steps:
                        values = log_training_window(
                            accelerator,
                            writer,
                            accumulator,
                            optimizer,
                            global_step,
                            epoch,
                            time.monotonic() - window_started,
                            window_steps,
                        )
                        window_started = None
                        window_steps = 0
                        accelerator.print(
                            f"metrics step={global_step} global_loss={values['loss_total']:.6f} "
                            f"samples/s={values['samples_per_second']:.2f}"
                        )
                    if validation_interval > 0 and global_step % validation_interval == 0:
                        accelerator.print(f"Running fixed validation at optimizer step {global_step}")
                        values = run_validation_loss(accelerator, model, val_loader, seed)
                        writer.write("val", global_step, values, epoch=epoch, optimizer_step=global_step)
                        last_validation_step = global_step
                        accelerator.print(f"validation step={global_step} loss={values['loss_total']}")
                    interval = int(cfg["training"].get("checkpointing_steps", 200))
                    if global_step % interval == 0:
                        checkpoint = output / f"checkpoint-{global_step:06d}"
                        accelerator.print(f"Saving lightweight checkpoint: {checkpoint}")
                        checkpoint_started = time.monotonic()
                        accelerator.save_state(checkpoint, exclude_frozen_parameters=True)
                        if accelerator.is_main_process:
                            write_trainer_state(
                                checkpoint,
                                {
                                    "config_hash": config_hash,
                                    "epoch": epoch,
                                    "batch_in_epoch": batch_index + 1,
                                    "global_step": global_step,
                                },
                            )
                            print(
                                f"Checkpoint complete: {checkpoint} elapsed={time.monotonic() - checkpoint_started:.1f}s",
                                flush=True,
                            )
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            limit = int(cfg["training"].get("checkpoints_total_limit", 2))
                            for path in prune_checkpoints(output, limit):
                                print(f"Removed old checkpoint: {path}", flush=True)
                        accelerator.wait_for_everyone()
                start_batch = 0
    
            save_trainable(accelerator, model, output / "final")
            if last_validation_step != global_step:
                accelerator.print(f"Running final fixed validation at optimizer step {global_step}")
                values = run_validation_loss(accelerator, model, val_loader, seed)
                writer.write("val", global_step, values, epoch=epochs - 1, optimizer_step=global_step)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                write_loss_summary(output / "metrics")
        finally:
            progress.close()
            writer.close()
            accelerator.end_training()
    
    
    if __name__ == "__main__":
        main()
    # END SOURCE: scripts/train_qwen_edit_pano_lora.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: scripts/infer_caupano_external.py
    # ------------------------------------------------------------------------------------------
    # #!/usr/bin/env python3
    """Generate a Stage 3 panorama from an arbitrary external perspective image."""
    
    # from __future__ import annotations
    
    import argparse
    import json
    import sys
    import time
    from pathlib import Path
    
    import torch
    import yaml
    from PIL import Image
    
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    
    from panogen_ft.dataset import DEFAULT_PROMPT  # noqa: E402
    from panogen_ft.external_inference import ExternalCausalPreprocessor, Stage3ExternalInferenceEngine  # noqa: E402
    from panogen_ft.metrics import cubemap_edge_consistency, seam_consistency  # noqa: E402
    
    
    def checkpoint_path(value: str, final_dir: Path, filename: str) -> Path:
        path = final_dir / filename if value == "auto" else Path(value)
        if path.is_dir():
            path = path / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    
    
    def atomic_save(image: Image.Image, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        image.save(temporary)
        temporary.replace(path)
    
    
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--config", default=str(ROOT / "configs/qwen_edit_2511_repair_2048.yaml"))
        parser.add_argument("--input", required=True, help="Local perspective reference image")
        parser.add_argument("--prompt", default=DEFAULT_PROMPT)
        parser.add_argument("--output", default="")
        parser.add_argument("--cache-dir", default=str(ROOT / "output/caupano-inference/condition-cache"))
        parser.add_argument("--models-root", default=str(ROOT / "models/teachers"))
        parser.add_argument("--lora", default="auto")
        parser.add_argument("--adapter", default="auto")
        parser.add_argument("--teacher-device", default="cuda:0")
        parser.add_argument("--generator-device", default="cuda:0")
        parser.add_argument("--hfov", type=float, default=None, help="Optional horizontal FOV override in degrees")
        parser.add_argument("--pitch", type=float, default=None, help="Optional pitch override in degrees")
        parser.add_argument("--roll", type=float, default=None, help="Optional roll override in degrees")
        parser.add_argument("--camera-mode", choices=("auto", "override", "unknown"), default="auto")
        parser.add_argument("--preserve-reference", choices=("soft", "hard"), default="soft")
        parser.add_argument("--width", type=int, default=2048)
        parser.add_argument("--height", type=int, default=1024)
        parser.add_argument("--steps", type=int, default=40)
        parser.add_argument("--seed", type=int, default=3407)
        parser.add_argument("--overwrite", action="store_true")
        args = parser.parse_args()
    
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if str(cfg["training"]["stage"]) != "highres":
            raise ValueError("External Stage 3 inference requires the highres config")
        if args.width != 2 * args.height:
            raise ValueError("Output resolution must be exactly 2:1")
        input_path = Path(args.input)
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        output_path = Path(args.output) if args.output else ROOT / "output/caupano-inference/external" / f"{input_path.stem}.png"
        final_dir = Path(cfg["training"]["output_dir"]) / "final"
        lora_path = checkpoint_path(args.lora, final_dir, "lora.safetensors")
        adapter_path = checkpoint_path(args.adapter, final_dir, "adapters.safetensors")
    
        with Image.open(input_path) as source:
            source = source.convert("RGB")
        preprocessor = ExternalCausalPreprocessor(args.models_root, args.teacher_device)
        engine = None
        try:
            reference, metadata = preprocessor.process(
                source,
                cache_root=args.cache_dir,
                record_id=input_path.stem,
                hfov_deg=args.hfov,
                pitch_deg=args.pitch,
                roll_deg=args.roll,
                camera_mode=args.camera_mode,
            )
            condition_key = str(metadata["condition_cache"]["key"])
            sidecar_path = output_path.with_suffix(".json")
            expected = {
                "condition_key": condition_key,
                "prompt": args.prompt,
                "steps": int(args.steps),
                "seed": int(args.seed),
                "resolution": [int(args.width), int(args.height)],
                "lora": str(lora_path),
                "adapter": str(adapter_path),
                "camera_mode": args.camera_mode,
                "preserve_reference": args.preserve_reference,
            }
            if output_path.is_file() and sidecar_path.is_file() and not args.overwrite:
                try:
                    existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    existing = {}
                if all(existing.get(name) == value for name, value in expected.items()) and int(
                    existing.get("adapter_injections", 0)
                ) > 0:
                    print(f"Reusing completed external inference: {output_path}", flush=True)
                    return
    
            # A one-GPU CLI run unloads teachers before loading the full Qwen
            # generator.  A two-GPU run can keep both resident for lower latency.
            if torch.device(args.teacher_device) == torch.device(args.generator_device):
                preprocessor.close()
            engine = Stage3ExternalInferenceEngine(
                model_dir=cfg["model"]["local_dir"],
                diffsynth_root=cfg["model"]["diffsynth_root"],
                lora_path=lora_path,
                adapter_path=adapter_path,
                device=args.generator_device,
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(torch.device(args.generator_device))
            started = time.perf_counter()
            image, injections = engine.generate(
                reference,
                metadata,
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                steps=args.steps,
                seed=args.seed,
                preserve_reference=args.preserve_reference,
            )
            latency_seconds = time.perf_counter() - started
            peak_memory_gib = (
                torch.cuda.max_memory_allocated(torch.device(args.generator_device)) / 2**30
                if torch.cuda.is_available()
                else 0.0
            )
            atomic_save(image, output_path)
            atomic_save(reference, output_path.with_name(output_path.stem + ".reference.png"))
            payload = {
                **expected,
                "input": str(input_path),
                "output": str(output_path),
                "adapter_injections": injections,
                "condition_cache": metadata["condition_cache"],
                "causal_valid": metadata["causal"]["valid"],
                "causal_quality": metadata["causal"]["quality"],
                "view_spec": metadata["view_spec"],
                "latency_seconds": latency_seconds,
                "peak_cuda_allocated_gib": peak_memory_gib,
                "seam_consistency": float(seam_consistency(image)),
                "cubemap_edge_consistency": float(cubemap_edge_consistency(image)),
            }
            sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"External Stage 3 inference complete: {output_path}", flush=True)
            print(
                f"adapter_injections={injections} condition_cache_hit={metadata['condition_cache']['hit']} ",
                f"seam={payload['seam_consistency']:.6f} cubemap={payload['cubemap_edge_consistency']:.6f}",
                flush=True,
            )
        finally:
            if engine is not None:
                engine.close()
            preprocessor.close()
    
    
    if __name__ == "__main__":
        main()
    # END SOURCE: scripts/infer_caupano_external.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: scripts/validate_caupano_stage.py
    # ------------------------------------------------------------------------------------------
    # #!/usr/bin/env python3
    """Offline, deterministic loss and A/B generation validation for every CauPano stage."""
    
    # from __future__ import annotations
    
    import argparse
    import gc
    import json
    import re
    import sys
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any
    
    import numpy as np
    import torch
    import yaml
    from accelerate import Accelerator
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader, Subset
    
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    
    from panogen_ft.caupano_model import (
        CausalAdapterController,
        activate_causal_adapter,
        build_training_module,
        load_adapter_checkpoint,
        load_inference_adapter,
        offline_environment,
    )
    from panogen_ft.dataset import DiffSynthPanoDataset
    from panogen_ft.metric_logging import append_jsonl
    from panogen_ft.metrics import (
        evaluate_generated,
        paired_metric_summary,
        relative_validation_verdict,
        validate_panorama,
    )
    from panogen_ft.qwen_edit_wrapper import load_pipeline_lora, load_qwen_edit_pipeline, run_qwen_edit_inference
    from panogen_ft.validation import fixed_validation_indices, run_validation_loss, seed_validation_sample, validation_seed
    
    
    @dataclass(frozen=True)
    class CheckpointSpec:
        stage: str
        lora: Path | None
        adapter: Path | None
    
    
    RAW_BASELINE_NAMES = {"base", "none", "qwen-base", "raw", "raw-base"}
    
    
    def stage_number(stage: str) -> int:
        return {"flow": 1, "causal": 2, "highres": 3}[stage]
    
    
    def _lora_path(value: str | Path | None) -> Path | None:
        if value is None or str(value).strip() in {"", "base", "none"}:
            return None
        path = Path(value)
        if path.is_dir():
            path = path / "lora.safetensors"
        return path
    
    
    def checkpoint_specs(cfg: dict, candidate: str, baseline: str) -> tuple[CheckpointSpec, CheckpointSpec]:
        stage = str(cfg["training"]["stage"])
        output = Path(cfg["training"]["output_dir"]) / "final"
        candidate_lora = output / "lora.safetensors" if candidate == "auto" else _lora_path(candidate)
        candidate_adapter = candidate_lora.parent / "adapters.safetensors" if stage != "flow" and candidate_lora else None
        baseline_name = str(baseline).strip().lower()
        if baseline_name in RAW_BASELINE_NAMES:
            # The untouched Qwen-Image-Edit model has neither a LoRA nor a causal
            # adapter.  Its validation objective must therefore be the original
            # Flow Matching loss, even when the candidate is Stage 2/3.
            baseline_lora, baseline_adapter, baseline_stage = None, None, "flow"
        elif baseline_name == "auto":
            if stage == "flow":
                baseline_lora, baseline_adapter, baseline_stage = None, None, "flow"
            elif stage == "causal":
                baseline_lora = _lora_path(cfg["training"].get("lora_checkpoint"))
                baseline_adapter, baseline_stage = None, "flow"
            else:
                baseline_lora = _lora_path(cfg["training"].get("lora_checkpoint"))
                baseline_adapter = Path(cfg["training"]["adapter_checkpoint"])
                baseline_stage = "highres"
        else:
            baseline_lora = _lora_path(baseline)
            baseline_adapter = baseline_lora.parent / "adapters.safetensors" if stage == "highres" and baseline_lora else None
            baseline_stage = "highres" if stage == "highres" else "flow"
        return (
            CheckpointSpec(stage, candidate_lora, candidate_adapter),
            CheckpointSpec(baseline_stage, baseline_lora, baseline_adapter),
        )
    
    
    def make_validation_dataset(cfg: dict) -> DiffSynthPanoDataset:
        stage = cfg["training"]["stage"]
        return DiffSynthPanoDataset(
            cfg["data"]["manifest"],
            cfg["data"]["image_root"],
            cfg["data"]["width"],
            cfg["data"]["height"],
            split="val",
            min_native_width=cfg["data"].get("min_native_width", 0),
            caption_dropout_probability=0.0,
            factor_dropout_probability=0.0,
            appearance_intervention_probability=0.0,
            causal_root=cfg["data"].get("causal_root", cfg["data"]["image_root"]),
            require_causal=stage in {"causal", "highres"},
            use_causal=stage in {"causal", "highres"},
            return_intervention_pair=stage in {"causal", "highres"},
        )
    
    
    def require_checkpoint(spec: CheckpointSpec, label: str) -> None:
        if spec.lora is not None and not spec.lora.is_file():
            raise FileNotFoundError(f"Missing {label} LoRA: {spec.lora}")
        if spec.adapter is not None and not spec.adapter.is_file():
            raise FileNotFoundError(f"Missing {label} causal adapter: {spec.adapter}")
    
    
    def evaluate_loss_checkpoint(
        accelerator: Accelerator,
        cfg: dict,
        dataset: DiffSynthPanoDataset,
        indices: list[int],
        spec: CheckpointSpec,
    ) -> dict[str, float | int | None]:
        model = build_training_module(cfg["model"]["local_dir"], accelerator.device, spec.stage, str(spec.lora) if spec.lora else None)
        if spec.adapter:
            load_adapter_checkpoint(model, spec.adapter, accelerator.device)
        rank_indices = indices[accelerator.process_index :: accelerator.num_processes]
        loader = DataLoader(Subset(dataset, rank_indices), batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda items: items)
        values = run_validation_loss(accelerator, model, loader, int(cfg["training"].get("seed", 3407)))
        controller = getattr(model, "adapter_controller", None)
        if controller is not None:
            controller.close()
        del model, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        return values
    
    
    def safe_id(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    
    
    def atomic_save(image: Image.Image, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        image.save(temporary)
        temporary.replace(path)
    
    
    def valid_existing_image(
        path: Path,
        spec: CheckpointSpec,
        steps: int,
        resolution: tuple[int, int],
        record_id: str,
    ) -> bool:
        sidecar = path.with_suffix(".json")
        if not path.is_file() or not sidecar.is_file():
            return False
        try:
            with Image.open(path) as image:
                if validate_panorama(image):
                    return False
                if image.size != resolution:
                    return False
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            expected = {
                "id": record_id,
                "stage": spec.stage,
                "lora": str(spec.lora) if spec.lora else None,
                "adapter": str(spec.adapter) if spec.adapter else None,
                "steps": int(steps),
                "resolution": list(resolution),
            }
            if any(payload.get(name) != value for name, value in expected.items()):
                return False
            return spec.adapter is None or int(payload.get("adapter_injections", 0)) > 0
        except (OSError, ValueError, json.JSONDecodeError):
            return False
    
    
    @torch.inference_mode()
    def generate_variant(
        accelerator: Accelerator,
        cfg: dict,
        dataset: DiffSynthPanoDataset,
        indices: list[int],
        spec: CheckpointSpec,
        output: Path,
        name: str,
        steps: int,
    ) -> None:
        variant_dir = output / name
        rank_indices = indices[accelerator.process_index :: accelerator.num_processes]
        width, height = int(cfg["data"]["width"]), int(cfg["data"]["height"])
        pending: list[int] = []
        for index in rank_indices:
            item = dataset.sample_pil(index)
            record_id = safe_id(str(item["record"]["id"]))
            if name == "baseline":
                reference_path = output / "reference" / f"{record_id}.png"
                target_path = output / "target" / f"{record_id}.png"
                if not reference_path.is_file():
                    atomic_save(item["edit_image"], reference_path)
                if not target_path.is_file():
                    atomic_save(item["image"], target_path)
            if not valid_existing_image(
                variant_dir / f"{record_id}.png",
                spec,
                steps,
                (width, height),
                str(item["record"]["id"]),
            ):
                pending.append(index)
        if not pending:
            accelerator.print(f"{name}: reusing {len(rank_indices)} completed images")
            accelerator.wait_for_everyone()
            return
    
        pipe = load_qwen_edit_pipeline(cfg["model"]["local_dir"], torch_dtype=torch.bfloat16, device=accelerator.device)
        if spec.lora is not None:
            load_pipeline_lora(pipe, spec.lora)
        adapter = controller = None
        if spec.adapter is not None:
            adapter = load_inference_adapter(spec.adapter, accelerator.device, torch.bfloat16)
            controller = CausalAdapterController(pipe.dit)
        seed = int(cfg["training"].get("seed", 3407))
        try:
            for ordinal, index in enumerate(pending, 1):
                item = dataset.sample_pil(index)
                record_id = safe_id(str(item["record"]["id"]))
                seed_validation_sample(str(item["record"]["id"]), seed)
                try:
                    if adapter is not None and controller is not None:
                        activate_causal_adapter(
                            adapter,
                            controller,
                            item,
                            (height // 16, width // 16),
                            accelerator.device,
                            torch.bfloat16,
                            1.0,
                        )
                    image = run_qwen_edit_inference(
                        pipe,
                        prompt=item["prompt"],
                        images=[item["edit_image"]],
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=1.0,
                        true_cfg_scale=1.0,
                        seed=validation_seed(str(item["record"]["id"]), seed),
                    )
                    injections = controller.injection_count if controller is not None else 0
                    errors = validate_panorama(image)
                    if errors:
                        raise RuntimeError(f"{record_id}: {'; '.join(errors)}")
                    if controller is not None and injections <= 0:
                        raise RuntimeError(f"{record_id}: causal adapter hook did not run")
                    destination = variant_dir / f"{record_id}.png"
                    atomic_save(image, destination)
                    destination.with_suffix(".json").write_text(
                        json.dumps(
                            {
                                "id": item["record"]["id"],
                                "stage": spec.stage,
                                "lora": str(spec.lora) if spec.lora else None,
                                "adapter": str(spec.adapter) if spec.adapter else None,
                                "steps": int(steps),
                                "resolution": [width, height],
                                "seed": validation_seed(str(item["record"]["id"]), seed),
                                "adapter_injections": injections,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    if name == "baseline":
                        atomic_save(item["edit_image"], output / "reference" / f"{record_id}.png")
                        atomic_save(item["image"], output / "target" / f"{record_id}.png")
                    print(f"[{name}] {ordinal}/{len(pending)} {record_id}", flush=True)
                finally:
                    if controller is not None:
                        controller.clear()
        finally:
            if controller is not None:
                controller.close()
            del pipe, adapter, controller
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
    
    
    def make_contact_sheet(paths: list[tuple[str, Path]], output: Path, prompt: str = "") -> None:
        images: list[tuple[str, Image.Image]] = []
        thumb_width, thumb_height, label_height, prompt_height = 320, 160, 24, 44
        for label, path in paths:
            with Image.open(path) as image:
                images.append((label, image.convert("RGB").resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)))
        sheet = Image.new("RGB", (thumb_width * len(images), thumb_height + label_height + prompt_height), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 5), "prompt: " + prompt[:220], fill="black")
        for index, (label, image) in enumerate(images):
            x = index * thumb_width
            sheet.paste(image, (x, prompt_height + label_height))
            draw.text((x + 6, prompt_height + 5), label, fill="black")
        atomic_save(sheet, output)
    
    
    def write_report_png(report: dict[str, Any], path: Path) -> None:
        try:
            import matplotlib
    
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required to write report.png") from exc
        paired = report.get("paired_metrics", {})
        names = list(paired)
        deltas = [float(paired[name]["delta_mean"]) for name in names]
        colors = ["#2ca02c" if (delta < 0 if name.endswith("_l1") else delta > 0) else "#d62728" for name, delta in zip(names, deltas)]
        figure, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].barh(names, deltas, color=colors)
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].set_title("Candidate - baseline (green is better)")
        loss = report.get("loss_validation", {})
        labels, values = [], []
        for variant in ("baseline", "candidate"):
            value = loss.get(variant, {}).get("loss_flow")
            if value is not None:
                labels.append(variant)
                values.append(value)
        axes[1].bar(labels, values, color=["#7f7f7f", "#1f77b4"][: len(values)])
        axes[1].set_title("Fixed validation flow loss")
        figure.suptitle(f"CauPano validation: {report['verdict']['status']}")
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
    
    
    def write_validation_tensorboard(report: dict[str, Any], output: Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError("TensorBoard is required; install requirements.txt") from exc
        writer = SummaryWriter(log_dir=str(output / "tensorboard"))
        try:
            for variant, values in report.get("loss_validation", {}).items():
                for name, value in values.items():
                    if isinstance(value, (int, float)) and value is not None:
                        writer.add_scalar(f"loss/{variant}/{name}", float(value), 0)
            for name, values in report.get("paired_metrics", {}).items():
                for field in ("baseline_mean", "candidate_mean", "delta_mean", "win_rate"):
                    if field in values:
                        writer.add_scalar(f"generation/{name}/{field}", float(values[field]), 0)
            writer.flush()
        finally:
            writer.close()
    
    
    def collect_generation_report(dataset: DiffSynthPanoDataset, indices: list[int], output: Path) -> tuple[list[dict], list[dict], list[str]]:
        baseline_items: list[dict] = []
        candidate_items: list[dict] = []
        errors: list[str] = []
        for index in indices:
            item = dataset.sample_pil(index)
            record_id = safe_id(str(item["record"]["id"]))
            reference = output / "reference" / f"{record_id}.png"
            target = output / "target" / f"{record_id}.png"
            paths = {
                "baseline": output / "baseline" / f"{record_id}.png",
                "candidate": output / "candidate" / f"{record_id}.png",
            }
            for variant, path in paths.items():
                if not path.is_file():
                    errors.append(f"missing {variant} image: {path.name}")
                    continue
                with Image.open(path) as prediction, Image.open(target) as target_image, Image.open(reference) as reference_image:
                    image_errors = validate_panorama(prediction)
                    errors.extend(f"{variant}/{record_id}: {message}" for message in image_errors)
                    metrics = evaluate_generated(prediction, target_image, reference_image, item["view_spec"])
                payload = {"id": str(item["record"]["id"]), "prompt": item["prompt"], **metrics}
                (baseline_items if variant == "baseline" else candidate_items).append(payload)
            if all(path.is_file() for path in paths.values()):
                make_contact_sheet(
                    [
                        ("reference", reference),
                        ("target", target),
                        ("baseline", paths["baseline"]),
                        ("candidate", paths["candidate"]),
                    ],
                    output / "grids" / f"{record_id}.png",
                    item["prompt"],
                )
        return baseline_items, candidate_items, errors
    
    
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--candidate", default="auto")
        parser.add_argument(
            "--baseline",
            default="auto",
            help="auto uses the preceding stage; base/raw compares against untouched Qwen-Image-Edit-2511",
        )
        parser.add_argument("--mode", choices=("loss", "generate", "all"), default="all")
        parser.add_argument("--output", default="")
        parser.add_argument("--max-samples", type=int, default=0, help="Override both loss and generation counts for a smoke test")
        parser.add_argument("--steps", type=int, default=0)
        args = parser.parse_args()
    
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        offline_environment(cfg["model"]["diffsynth_root"])
        accelerator = Accelerator(mixed_precision="bf16")
        stage = str(cfg["training"]["stage"])
        number = stage_number(stage)
        output = Path(args.output) if args.output else ROOT / "output" / "caupano-validation" / f"stage{number}"
        if accelerator.is_main_process:
            output.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()
        candidate, baseline = checkpoint_specs(cfg, args.candidate, args.baseline)
        loss_results: dict[str, Any] = {}
        try:
            require_checkpoint(candidate, "candidate")
            require_checkpoint(baseline, "baseline")
            dataset = make_validation_dataset(cfg)
            dataset.set_epoch(0)
            validation_cfg = cfg.get("validation", {})
            loss_count = int(args.max_samples or validation_cfg.get("loss_samples", {1: 64, 2: 32, 3: 16}[number]))
            generation_count = int(args.max_samples or validation_cfg.get("generation_samples", {1: 32, 2: 32, 3: 16}[number]))
            steps = int(args.steps or validation_cfg.get("generation_steps", 40))
            seed = int(cfg["training"].get("seed", 3407))
            loss_indices = fixed_validation_indices(dataset, loss_count, seed)
            generation_indices = fixed_validation_indices(dataset, generation_count, seed)
            if args.mode in {"loss", "all"}:
                accelerator.print(f"Evaluating baseline fixed loss on {len(loss_indices)} samples")
                loss_results["baseline"] = evaluate_loss_checkpoint(accelerator, cfg, dataset, loss_indices, baseline)
                accelerator.print(f"Evaluating candidate fixed loss on {len(loss_indices)} samples")
                loss_results["candidate"] = evaluate_loss_checkpoint(accelerator, cfg, dataset, loss_indices, candidate)
            if args.mode in {"generate", "all"}:
                generate_variant(accelerator, cfg, dataset, generation_indices, baseline, output, "baseline", steps)
                generate_variant(accelerator, cfg, dataset, generation_indices, candidate, output, "candidate", steps)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                baseline_items: list[dict] = []
                candidate_items: list[dict] = []
                technical_errors: list[str] = []
                paired: dict[str, Any] = {}
                if args.mode in {"generate", "all"}:
                    baseline_items, candidate_items, technical_errors = collect_generation_report(dataset, generation_indices, output)
                    paired = paired_metric_summary(baseline_items, candidate_items)
                verdict = relative_validation_verdict(
                    paired,
                    loss_results.get("baseline", {}).get("loss_flow"),
                    loss_results.get("candidate", {}).get("loss_flow"),
                    technical_errors,
                )
                report = {
                    "stage": number,
                    "stage_name": stage,
                    "resolution": [int(cfg["data"]["width"]), int(cfg["data"]["height"])],
                    "candidate": {
                        "stage": candidate.stage,
                        "lora": str(candidate.lora),
                        "adapter": str(candidate.adapter) if candidate.adapter else None,
                    },
                    "baseline": {
                        "stage": baseline.stage,
                        "lora": str(baseline.lora) if baseline.lora else None,
                        "adapter": str(baseline.adapter) if baseline.adapter else None,
                    },
                    "loss_validation": loss_results,
                    "paired_metrics": paired,
                    "verdict": verdict,
                    "items": {"baseline": baseline_items, "candidate": candidate_items},
                }
                (output / "report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
                )
                metrics_path = output / "metrics.jsonl"
                metrics_path.write_text("", encoding="utf-8")
                for variant, items in (("baseline", baseline_items), ("candidate", candidate_items)):
                    for item in items:
                        append_jsonl(metrics_path, {"split": "generation", "variant": variant, **item})
                for variant, values in loss_results.items():
                    append_jsonl(metrics_path, {"split": "loss", "variant": variant, **values})
                write_report_png(report, output / "report.png")
                write_validation_tensorboard(report, output)
                print(f"Validation {verdict['status']}: {output / 'report.json'}", flush=True)
        except Exception as exc:
            if accelerator.is_main_process:
                output.mkdir(parents=True, exist_ok=True)
                failure = {"stage": number, "verdict": {"status": "FAIL", "reasons": [f"{type(exc).__name__}: {exc}"]}}
                (output / "report.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        finally:
            accelerator.end_training()
    
    
    if __name__ == "__main__":
        main()
    # END SOURCE: scripts/validate_caupano_stage.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: scripts/select_probe_initialization.py
    # ------------------------------------------------------------------------------------------
    # #!/usr/bin/env python3
    """Select the 2k-step repair initialization using the locked composite metric."""
    
    # from __future__ import annotations
    
    import argparse
    import json
    import os
    import tempfile
    from pathlib import Path
    
    
    WEIGHTS = {
        "unknown_lpips": 0.30,
        "polar_cubemap_lpips": 0.25,
        "visible_lpips": 0.20,
        "seam_error": 0.15,
        "cubemap_edge_error": 0.10,
    }
    
    
    def read_metrics(path: str | Path) -> dict[str, float]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metrics = payload.get("metrics", payload)
        missing = sorted(set(WEIGHTS) - set(metrics))
        if missing:
            raise ValueError(f"{path} lacks metrics: {missing}")
        values = {name: float(metrics[name]) for name in WEIGHTS}
        if any(value <= 0 for value in values.values()):
            raise ValueError(f"All lower-is-better metrics must be positive: {path}")
        return values
    
    
    def normalized_score(candidate: dict[str, float], baseline: dict[str, float]) -> float:
        return sum(WEIGHTS[name] * candidate[name] / baseline[name] for name in WEIGHTS)
    
    
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--baseline", required=True, help="Current Stage 3 evaluation JSON")
        parser.add_argument("--probe-a", required=True, help="Stage-3-LoRA probe evaluation JSON")
        parser.add_argument("--probe-b", required=True, help="Original-Qwen probe evaluation JSON")
        parser.add_argument("--probe-a-lora", required=True)
        parser.add_argument("--probe-b-lora", required=True)
        parser.add_argument("--probe-a-adapter", required=True)
        parser.add_argument("--probe-b-adapter", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args()
    
        baseline = read_metrics(args.baseline)
        score_a = normalized_score(read_metrics(args.probe_a), baseline)
        score_b = normalized_score(read_metrics(args.probe_b), baseline)
        relative_gap = abs(score_a - score_b) / max(min(score_a, score_b), 1e-12)
        selected = "A" if score_a <= score_b or relative_gap < 0.01 else "B"
        result = {
            "selected": selected,
            "lora_checkpoint": args.probe_a_lora if selected == "A" else args.probe_b_lora,
            "adapter_checkpoint": args.probe_a_adapter if selected == "A" else args.probe_b_adapter,
            "score_a": score_a,
            "score_b": score_b,
            "relative_gap": relative_gap,
            "tie_rule": "A when normalized score gap is below 1%",
            "weights": WEIGHTS,
        }
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    
    if __name__ == "__main__":
        main()
    # END SOURCE: scripts/select_probe_initialization.py

    # ------------------------------------------------------------------------------------------
    # BEGIN SOURCE: scripts/summarize_distribution_metrics.py
    # ------------------------------------------------------------------------------------------
    # #!/usr/bin/env python3
    """Compute FID/KID/FAED from extracted features and aggregate Q-Align/NIQE."""
    
    # from __future__ import annotations
    
    import argparse
    import json
    from pathlib import Path
    
    import numpy as np
    
    
    def load_features(path: str | Path) -> np.ndarray:
        payload = np.load(path, allow_pickle=False)
        value = payload["features"] if isinstance(payload, np.lib.npyio.NpzFile) else payload
        value = np.asarray(value, dtype=np.float64)
        if value.ndim != 2 or value.shape[0] < 2 or not np.isfinite(value).all():
            raise ValueError(f"Invalid [N,D] features: {path} -> {value.shape}")
        return value
    
    
    def frechet_distance(real: np.ndarray, generated: np.ndarray) -> float:
        mean_real, mean_generated = real.mean(0), generated.mean(0)
        covariance_real = np.cov(real, rowvar=False)
        covariance_generated = np.cov(generated, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_real)
        root_real = (eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))) @ eigenvectors.T
        middle = root_real @ covariance_generated @ root_real
        middle_eigenvalues = np.linalg.eigvalsh((middle + middle.T) * 0.5)
        trace_root = float(np.sqrt(np.clip(middle_eigenvalues, 0, None)).sum())
        difference = mean_real - mean_generated
        return float(difference @ difference + np.trace(covariance_real) + np.trace(covariance_generated) - 2 * trace_root)
    
    
    def kid(real: np.ndarray, generated: np.ndarray) -> float:
        dimension = real.shape[1]
        if generated.shape[1] != dimension:
            raise ValueError("KID feature dimensions do not match")
        kernel_rr = (real @ real.T / dimension + 1.0) ** 3
        kernel_gg = (generated @ generated.T / dimension + 1.0) ** 3
        kernel_rg = (real @ generated.T / dimension + 1.0) ** 3
        n, m = len(real), len(generated)
        return float(
            (kernel_rr.sum() - np.trace(kernel_rr)) / (n * (n - 1))
            + (kernel_gg.sum() - np.trace(kernel_gg)) / (m * (m - 1))
            - 2.0 * kernel_rg.mean()
        )
    
    
    def quality_means(path: str | Path) -> dict[str, float]:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        result = {}
        for name in ("q_align", "niqe"):
            values = [float(row[name]) for row in rows if row.get(name) is not None]
            if len(values) != len(rows):
                raise ValueError(f"Every quality row must contain {name}")
            result[name] = float(np.mean(values))
        return result
    
    
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--real-inception", required=True)
        parser.add_argument("--generated-inception", required=True)
        parser.add_argument("--real-faed", required=True, help="Real ERP/tangent panorama embeddings")
        parser.add_argument("--generated-faed", required=True)
        parser.add_argument("--quality-jsonl", required=True, help="One q_align and niqe value per generation")
        parser.add_argument("--output", required=True)
        args = parser.parse_args()
        real_inception, generated_inception = load_features(args.real_inception), load_features(args.generated_inception)
        real_faed, generated_faed = load_features(args.real_faed), load_features(args.generated_faed)
        if len(generated_inception) < 5000 or len(generated_faed) < 5000:
            raise ValueError("Distribution evaluation requires at least 5000 generated results")
        report = {
            "count": int(len(generated_inception)),
            "fid": frechet_distance(real_inception, generated_inception),
            "kid": kid(real_inception, generated_inception),
            "faed": frechet_distance(real_faed, generated_faed),
            **quality_means(args.quality_jsonl),
        }
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    
    
    if __name__ == "__main__":
        main()
