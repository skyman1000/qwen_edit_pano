"""Prepare building-disjoint inventories; these are NOT a new training entry point."""
import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://raw.githubusercontent.com/niessner/Matterport/master/tasks/benchmark/"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "qwen_pano/data/cached_official_full/train.jsonl")
    p.add_argument("--raw-root", type=Path, default=ROOT / "benchmark_assets/Matterport3D_raw")
    p.add_argument("--output", type=Path, default=ROOT / "qwen_edit_pano/data/official_building_split")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    owner, sources = {}, {}
    for split in ("train", "val", "test"):
        name = f"scenes_{split}.txt"
        cached = args.output / name
        if not cached.exists():
            content = urllib.request.urlopen(BASE + name, timeout=60).read()
            cached.write_bytes(content)
        content = cached.read_bytes()
        houses = content.decode().split()
        for house in houses:
            if house in owner:
                raise ValueError("Duplicate or overlapping official house: " + house)
            owner[house] = split
        sources[split] = {"url": BASE + name, "sha256": hashlib.sha256(content).hexdigest(), "houses": len(houses)}
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    keys = [(r["scene_id"], r["source_view_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate panorama identity")
    unknown = sorted({h for h, _ in keys} - owner.keys())
    if unknown:
        raise ValueError(f"Houses absent from official splits: {unknown}")
    inventory, invalid = {}, []
    for house in sorted({h for h, _ in keys}):
        archive = args.raw_root / "v1/scans" / house / "matterport_skybox_images.zip"
        if not archive.is_file():
            continue
        try:
            with zipfile.ZipFile(archive) as zf:
                members = {}
                for entry in zf.infolist():
                    parts = [v for v in entry.filename.split("/") if v]
                    if len(parts) == 3 and parts[:2] == [house, "matterport_skybox_images"] and entry.file_size > 0:
                        if parts[-1] in members:
                            raise ValueError("Duplicate member basename")
                        members[parts[-1]] = entry.filename
            inventory[house] = (str(archive.resolve()), members)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            invalid.append({"archive": str(archive), "error": str(exc)})
    counts = {}
    for split in ("train", "val", "test"):
        full, available = [], []
        for row in rows:
            house, view = row["scene_id"], row["source_view_id"]
            if owner[house] != split:
                continue
            item = {**row, "split": split, "original_split": row.get("split")}
            full.append(item)
            if house in inventory:
                archive, members = inventory[house]
                names = [f"{view}_skybox{i}_sami.jpg" for i in range(6)]
                if all(name in members for name in names):
                    available.append({**item, "local_condition": {"archive": archive, "member": members[names[2]], "face_index": 2},
                                      "pair_alignment_verified": False})
        write_jsonl(args.output / f"{split}.full.jsonl", full)
        write_jsonl(args.output / f"{split}.available.jsonl", available)
        counts[split] = {"full": len(full), "available": len(available), "available_houses": len({r["scene_id"] for r in available})}
    report = {"sources": sources, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "counts": counts, "invalid_archives": invalid, "crc_verified": False,
              "training_ready": False, "note": "Identity coverage only. Existing text-only training ignores local_condition; do not use it for paired training. Full lists may lack local files."}
    (args.output / "split_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if invalid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
