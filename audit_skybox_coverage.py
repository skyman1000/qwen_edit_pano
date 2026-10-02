"""Read-only ZIP inventory; does not certify image/ERP geometric alignment."""
import argparse
import collections
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACE = re.compile(r"^([0-9a-fA-F]{32})_skybox([0-5])_sami\.jpg$")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "qwen_pano/data/cached_official_full/train.jsonl")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "benchmark_assets/Matterport3D_raw")
    parser.add_argument("--output", type=Path, default=ROOT / "qwen_edit_pano/data/skybox_coverage.json")
    parser.add_argument("--verify-crc", action="store_true", help="Read every ZIP member; can take considerable time")
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty training manifest")
    keys = [(row["scene_id"], row["source_view_id"]) for row in rows]
    if any(not house or not view for house, view in keys) or len(set(keys)) != len(keys):
        raise ValueError("Missing or duplicate scene_id + source_view_id; cannot claim unique pair coverage")
    houses = sorted({house for house, _ in keys})
    faces = collections.defaultdict(set)
    invalid, missing_archives = [], []
    for house in houses:
        archive = args.raw_root / "v1/scans" / house / "matterport_skybox_images.zip"
        if not archive.is_file():
            missing_archives.append(house)
            continue
        try:
            with zipfile.ZipFile(archive) as zf:
                if args.verify_crc:
                    bad = zf.testzip()
                    if bad is not None:
                        raise ValueError("CRC failure: " + bad)
                local = collections.defaultdict(set)
                for entry in zf.infolist():
                    parts = [part for part in entry.filename.split("/") if part]
                    match = FACE.fullmatch(parts[-1]) if parts else None
                    if not match:
                        continue
                    if parts != [house, "matterport_skybox_images", parts[-1]]:
                        raise ValueError("Unexpected member identity: " + entry.filename)
                    if entry.file_size <= 0:
                        raise ValueError("Empty image: " + entry.filename)
                    view, face = match.group(1), int(match.group(2))
                    if face in local[view]:
                        raise ValueError("Duplicate face: " + entry.filename)
                    local[view].add(face)
                if not local:
                    raise ValueError("No recognized skybox members")
                for view, found in local.items():
                    faces[(house, view)] = found
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, EOFError) as exc:
            invalid.append({"archive": str(archive), "error": str(exc)})
    missing = [{"scene_id": house, "source_view_id": view,
                "missing_faces": sorted(set(range(6)) - faces[(house, view)])}
               for house, view in keys if faces[(house, view)] != set(range(6))]
    summary = {
        "manifest": str(args.manifest.resolve()), "raw_root": str(args.raw_root.resolve()),
        "total_unique_erp": len(keys), "required_houses": len(houses),
        "matched_face2": sum(2 in faces[key] for key in keys),
        "matched_all_six_faces": len(keys) - len(missing),
        "missing_archives": missing_archives, "invalid_archives": invalid,
        "crc_verified": args.verify_crc, "missing_pairs": missing,
        "all_manifest_rows_have_six_faces": not missing and not invalid,
        "geometric_alignment_verified": False,
        "note": "File coverage only; not a training-ready paired manifest or an independent test split.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("missing_pairs", "missing_archives", "invalid_archives")}, indent=2))
    print(f"Missing house ZIPs: {len(missing_archives)}; invalid ZIPs: {len(invalid)}; report: {args.output}")
    if invalid:
        print("Downloader skips existing files. Inspect invalid_archives in the report; move confirmed broken ZIPs aside before retrying.")
    if (args.fail_on_invalid and invalid) or (args.require_full and (missing or invalid)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
