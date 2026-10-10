#!/usr/bin/env bash
# Fixed, isolated 14 train / 3 val experiment. User submits srun explicitly.
set -euo pipefail
cd "$(dirname "$0")/../.."
pipeline_python=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
pipeline_pairs=qwen_edit_pano/data/paired_gt_14train3val_v1
pipeline_sources=qwen_edit_pano/data/gt_sources_14train3val_v1
pipeline_gt=qwen_edit_pano/data/gt_world_14train3val_v1
pipeline_bundle=qwen_edit_pano/data/oracle_14train3val_epoch006_v1
pipeline_checkpoint=qwen_edit_pano/outputs/paired_full_63637/checkpoint-epoch006

check_selection() {
  "$pipeline_python" - "$pipeline_pairs" <<'PY'
import json, hashlib, sys
from pathlib import Path
p=Path(sys.argv[1]); report=json.loads((p/'selection.json').read_text())
seen=set()
for split, expected in [('train',14),('val',3)]:
    source=Path(next(k for k in report['source_manifests'] if k.endswith('/'+split+'.jsonl')))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=report['source_manifests'][str(source)]:
        raise SystemExit('Original paired manifest changed')
    canonical={r['id']:r for r in map(json.loads,source.read_text().splitlines())}
    rows=[json.loads(s) for s in (p/(split+'.jsonl')).read_text().splitlines() if s.strip()]
    houses=set(report['houses'][split]); ids=[r['id'] for r in rows]
    if (len(houses)!=expected or houses & seen or len(ids)!=len(set(ids)) or
        len(rows)!=report['counts'][split] or set(ids)!={k for k,v in canonical.items() if v['scene_id'] in houses} or
        any(r!=canonical[r['id']] or r['split']!=split for r in rows)):
        raise SystemExit('Selected paired subset changed or split mismatch')
    seen.update(houses)
if (p/'scans.txt').read_text().splitlines()!=sorted(seen):
    raise SystemExit('Download scan list differs from selected buildings')
print('Selection verified: 14 train / 3 val buildings; 1290 / 174 candidate pairs.',flush=True)
PY
}

plan_sources() {
  local resume_args=()
  if [[ -d "$pipeline_sources" ]]; then resume_args+=(--resume); fi
  "$pipeline_python" -u -m qwen_edit_pano.world_pipeline.expand_gt_sources \
    --pairs "$pipeline_pairs" --work-root "$pipeline_sources" --threads 4 "${resume_args[@]}" "$@"
}

check_selection
case "${1:-plan}" in
  plan) plan_sources ;;
  download)
    "$pipeline_python" -u download_mp_py3.py \
      -o benchmark_assets/Matterport3D_raw --scan-list "$pipeline_pairs/scans.txt" \
      --type house_segmentations region_segmentations matterport_camera_intrinsics \
        matterport_camera_poses undistorted_camera_parameters undistorted_color_images \
        undistorted_depth_images undistorted_normal_images \
      --retries 8 --timeout 60 --retry-delay 5 --verify-existing
    plan_sources
    ;;
  build) plan_sources --build ;;
  export)
    "$pipeline_python" - "$pipeline_sources" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'BUILD_RESULT.json'
if not p.exists() or not json.loads(p.read_text())['all_requested_built']:
    raise SystemExit('Build incomplete; inspect build_progress.json / BUILD_RESULT.json first')
PY
    resume_args=()
    if [[ -d "$pipeline_gt" ]]; then resume_args+=(--resume); fi
    "$pipeline_python" -u -m qwen_edit_pano.world_pipeline.prepare_gt \
      --pairs "$pipeline_pairs" --world-root "$pipeline_sources" \
      --train-limit 0 --val-limit 0 --threads 4 --require-all-pairs \
      --output "$pipeline_gt" "${resume_args[@]}"
    ;;
  bundle)
    "$pipeline_python" - "$pipeline_gt" "$pipeline_pairs" <<'PY'
import json,sys,collections
from pathlib import Path
gt,pairs=map(Path,sys.argv[1:]); selection=json.loads((pairs/'selection.json').read_text())
review=json.loads((gt/'review_decisions.json').read_text())
canonical={r['id']:r for s in ('train','val') for r in map(json.loads,(pairs/(s+'.jsonl')).read_text().splitlines())}
accepted=[r for r in map(json.loads,(gt/'gt_manifest.jsonl').read_text().splitlines())
          if review.get(r['id'],{}).get('decision')=='approved' and not r['review_flags']]
for split in ('train','val'):
    houses={canonical[r['id']]['scene_id'] for r in accepted if r['split']==split}
    missing=set(selection['houses'][split])-houses
    if missing:
        raise SystemExit(f'No approved flag-free samples in {split} buildings: {sorted(missing)}. Review data; do not auto-approve.')
print('Reviewed eligible sample counts:',dict(collections.Counter(r['split'] for r in accepted)))
PY
    # Keep ORIGINAL full paired manifests/cache: they are bound to epoch006.
    "$pipeline_python" -m qwen_edit_pano.world_pipeline.oracle_data \
      --gt "$pipeline_gt" --review "$pipeline_gt/review_decisions.json" \
      --base-checkpoint "$pipeline_checkpoint" --output "$pipeline_bundle"
    "$pipeline_python" -m qwen_edit_pano.world_pipeline.oracle_train \
      --bundle "$pipeline_bundle" --purpose oracle --preflight
    ;;
  train-full|train-constant)
    condition="${1#train-}"
    resume_args=()
    run_output="qwen_edit_pano/outputs/oracle_14train3val_epoch006_${condition}_v1"
    # Optional positional resume checkpoint and NEW output directory.
    if [[ $# -ne 1 && $# -ne 3 ]]; then echo 'Usage: train-full|train-constant [resume-checkpoint new-output]' >&2; exit 2; fi
    if [[ $# -eq 3 ]]; then resume_args=(--resume "$2"); run_output="$3"; fi
    "$pipeline_python" -u -m qwen_edit_pano.world_pipeline.oracle_train \
      --bundle "$pipeline_bundle" --condition "$condition" --purpose oracle \
      --steps 1000 --warmup-steps 100 --save-every 100 --accumulation 4 --lr 1e-4 --seed 0 \
      --activation-storage cpu --output "$run_output" "${resume_args[@]}"
    ;;
  eval-full|eval-constant)
    condition="${1#eval-}"
    modes=(correct)
    if [[ "$condition" == full ]]; then modes=(baseline correct shuffled); fi
    # Optional checkpoint and NEW output for resumed runs or other saved steps.
    if [[ $# -ne 1 && $# -ne 3 ]]; then echo 'Usage: eval-full|eval-constant [checkpoint new-output]' >&2; exit 2; fi
    eval_checkpoint="qwen_edit_pano/outputs/oracle_14train3val_epoch006_${condition}_v1/checkpoint-step001000"
    eval_output="qwen_edit_pano/outputs/oracle_14train3val_epoch006_${condition}_val_v1"
    if [[ $# -eq 3 ]]; then eval_checkpoint="$2"; eval_output="$3"; fi
    "$pipeline_python" -u -m qwen_edit_pano.world_pipeline.oracle_inference \
      --checkpoint "$eval_checkpoint" --split val --limit 174 \
      --seed 0 --steps 28 --cfg 4 --modes "${modes[@]}" --output "$eval_output"
    ;;
  *) echo 'Usage: plan|download|build|export|bundle|train-full|train-constant|eval-full|eval-constant' >&2; exit 2 ;;
esac
