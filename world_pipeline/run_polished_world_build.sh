#!/usr/bin/env bash
# User-run CPU construction. Never downloads or requests/uses a GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."

world_build_python="${WORLD_DATA_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python}"
world_build_output="${WORLD_BUILD_OUTPUT:-qwen_edit_pano/data/world_polished_full_v2}"
world_build_workers="${WORLD_BUILD_WORKERS:-2}"
world_build_threads="${WORLD_BUILD_THREADS:-4}"
action="${1:-build}"

case "$action" in
  check)
    "$world_build_python" -u -m qwen_edit_pano.world_pipeline.prepare_world_downloads
    ;;
  build)
    "$world_build_python" -u -m qwen_edit_pano.world_pipeline.prepare_world_downloads
    "$world_build_python" - <<'PY'
import json
from pathlib import Path
p=Path('qwen_edit_pano/data/world_polished_full_candidates_v2/download_plan/SUMMARY.json')
r=json.loads(p.read_text())
if not r['ready_for_minimal_build']:
    raise SystemExit('Required sources are still missing; inspect '+str(p))
PY
    build_args=(--output "$world_build_output" --workers "$world_build_workers" --threads "$world_build_threads")
    if [[ -f "$world_build_output/CONFIG.json" ]]; then build_args+=(--resume); fi
    "$world_build_python" -u -m qwen_edit_pano.world_pipeline.build_polished_world "${build_args[@]}" \
      2>&1 | tee -a "${world_build_output}.log"
    ;;
  status)
    "$world_build_python" - "$world_build_output" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); result=p/'BUILD_RESULT.json'; progress=p/'progress.json'
if result.is_file():
    r=json.loads(result.read_text())
    print(json.dumps({k:v for k,v in r.items() if k!='failures'},ensure_ascii=False,indent=2))
    print('failure_count:',len(r['failures']))
if progress.is_file():
    houses=json.loads(progress.read_text())
    print('houses_reported:',len(houses))
samples=p/'samples'
if samples.is_dir():
    print('sample_records_present:',sum(1 for _ in samples.glob('*/record.json')))
    print('sample_failure_reports:',sum(1 for _ in samples.glob('*/failure.json')))
if not result.is_file() and not progress.is_file() and not samples.is_dir():
    raise SystemExit('Build has not started at '+str(p))
PY
    ;;
  *)
    printf '%s\n' 'Usage: bash qwen_edit_pano/world_pipeline/run_polished_world_build.sh check|build|status' >&2
    exit 2
    ;;
esac
