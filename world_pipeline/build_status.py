"""Read-only source-build progress. No model, scheduler or geometry work."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('qwen_edit_pano/data/gt_sources_14train3val_v1'))
    args = parser.parse_args()
    source = json.loads((args.root/'SOURCE_RUN.json').read_text())
    expected = Counter(r['scene_id'] for r in source['rows'])
    path = args.root/'build_progress.json'
    progress = json.loads(path.read_text()) if path.exists() else {}
    done = failed = blocked = 0
    print('building       complete sample_fail building_blocked expected status')
    for house in sorted(expected):
        entry = progress.get(house, {})
        n = len(entry.get('completed', []))
        f = len(entry.get('failures', []))
        b = expected[house] if entry.get('status') in ('FAILED_CANONICAL', 'MISSING_ARCHIVES') else 0
        done += n; failed += f; blocked += b
        print(f'{house:14} {n:8} {f:11} {b:16} {expected[house]:8} {entry.get("status", "NOT_REPORTED")}')
        if entry.get('error'):
            print('  ' + entry['error'])
    total = sum(expected.values())
    print(json.dumps(dict(total=total, source_worlds_completed=done, sample_failures=failed,
        samples_blocked_by_building_failure=blocked,
        remaining_unreported=total-done-failed-blocked,
        progress_age_seconds=round(time.time()-path.stat().st_mtime, 1) if path.exists() else None,
        filesystem_free_gib=round(shutil.disk_usage(args.root).free/2**30, 2),
        build_finished=(args.root/'BUILD_RESULT.json').exists(),
        note='Completed source worlds are NOT exported/approved training GT. Counts are a live snapshot.'), indent=2))


if __name__ == '__main__':
    main()
