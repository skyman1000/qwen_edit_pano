"""Record a user's per-sample visual review. Never certifies training readiness."""
import argparse
from datetime import datetime, timezone, timedelta

from .common import read, rows, sha, write


def main():
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--id',action='append',required=True,help='Repeat for specifically inspected samples')
    p.add_argument('--decision',choices=['approved','rejected','pending'],required=True)
    p.add_argument('--notes',required=True)
    args=p.parse_args()
    report=read(args.root/'EXPORT_COMPLETE.json')
    if sha(args.root/'gt_manifest.jsonl') != report['gt_manifest_sha256']:
        raise ValueError('Export manifest changed')
    index={r['id']:r for r in rows(args.root/'gt_manifest.jsonl')}
    decisions=read(args.root/'review_decisions.json')
    for sid in args.id:
        if sid not in index:
            raise ValueError(f'Unknown sample: {sid}')
        if args.decision=='approved' and index[sid]['review_flags']:
            raise ValueError(f'Unresolved numerical flags for {sid}; inspect bridge.json, do not override')
        for name,digest in index[sid]['gt_sha256'].items():
            if sha(args.root/index[sid]['directory']/f'{name}.json') != digest:
                raise ValueError('GT changed after export')
    for sid in args.id:
        decisions[sid]=dict(decision=args.decision,notes=args.notes,
                           recorded_at=datetime.now(timezone(timedelta(hours=8))).isoformat())
    write(args.root/'review_decisions.json',decisions)
    print(f'Recorded {len(args.id)} explicit visual decisions; training readiness remains false.')


if __name__=='__main__':
    main()
