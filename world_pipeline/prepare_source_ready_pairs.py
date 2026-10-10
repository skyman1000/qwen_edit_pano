"""Explicitly select successful source builds; never approve GT or relax checks."""
import argparse
from collections import Counter
from pathlib import Path
from .common import read, rows, sha, new_output
from .checkpoint_io import atomic_json, atomic_rows, output_lock, verify_files
from .expand_gt_sources import validate_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    with output_lock(args.sources):
        result = read(args.sources/'BUILD_RESULT.json')
        source_config = read(args.sources/'SOURCE_RUN.json')
        selected = [r for split in ('train','val') for r in rows(args.pairs/f'{split}.jsonl')]
        expected = sorted([dict(id=r['id'],scene_id=r['scene_id'],source_view_id=r['source_view_id'],split=r['split'])
                           for r in selected], key=lambda r:(r['scene_id'],r['id']))
        if expected != source_config['rows']:
            raise ValueError('Build selection differs from paired manifests')
        completed, excluded = set(), []
        for house, entry in result['houses'].items():
            done = entry.get('completed', [])
            if len(done) != len(set(done)) or completed.intersection(done):
                raise ValueError('Duplicate completed ID')
            if entry['status'] not in ('PARTIAL','REVIEW_REQUIRED','FAILED_CANONICAL','MISSING_ARCHIVES'):
                raise ValueError('Unexpected build status')
            completed.update(done)
        if completed - {r['id'] for r in selected}:
            raise ValueError('Unknown completed sample')
        accepted = []
        for row in selected:
            sid,h,u = row['id'],row['scene_id'],row['source_view_id']
            entry = result['houses'][h]
            failure = next((f['error'] for f in entry.get('failures',[]) if f['id']==sid),None)
            if sid not in completed:
                excluded.append(dict(id=sid,split=row['split'],scene_id=h,
                                     reason=failure or entry.get('error') or entry['status']))
                continue
            if failure or entry['status'] not in ('PARTIAL','REVIEW_REQUIRED'):
                raise ValueError('Inconsistent success/failure record')
            geo = args.sources/'geometry_pilot'/h/u
            world = args.sources/'world_state_pilot'/h/u
            validate_report(geo/'geometry_report.json', {'BUILT_FOR_REVIEW'}, sid)
            validate_report(world/'build_report.json', {'BUILT'}, sid)
            marker = args.sources/'checkpoints'/f'{sid}_world_state_pilot.json'
            verify_files(world, read(marker)['files'])
            if read(world/'world_state.json')['sample_id'] != sid:
                raise ValueError('World identity mismatch')
            accepted.append(row)
        if not all(any(r['split']==split for r in accepted) for split in ('train','val')):
            raise ValueError('Need source-ready train and val')
        output = new_output(args.output)
        for split in ('train','val'):
            atomic_rows(output/f'{split}.jsonl', [r for r in accepted if r['split']==split])
        report = dict(status='SOURCE_READY_SUBSET_NOT_GT_APPROVAL', original_count=len(selected),
            selected_count=len(accepted), excluded_count=len(excluded),
            counts=dict(Counter(r['split'] for r in accepted)),
            houses={split:sorted({r['scene_id'] for r in accepted if r['split']==split}) for split in ('train','val')},
            missing_houses={split:sorted({r['scene_id'] for r in selected if r['split']==split} -
                                        {r['scene_id'] for r in accepted if r['split']==split}) for split in ('train','val')},
            excluded=excluded, sources=str(args.sources.resolve()),
            protected_files={str(f.resolve()):sha(f) for f in [args.sources/'BUILD_RESULT.json',
                args.sources/'SOURCE_RUN.json',args.pairs/'train.jsonl',args.pairs/'val.jsonl']},
            note='Explicit reduced coverage. Same GT fields and thresholds. Export/review still required.')
        atomic_json(output/'selection.json', report)
        print({k:report[k] for k in ('status','selected_count','excluded_count','counts','houses','missing_houses')})


if __name__ == '__main__':
    main()
