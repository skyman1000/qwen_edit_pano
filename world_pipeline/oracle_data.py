"""Prepare a NEW reviewed GT oracle bundle; reuse existing paired RGB and caches."""
import argparse
from pathlib import Path
from .common import read,rows,sha,write,write_rows,new_output
from .oracle_model import tensorize,FEATURE_VERSION


def load_bundle(root):
    root=Path(root)
    report=read(root/'BUNDLE.json')
    if report['feature_version']!=FEATURE_VERSION:
        raise ValueError('Feature version mismatch')
    for path,digest in report['protected_files'].items():
        if sha(path)!=digest:
            raise ValueError(f'Oracle input changed: {path}')
    entries=rows(root/'samples.jsonl')
    for entry in entries:
        if sha(entry['graph'])!=entry['graph_sha256']:
            raise ValueError('GT graph changed')
        graph=read(entry['graph'])
        if graph['id']!=entry['id'] or graph['split']!=entry['split'] or graph['stage']!='gt' or graph['scope']!='full':
            raise ValueError('Oracle requires matched GT G_full, never Observer predictions')
        tensorize(graph,report['max_objects'])
    return report,entries


def main():
    from qwen_edit_pano.paired_data import paired_rows,cache_key,CONDITIONING
    from qwen_edit_pano.common import versions,require_versions
    require_versions()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--paired-train',type=Path,default=Path('qwen_edit_pano/data/paired_full_v1/train.jsonl'))
    p.add_argument('--paired-val',type=Path,default=Path('qwen_edit_pano/data/paired_full_v1/val.jsonl'))
    p.add_argument('--cache',type=Path,default=Path('qwen_edit_pano/cache/paired_full_v1'))
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--max-objects',type=int,default=128)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.max_objects<1:
        p.error('max-objects must be positive')
    checkpoint=args.base_checkpoint.resolve()
    progress=read(checkpoint/'COMPLETE.json')
    if progress.get('completed_epochs',0)<1:
        raise ValueError('Use a completed formal panorama epoch, not smoke')
    pano=read(checkpoint/'pano_config.json'); training=read(checkpoint/'training_config.json')
    if pano['conditioning']!=CONDITIONING:
        raise ValueError('Not a paired Edit checkpoint')
    cache=read(args.cache/'cache_config.json'); cache_done=read(args.cache/'complete.json')
    train=paired_rows(args.paired_train,'train'); val=paired_rows(args.paired_val,'val')
    # Heldout identity report is fixed by the selected training checkpoint.
    heldout_path=Path(training['heldout_manifest'])
    heldout=rows(heldout_path)
    heldout_houses={r['scene_id'] for r in heldout}
    if any(r['scene_id'] in heldout_houses for r in train):
        raise ValueError('Training house appears in heldout')
    if {r['scene_id'] for r in train}&{r['scene_id'] for r in val}:
        raise ValueError('Train/val house overlap')
    if (cache['manifest_sha256']!=sha(args.paired_train) or
        training['panorama_sha256']!=sha(args.paired_train) or
        cache['base_snapshot']!=pano['base_snapshot'] or cache['versions']!=versions() or
        cache['conditioning']!=CONDITIONING or cache['prompt']!=pano['prompt'] or
        cache_done['config_sha256']!=sha(args.cache/'cache_config.json') or
        cache_done['count']!=2*len(train)):
        raise ValueError('Cache/checkpoint/manifest mismatch')
    for name,digest in cache['cache_sources'].items():
        if sha(Path('qwen_edit_pano')/name)!=digest:
            raise ValueError('Cached conditioning source changed')
    complete=read(args.gt/'EXPORT_COMPLETE.json')
    if sha(args.gt/'gt_manifest.jsonl')!=complete['gt_manifest_sha256']:
        raise ValueError('GT manifest changed')
    review=read(args.review); canonical={r['id']:r for r in train+val}
    if len(canonical)!=len(train)+len(val):
        raise ValueError('Duplicate paired identity')
    vocabulary=read(args.gt/'vocabulary.json')
    protected={str(path.resolve()):sha(path) for path in [args.paired_train,args.paired_val,
        heldout_path,args.review,args.gt/'gt_manifest.jsonl',args.gt/'vocabulary.json',
        args.gt/'contract.json',args.cache/'cache_config.json',
        checkpoint/'pano_config.json',checkpoint/'training_config.json',checkpoint/'COMPLETE.json',
        checkpoint/'adapter_resume.safetensors',checkpoint/'pytorch_lora_weights.safetensors']}
    entries=[]; excluded=[]
    for record in rows(args.gt/'gt_manifest.jsonl'):
        sid=record['id']
        if review.get(sid,{}).get('decision')!='approved' or record['review_flags']:
            excluded.append(dict(id=sid,reason='unreviewed_or_flags',flags=record['review_flags']));continue
        row=canonical[sid]
        if row['split']!=record['split']:
            raise ValueError('Split mismatch')
        directory=args.gt/record['directory']
        provenance=read(directory/'provenance.json')
        for key in ('image','local_condition','target_alignment'):
            if row[key]!=provenance['paired_row'][key]:
                raise ValueError(f'GT vs paired source/alignment mismatch: {sid}/{key}')
        graphpath=directory/'G_full.json'; graph=read(graphpath)
        if (sha(graphpath)!=record['gt_sha256']['G_full'] or graph['stage']!='gt' or
            graph['vocabulary_sha256']!=vocabulary['category_mapping_sha256']):
            raise ValueError('Invalid GT graph')
        tensorize(graph,args.max_objects)
        condition=args.cache/(cache_key(row,False)+'.safetensors')
        if row['split']=='train':
            protected[str(condition.resolve())]=sha(condition)
        # No caption or Observer output is included in this bundle.
        paired={k:row[k] for k in ('id','split','scene_id','source_view_id','image','local_condition','target_alignment')}
        entries.append(dict(id=sid,split=row['split'],paired=paired,graph=str(graphpath.resolve()),
            graph_sha256=sha(graphpath),condition_cache=str(condition.resolve()) if row['split']=='train' else None))
    if not any(e['split']=='train' for e in entries):
        raise ValueError('No reviewed train samples')
    out=new_output(args.output); write_rows(out/'samples.jsonl',entries)
    protected[str(out/'samples.jsonl')]=sha(out/'samples.jsonl')
    report=dict(feature_version=FEATURE_VERSION,base_checkpoint=str(checkpoint),
        base_snapshot=pano['base_snapshot'],height=pano['height'],padding_columns=pano['padding_columns'],
        max_objects=args.max_objects,train_count=sum(e['split']=='train' for e in entries),
        val_count=sum(e['split']=='val' for e in entries),excluded=excluded,protected_files=protected,
        status='REVIEWED_PILOT_NOT_FORMAL_GT_CERTIFICATION',augmentation='none',
        description='GT-only oracle; original Local RGB conditioning; object category/center/AABB/masks; no room',
        source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('oracle_*.py')})
    write(out/'BUNDLE.json',report)
    print({k:report[k] for k in ('train_count','val_count','status')})


if __name__=='__main__':
    main()
