"""User-run CPU repairs in a separate root; preserve all original evidence."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
from .common import ROOT, read, sha
from .checkpoint_io import atomic_json, output_lock, file_hashes, verify_files, archive_partial
from .expand_gt_sources import Builder, GROUPS, ARCHIVES, paired_selection, validate_report, source_stats
from .source_repair_ops import (PROFILE, link_files, label_conflicts,
                               accept_quarantined_region, refusion, repaired_world)


class RepairBuilder(Builder):
    def __init__(self,args,roots):
        super().__init__(args)
        self.roots=roots

    def old_stage(self, relative, report):
        return next((root/relative for root in self.roots if (root/relative/report).is_file()),None)

    def local_run(self,module,arguments,label,allow_failure=False):
        with (self.logs/(label+'.log')).open('a') as f:
            result=subprocess.run([sys.executable,'-m',module,*map(str,arguments)],cwd=ROOT,
                env=dict(os.environ,OMP_NUM_THREADS=str(self.args.threads)),stdout=f,stderr=subprocess.STDOUT)
        if result.returncode and not allow_failure:
            raise RuntimeError(f'{module} failed; see {label}.log')

    def canonical(self,house):
        oldregion=self.old_stage(Path('region_pilot')/house,'region_validation.json')
        if oldregion is None or read(oldregion/'region_validation.json')['status']=='PASS':
            return super().canonical(house)
        paths={k:self.work/f'{k}_pilot'/house for k in GROUPS}
        marker=self.state/f'{house}_canonical.json'
        stats=source_stats(self.args.raw_root/house)
        if marker.exists():
            saved=read(marker)
            if saved['archives']!=stats:raise ValueError('Raw assets changed')
            verify_files(self.work,saved['files'])
            return paths['canonical']
        conflicts=label_conflicts(oldregion)
        if sorted(read(oldregion/'region_validation.json')['errors'])!=sorted(c['validation_error'] for c in conflicts) or not conflicts:
            raise ValueError('Non-category region errors remain; refusing automatic repair')
        for key in ('camera','house'):
            old=self.old_stage(Path(f'{key}_pilot')/house,f'{key}_validation.json')
            if old is None:raise ValueError('Missing validated parser stage')
            report=read(old/f'{key}_validation.json')
            if report['errors']:raise ValueError('Parser errors remain')
            dest=paths[key];dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.symlink_to(old.resolve(),target_is_directory=True)
        for key in ('region','object_link','canonical'):
            archive_partial(paths[key])
        link_files(oldregion,paths['region'],{'region_index.json','region_validation.json'})
        atomic_json(paths['region']/'region_index.json',read(oldregion/'region_index.json'))
        mapping=ROOT/'benchmark_assets/Matterport3D_metadata/official/category_mapping.tsv'
        self.local_run('qwen_pano.caupano.tools.validate_regions',
            ['--input',paths['region'],'--category-mapping',mapping],house+'_region_recheck',True)
        accept_quarantined_region(paths['region'],conflicts)
        self.run('data.matterport.object_linker',['--regions',paths['region'],'--category-mapping',mapping,
            '--output',paths['object_link']],house+'_object_link_repair')
        self.run('tools.validate_object_links',['--input',paths['object_link']],house+'_object_link_validate')
        self.run('data.canonical_index',['--cameras',paths['camera'],'--objects',paths['object_link'],
            '--output',paths['canonical']],house+'_canonical_repair')
        counts=read(paths['house']/'parse_summary.json')['declared_counts']
        self.run('tools.validate_canonical_index',['--input',paths['canonical'],
            '--expected-panoramas',counts['panoramas'],'--expected-observations',counts['images']],house+'_canonical_validate')
        validate_report(paths['canonical']/'index_validation.json',{'PASS'})
        files={str(path.relative_to(self.work)/name):digest for path in paths.values() for name,digest in file_hashes(path).items()}
        atomic_json(marker,dict(archives=stats,files=files,repair_profile=PROFILE))
        return paths['canonical']

    def sample(self,row,index):
        h,u,sid=row['scene_id'],row['source_view_id'],row['id']
        alignment=self.work/'erp_alignment_pilot'/h/u
        geometry=self.work/'geometry_pilot'/h/u
        world=self.work/'world_state_pilot'/h/u
        stages=[(alignment,'alignment_report.json',{'CANDIDATE_REQUIRES_VISUAL_REVIEW'}),
                (geometry,'geometry_report.json',{'BUILT_FOR_REVIEW'}),(world,'build_report.json',{'BUILT'})]
        for directory,report,statuses in stages:
            marker=self.state/f'{sid}_{directory.parts[-3]}.json'
            if marker.exists():
                verify_files(directory,read(marker)['files']);validate_report(directory/report,statuses,sid);continue
            if (directory==world and not directory.is_symlink() and (directory/report).exists()
                    and read(directory/report).get('repair_profile')!=PROFILE):
                archive_partial(directory)
            # A completed repair attempt that still fails is not repeated on resume.
            if (directory/report).exists() and not directory.is_symlink():
                if directory==geometry:
                    saved=read(directory/report)
                    if saved.get('review_flags') and 'repair' not in saved:
                        # Interrupted after the original builder committed but
                        # before observation refusion was started.
                        raw=directory.with_name(directory.name+'.before_refusion')
                        if raw.exists():raise ValueError('Original refusion evidence path already exists')
                        directory.rename(raw)
                        refusion(index,raw,directory)
                validate_report(directory/report,statuses,sid)
            else:
                old=self.old_stage(directory.relative_to(self.work),report)
                reusable=False
                if old:
                    try:validate_report(old/report,statuses,sid);reusable=True
                    except ValueError:pass
                if reusable:
                    directory.parent.mkdir(parents=True,exist_ok=True)
                    if not directory.exists():directory.symlink_to(old.resolve(),target_is_directory=True)
                else:
                    archive_partial(directory)
                    if directory==alignment:
                        self.local_run('qwen_edit_pano.world_pipeline.repair_alignment',
                            ['--index',index,'--panorama-uuid',u,'--output',directory],sid+'_alignment_recovery')
                    elif directory==geometry:
                        raw=directory.with_name(directory.name+'.before_refusion')
                        if old is None and (raw/report).is_file():
                            old=raw
                        if old:
                            refusion(index,old,directory)
                        else:
                            self.run('data.build_geometry_pilot',['--index',index,'--alignment',alignment,
                                '--output',directory,'--height',512,'--threads',self.args.threads],sid+'_geometry')
                            if read(directory/report)['review_flags']:
                                if raw.exists():raise ValueError('Original refusion evidence path already exists')
                                directory.rename(raw)
                                refusion(index,raw,directory)
                    else:
                        repaired_world(index,geometry,directory)
            validate_report(directory/report,statuses,sid)
            atomic_json(marker,dict(files=file_hashes(directory),repair_profile=PROFILE))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,default=ROOT/'qwen_edit_pano/data/paired_gt_14train3val_v1')
    p.add_argument('--reuse-root',type=Path,default=ROOT/'qwen_edit_pano/data/gt_sources_14train3val_encoder_v1')
    p.add_argument('--work-root',type=Path,default=ROOT/'qwen_edit_pano/data/gt_sources_14train3val_repaired_v1')
    p.add_argument('--raw-root',type=Path,default=ROOT/'benchmark_assets/Matterport3D_raw/v1/scans')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--only-houses',nargs='+',help='Optional CPU recovery test; use a separate work root')
    args=p.parse_args()
    for k in ('pairs','reuse_root','work_root','raw_root'):setattr(args,k,getattr(args,k).resolve())
    roots=[];root=args.reuse_root
    while root not in roots:
        roots.append(root)
        cfg=root/'SOURCE_RUN.json'
        if not cfg.exists():break
        signature=read(cfg)
        for filename,digest in signature['code'].items():
            if sha(filename)!=digest:raise ValueError(f'Old source code changed: {filename}')
        root=Path(signature['reuse_root']).resolve()
    if (args.threads<1 or not args.work_root.is_relative_to(ROOT/'qwen_edit_pano/data') or
        args.work_root==ROOT/'qwen_edit_pano/data' or any(args.work_root==r or args.work_root in r.parents or r in args.work_root.parents for r in roots)):
        raise ValueError('Use a separate work root under data')
    selected=paired_selection(args.pairs)
    if args.only_houses:
        if set(args.only_houses)-{r['scene_id'] for r in selected}:raise ValueError('Unknown requested house')
        selected=[r for r in selected if r['scene_id'] in args.only_houses]
    with ExitStack() as stack:
        for root in roots:
            if (root/'SOURCE_RUN.json').exists():stack.enter_context(output_lock(root))
        args.work_root.mkdir(parents=True,exist_ok=True)
        stack.enter_context(output_lock(args.work_root))
        files=[Path(__file__),Path(__file__).with_name('source_repair_ops.py'),Path(__file__).with_name('repair_alignment.py')]
        signature=dict(profile=PROFILE,rows=selected,raw_root=str(args.raw_root),reuse_root=str(args.reuse_root),
            original_build_sha256=sha(args.reuse_root/'BUILD_RESULT.json'),code={str(f):sha(f) for f in files})
        marker=args.work_root/'SOURCE_RUN.json'
        if marker.exists() and read(marker)!=signature:raise ValueError('Repair signature changed; use a new work root')
        atomic_json(marker,signature);builder=RepairBuilder(args,roots);results={}
        for house in sorted({r['scene_id'] for r in selected}):
            try:index=builder.canonical(house)
            except (ValueError,OSError,RuntimeError,KeyError) as exc:
                results[house]=dict(status='FAILED_CANONICAL',error=str(exc))
            else:
                done=[];failures=[]
                for row in (r for r in selected if r['scene_id']==house):
                    try:builder.sample(row,index);done.append(row['id'])
                    except (ValueError,OSError,RuntimeError,KeyError) as exc:
                        failures.append(dict(id=row['id'],error=str(exc)))
                    results[house]=dict(status='PARTIAL' if failures else 'REVIEW_REQUIRED',completed=done,failures=failures)
                    atomic_json(args.work_root/'build_progress.json',results)
                    print(f'{house}: completed={len(done)} unresolved={len(failures)}',flush=True)
            atomic_json(args.work_root/'build_progress.json',results)
        complete=all(v['status']=='REVIEW_REQUIRED' for v in results.values())
        atomic_json(args.work_root/'BUILD_RESULT.json',dict(profile=PROFILE,all_requested_built=complete,
            ready_for_training=False,houses=results))
        if not complete:raise SystemExit('Repair pass finished; unresolved samples remain isolated. Inspect BUILD_RESULT.json.')


if __name__=='__main__':main()
