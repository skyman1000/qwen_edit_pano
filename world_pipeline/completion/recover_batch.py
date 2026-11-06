"""CPU-only, explicit parser migration of saved validation model responses.

Never edits the source batch or claims replayed responses are new predictions.
Verified completed inventories/detections are copied unchanged. Saved normal-EOS
responses are re-parsed/recompiled; missing geometry is left for manual GPU work.
"""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
import shutil

from PIL import Image

from ..common import read,sha,write,new_output
from .batch_world import inputs,configuration,verify_stage
from .placement_program import PROTOCOL
from .semantic_world import run as run_scene,reuse_inventory


def saved_response(directory,stem):
    generation=read(directory/(stem+'.generation.json'))
    if generation.get('finish_reason')!='eos':
        raise ValueError('Do not reinterpret unfinished responses: '+str(directory))
    return read(directory/(stem+'.json'))


def copy_response(source,destination,stem):
    for suffix in ('.txt','.json','.request.json','.generation.json'):
        path=source/(stem+suffix)
        if path.is_file():shutil.copy2(path,destination/path.name)


class SavedReply:
    def __init__(self,stem,answer):self.stem=stem;self.answer=answer;self.calls=0
    def ask(self,image,prompt,path,**kwargs):
        if path.stem!=self.stem or self.calls:
            raise ValueError('CPU replay permits exactly one explicitly saved stage response')
        self.calls+=1
        return deepcopy(self.answer)


def recover(source,output):
    source=Path(source).resolve()
    original_files={str(p):sha(p)for p in source.rglob('*')if p.is_file()}
    old_config=read(source/'run_config.json');old_progress=read(source/'progress.json')
    manifest=Path(old_config['manifest']);cases=inputs(manifest)
    current=configuration(manifest,cases)
    for key in ('manifest_sha256','image_sha256','qwen_model','seed','max_observed','max_additions'):
        if old_config[key]!=current[key]:raise ValueError('Recovery input/parameter mismatch: '+key)
    for stages in old_progress.values():
        for row in stages.values():
            if row['status']=='completed':
                directory=source/row['directory']
                if any(sha(directory/name)!=digest for name,digest in row['artifacts_sha256'].items()):
                    raise ValueError('Source completed artifacts changed: '+str(directory))
    output=new_output(output);write(output/'run_config.json',current)
    progress={};actions=[]
    for case in cases:
        case_dir=output/case['id'];case_dir.mkdir()
        previous={};record={};progress[case['id']]=record
        for stage in ('inventory','observer','completion'):
            old=old_progress.get(case['id'],{}).get(stage)
            if not old:break
            src=source/old['directory'];dst=case_dir/(stage+'_001')
            if stage=='observer' and old['status']!='completed':break
            replay=stage=='completion' or old['status']!='completed'
            if not replay:
                shutil.copytree(src,dst)
                execution='copied_verified_original_stage'
            else:
                stem='observation' if stage=='inventory' else 'placement_program'
                if not (src/(stem+'.json')).is_file():break
                answer=saved_response(src,stem)
                dst.mkdir();copy_response(src,dst,stem)
                cfg=deepcopy(read(src/'config.json'))
                if cfg['image_sha256']!=case['image_sha256']:
                    raise ValueError('Cached response belongs to another RGB')
                cfg['execution_mode']='cpu_reparse_saved_model_response'
                cfg['replay']=dict(source=str(src),raw_response_sha256=sha(src/(stem+'.json')),
                    source_config_sha256=sha(src/'config.json'),gpu_model_loaded=False,
                    new_model_prediction=False,
                    original_prompt_version=cfg.get('completion_prompt_version',cfg.get('protocol')))
                if stage=='completion':cfg['protocol']=PROTOCOL
                write(dst/'config.json',cfg)
                with Image.open(src/'local.png')as handle:image=handle.convert('RGB')
                image.save(dst/'local.png')
                fixed=read(previous['observer']/'G_obs.json') if stage=='completion' else None
                observed=reuse_inventory(previous['observer']/'G_obs.json',image,cfg['vocabulary_sha256'],32) if fixed else None
                with (dst/'cpu_replay.log').open('w')as log,redirect_stdout(log):
                    print('CPU REPLAY ONLY; no new model/GPU execution; original reply retained.')
                    run_scene(SavedReply(stem,answer),image,case['hfov_deg'],dst,cfg['vocabulary_sha256'],
                        max_observed=cfg['max_observed'],max_added=cfg['max_additions'],seed=cfg['seed'],
                        reused=observed,fixed_observed_state=fixed,inventory_only=stage=='inventory',
                        completion_mode='program' if stage=='completion' else 'direct')
                execution='cpu_reparsed_original_response'
            artifacts=verify_stage(dst,stage,case,previous)
            record[stage]=dict(status='completed',directory=str(dst.relative_to(output)),
                artifacts_sha256=artifacts,execution=execution,source_directory=str(src),
                source_config_sha256=sha(src/'config.json'))
            if replay:record[stage]['log']=str((dst/'cpu_replay.log').relative_to(output))
            previous[stage]=dst
            actions.append(dict(id=case['id'],stage=stage,execution=execution))
        for stage in ('observer','completion'):
            if stage not in record:
                record[stage]=dict(status='pending',error='Awaiting manual GPU stage; prior parsed observation retained')
        write(output/'progress.json',progress)
    write(output/'recovery.json',dict(source=str(source),source_run_config_sha256=sha(source/'run_config.json'),
        source_files_sha256=original_files,actions=actions,gpu_executed=False,
        note='New parser/compilation of saved replies, not new model predictions; original prompt versions retained'))
    from .batch_world_review import review
    summary=review(manifest,output)
    if any(sha(Path(p))!=digest for p,digest in original_files.items()):
        raise AssertionError('Source batch changed during CPU replay')
    print(dict(output=str(output),completed_cases=summary['completed_cases'],
               pending_cases=summary.get('pending_cases',summary['failed_cases']),gpu_executed=False))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();recover(args.source,args.output)


if __name__=='__main__':main()
