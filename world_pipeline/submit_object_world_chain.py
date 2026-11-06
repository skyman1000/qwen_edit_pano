"""User-run submission of three dependent jobs within the debug 72-hour limit.

Does not run at import. --dry-run performs local checks and prints the plan only.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from .build_polished_world import ROOT,atomic_json,read
from .object_world.data import DEFAULT_PACK,load_pack


DEFAULT_OUTPUT=ROOT/'qwen_edit_pano/outputs/object_world_polished_semantic_region_v1'
SUBMIT=ROOT/'qwen_edit_pano/world_pipeline/submit_object_world_train.sh'


def preflight(pack, output):
    plan,records=load_pack(pack)
    train=[row for row in records if row['split']=='train']
    cache=read(pack/'CACHE_COMPLETE.json')
    if (cache['samples']!=len(train) or cache['ids']!=[row['id'] for row in train]
            or cache['base_snapshot']!=plan['base_snapshot']
            or cache['semantic_policy']!=plan['semantic_policy']):
        raise ValueError('Full cache completion does not match the training pack')
    missing={key:sum(not Path(row[key]).is_file() for row in train)
             for key in ('condition_cache','posterior_cache','region_cache')}
    if any(missing.values()) or not (pack/'class_features.safetensors').is_file():
        raise ValueError('Training caches missing: '+str(missing))
    # The fixed GPU batch script uses the production pack, not a diagnostic pack.
    if pack.resolve()!=DEFAULT_PACK.resolve():raise ValueError('Batch runner expects the production training pack')
    smoke=read(DEFAULT_OUTPUT.parent/(DEFAULT_OUTPUT.name+'_smoke')/'checkpoint-step000002/SMOKE_COMPLETE.json')
    required=('zero_gate_baseline_exact','base_frozen','semantic_features_frozen','encoder_gradient',
              'semantic_projection_gradient','parameters_changed','object_region_loss_enabled')
    if any(smoke.get(key) is not True for key in required):raise ValueError('GPU smoke has not passed all required checks')
    outputs=[output,output.with_name(output.name+'_part2'),output.with_name(output.name+'_part3')]
    allowed=ROOT/'qwen_edit_pano/outputs'
    if any(not path.is_relative_to(allowed) or path==allowed or path.exists() for path in outputs):
        raise ValueError('Use three NEW output directories below qwen_edit_pano/outputs')
    if (output.parent/(output.name+'_submission.json')).exists():
        raise ValueError('Submission record already exists; inspect it before submitting again')
    return outputs,dict(train=len(train),val=plan['val_count'],cache_complete=True,smoke_passed=True)


def submit_jobs(outputs, submit, on_submitted):
    """Stop on submission failure; never hide already-submitted job IDs."""
    previous=None;records=[]
    for index,(steps,output) in enumerate(zip((3500,7000,10000),outputs)):
        command=['sbatch','--parsable',f'--job-name=object-world-{index+1}']
        if previous:
            command.extend([f'--dependency=afterok:{previous}','--kill-on-invalid-dep=yes'])
        command.extend([str(SUBMIT),str(steps),str(output)])
        if index:
            before=(3500,7000)[index-1]
            command.append(str(outputs[index-1]/f'checkpoint-step{before:06d}'))
        print(shlex.join(command),flush=True)
        response=submit(command).strip()
        job=response.split(';',1)[0]
        if not job.isdigit():raise ValueError('Unexpected sbatch response: '+response)
        record=dict(job_id=job,target_step=steps,output=str(output),dependency=previous)
        records.append(record);on_submitted(records.copy())
        print('SUBMITTED '+json.dumps(record),flush=True)
        previous=job
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();args.output=args.output.resolve()
    outputs,status=preflight(DEFAULT_PACK,args.output)
    print(json.dumps(status),flush=True)
    submission=args.output.parent/(args.output.name+'_submission.json')
    if args.dry_run:
        counter=iter(('JOB1','JOB2','JOB3'))
        # Show dependency structure without calling sbatch or writing a record.
        previous=None
        for index,(step,out) in enumerate(zip((3500,7000,10000),outputs)):
            command=['sbatch','--parsable',f'--job-name=object-world-{index+1}']
            if previous:command.extend([f'--dependency=afterok:{previous}','--kill-on-invalid-dep=yes'])
            command.extend([str(SUBMIT),str(step),str(out)])
            if index:command.append(str(outputs[index-1]/f'checkpoint-step{(3500,7000)[index-1]:06d}'))
            print(shlex.join(command));previous=next(counter)
        print('DRY RUN ONLY. No job submitted or files created.');return
    # Reserve submission identity before the first scheduler side effect. A retry
    # cannot accidentally submit duplicate trains if a later sbatch call fails.
    with submission.open('x') as handle:
        json.dump(dict(status='submitting',jobs=[],checks=status),handle)
    def save(jobs):
        atomic_json(submission,dict(status='submitted_all' if len(jobs)==3 else 'submitting',
                    jobs=jobs,checks=status,final_output=str(outputs[-1]),total_steps=10000))
    def execute(command):
        return subprocess.check_output(command,text=True)
    try:
        jobs=submit_jobs(outputs,execute,save)
    except Exception:
        print('Submission interrupted. Inspect '+str(submission)+' and any SUBMITTED IDs before retrying.',flush=True)
        raise
    print('ALL THREE JOBS SUBMITTED. Safe to disconnect after this message.',flush=True)
    print('Job IDs: '+','.join(job['job_id'] for job in jobs))
    print('Final output: '+str(outputs[-1]))


if __name__=='__main__':main()
