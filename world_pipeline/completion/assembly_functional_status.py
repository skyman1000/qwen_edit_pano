"""Read-only dependency checks, and explicit blocked records; no model loading."""
import argparse
from html import escape
import json
from pathlib import Path

from ..common import read,write
from .assembly_functional import INTERFACE
from .world_scm import validate


def candidate_status(root,backend):
    folder=Path(root).resolve()/backend/'completion'
    names=('causal_model.json','hypotheses.json','furniture_mechanisms.json','G_obs.json','case.json')
    missing=[n for n in names if not (folder/n).is_file()]
    result=dict(ready=False,source=str(folder),missing=missing)
    summary_path=folder/'summary.json'
    if summary_path.is_file():
        summary=read(summary_path)
        result.update(upstream_status=summary.get('status'),upstream_stage=summary.get('failed_stage'),
            upstream_error=summary.get('error'))
    if missing:return result
    try:
        model=read(folder/'causal_model.json');validate(model)
        expected=INTERFACE
        if read(Path(root)/'experiment.json').get('completion_model')=='population':
            from .assembly_population import INTERFACE as expected
        if model.get('response_interface')!=expected:raise ValueError('Candidate response interface differs')
        for name in names[1:]:read(folder/name)
    except (ValueError,KeyError,TypeError) as exc:
        result['validation_error']=str(exc);return result
    if model.get('candidate_support',{}).get('all_nonempty_proposals_missing'):
        result['validation_error']='All fixed-budget proposals are empty; hidden-world candidate support is unavailable'
        return result
    result['ready']=True
    return result


def dependency_error(status):
    detail=status.get('upstream_error') or status.get('validation_error') or 'upstream did not produce a candidate model'
    return ('Upstream functional candidates unavailable: '+status['source']+
        '; missing='+str(status['missing'])+'; '+
        detail)


def mark_blocked(root,source,backend):
    status=candidate_status(source,backend)
    if status['ready']:raise ValueError('Cannot mark a ready candidate source as blocked')
    folder=Path(root)/backend/'completion';folder.mkdir(parents=True,exist_ok=False)
    summary=dict(status='blocked_upstream_failed',failed_stage='candidate_reuse',world_state_ready=False,
        worlds=[],blocked_worlds=[],failures=[],error=dependency_error(status),upstream=status,
        completion_model=read(Path(root)/'experiment.json').get('completion_model'),model_loaded=False,api_called=False)
    write(folder/'summary.json',summary)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>先验对照未启动</h1><p>'+
        escape(summary['error'])+'</p><p>上游未提供可复用的候选目录；未加载模型或调用 API。</p>')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['check','block'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--backend',choices=['api','qwen'],required=True)
    p.add_argument('--source-root',type=Path);a=p.parse_args()
    if a.action=='block':
        if a.source_root is None:p.error('--source-root required for block')
        print(json.dumps(mark_blocked(a.root,a.source_root,a.backend),ensure_ascii=False));return
    status=candidate_status(a.root,a.backend);print(json.dumps(status,ensure_ascii=False))
    if not status['ready']:raise SystemExit(1)


if __name__=='__main__':main()
