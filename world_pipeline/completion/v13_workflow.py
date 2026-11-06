"""Pinned original v13 and an optional fixed-budget local proposal experiment.

Only proposal supply changes in the experiment. The original v13 H/A/V,
smoothing, program pooling, sampling, geometry and report remain the executor.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import traceback

from ..common import ROOT,read,write,sha
from . import organized_local_world as baseline
from .assembly_experiment import check as check_context,stage_seed
from .fresh_pipeline import provider

MANIFEST=ROOT/'qwen_edit_pano/config/world_completion_v13.json'


def verify(manifest=MANIFEST):
    pinned=read(manifest)
    changed=[p for p,h in pinned['source_sha256'].items() if not Path(p).is_file() or sha(p)!=h]
    if changed:raise ValueError('Restored v13 source differs from pinned baseline: '+str(changed))
    return dict(status='verified_original_v13_no_model_calls',model=pinned['model'],source_files=len(pinned['source_sha256']))


def extension_hashes():
    return {str(p):sha(p) for p in (Path(__file__).resolve(),ROOT/'qwen_edit_pano/scripts/world_completion.sh',MANIFEST)}


def prepare(root,source,worlds=4,draws=4,seed=3407):
    verify();result=baseline.prepare(root,source,worlds,draws,seed)
    write(root/'v13_proposal_budget.json',dict(kind='v13_fixed_two_local_proposal_rounds_v1',rounds=2,
        local_provider='qwen',API_used=False,base_model_changed=False,geometry_changed=False,
        fixed_budget_not_retry_until_success=True,entry_sha256=extension_hashes()))
    return dict(result,proposal_rounds=2,scope='candidate_supply_experiment_original_v13_executor')


def check(root):
    verify();suite=baseline.check_run(root);meta=read(root/'v13_proposal_budget.json')
    if meta.get('rounds')!=2 or meta.get('entry_sha256')!=extension_hashes():
        raise ValueError('Proposal budget/source changed; use a new output directory')
    return suite


def merge_rounds(rounds):
    """Namespace hypothesis IDs only; preserve all instance/host/relation data."""
    hypotheses=[];programs={}
    for index,(hs,ps) in enumerate(rounds):
        mapping={h['id']:'round'+str(index)+'_'+h['id'] for h in hs}
        for h in hs:h=deepcopy(h);h['id']=mapping[h['id']];hypotheses.append(h)
        for fid,p in ps.items():
            value=deepcopy(p);value['hypothesis']['id']=mapping[p['hypothesis']['id']]
            value['proposal_source']=dict(value.get('proposal_source',{}),backend='qwen',proposal_round=index,
                original_program_id=fid,local_model_only=True)
            programs['round'+str(index)+'_'+fid]=value
    return hypotheses,programs


def _saved_or_propose(comp,client):
    folder=comp/'proposals/qwen'
    if (folder/'summary.json').exists():return read(folder/'summary.json')
    if folder.exists():raise ValueError('Interrupted proposal directory has no final summary; preserve it and start a new run: '+str(folder))
    result=baseline.pipeline.fresh.propose(comp,'qwen',client)
    for name,empty in [('hypotheses.json',[]),('programs.json',{})]:
        if not (folder/name).exists():write(folder/name,empty)
    return result


def propose(root):
    suite=check(root);client=None;records=[]
    for scene in suite['scenes']:
        marker=root/'phase_records'/(scene['id']+'_propose_qwen.json')
        if marker.exists():records.append(read(marker));continue
        comp=root/'scenes'/scene['id']/'completion'
        try:
            config=check_context(comp/'context')
            if client is None:client=provider(config,'qwen')
            snapshots=comp/'proposal_rounds';snapshots.mkdir(exist_ok=True)
            first=snapshots/'round_00';first.mkdir(exist_ok=True)
            if not (first/'summary.json').exists():
                _saved_or_propose(comp,client)
                for name in ('hypotheses.json','programs.json','summary.json'):shutil.copyfile(comp/'proposals/qwen'/name,first/name)
            extra=snapshots/'round_01'
            if not extra.exists():
                extra.mkdir();shutil.copytree(comp/'context',extra/'context');(extra/'shared').mkdir()
                for name in ('case.json','observation_board.png'):shutil.copyfile(comp/'shared'/name,extra/'shared'/name)
                run=read(comp/'run.json');run['seed']=stage_seed(suite['seed'],scene['id']+':proposal_round_1')
                write(extra/'run.json',run)
            second_summary=_saved_or_propose(extra,client);second=extra/'proposals/qwen'
            rounds=[(read(first/'hypotheses.json'),read(first/'programs.json')),
                (read(second/'hypotheses.json'),read(second/'programs.json'))]
            hs,ps=merge_rounds(rounds);dest=comp/'proposals/qwen'
            write(dest/'hypotheses.json',hs);write(dest/'programs.json',ps)
            summaries=[read(first/'summary.json'),second_summary]
            status='complete' if all(s['status']=='complete' for s in summaries) else 'complete_with_warnings' if ps else 'failed'
            summary=dict(status=status,backend='qwen',programs=len(ps),
                candidate_instances=sum(len(g['objects']) for p in ps.values() for g in p['groups']),
                model_id=config['local_model'],proposal_rounds=2,round_summaries=summaries,API_used=False,
                scope='two_fixed_local_proposal_rounds_original_v13_pool_and_mechanisms_unchanged')
            write(dest/'summary.json',summary)
            write(comp/'proposal_rounds.json',dict(rounds=2,primary_snapshot=str(first),extra_snapshot=str(second),
                source_sha256={str(p):sha(p) for d in (first,second) for p in d.glob('*.json')},
                duplicate_handling='original_v13_pooling_no_extra_instance_creation',empty_rounds_retained=True))
            record=dict(scene=scene['id'],phase='propose',backend='qwen',**{k:v for k,v in summary.items() if k!='backend'})
        except Exception as exc:
            record=dict(scene=scene['id'],phase='propose',backend='qwen',status='failed',error=type(exc).__name__+': '+str(exc))
            marker.with_suffix('.txt').write_text(traceback.format_exc())
        write(marker,record);records.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
    check(root)
    return dict(status='partial_or_failed' if any(r['status']=='failed' for r in records) else 'complete_with_warnings' if any(r['status']!='complete' for r in records) else 'complete',records=records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify','prepare','check','report')+baseline.PHASES)
    parser.add_argument('--root',type=Path);parser.add_argument('--observations-from',type=Path)
    parser.add_argument('--worlds',type=int,default=4);parser.add_argument('--draws',type=int,default=4)
    a=parser.parse_args()
    if a.action=='verify':result=verify()
    else:
        if a.root is None:parser.error('--root required')
        root=a.root.resolve()
        if a.action=='prepare':
            if a.observations_from is None:parser.error('--observations-from required')
            result=prepare(root,a.observations_from,a.worlds,a.draws)
        else:
            check(root)
            if a.action=='check':result=dict(status='verified_no_model_calls')
            elif a.action=='propose':result=propose(root)
            elif a.action=='report':result=baseline.report(root)
            else:result=baseline.phase(root,a.action)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False),flush=True)
    return 1 if result['status'] in ('failed','partial','partial_or_failed') else 0


if __name__=='__main__':sys.exit(main())
