"""Explicit CPU mechanism fixtures. NEVER real Qwen hidden-world predictions."""
from pathlib import Path
import argparse

from ..common import new_output,write
from .frozen_causal_world import load_case
from .scene_program_audit import audit
from .scene_program_world import code_hashes,run


def obj(key,name,size,offset,anchor='camera',support='floor',target=None,category=None):
    return dict(key=key,name=name,category=category or name,variant=name,size_m=size,anchor=anchor,
                offset_m=offset,support=dict(kind=support,target=target))


def fixture(case):
    ids=[o['id'] for o in case['context']]
    registry=dict(instances=[dict(id=oid,variant=o['description'],affordances=[],
        on_floor=(i==0 or oid in ('observed_0001','observed_0008'))) for i,(oid,o) in enumerate(zip(ids,case['context']))],
        configurations=[dict(description='TEST FIXTURE: optional surrounding reading/storage',
            explanation='Explicit CPU fixture, not model inference.',evidence_ids=ids[:1]),
            dict(description='TEST FIXTURE: optional surrounding work nook',
            explanation='Explicit CPU fixture, not model inference.',evidence_ids=ids[:1])])
    light=dict(name='reading light options',explanation='TEST joint variants and support mechanism.',parents=[],modes=[
        dict(description='side table with table lamp',objects=[
            obj('table','side table',[.55,.6,.55],[2.5,0,-3.1],category='table'),
            obj('light','table lamp',[.25,.45,.25],[0,0,0],anchor='table',support='object',target='table',category='lamp')]),
        dict(description='freestanding floor lamp',objects=[obj('light','floor lamp',[.4,1.6,.4],[2.8,0,-2.2],category='lamp')])])
    storage=dict(name='storage cluster',explanation='TEST optional surrounding storage and decor.',parents=[0],modes=[
        dict(description='bookshelf and plant',objects=[
            obj('shelf','bookshelf',[1.1,1.8,.35],[-2.5,0,-3.2]),
            obj('plant','potted plant',[.35,.8,.35],[1.1,0,0],anchor='shelf',category='plant')])])
    work=dict(name='desk cluster',explanation='TEST joint host and accessory recipe.',parents=[],modes=[
        dict(description='desk monitor and chair',objects=[
            obj('desk','desk',[1.2,.75,.65],[0,0,-3.8]),
            obj('monitor','monitor',[.55,.4,.15],[0,0,-.05],anchor='desk',support='object',target='desk'),
            obj('chair','chair',[.55,.85,.55],[0,0,1.2],anchor='desk')])])
    plant=dict(name='independent decor',explanation='TEST optional secondary object.',parents=[],modes=[
        dict(description='one potted plant',objects=[obj('plant','potted plant',[.4,.9,.4],[3.3,0,-3.1],category='plant')])])
    return dict(registry=registry,programs={'c0':dict(groups=[light,storage]),'c1':dict(groups=[work,plant])},
        tables={'c0':{'g0':{'root':[.1,.6,.3]},'g1':{'0':[.6,.4],'1':[.2,.8],'2':[.5,.5]}},
                'c1':{'g0':{'root':[.15,.85]},'g1':{'root':[.6,.4]}}})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',type=Path);a=p.parse_args();case=load_case(a.case);output=new_output(a.output)
    budgets=dict(configurations=3,groups=3,modes=2,max_new=24);search=dict(region_keep=3,beam_width=4,rounds=8)
    write(output/'config.json',dict(case=case,code_sha256=code_hashes(),budgets=budgets,search=search,
        arguments=dict(worlds=3),execution_kind='cpu_fixture_not_model_prediction'))
    result=run(None,case,output,budgets,search,fixture=fixture(case))
    if not result['world_state_ready']:raise SystemExit('CPU fixture failed; inspect saved errors')
    print(audit(output,allow_fixture=True))


if __name__=='__main__':main()
