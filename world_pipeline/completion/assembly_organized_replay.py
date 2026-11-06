"""CPU-only old-size replay: joint packing, no inference, no source mutation."""
import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path

from ..common import read,write,sha
from . import assembly_organized_geometry as geometry
from .scene_completion import export_states
from .assembly_taxonomy import annotate
from .assembly_holistic_visual import table
from .assembly_pilot_visual import footprint,shared_bounds
from .batch_world_review import interactive_boxes


def replay(source,out):
    source=source.resolve();out=out.resolve()
    if out==source or source in out.parents:raise ValueError('Replay output must be separate from the original run')
    if out.exists():raise FileExistsError(out)
    before={str(p):sha(p) for p in source.rglob('*.json')};out.mkdir(parents=True);r=read(source/'run.json');results=[];gallery=[]
    for b in ('qwen','api'):
        base=source/'layout_control'/b/'completion';observed=read(base/'G_obs.json');case=read(base/'case.json');plane=read(base/'support_plane.json')
        for i in range(1,r['worlds']+1):
            f=base/f'world_{i:02d}';world=read(f/'semantic_world.json');layout=read(f/'execution.json')['layout'];old=read(f/'numeric_review.json');solved=geometry.compile_layout(case,observed,world,layout,plane)
            dest=out/(b+'_'+f.name);dest.mkdir();write(dest/'semantic_world.json',world);write(dest/'execution.json',dict(layout=layout,solved=solved));write(dest/'numeric_review.json',solved['review'])
            state,_,_=export_states(observed,solved['plan'],solved['items'],solved['mappings'],solved['review']);objects={o['id']:o for o in world['objects']}
            for o in state['objects'][len(observed['objects']):]:o['raw_category']=objects[o['track_id']]['category']
            state=geometry.annotate_state(annotate(state,world),layout);state.update(world_state_ready=bool(solved['review']['geometry_contract_passed']),replay_only=True,
                replay_scope='Same old inventory, dimensions, yaw, host, floor prior and spatial program; joint packing plus unchanged global solver')
            assert state['objects'][:len(observed['objects'])]==observed['objects']
            assert {o['track_id'] for o in state['objects'][len(observed['objects']):]}==set(objects)
            for o in state['objects'][len(observed['objects']):]:
                expected=next(x for g in layout.values() for x in g['objects'] if x['id']==o['track_id'])
                assert o['intrinsic_size_m']['value']==expected['intrinsic_size_m']
            passed=solved['review']['geometry_contract_passed'];write(dest/('G_full.json' if passed else 'G_full_candidates.json'),state)
            row=dict(backend=b,world=f.name,objects=len(objects),before_pass=old['geometry_contract_passed'],after_pass=passed,
                before_overlap_pairs=len(old['program_solver']['final_overlaps']),after_overlap_pairs=len(solved['review']['program_solver']['final_overlaps']),
                old_sizes_frozen=True,semantic_quality_verified=False,packing=solved['review']['joint_host_packing']);results.append(row);gallery.append((dest,state,case,row))
    bounds=shared_bounds([s for _,s,_,_ in gallery]);parts=['<meta charset="utf-8"><title>固定旧尺寸的 CPU 布局重放</title><style>body{font:16px sans-serif;margin:24px}td,th{border:1px solid #ccc;padding:6px}img{max-width:750px;width:100%}iframe{width:100%;height:480px}</style>',
        '<h1>CPU 原尺寸重放：联合宿主物品放置</h1><p>无新 Qwen/API。所有库存、尺寸、朝向、宿主、固定观测及方位沿用原样。通过只表示当前框约束，不认证原来的错误尺度或完整世界合理性。</p>',
        table(['后端','世界','对象','原约束通过','新约束通过','原碰撞对','新碰撞对'],[[x['backend'],x['world'],x['objects'],x['before_pass'],x['after_pass'],x['before_overlap_pairs'],x['after_overlap_pairs']] for x in results])]
    for dest,state,case,row in gallery:
        title=dest.name+('' if row['after_pass'] else ' BLOCKED CANDIDATE');write(dest/'plot_labels.json',footprint(state,bounds,case['hfov'],dest/'layout_xz.png',title))
        interactive_boxes(state,case['hfov'],case['image_size'],dest/'scene_3d.html',title=title,scene_bounds=bounds)
        parts+=['<h2>'+escape(title)+'</h2><img src="'+dest.name+'/layout_xz.png"><details><summary>旋转三维</summary><iframe loading="lazy" src="'+dest.name+'/scene_3d.html"></iframe></details>']
    assert all(Path(p).is_file() and sha(Path(p))==h for p,h in before.items())
    write(out/'replay.json',dict(worlds=results,source_JSON_sha256=before,new_provider_calls=0,old_dimensions_preserved=True,training_performed=False))
    (out/'index.html').write_text(''.join(parts));return dict(out=str(out),worlds=len(results),before_pass=sum(x['before_pass'] for x in results),after_pass=sum(x['after_pass'] for x in results),original_JSON_preserved=len(before),new_provider_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(replay(a.source_run,a.out)))


if __name__=='__main__':main()
