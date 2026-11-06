"""CPU-only saved-run diagnosis/replay; never invokes a provider or exports G_full.

Run in the qwen3vl environment, with --source and a NEW --out. Replay fixes only
the solver, keeping the saved object inventory, dimensions, yaw, hosts, room and
optional constraints. Proxy passing is not a fresh model inference or acceptance.
"""
import argparse
from collections import Counter
from copy import deepcopy
from html import escape
import json
from pathlib import Path

from PIL import Image

from ..common import ROOT,read,write,sha
from . import global_world_model as model,global_world_geometry as geometry
from .global_world import make_board


def diagnose(source,out):
    source = Path(source).resolve();out = Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    files = [p for p in source.rglob('*') if p.is_file()]
    hashes = {str(p):sha(p) for p in files}
    suite = read(source/'suite.json');scene_rows = [];replay = [];failures = Counter();cards = []
    provenance = read(ROOT/'qwen_edit_pano/outputs/meeting_comparison_20261028_001/provenance.json')
    for s in suite['scenes']:
        scene = source/'scenes'/s['id'];comp = scene/'completion'
        observed = read(comp/'G_obs.json')
        packet = model.observation_packet(observed,read(scene/'observation/qwen/observer2d/grounding.json'))
        hypotheses = {h['id']:h for h in read(comp/'hypotheses.json')}
        summary = read(comp/'summary.json')
        row = dict(scene=s['id'],observed=len(observed['objects']),draws=summary['attempts'],accepted=summary['accepted'])
        row['world_failures'] = Counter()
        for result in summary['results']:
            for failure in result['failures']:
                labels = failure.get('issues',[failure.get('error','unknown')])
                for issue in labels:
                    label = issue.get('issue') if isinstance(issue,dict) else issue
                    row['world_failures'][label] += 1;failures[label] += 1
        old_root = Path(provenance['scene_sources'][s['id']])
        old_path = old_root/'scenes'/s['id']/'completion/end_to_end/qwen/completion/world_01/G_full.json'
        old = read(old_path)
        row['v13'] = dict(source=str(old_path),sha256=sha(old_path),
            observed=sum(o['local_evidence']=='visible' for o in old['objects']),
            new=sum(o['local_evidence']=='unobserved' for o in old['objects']),
            same_frozen_observations=[o for o in old['objects'] if o['local_evidence']=='visible']==observed['objects'])
        row['hypothesis_mechanism_failures'] = read(comp/'domain.json')['failures']
        row['recognition_votes'] = read(comp/'mechanisms.json')['recognition']['counts']
        row['metadata_separated_inventory_replay'] = []
        from .global_world_proposals import inventory_answer
        for h in hypotheses.values():
            path = comp/'calls'/('programs_'+h['id']+'.try0.json')
            try:
                values = inventory_answer(read(path),h,packet,suite['alternatives'],suite['maximum_objects'])
                row['metadata_separated_inventory_replay'].append(dict(hypothesis=h['id'],valid_inventories=len(values),
                    objects=sum(len(p['objects']) for p in values),fresh_dimension_or_layout_inference=False))
            except ValueError as exc:
                row['metadata_separated_inventory_replay'].append(dict(hypothesis=h['id'],error=str(exc)))
        visualized = set()
        for directory in sorted(comp.glob('world_*')):
            if not (directory/'initial_placement.json').exists():continue
            draw = read(directory/'draw.json');h = hypotheses[draw['hypothesis']]
            p = read(directory/'joint_program.json');proposal = read(directory/'initial_placement.json')
            immutable = deepcopy((observed,p,h,proposal))
            camera = dict(hfov=s['hfov'],image_size=s['image_size'])
            layout,numeric = geometry.solve_layout(h,p,packet,proposal,3407,starts=3,maxiter=180,camera=camera)
            state = model.export_graph(observed,h,p,layout)
            visibility = geometry.visibility_audit(state,s['hfov'],s['image_size'])
            item = dict(scene=s['id'],world=directory.name,geometry_passed=numeric['passed'],
                visibility_passed=visibility['passed'],original_result=read(directory/'result.json')['status'],
                numeric=numeric,visibility=visibility,qwen_critic_not_run=True,accepted_world=False)
            proofs = []
            for zone in p['access_zones']:
                owner = next(o for o in p['objects'] if o['id']==zone['owner'])
                if owner['support']['kind']=='floor' and zone.get('offset_m'):
                    clearance = owner['size_m'][1]/2+zone['offset_m'][1]-zone['size_m'][1]/2
                    if clearance < -.03:
                        proofs.append(dict(issue='fixed_ground_object_and_owner_relative_zone_force_zone_below_floor',
                            bottom_minus_floor_m=clearance,independent_of_horizontal_placement=True))
            item['analytic_conflicts'] = proofs
            target = out/s['id']/directory.name;target.mkdir(parents=True)
            write(target/'CPU_replay.json',item);write(target/'candidate_state_after_CPU_solver.json',state)
            if p['id'] not in visualized:
                with Image.open(scene/'observation/input.png') as im:image = im.convert('RGB')
                before = read(directory/'G_full_candidate.json')
                make_board(before,image,s['hfov'],target/'before.png')
                make_board(state,image,s['hfov'],target/'after.png')
                item['boards'] = [str((target/'before.png').relative_to(out)),str((target/'after.png').relative_to(out))]
                visualized.add(p['id'])
            if (observed,p,h,proposal) != immutable:raise AssertionError('Immutable replay input changed')
            if state['objects'][:len(observed['objects'])] != observed['objects']:raise AssertionError('Observation changed')
            replay.append(item)
        scene_rows.append(row)
        card = '<article><h2>'+escape(s['id'])+'</h2><p>v13：观测 '+str(row['v13']['observed'])+'，新增 '+str(row['v13']['new'])
        card += '；002：观测 '+str(row['observed'])+'，通过 '+str(row['accepted'])+'/'+str(row['draws'])+'。冻结观测不同。</p>'
        card += '<pre>'+escape(json.dumps(row['world_failures'],ensure_ascii=False,indent=2))+'</pre>'
        for item in [r for r in replay if r['scene']==s['id']]:
            card += '<p>'+item['world']+'：CPU 几何 '+str(item['geometry_passed'])+'，可见性 '+str(item['visibility_passed'])+'；未进行 Qwen 审查，不能算完整世界验收。</p>'
            if item['analytic_conflicts']:card += '<pre>'+escape(json.dumps(item['analytic_conflicts'],ensure_ascii=False,indent=2))+'</pre>'
            if item.get('boards'):
                card += '<div class="grid"><div>002 原候选<img src="'+item['boards'][0]+'"></div><div>固定同一库存与尺寸，CPU 求解后<img src="'+item['boards'][1]+'"></div></div>'
        cards.append(card+'</article>')
    generation = Counter(read(p).get('finish_reason') for p in source.glob('scenes/*/completion/calls/*.generation.json'))
    unchanged = hashes=={str(p):sha(p) for p in files}
    if not unchanged:raise AssertionError('Source artifacts changed')
    previous = read(ROOT/'qwen_edit_pano/reports/causal_model_audit_20261102.json')['source_sha256']
    historic_unchanged = all(sha(Path(p))==h for p,h in previous.items())
    audit = dict(source=str(source),scope='saved_outputs_plus_CPU_solver_replay_only',model_called=False,API_used=False,
        scenes=scene_rows,failures=dict(failures),generation_finish_reasons=dict(generation),
        replay=replay,replay_geometry_and_visibility_passed=sum(r['geometry_passed'] and r['visibility_passed'] for r in replay),
        replay_still_rejected=sum(not(r['geometry_passed'] and r['visibility_passed']) for r in replay),
        source_sha256=hashes,source_unchanged=unchanged,historical_38_audit_sources_unchanged=historic_unchanged,
        new_Qwen_compatibility_checks=0,new_accepted_worlds=0)
    write(out/'audit.json',audit)
    page = '<!doctype html><meta charset="utf-8"><title>002 诊断与 CPU 回放</title><style>body{font:16px/1.6 system-ui;margin:24px}article{border-top:1px solid #aaa}pre{white-space:pre-wrap}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{width:100%}</style>'
    page += '<h1>五场景系统性诊断与固定输入 CPU 回放</h1><p>没有调用 Qwen、WildDet3D 或 API。没有生成验收通过的 G_full。仅回放保存结果，检查修复后的约束求解器。原始实验文件不变。</p>'
    page += '<p>11 份已有布局中，'+str(audit['replay_geometry_and_visibility_passed'])+' 份在同一物体/尺寸/宿主/房间下通过几何和可见性代理；其余仍被拒绝。</p><a href="audit.json">完整证据与源码哈希</a>'
    (out/'index.html').write_text(page+''.join(cards))
    print(json.dumps(dict(report=str(out/'index.html'),source_unchanged=unchanged,
        proxy_passing=audit['replay_geometry_and_visibility_passed'],still_rejected=audit['replay_still_rejected'],
        model_called=False,new_accepted_worlds=0),ensure_ascii=False))
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args = p.parse_args();diagnose(args.source,args.out)


if __name__=='__main__':main()
