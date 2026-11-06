"""Read-only CPU forensics for saved observations/worlds; no inference or cleaning.

Statistical flags are review cues, not automated object existence decisions.
Per-scene visual notes below document manual inspection of the five saved RGBs.
No corrected G_obs or accepted world is produced by this audit.
"""
import argparse
from collections import Counter,defaultdict
from html import escape
import json
import os
from pathlib import Path
import shutil

from PIL import Image,ImageDraw

from ..common import read,write,sha

NOTES = {
 '8194nk5LbLH_c9e8dc09263e4d0da77d16de0ecddd39':
  '45条记录中28条是outlet。obs_018–044沿同一窄竖条延续到地毯底部；末两条完全重复。原图不支持这串插座。obs_017是墙上的可见插座，不能因同类误检而一并删除。',
 'X7HyMhZNoso_2bb4931911e4412cb51f6437a82c5b6c':
  '8条主要实例与图像大体对应，没有等候区那种序列式膨胀。它的全部世界候选仍失败，说明下游失败不能统统归因于观测数量。',
 'x8F5xyUWy9e_3a127fd2c5f34d978cba253145dae7d6':
  '未见同类长序列。浴缸等大件及towel框的实例边界需要复核；soap dish小框也需局部证据。7条并非人工确认的真值数量。',
 'zsNo4HB9uLZ_ead481533f834704bd489d3d44b6a03a':
  '19条包含6个枕头、2个床头柜、2盏灯、2个画框等，数量多本身不足以判错；需要检查小物体身份与三维几何，不能按数量上限压缩。',
 '8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029':
  'obs_009与obs_010是同一远端吧凳的整体/局部框，后者完全包含于前者。前景裁切吧凳的覆盖也需核对；记录条数不能代替实例身份验收。'
}


def overlap(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1])
    return dict(iou=inter/(aa+bb-inter),intersection_over_smaller=inter/min(aa,bb),exact=a==b)


def run(source,out):
    source=Path(source).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    assets=out/'assets';assets.mkdir();tracked={};scenes=[];cards=[];all_worlds=[]
    def load(path):
        path=Path(path);tracked[str(path)]=sha(path);return read(path)
    def href(path):return os.path.relpath(Path(path).resolve(),out)
    suite=load(source/'suite.json')
    for s in suite['scenes']:
        scene=source/'scenes'/s['id'];obs=scene/'observation';comp=scene/'completion'
        two=load(obs/'qwen/observer2d/G_2d.json');three=load(obs/'qwen/observer3d/G_obs.json')
        native=load(obs/'qwen/observer3d/native_boxes.json');semantic=load(obs/'qwen/observer2d/semantic_audit.json')
        row=dict(scene=s['id'],objects=len(two['objects']),categories=dict(Counter(o['raw_category'] for o in two['objects'])),
            visual_review_note=NOTES.get(s['id'],'Not manually inspected'),nested_same_category=[],same_x_strips=[])
        imported=load(obs/'imported_observation.json') if (obs/'imported_observation.json').exists() else None
        if imported:
            origin=Path(imported['source']);row['imported_from']=str(origin)
            row['same_2d_bytes']=sha(origin/'qwen/observer2d/G_2d.json')==sha(obs/'qwen/observer2d/G_2d.json')
            row['same_3d_bytes']=sha(origin/'qwen/observer3d/G_obs.json')==sha(obs/'qwen/observer3d/G_obs.json')
        objects=two['objects'];groups=defaultdict(list)
        for i,o in enumerate(objects):
            b=o['bbox2d_visible_xyxy_norm']['value'];groups[(o['raw_category'],b[0],b[2])].append(o)
            for previous in objects[:i]:
                if previous['raw_category']!=o['raw_category']:continue
                metric=overlap(previous['bbox2d_visible_xyxy_norm']['value'],b)
                if metric['intersection_over_smaller']>=.95:
                    row['nested_same_category'].append(dict(ids=[previous['track_id'],o['track_id']],category=o['raw_category'],**metric))
        suspicious=set()
        for (category,x0,x1),members in groups.items():
            if len(members)<5:continue
            ids=[o['track_id'] for o in members];suspicious.update(ids)
            row['same_x_strips'].append(dict(category=category,x_interval=[x0,x1],ids=ids,count=len(ids),
                unique_boxes=len({tuple(o['bbox2d_visible_xyxy_norm']['value']) for o in members}),
                scope='statistical_review_flag_not_automatic_rejection'))
        vals=sorted(v['prompt_prediction_iou'] for v in native)
        row['prompt_prediction_iou']=dict(minimum=min(vals) if vals else None,median=vals[len(vals)//2] if vals else None,
            zero=sum(v==0 for v in vals),below_half=sum(v<.5 for v in vals),
            interpretation='prompt_prediction_agreement; not_independent_object_existence_evidence')
        row['upstream_semantic_audit']=semantic
        sid=s['id'];image_path=obs/'input.png';tracked[str(image_path)]=sha(image_path)
        with Image.open(image_path) as im:image=im.convert('RGB')
        for original,label in [(image_path,'rgb'),(obs/'qwen/observer2d/boxes2d.png','boxes2d'),
                               (obs/'views/local_projection.png','boxes3d')]:
            if original.exists():shutil.copyfile(original,assets/(sid+'_'+label+'.png'))
        card='<article><h2>'+escape(sid)+'</h2><p>'+escape(row['visual_review_note'])+'</p>'
        card+='<p>模型记录 '+str(row['objects'])+'；类别 '+escape(json.dumps(row['categories'],ensure_ascii=False))+'</p>'
        card+='<div class="grid"><figure>原图<img src="assets/'+sid+'_rgb.png"></figure><figure>保存的二维预测<img src="assets/'+sid+'_boxes2d.png"></figure></div>'
        if suspicious:
            marked=image.copy();draw=ImageDraw.Draw(marked)
            for o in objects:
                b=o['bbox2d_visible_xyxy_norm']['value'];box=[b[0]*image.width,b[1]*image.height,b[2]*image.width,b[3]*image.height]
                draw.rectangle(box,outline='#cf263a' if o['track_id'] in suspicious else '#777777',width=2)
            marked.save(assets/(sid+'_strip.png'))
            card+='<p>红色：同x坐标的重复类别序列，仅作诊断标记；没有自动删除或导出新观测。</p><img class="medium" src="assets/'+sid+'_strip.png">'
            crop_ids=['obs_017','obs_018','obs_021','obs_028','obs_035','obs_043']
            card+='<div class="crops">'
            for oid in crop_ids:
                o=next((o for o in objects if o['track_id']==oid),None)
                if o is None:continue
                b=o['bbox2d_visible_xyxy_norm']['value'];p=[b[0]*image.width,b[1]*image.height,b[2]*image.width,b[3]*image.height]
                pad=40;l=max(0,int(p[0]-pad));t=max(0,int(p[1]-pad));r=min(image.width,int(p[2]+pad));bottom=min(image.height,int(p[3]+pad))
                crop=image.crop((l,t,r,bottom));d=ImageDraw.Draw(crop);d.rectangle((p[0]-l,p[1]-t,p[2]-l,p[3]-t),outline='#e02030',width=2)
                crop=crop.resize((crop.width*3,crop.height*3),Image.Resampling.LANCZOS)
                name=sid+'_'+oid+'_crop.png';crop.save(assets/name)
                card+='<figure><figcaption>'+escape(oid+' '+o['raw_category'])+'</figcaption><img src="assets/'+name+'"></figure>'
            card+='</div>'
            # A coordinate-envelope ablation, not a cleaned observation graph.
            remaining=[o for o in three['objects'] if o['track_id'] not in suspicious]
            def bottom(o):return o['center_local_m']['value'][1]-o['size_aabb_local_m']['value'][1]/2
            full_min=min(three['objects'],key=bottom);other_min=min(remaining,key=bottom)
            row['envelope_bottom_ablation']=dict(full_bottom_id=full_min['track_id'],full_bottom_m=bottom(full_min),
                without_flagged_strip_bottom_id=other_min['track_id'],without_flagged_strip_bottom_m=bottom(other_min),
                difference_m=bottom(full_min)-bottom(other_min),cleaned_G_obs_exported=False)
            card+='<p>包络下界敏感性（仅数值诊断）：'+escape(json.dumps(row['envelope_bottom_ablation'],ensure_ascii=False))+'</p>'
        card+='<details><summary>三维框与二维提示一致性；嵌套/重复证据</summary>'
        if (assets/(sid+'_boxes3d.png')).exists():card+='<img src="assets/'+sid+'_boxes3d.png">'
        card+='<pre>'+escape(json.dumps(dict(iou=row['prompt_prediction_iou'],nested=row['nested_same_category'],audit=semantic),ensure_ascii=False,indent=2))+'</pre></details>'
        phase=load(scene/'complete_result.json');domain=load(comp/'domain.json')
        row['completion_status']=phase['status'];row['proposed_programs']=[]
        for path in sorted(comp.glob('h*_inventories.json')):
            for p in load(path):
                row['proposed_programs'].append(dict(id=p['id'],objects=len(p['objects']),categories=[o['category'] for o in p['objects']]))
        row['domain_failures']=domain['failures'];row['executable_programs']=sum(len(ps) for ps in domain['programs'].values())
        if (comp/'summary.json').exists():
            summary=load(comp/'summary.json');row.update(accepted=summary['accepted'],draws=summary['attempts'],
                distinct_accepted_programs=summary['distinct_accepted_programs'],mechanism_coverage=summary['mechanism_coverage'])
            unique={}
            for world in summary['results']:
                if world['status']=='accepted_proxy_compatible':unique.setdefault(world['program'],world)
            row['accepted_state_supports']=[]
            for program,w in unique.items():
                folder=comp/('world_'+str(w['index']).zfill(2));state=load(folder/'G_full.json');critic=load(folder/'qwen_compatibility.json')
                floor=state['room_hypothesis']['floor_y']
                new=[dict(id=o['track_id'],category=o['raw_category'],support=o['support'],
                    center_m=o['center_local_m']['value'],size_m=o['size_aabb_local_m']['value'],
                    bottom_minus_hypothesized_floor_m=o['center_local_m']['value'][1]-o['size_aabb_local_m']['value'][1]/2-floor)
                    for o in state['objects'] if o['local_evidence']=='unobserved']
                record=dict(program=program,new=new,critic=critic,state=str(folder/'G_full.json'))
                row['accepted_state_supports'].append(record);all_worlds.append(record)
            card+='<p>导出 '+str(row['accepted'])+'/'+str(row['draws'])+' 次；不同通过程序 '+str(row['distinct_accepted_programs'])
            card+='；保留的H原估计质量 '+format(row['mechanism_coverage']['retained_elicited_mass'],'.3f')+'。上述数字不是语义质量认证。</p>'
            card+='<details><summary>通过世界中的支撑声明及Qwen审查原文</summary><pre>'+escape(json.dumps(row['accepted_state_supports'],ensure_ascii=False,indent=2))+'</pre></details>'
        else:row.update(accepted=0,draws=0,distinct_accepted_programs=0)
        card+='<p><a href="'+href(obs/'qwen/observer2d/G_2d.json')+'">原始二维图</a> · <a href="'+href(comp/'domain.json')+'">候选及失败</a></p>'
        cards.append(card+'</article>');scenes.append(row)
    immutable=all(sha(Path(p))==h for p,h in tracked.items())
    if not immutable:raise AssertionError('Source changed during read-only audit')
    audit=dict(source=str(source),model_called=False,API_used=False,production_code_modified=False,
        source_files_unchanged=immutable,source_sha256=tracked,scenes=scenes,
        proposed_programs=sum(len(s['proposed_programs']) for s in scenes),executable_programs=sum(s['executable_programs'] for s in scenes),
        world_draws=sum(s['draws'] for s in scenes),exported_world_draws=sum(s['accepted'] for s in scenes),
        distinct_exported_programs=sum(s['distinct_accepted_programs'] for s in scenes),
        valid_observation_graph_exported=False,quality_scope='manual_visual_forensics_and_saved_contracts; not_GT_precision_recall_benchmark')
    write(out/'audit.json',audit)
    header='<!doctype html><meta charset="utf-8"><title>003 观测真实性与世界状态审计</title><style>body{font:16px/1.7 system-ui;margin:24px;color:#18232c}article{border-top:1px solid #aaa;padding:16px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}img{max-width:100%}.medium{max-width:700px}.crops{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
    header+='<h1>003 观测真实性与世界状态审计</h1><p>未调用模型，未改动原结果，未输出清洗后的G_obs。红框及统计均为诊断；物体真实性来自原图逐项核对，不以数量或JSON通过判定。</p>'
    header+='<p>五图98条模型预测记录；实际发生24次世界采样并导出，只有7个不同程序。灰沙发客厅在采样前失败。现有通过标记不能视为合理完整世界验收。</p><a href="audit.json">结构化证据与哈希</a>'
    (out/'index.html').write_text(header+''.join(cards))
    print(json.dumps({k:audit[k] for k in ('source_files_unchanged','proposed_programs','executable_programs','world_draws','distinct_exported_programs','model_called')},ensure_ascii=False))
    print(out/'index.html')
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.out)


if __name__=='__main__':main()
