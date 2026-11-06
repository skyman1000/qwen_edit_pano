"""CPU-only visual review of predicted boxes; no GT/model or input mutation."""
import csv
import html
import itertools
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ..common import ROOT, read, rows, sha, write, validate_graph
from .geometry import CANONICAL_CATEGORIES


def inventory_image(image_path, graph, output):
    image = Image.open(image_path).convert('RGB')
    image.thumbnail((1024, 1024))
    draw = ImageDraw.Draw(image)
    for obj in graph['objects']:
        box = obj['bbox2d_visible_xyxy_norm']['value']
        if box is None:
            continue
        rect = np.array(box)*[image.width,image.height,image.width,image.height]
        draw.rectangle(rect.tolist(), outline='red', width=2)
        label = obj['track_id']+' '+CANONICAL_CATEGORIES.get(obj['category_id']['value'],'unknown')
        draw.text(tuple(rect[:2]), label, fill='red', stroke_width=1, stroke_fill='white')
    image.save(output)


def interactive_boxes(graph, hfov, image_size, output, title=None, scene_bounds=None):
    import plotly.graph_objects as go
    graph = validate_graph(graph)
    fig = go.Figure()
    signs = np.array(list(itertools.product((-1,1), repeat=3)))
    edges = [(i,j) for i,j in itertools.combinations(range(8),2) if (signs[i]!=signs[j]).sum()==1]
    shown = set()
    for obj in graph['objects']:
        center, size = obj['center_local_m']['value'],obj['size_aabb_local_m']['value']
        if center is None or size is None:
            continue
        role = 'observed' if obj['local_evidence']=='visible' else 'hypothesized'
        if role=='hypothesized' and not obj['category_id']['valid']:
            role='unmapped candidate'
        role=obj.get('visual_role',role) if obj['local_evidence']!='visible' else role
        corners = np.array(center)+signs*np.array(size)/2
        points=[]
        for i,j in edges:
            points.extend((corners[i].tolist(),corners[j].tolist(),[None,None,None]))
        # Display camera y vertically, without altering any saved camera xyz.
        x,y,z = zip(*points)
        category=CANONICAL_CATEGORIES.get(obj['category_id']['value'],'unmapped')
        label=html.escape(obj['track_id']+' '+category)
        hover=label+'<br>'+html.escape(obj.get('description',''))+'<br>camera xyz='+str(np.round(center,3).tolist())+'<br>full size='+str(np.round(size,3).tolist())
        if 'taxonomy_review' in obj:
            audit=obj['taxonomy_review']
            hover+='<br>raw label='+html.escape(str(audit['raw_label']))+'<br>dataset label='+html.escape(str(audit['dataset_category_label']))+'<br>mapping='+html.escape(audit['mapping_status'])
        fig.add_trace(go.Scatter3d(x=x,y=z,z=y,mode='lines',name=role,legendgroup=role,
            showlegend=role not in shown,line=dict(color=obj.get('visual_color',{'observed':'#287db5','hypothesized':'#e88c13','unmapped candidate':'#b14a62'}.get(role,'#e88c13')) if obj['local_evidence']!='visible' else '#287db5',width=5),
            hovertemplate=hover+'<extra></extra>'))
        shown.add(role)
    th=math.tan(math.radians(hfov)/2);tv=th*image_size[1]/image_size[0]
    for x in (-th,th):
        for y in (-tv,tv):
            fig.add_trace(go.Scatter3d(x=[0,x],y=[0,1],z=[0,y],mode='lines',
                line=dict(color='gray',width=2),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode='markers+text',text=['camera'],
        marker=dict(color='red',size=5),name='camera',hovertemplate='camera origin<extra></extra>'))
    fig.update_layout(title=title or 'Predicted camera AABBs: blue observed / orange hypothesized; gray input FOV',
        scene=dict(xaxis_title='camera x: right (m)',yaxis_title='camera z: forward (m)',
                   zaxis_title='camera y: IMAGE-UP (m), not verified gravity',aspectmode='data'),
        margin=dict(l=0,r=0,b=0,t=60),legend=dict(groupclick='togglegroup'))
    if scene_bounds is not None:
        extent={k:scene_bounds[k][1]-scene_bounds[k][0] for k in ('x','y','z')}
        fig.update_layout(scene=dict(xaxis_range=scene_bounds['x'],yaxis_range=scene_bounds['z'],
            zaxis_range=scene_bounds['y'],aspectmode='manual',
            aspectratio=dict(x=extent['x'],y=extent['z'],z=extent['y'])))
    fig.write_html(str(output),include_plotlyjs=True,full_html=True,auto_open=False)
    return fig


def panel(tiles, output, caption):
    canvas=Image.new('RGB',(1440,1040),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,8),caption,fill='black')
    for i,(label,path) in enumerate(tiles[:6]):
        x=(i%3)*480;y=40+(i//3)*500
        draw.text((x+8,y+4),label,fill='black')
        if path and path.is_file():
            with Image.open(path) as image:
                fitted=ImageOps.contain(image.convert('RGB'),(464,464))
                canvas.paste(fitted,(x+8,y+28))
        else:
            draw.text((x+8,y+40),'Not available / stage did not complete',fill='darkred')
    canvas.save(output,quality=90)


def review(manifest, output):
    output=Path(output).resolve()
    config=read(output/'run_config.json')
    if config['manifest_sha256']!=sha(manifest):
        raise ValueError('Review manifest differs from frozen batch')
    progress=read(output/'progress.json');cases=rows(manifest);records=[];cards=[]
    for case in cases:
        case_dir=output/case['id'];case_dir.mkdir(exist_ok=True)
        info=progress.get(case['id'],{})
        def success(stage):
            record=info.get(stage,{})
            return output/record['directory'] if record.get('status')=='completed' else None
        inventory,observer,completion=[success(k)for k in ('inventory','observer','completion')]
        observed_graph=read((observer or inventory)/'G_obs.json') if observer or inventory else None
        if observed_graph:
            inventory_image(case['local_rgb'],observed_graph,case_dir/'inventory_2d.png')
        full=read(completion/'G_full.json') if completion else None
        summary=read(completion/'summary.json') if completion else {}
        if full:
            with Image.open(case['local_rgb']) as im: image_size=im.size
            interactive_boxes(full,case['hfov_deg'],image_size,case_dir/'scene_3d.html')
        tiles=[('1 Input RGB',Path(case['local_rgb'])),
               ('2 Qwen category + 2D instance boxes',case_dir/'inventory_2d.png'),
               ('3 WildDet3D AABBs projected onto RGB',observer/'projection_boxes.png' if observer else None),
               ('4 Full layout: camera x/z; NOT recovered walls',completion/'layout_refined.png' if completion else None),
               ('5 Camera-y intervals; NOT a measured floor',completion/'camera_y_refined.png' if completion else None),
               ('6 Prompt boxes RED / detector-selected boxes GREEN',observer/'instance_binding.png' if observer else None)]
        panel(tiles,case_dir/'review.jpg',case['id'])
        errors=[]
        for stage,data in info.items():
            if data.get('status')=='completed':continue
            detail=data.get('error','incomplete')
            if data.get('directory'):
                stage_summary=output/data['directory']/'summary.json'
                if stage_summary.is_file():detail=read(stage_summary).get('error',detail)
            errors.append(stage+': '+detail)
        pending=(not completion and not any(v.get('status')=='failed'for v in info.values())
                 and any(v.get('status')=='pending'for v in info.values()))
        record=dict(id=case['id'],label=case.get('display_label',''),
            status='completed_artifacts_not_quality_certified' if completion else 'pending_manual_gpu' if pending else 'failed_or_incomplete',
            observed=summary.get('observed',len(observed_graph['objects']) if observed_graph else None),added=summary.get('hypothesized'),
            stage_status='; '.join(k+'='+v.get('status','unknown')for k,v in info.items()),
            outside_fov_failed=summary.get('outside_fov_failed'),
            suspected_aabb_overlap_pairs=summary.get('suspected_aabb_overlap_pairs'),
            quality_status=summary.get('quality_status'),error='; '.join(errors))
        records.append(record)
        added=full['objects'][len(observed_graph['objects']):] if full else []
        text=json.dumps([dict(id=o['track_id'],category=CANONICAL_CATEGORIES[o['category_id']['value']],
            center=o['center_local_m']['value'],full_size=o['size_aabb_local_m']['value'])for o in added],
            ensure_ascii=False,indent=2)
        links=[]
        for stage,data in info.items():
            log=data.get('log')
            if log:links.append('<a href="'+html.escape(log)+'">'+stage+' log</a>')
        if completion:
            rel=str(completion.relative_to(output))
            links += [f'<a href="{rel}/G_obs.json">G_obs</a>',f'<a href="{rel}/G_full.json">G_full</a>',
                      f'<a href="{case["id"]}/scene_3d.html">交互式三维框</a>']
        cards.append('<article><h2>'+html.escape(case.get('display_label',case['id']))+'</h2><p>'
            +html.escape(case['id'])+'</p><pre>'+html.escape(json.dumps(record,ensure_ascii=False,indent=2))
            +'</pre><img src="'+case['id']+'/review.jpg"><p>'+' | '.join(links)
            +'</p><details><summary>新增对象的实际类别、xyz和完整尺寸</summary><pre>'+html.escape(text)+'</pre></details></article>')
    write(output/'summary.json',dict(cases=len(cases),completed_cases=sum(bool(success_row['status'].startswith('completed'))for success_row in records),
        failed_cases=sum(r['status']=='failed_or_incomplete'for r in records),
        pending_cases=sum(r['status']=='pending_manual_gpu'for r in records),scene_gt_used=False,
        quality_certified=False,panorama_generated=False,cases_detail=records))
    with (output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    page='''<!doctype html><meta charset="utf-8"><title>Local RGB → observed boxes → hypothesized world</title>
<style>body{font-family:sans-serif;max-width:1500px;margin:auto;padding:20px}article{border-top:2px solid #ddd;margin-top:24px}img{max-width:100%}pre{white-space:pre-wrap;background:#f5f5f5;padding:12px}a{margin-right:10px}</style>
<h1>局部图片 → 观测状态 → 视野外补全：固定验证样例</h1>
<p>蓝框是观测，橙框是假设。状态完成表示文件已写出，不代表语义或几何正确。三维图可以拖动旋转、滚轮缩放、点击图例开关对象组，离线可用。没有读取场景GT、生成全景或做物理支撑认证。</p>'''
    (output/'index.html').write_text(page+'\n'.join(cards),encoding='utf-8')
    return read(output/'summary.json')
