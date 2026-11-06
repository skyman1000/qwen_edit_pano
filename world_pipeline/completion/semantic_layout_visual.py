"""Render predicted box coordinates, including directions outside the input RGB."""
from html import escape
import itertools
import math

import numpy as np
from PIL import Image, ImageDraw

from .batch_world_review import interactive_boxes
from .box_layout import projected_box
from .geometry import CANONICAL_CATEGORIES

SIGNS=np.array(list(itertools.product((-1.,1.),repeat=3)))
EDGES=[(i,j) for i,j in itertools.combinations(range(8),2) if (SIGNS[i]!=SIGNS[j]).sum()==1]


def color(obj):
    return '#287db5' if obj['local_evidence']=='visible' else obj.get('visual_color', '#b14a62' if not obj['category_id']['valid'] else '#e88c13')


def erp_uv(points,width,height):
    points=np.asarray(points,dtype=float)
    x,y,z=points.T
    return np.column_stack(((.5+np.arctan2(x,z)/(2*np.pi))*width,
        (.5-np.arctan2(y,np.hypot(x,z))/np.pi)*height))


def erp_map(state,hfov,image_size,output,width=1200,height=600):
    """A 2:1 direction chart of actual box geometry, not a synthesized RGB panorama."""
    a=(np.arange(width)+.5)/width*2*np.pi-np.pi
    e=np.pi/2-(np.arange(height)+.5)/height*np.pi
    x=np.cos(e[:,None])*np.sin(a[None,:]);y=np.sin(e[:,None])+np.zeros((1,width))
    z=np.cos(e[:,None])*np.cos(a[None,:])
    th=math.tan(math.radians(hfov)/2);tv=th*image_size[1]/image_size[0]
    fov=(z>0)&(np.abs(x)<=th*z)&(np.abs(y)<=tv*z)
    pixels=np.full((height,width,3),250,dtype=np.uint8);pixels[fov]=[225,235,245]
    canvas=Image.fromarray(pixels);draw=ImageDraw.Draw(canvas)
    for deg in range(-180,181,30):
        u=(.5+deg/360)*width
        draw.line((u,0,u,height),fill='#dde2e8');draw.text((max(2,min(width-38,u+2)),height-16),str(deg),fill='#657487')
    for deg in range(-60,61,30):
        v=(.5-deg/180)*height
        draw.line((0,v,width,v),fill='#dde2e8')
    for obj in state['objects']:
        if obj['center_local_m']['value'] is None or obj['size_aabb_local_m']['value'] is None:
            continue
        center=np.array(obj['center_local_m']['value']);size=np.array(obj['size_aabb_local_m']['value'])
        corners=center+SIGNS*size/2
        for i,j in EDGES:
            segment=np.linspace(corners[i],corners[j],33)
            uv=erp_uv(segment,width,height)
            for p,q in zip(uv[:-1],uv[1:]):
                if abs(q[0]-p[0])<width/2: # Break at the ERP seam; never draw across the image middle.
                    draw.line([tuple(p),tuple(q)],fill=color(obj),width=2)
        if np.linalg.norm(center)>1e-8:
            u,v=erp_uv([center],width,height)[0]
            draw.text((max(0,min(width-130,u)),max(25,min(height-32,v))),obj['track_id'],fill=color(obj))
    draw.text((8,7),'ERP (not RGB): blue observed / orange dataset label / purple outside raw label table / red unresolved',fill='#172033')
    canvas.save(output)


def render(state,image,hfov,output,report,position_note=None):
    taxonomy=any('taxonomy_review' in o for o in state['objects'])
    legend=('蓝色观测 / 橙色类别表内 / 紫色原始名表外但可映射 / 红色未解析' if taxonomy else
        '蓝色观测 / 橙色补全 / 红色未映射候选')
    interactive_boxes(state,hfov,image.size,output/'scene_3d.html',
        title=legend+'；均为预测框，非真实重建')
    erp_map(state,hfov,image.size,output/'erp_state_map.png')
    objects=state['objects']
    centers=np.array([o['center_local_m']['value'] for o in objects])
    sizes=np.array([o['size_aabb_local_m']['value'] for o in objects])
    lo=np.minimum((centers-sizes/2)[:,[0,2]].min(0),[0,0])-.5
    hi=np.maximum((centers+sizes/2)[:,[0,2]].max(0),[0,0])+.5
    scale=760/max(hi-lo)
    def uv(p):return (70+(p[0]-lo[0])*scale,850-(p[1]-lo[1])*scale)
    xz=Image.new('RGB',(1000,940),'white');draw=ImageDraw.Draw(xz)
    draw.text((12,12),'Camera x/z projection; NOT a recovered floor plan. No assumed room walls or floor.',fill='black')
    overlay=image.copy();od=ImageDraw.Draw(overlay)
    th=math.tan(math.radians(hfov)/2);tv=th*image.height/image.width
    for obj,center,size in zip(objects,centers,sizes):
        p=uv((center-size/2)[[0,2]]);q=uv((center+size/2)[[0,2]])
        draw.rectangle((p[0],q[1],q[0],p[1]),outline=color(obj),width=2)
        draw.text(uv(center[[0,2]]),obj['track_id'],fill=color(obj))
        box=projected_box(center,size,th,tv)
        if box is not None and np.all(box[2:]>0) and np.all(box[:2]<1):
            r=np.clip(box,0,1)*[image.width-1,image.height-1,image.width-1,image.height-1]
            od.rectangle(r.tolist(),outline=color(obj),width=2)
            od.text(tuple(r[:2]),obj['track_id'],fill=color(obj))
    origin=uv([0,0]);draw.ellipse((origin[0]-5,origin[1]-5,origin[0]+5,origin[1]+5),fill='black')
    for sign in (-1,1):
        draw.line([origin,uv([sign*th,1])],fill='gray',width=2)
    xz.save(output/'layout_xz.png');overlay.save(output/'local_projection.png')
    rows=[]
    for obj in objects:
        rows.append('<tr style="color:'+color(obj)+'"><td>'+escape(obj['track_id'])+'</td><td>'
            +escape(CANONICAL_CATEGORIES.get(obj['category_id']['value'],'未映射'))+'</td><td>'
            +escape(obj.get('description',''))+'</td><td>'+escape(str(np.round(obj['center_local_m']['value'],3).tolist()))
            +'</td><td>'+escape(str(np.round(obj['size_aabb_local_m']['value'],3).tolist()))+'</td></tr>')
    content='''<!doctype html><meta charset="utf-8"><title>观测与补全三维状态</title>
<style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:24px auto;padding:15px}img{max-width:100%}iframe{width:100%;height:700px;border:0}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}.warn{background:#fff0dc;padding:15px}</style>
<h1>蓝色观测 / 橙色补全 / 红色未映射候选</h1>
'''
    if taxonomy:content=content.replace('蓝色观测 / 橙色补全 / 红色未映射候选',legend)
    position_note = position_note or '新增框由Qwen预测，不是WildDet3D测量或真实重建。'
    content+='<p class="warn">'+escape(position_note)+' 红色候选保留在G_full_candidates中，不进入Adapter子集G_full。坐标y是图像上方向，地板、重力和真实墙体未标定。</p>'
    content+='<p>'+escape(str(report))+'</p><a href="scene_3d.html">单独打开交互三维框</a>'
    if report.get('hypotheses_withdrawn'):
        content+='<p class="warn">有新增假设被显式撤回，未计作成功放置；撤回理由尚未验证。'
        content+='<a href="withdrawn_hypotheses.json">查看原假设及撤回理由</a></p>'
    content+='<iframe src="scene_3d.html"></iframe><h2>完整360°方向图（2:1 ERP投影，不是生成图片）</h2><img src="erp_state_map.png">'
    content+='<h2>相机x/z投影</h2><img src="layout_xz.png"><h2>原局部图重投影</h2><p>视野外／相机后方的对象不会出现在这张局部图上。</p><img src="local_projection.png">'
    content+='<h2>实际数值状态</h2><table><tr><th>ID</th><th>类别</th><th>名称</th><th>中心xyz(m)</th><th>完整AABB尺寸xyz(m)</th></tr>'+''.join(rows)+'</table>'
    (output/'index.html').write_text(content)
