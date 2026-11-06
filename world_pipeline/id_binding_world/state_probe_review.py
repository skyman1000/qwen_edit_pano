"""CPU perspective galleries and a standalone interactive review (no server)."""
import html
import itertools
import json
import math

import numpy as np

from ..build_polished_world import atomic_json, read


def review_groups(original,names):
    from .state_probe import floor_addition,legacy_variants
    objects={o['instance_id']:o for o in original['objects']}
    plant=next(r for n,_,r in legacy_variants(original) if n=='add_plant')['objects'][-1]
    floor=floor_addition(original,14)
    anchors=['gt_erp',*[n for n in names if n.startswith('correct_step')],
             *[n for n in ('baseline','correct') if n in names]]
    specs=[
        ('front','Local 正前方：检查原图保持',None,90.,['shuffled']),
        ('tv','电视：删除、移动、缩放、换镜子 / 挂画',objects['360'],70.,
         ['remove_tv','move_tv','half_tv','mirror_tv','picture_tv']),
        ('tabletop','Local 内桌上新增植物（与原 Local 冲突）',plant,70.,['add_plant']),
        ('hidden_chair','相机后方椅子：植物 / 柜子 / 凳子',objects['328'],65.,
         ['remove_hidden_chair','hidden_chair_to_plant','hidden_chair_to_cabinet','hidden_chair_to_stool']),
        ('dining','餐桌：删除 / 替换成床、沙发',objects['335'],85.,
         ['remove_dining_table','dining_table_to_bed','dining_table_to_sofa']),
        ('floor','同一空地新增：植物 / 椅子 / 柜子',floor,75.,
         ['add_floor_plant','add_floor_chair','add_floor_cabinet'])]
    groups=[]
    for key,title,obj,fov,edits in specs:
        selected=[n for n in edits if n in names]
        if key not in ('front','tv','tabletop') and not selected:continue
        if obj is None:yaw=pitch=0.
        else:
            x,y,z=obj['center_local_m']
            yaw=math.degrees(math.atan2(x,z));pitch=math.degrees(math.atan2(y,math.hypot(x,z)))
        groups.append(dict(key=key,title=title,yaw=yaw,pitch=pitch,fov=fov,
                           target=obj,cases=anchors+selected))
    return groups


def draw_box(image,box,view,color,label):
    """Project a camera-local AABB onto the SAME perspective used for review."""
    from PIL import ImageDraw
    yaw,pitch=map(math.radians,(view['yaw'],view['pitch']))
    right=np.array([math.cos(yaw),0,-math.sin(yaw)])
    up=np.array([-math.sin(yaw)*math.sin(pitch),math.cos(pitch),-math.cos(yaw)*math.sin(pitch)])
    forward=np.array([math.sin(yaw)*math.cos(pitch),math.sin(pitch),math.cos(yaw)*math.cos(pitch)])
    corners=np.array(box['center_local_m'])+np.array(list(itertools.product((-1.,1.),repeat=3)))*np.array(box['size_aabb_local_m'])/2
    camera=corners@np.stack((right,up,forward),1)
    width,height=image.size;focal=1/math.tan(math.radians(view['fov'])/2)
    draw=ImageDraw.Draw(image)
    def project(p):
        xy=np.array([(p[0]/p[2]*focal+1)*width/2-.5,(1-p[1]/p[2]*focal)*height/2-.5])
        return tuple(np.clip(xy,-10*max(width,height),10*max(width,height)))
    for i,a in enumerate(camera):
        for j,b in enumerate(camera):
            if j<=i or (i^j) not in (1,2,4):continue
            a,b=a.copy(),b.copy()
            if max(a[2],b[2])<=.01:continue
            if a[2]<.01:a=a+(.01-a[2])/(b[2]-a[2])*(b-a)
            if b[2]<.01:b=b+(.01-b[2])/(a[2]-b[2])*(a-b)
            draw.line([project(a),project(b)],fill=color,width=2)
    draw.rectangle((0,0,width,18),fill='black');draw.text((3,3),label,fill=color)
    return image


def write_review(output,original,names):
    import torch
    from PIL import Image,ImageDraw
    from qwen_pano.geometry import perspective
    groups=review_groups(original,names)
    files=['gt_erp',*names]
    config=read(output/'run_config.json') if (output/'run_config.json').exists() else {}
    states=read(output/'input_states.json') if (output/'input_states.json').exists() else {}
    descriptions={c['name']:c['description'] for c in config.get('cases',[])}
    category_names=config.get('category_names',{})
    overview=Image.new('RGB',(256*len(files),292*len(groups)),'white');draw=ImageDraw.Draw(overview)
    for column,name in enumerate(files):
        with Image.open(output/(name+'.png')) as image:
            erp=torch.from_numpy(np.array(image.convert('RGB'),copy=True)).permute(2,0,1).float()[None]/255
        for row,group in enumerate(groups):
            with torch.inference_mode():
                cut=perspective(erp,group['yaw'],group['pitch'],fov=group['fov'],size=384)
            rgb=(cut[0].permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)
            crop=Image.fromarray(rgb);directory=output/'reviews'/group['key'];directory.mkdir(parents=True,exist_ok=True)
            crop.save(directory/(name+'.jpg'),quality=94)
            overview.paste(crop.resize((256,256)),(256*column,292*row+36))
            draw.text((256*column+2,292*row+2),name,fill='black')
            draw.text((256*column+2,292*row+18),group['key'],fill='black')
            box=group['target']
            if box is not None:
                source=original if name=='gt_erp' or name.startswith('correct_step') else states.get(name)
                actual=next((o for o in (source or {}).get('objects',[]) if o['instance_id']==box['instance_id']),None)
                if actual is not None:
                    label=f"input: {category_names.get(str(actual['category_id']),actual['category_id'])} / id{actual['instance_id']}"
                    color=(255,155,20)
                else:
                    actual=box;color=(240,85,85) if name.startswith('remove_') else (180,180,180)
                    label='removed: original target box' if name.startswith('remove_') else 'reference region (not a detected object)'
                draw_box(crop.copy(),actual,group,color,label).save(directory/(name+'_boxes.jpg'),quality=94)
            else:crop.save(directory/(name+'_boxes.jpg'),quality=94)
    overview.save(output/'direction_review.jpg',quality=92)
    for group in groups:
        sheet=Image.new('RGB',(384*len(group['cases']),414),'white');draw=ImageDraw.Draw(sheet)
        for column,name in enumerate(group['cases']):
            with Image.open(output/'reviews'/group['key']/(name+'.jpg')) as image:sheet.paste(image,(384*column,30))
            draw.text((384*column+3,6),name,fill='black')
        sheet.save(output/'reviews'/(group['key']+'_comparison.jpg'),quality=94)
    atomic_json(output/'direction_review.json',dict(groups=groups,
        note='Identical perspective cameras for all modes. Box overlays project INPUT AABBs, never generated-object detections. Synthetic edits have no edited GT.'))
    data=dict(groups=groups,files=files,descriptions=descriptions)
    options=''.join(f'<option value="{html.escape(n)}">{html.escape(n)}</option>' for n in files)
    sections=[]
    for group in groups:
        cards=[]
        for name in group['cases']:
            path=f"reviews/{group['key']}/{name}"
            cards.append(f'<figure><a class="crop-link" href="{path}.jpg" target="_blank"><img class="crop" loading="lazy" '
                f'src="{path}.jpg" data-plain="{path}.jpg" data-boxes="{path}_boxes.jpg"></a>'
                f'<figcaption><b>{html.escape(name)}</b><br>{html.escape(descriptions.get(name,"Original unedited GT"))}</figcaption></figure>')
        sections.append(f'<section id="{group["key"]}"><h2>{html.escape(group["title"])}</h2>'
            f'<p><a href="reviews/{group["key"]}_comparison.jpg" target="_blank">打开本组横向拼图</a></p>'
            '<div class="cards">'+''.join(cards)+'</div></section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>World 状态干预可视化</title><style>
body{font-family:system-ui,sans-serif;margin:24px auto;padding:0 18px;max-width:1500px;background:#f3f5f8;color:#172033}
h1{font-size:26px}section,.panel{background:white;padding:18px;margin:18px 0;border-radius:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:14px}figure{margin:0;min-width:0}
img{max-width:100%;height:auto}.crop{width:100%}figcaption{font-size:13px;line-height:1.6;overflow-wrap:anywhere}
select,button{padding:7px;margin:5px;max-width:100%}.compare{position:relative;background:#111;width:100%;overflow:hidden}
.compare img{width:100%;display:block}.compare .over{position:absolute;top:0;left:0}.line{position:absolute;top:0;bottom:0;width:2px;background:#ffb020;left:50%}
input[type=range]{width:100%}.note{line-height:1.7;color:#525b6e}nav a{margin-right:15px}a{color:#1759ac}summary{cursor:pointer}
</style><h1>固定客厅：GT 状态增删改对比</h1>
<p class="note">所有生成固定 Local、prompt、seed 和采样参数。GT 图仅对应未编辑场景。
橙色框是输入状态的 AABB 投影，红色框是被删除对象的原位置，灰色框是参考区域；都不是生成图的检测结果。
单次替换固定几何，床替换餐桌等属于场景变更压力测试，不代表存在对应真实 GT。</p>
<nav>GROUP_LINKS</nav><div class="panel"><h2>滑动对比</h2>
<label>底图 <select id="left">OPTIONS</select></label><label>叠图 <select id="right">OPTIONS</select></label>
<label>视角 <select id="view"><option value="erp">完整 ERP</option>VIEW_OPTIONS</select></label>
<div class="compare"><img id="under"><img id="over" class="over"><span class="line" id="line"></span></div>
<input id="wipe" type="range" min="0" max="100" value="50"><p id="pair" class="note"></p></div>
<div class="panel"><label><input type="checkbox" id="boxes">显示输入状态框（位置参考）</label>
<details><summary>Local 输入</summary><img src="local.png" style="max-width:512px"></details>
<p><a href="direction_review.jpg" target="_blank">打开所有模式 × 所有视角总览</a> · <a href="run_config.json">运行与改动记录</a></p></div>
SECTIONS
<details class="panel"><summary>所有完整 ERP</summary>FULL_IMAGES</details>
<script>const reviewData=REVIEW_DATA;
const get=id=>document.getElementById(id);const names=reviewData.files;
get('left').value=names.includes('correct_step001000')?'correct_step001000':(names.includes('baseline')?'baseline':names[0]);
get('right').value=names.includes('correct')?'correct':names[names.length-1];
function path(name){const view=get('view').value;return view==='erp'?name+'.png':'reviews/'+view+'/'+name+(get('boxes').checked?'_boxes.jpg':'.jpg');}
function update(){get('under').src=path(get('left').value);get('over').src=path(get('right').value);
get('pair').textContent='底图：'+get('left').value+' ｜叠图：'+get('right').value+'（滑块控制叠图覆盖范围）';}
for(const id of ['left','right','view'])get(id).onchange=update;
get('wipe').oninput=()=>{get('over').style.clipPath='inset(0 '+(100-Number(get('wipe').value))+'% 0 0)';get('line').style.left=get('wipe').value+'%';};
get('boxes').onchange=()=>{for(const image of document.querySelectorAll('.crop')){image.src=get('boxes').checked?image.dataset.boxes:image.dataset.plain;image.parentElement.href=image.src;}update();};
update();get('wipe').oninput();</script></html>'''
    page=page.replace('GROUP_LINKS',''.join(f'<a href="#{g["key"]}">{html.escape(g["title"])}</a>' for g in groups))
    page=page.replace('VIEW_OPTIONS',''.join(f'<option value="{g["key"]}">{html.escape(g["title"])}</option>' for g in groups))
    page=page.replace('OPTIONS',options).replace('SECTIONS',''.join(sections))
    page=page.replace('FULL_IMAGES',''.join(f'<h3>{html.escape(n)}</h3><a href="{n}.png" target="_blank"><img loading="lazy" src="{n}.png"></a>' for n in files))
    page=page.replace('REVIEW_DATA',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    (output/'comparison.html').write_text(page)
