"""Export five-column PPT panels and browse all existing matched validation outputs; CPU only."""
import argparse
import collections
import csv
import html
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT=Path(__file__).resolve().parents[2]
BENCHMARK=ROOT/'qwen_edit_pano/outputs/native_pano_gt_A10000_val923_v1'


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--per-house',type=int,default=6)
    p.add_argument('--cell-width',type=int,default=1024)
    p.add_argument('--output',type=Path,default=BENCHMARK/'meeting_2026-10-27')
    args=p.parse_args()
    if args.per_house<1 or args.cell_width<256:p.error('Invalid panel count/width')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    panels=output/'panels';previews=output/'previews'
    panels.mkdir(exist_ok=True);previews.mkdir(exist_ok=True)
    plan=read(BENCHMARK/'plan.json');sources=read(BENCHMARK/'GENERATION_COMPLETE.json')['sources']
    ids=plan['ids'];houses=collections.defaultdict(list)
    for sid in ids:
        if any(sid not in sources[a] for a in ('native','finetuned','gt_state')):raise ValueError('Missing matched sample: '+sid)
        houses[sid.split('_')[0]].append(sid)
    chosen=[]
    for house,items in sorted(houses.items()):
        items=sorted(items);count=min(args.per_house,len(items))
        indices=[round(i*(len(items)-1)/(count-1)) for i in range(count)] if count>1 else [len(items)//2]
        chosen.extend(items[i] for i in indices)
    columns=[('局部图片','local'),('GT全景','gt'),('原生Qwen','native'),
             ('Qwen全景微调','finetuned'),('Qwen微调＋GT状态','gt_state')]
    def paths(sid):
        return [sources['finetuned'][sid]['local'],sources['finetuned'][sid]['gt'],
                sources['native'][sid]['image'],sources['finetuned'][sid]['image'],sources['gt_state'][sid]['image']]
    font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
    font=ImageFont.truetype(font_path,max(16,args.cell_width//30))
    cell=args.cell_width;height=cell//2;header=max(36,cell//18)
    inventory=[]
    for index,sid in enumerate(chosen,1):
        case=f'{index:02d}_{sid}';canvas=Image.new('RGB',(5*cell,height+header),'white');draw=ImageDraw.Draw(canvas)
        inputs=paths(sid)
        for j,((title,_),path) in enumerate(zip(columns,inputs)):
            with Image.open(path) as image:
                thumb=ImageOps.contain(image.convert('RGB'),(cell,height),method=Image.Resampling.LANCZOS)
            box=draw.textbbox((0,0),title,font=font);width=box[2]-box[0]
            draw.text((j*cell+(cell-width)//2,4),title,font=font,fill='black')
            canvas.paste(thumb,(j*cell+(cell-thumb.width)//2,header+(height-thumb.height)//2))
            if j:draw.line((j*cell,0,j*cell,height+header),fill='#dddddd')
        canvas.save(panels/(case+'.png'))
        canvas.resize((2560,round(canvas.height*2560/canvas.width)),Image.Resampling.LANCZOS).save(previews/(case+'.jpg'),quality=93)
        inventory.append(dict(index=index,id=sid,house=sid.split('_')[0],panel=str(panels/(case+'.png')),
                              **{key:path for (_,key),path in zip(columns,inputs)}))
        if index%11==0:print(f'Exported {index}/{len(chosen)} panels',flush=True)
    with (output/'selection_and_sources.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(inventory[0]));writer.writeheader();writer.writerows(inventory)
    intro=f'<p>完整三方法评估：{len(ids)}张、{len(houses)}栋验证建筑。导出{len(chosen)}张高清拼图，按建筑均匀取样，不按生成分数筛选。单个案例不代表总体结论。顺序：局部图片、GT、原生Qwen、全景微调、微调＋GT状态。微调固定epoch015/0.5，GT Adapter=A10000。</p>'
    style='<style>body{font-family:sans-serif;margin:24px}img{max-width:100%;height:auto}section{margin-bottom:28px}header{position:sticky;top:0;background:white;padding:8px;z-index:5}table{border-collapse:collapse;table-layout:fixed;width:100%}td,th{border:1px solid #ddd;padding:4px;width:20%;vertical-align:top}td img{width:100%;aspect-ratio:2/1;object-fit:contain}h2{overflow-wrap:anywhere}</style>'
    controls='<header>搜索图片ID或建筑：<input id="query" oninput="filterRows()"> <span id="counter"></span></header>'
    js='<script>function filterRows(){let q=document.getElementById("query").value.toLowerCase(),n=0;document.querySelectorAll("[data-key]").forEach(e=>{let show=e.dataset.key.toLowerCase().includes(q);e.hidden=!show;if(show)n++});document.getElementById("counter").textContent=n+" 项"}filterRows()</script>'
    page=['<!doctype html><meta charset="utf-8"><title>PPT 五列对比图</title>',style,'<h1>PPT 五列对比图</h1>',intro,
          '<p><a href="all_923.html">浏览全部923张</a> · 高清PNG位于panels目录，点击预览打开原图。</p>',controls]
    for row in inventory:
        case=f"{row['index']:02d}_{row['id']}"
        page.append(f'<section data-key="{html.escape(row["id"],quote=True)}"><h2>{case}</h2>'
                    f'<a href="panels/{case}.png"><img loading="lazy" src="previews/{case}.jpg"></a></section>')
    page.append(js);(output/'review.html').write_text('\n'.join(page))
    allpage=['<!doctype html><meta charset="utf-8"><title>全部923张五列对比</title>',style,
             '<h1>全部923张五列对比</h1>',intro,'<p><a href="review.html">返回已导出的高清拼图</a>；点击各列图片可打开原始分辨率。</p>',controls]
    for index,sid in enumerate(ids,1):
        allpage.append(f'<section data-key="{html.escape(sid,quote=True)}"><h2>{index:03d} · {sid}</h2><table><tr>'+''.join('<th>'+title+'</th>' for title,_ in columns)+'</tr><tr>')
        for path in paths(sid):
            relative=html.escape(os.path.relpath(path,output),quote=True)
            allpage.append(f'<td><a href="{relative}"><img loading="lazy" src="{relative}"></a></td>')
        allpage.append('</tr></table></section>')
    allpage.append(js);(output/'all_923.html').write_text('\n'.join(allpage))
    split_summary={}
    for stage,directory in [('panorama','paired_full_v1'),('adapter','world_polished_repair_v2')]:
        counts={};building_sets={}
        for split in ('train','val'):
            rows=[json.loads(l) for l in (ROOT/'qwen_edit_pano/data'/directory/(split+'.jsonl')).read_text().splitlines() if l.strip()]
            building_sets[split]={r['scene_id'] for r in rows}
            counts[split]=dict(images=len(rows),houses=len(building_sets[split]))
        if building_sets['train']&building_sets['val']:raise ValueError('House overlap')
        split_summary[stage]=counts
    (output/'dataset_counts.json').write_text(json.dumps(split_summary,indent=2)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(exported_panels=len(chosen),validation_images=len(ids),houses=len(houses),
                 per_house=args.per_house,panel_size=[5*cell,height+header],new_inference=False,
                 selection='six evenly spaced sorted IDs per house; no output or metric based filtering'),indent=2)+'\n')
    print(output/'review.html',flush=True)


if __name__=='__main__':main()
