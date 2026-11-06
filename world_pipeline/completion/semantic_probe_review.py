"""CPU-only visualization of saved semantic hypotheses and fixed observed boxes."""
import argparse
import base64
import csv
from html import escape as esc
from pathlib import Path
import re
import shutil

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..common import new_output, read, sha, validate_graph, write
from .batch_world_review import interactive_boxes

MODES = ('state', 'image_state')
LABELS = dict(state='仅状态（含对象描述）', image_state='图片＋状态')
CSS = '''body{font:16px/1.65 sans-serif;background:#f2f5f9;color:#172033;margin:0}
main{max-width:1500px;margin:auto;padding:28px}a{color:#1766ad}h1,h2,h3{line-height:1.35}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
section,.card{background:white;border:1px solid #dde3ec;border-radius:12px;padding:20px;margin:16px 0}
.item{border-left:4px solid #d89b2a;background:#fffaf0;padding:12px;margin:12px 0}
.state .item{border-color:#8270bb}.note{background:#eaf2ff;padding:14px;border-radius:8px}
.warn{background:#fff0df;padding:14px}.small{font-size:14px;color:#526072}
img,svg{max-width:100%;height:auto}iframe{width:100%;height:650px;border:0}
table{width:100%;border-collapse:collapse}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}
pre{white-space:pre-wrap;overflow-wrap:anywhere}svg rect{fill:transparent;stroke:#167cb9;stroke-width:2}
svg g:hover rect{stroke:#e18e13;stroke-width:5}svg text{fill:white;font-size:14px;paint-order:stroke;stroke:#07517d;stroke-width:3}
@media(max-width:850px){.grid{display:block}main{padding:12px}}'''


def page(title, body):
    return ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title><style>'+CSS+
            '</style><main>'+body+'</main></html>')


def annotated_svg(image, context):
    with Image.open(image) as im:
        w, h = im.size
    uri = 'data:image/png;base64,'+base64.b64encode(Path(image).read_bytes()).decode()
    parts = [f'<svg viewBox="0 0 {w} {h}"><image width="{w}" height="{h}" href="{uri}"/>']
    for obj in context:
        box = obj['visible_bbox_xyxy_norm']
        if box is None:
            continue
        x,y,x1,y1 = [v*s for v,s in zip(box,(w,h,w,h))]
        label = obj['id']+' '+obj['category']+' '+obj['description']
        short = obj['id'].split('_')[-1]
        parts.append(f'<g><title>{esc(label)}</title><rect x="{x}" y="{y}" '
                     f'width="{x1-x}" height="{y1-y}"/><text x="{x+2}" y="{max(16,y+16)}">'
                     +esc(short)+'</text></g>')
    return ''.join(parts)+'</svg>'


def font(size):
    path = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    return ImageFont.truetype(str(path) if path.exists() else 'DejaVuSans.ttf',size)


def semantic_graph(context, answer):
    """A colored inventory/reference graph, never a fabricated spatial layout."""
    additions=answer.get('additions',[])
    height=100+max(len(context),len(additions),1)*68
    nodes={o['id']:(20,80+i*68) for i,o in enumerate(context)}
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1240 {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        '<text x="20" y="28" font-size="20" fill="#172033">蓝色：观测；橙色：新增语义候选。排版不表示方位、距离或三维位置。</text>',
        '<text x="20" y="57" font-size="16">观测清单</text><text x="650" y="57" font-size="16">补全清单</text>']
    for i,obj in enumerate(additions):
        y=80+i*68
        text=obj.get('reason','')+' '+obj.get('placement','')
        for ref in set(re.findall(r'\bobserved_\d+\b',text)) & nodes.keys():
            _,oy=nodes[ref]
            parts.append(f'<line x1="580" y1="{oy+24}" x2="650" y2="{y+24}" stroke="#9aa7b8"/>')
    for x,items,color,fill in ((20,context,'#217db6','#e7f3ff'),(650,additions,'#ce8117','#fff0d9')):
        for i,obj in enumerate(items):
            y=80+i*68
            label=obj.get('name',obj.get('description',''))
            heading=obj['id']+' · '+obj.get('category','新增假设')
            parts.append(f'<g><title>{esc(label)}</title><rect x="{x}" y="{y}" width="560" height="56" rx="8" '
                f'stroke="{color}" fill="{fill}"/><text x="{x+10}" y="{y+21}" font-size="15" fill="{color}">'
                +esc(heading)+f'</text><text x="{x+10}" y="{y+44}" font-size="14">'+esc(label[:65])+'</text></g>')
    return ''.join(parts)+'</svg>'


def draw_wrap(draw, xy, text, width, fill='#26344a', size=23):
    x,y = xy
    face = font(size)
    line = ''
    for char in text:
        if char == '\n' or draw.textlength(line+char,font=face)>width:
            draw.text((x,y),line,font=face,fill=fill)
            y += int(size*1.6)
            line = '' if char=='\n' else char
        else:
            line += char
    if line:
        draw.text((x,y),line,font=face,fill=fill)
        y += int(size*1.6)
    return y


def static_panel(case, responses, note, destination):
    canvas=Image.new('RGB',(1680,1060),'#f2f5f9')
    draw=ImageDraw.Draw(canvas)
    title=note.get('title',case['id'][:18])
    modes=[m for m in MODES if m in responses]
    draw_wrap(draw,(30,20),title+'：观测与语义补全对照',1600,size=32)
    draw_wrap(draw,(30,77),'新增物体没有三维坐标；本图是内容对照，不是完整布局。',1600,size=23)
    with Image.open(case['preview']) as im:
        image=ImageOps.contain(im.convert('RGB'),(520,620))
        canvas.paste(image,(30,145))
    column_width=1000 if len(modes)==1 else 500
    for i,mode in enumerate(modes):
        x=580+i*540
        draw_wrap(draw,(x,145),LABELS[mode],500,size=27)
        raw=responses.get(mode,{})
        y=draw_wrap(draw,(x,198),'场景：'+raw.get('scene_type','未完成'),column_width,size=22)
        translations=note.get(mode,{}).get('names_zh',{})
        for item in raw.get('additions',[]):
            y=draw_wrap(draw,(x,y+6),'• '+translations.get(item['id'],item['name']),column_width,size=21)
        y=draw_wrap(draw,(x,max(y+20,480)),'审阅：'+note.get(mode,{}).get('summary','待人工审阅'),column_width,size=22)
    draw_wrap(draw,(30,860),'结论：'+note.get('conclusion','请检查是否真正延展了未见场景。'),1600,size=25)
    canvas.save(destination,quality=92)


def render(source, output, notes_path=None):
    config=read(source/'config.json')
    inputs=config['inputs']
    notes=read(notes_path) if notes_path else {}
    if notes and Path(notes['source']).resolve()!=source:
        raise ValueError('Review notes belong to another experiment')
    tracked={str(p):sha(p) for p in (source/'config.json',source/'summary.json')}
    prepared=[]
    for case in inputs:
        case_id=case['id']
        if Path(case_id).name!=case_id:
            raise ValueError('Invalid case ID')
        input_path=source/case_id/'input.json'
        if read(input_path)!=case:
            raise ValueError('Saved input differs from experiment config: '+case_id)
        tracked[str(input_path)]=sha(input_path)
        for path,digest in case['source_sha256'].items():
            if sha(Path(path))!=digest:
                raise ValueError('Original observation/image changed: '+path)
            tracked[path]=digest
        graph=validate_graph(read(case['state']))
        if graph['stage']!='prediction' or graph['scope']!='observed':
            raise ValueError('Use predicted observations')
        answers={}
        for mode in MODES:
            path=source/case_id/mode/'answer.json'
            if path.exists():
                tracked[str(path)]=sha(path)
                answers[mode]=read(path)
                if notes and notes['answer_sha256'].get(str(path))!=sha(path):
                    raise ValueError('Review notes do not match saved answer: '+str(path))
            for suffix in ('request.json','generation.json','txt'):
                artifact=source/case_id/mode/('answer.'+suffix)
                if artifact.exists():
                    tracked[str(artifact)]=sha(artifact)
        prepared.append((case,graph,answers,notes.get('cases',{}).get(case_id,{})))
    output=new_output(output)
    cards=[]
    rows=[]
    for case,graph,answers,note in prepared:
        cid=case['id']; directory=output/cid;directory.mkdir()
        shutil.copyfile(case['image'],directory/'local.png')
        case=dict(case,preview=str(directory/'local.png'))
        static_panel(case,answers,note,directory/'comparison.jpg')
        with Image.open(case['image']) as im: image_size=im.size
        interactive_boxes(graph,case['hfov'],image_size,directory/'observed_3d.html',
            title='仅 G_obs 观测三维框；新增对象尚无三维位置')
        projection=Path(case['state']).parent/'projection_boxes.png'
        if projection.exists():
            tracked[str(projection)]=sha(projection)
            shutil.copyfile(projection,directory/'observed_projection.png')
        mode_cards=[]
        for mode,raw in answers.items():
            (directory/(mode+'_semantic_graph.svg')).write_text(semantic_graph(case['context'],raw))
            n=note.get(mode,{})
            obj_cards=[]
            for item in raw.get('additions',[]):
                title=n.get('names_zh',{}).get(item['id'],item['name'])
                review_note=n.get('objects',{}).get(item['id'],'待检查：存在理由、观测冲突与未见区域位置。')
                refs=sorted(set(re.findall(r'\b(?:observed|addition|add)_\d+\b',item['reason']+' '+item['placement'])))
                obj_cards.append('<div class="item"><b>'+esc(item['id']+' · '+title)+'</b><br>'
                    +esc(item['name'])+'<p><b>模型理由：</b>'+esc(item['reason'])+'</p><p><b>模型位置描述：</b>'
                    +esc(item['placement'])+'</p><p class="small">引用ID：'+esc(', '.join(refs) or '未明确引用')
                    +'</p><p class="note"><b>审阅：</b>'+esc(review_note)+'</p></div>')
                rows.append(dict(case_id=cid,mode=mode,object_id=item['id'],name=item['name'],
                    reason=item['reason'],placement=item['placement'],review=review_note))
            mode_cards.append('<section class="'+mode+'"><h2>'+LABELS[mode]+' · '
                +str(len(raw.get('additions',[])))+' 个列表条目</h2><p><b>场景判断：</b>'+esc(raw.get('scene_type',''))
                +'</p><p>'+esc(raw.get('scene_hypothesis',''))+'</p><p class="note">'+esc(n.get('summary','待审阅'))
                +'</p>'+''.join(obj_cards)+'</section>')
        observed_rows=''.join('<tr><td>'+esc(o['id'])+'</td><td>'+esc(o['category'])+'</td><td>'
            +esc(o['description'])+'</td></tr>' for o in case['context'])
        body='<a href="../index.html">← 全部样例</a><h1>'+esc(note.get('title',cid))+'</h1><p>'+esc(cid)+'</p>'
        body+='<p class="warn">原始模型回答保留原文；审阅是人工分析。新增项只有文字位置，不绘制新增三维框。</p>'
        body+='<div class="grid"><section><h2>原图</h2><img src="local.png"></section><section><h2>观测二维框</h2>'
        body+=annotated_svg(directory/'local.png',case['context'])+'<p>蓝框来自已有观测。悬停查看ID/类别/描述。</p></section></div>'
        body+='<details><summary>观测对象清单</summary><table>'+observed_rows+'</table></details>'
        body+='<div class="grid">'+''.join(mode_cards)+'</div><section><h2>本图结论</h2><p>'+esc(note.get('conclusion','待审阅'))+'</p></section>'
        for mode in answers:
            body+='<section><h2>'+LABELS[mode]+'：蓝色观测／橙色补全清单</h2><p>连线只表示明确引用观测ID；不表示空间关系。</p><img src="'+mode+'_semantic_graph.svg"></section>'
        body+='<section><h2>可旋转的 G_obs 三维框</h2><p>只展示固定观测；y为图像上方向，未认证重力/地板。拖动旋转，滚轮缩放。</p>'
        body+='<a href="observed_3d.html">单独打开三维图</a><iframe loading="lazy" src="observed_3d.html"></iframe></section>'
        if projection.exists():
            body+='<section><h2>已有 WildDet3D 框重投影</h2><img src="observed_projection.png"></section>'
        (directory/'index.html').write_text(page(note.get('title',cid),body))
        cards.append('<section><h2>'+esc(note.get('title',cid))+'</h2><a href="'+cid+'/index.html">'
            '<img src="'+cid+'/comparison.jpg"></a><p>'+esc(note.get('conclusion','待审阅'))+'</p><a href="'
            +cid+'/index.html">打开原图、逐对象分析、两组对照及三维观测图 →</a></section>')
    overview=Image.new('RGB',(1680,1060*len(prepared)),'white')
    for i,(case,*_) in enumerate(prepared):
        with Image.open(output/case['id']/'comparison.jpg') as im: overview.paste(im,(0,i*1060))
    overview.save(output/'all_cases.jpg',quality=88)
    intro=('<h1>局部观测 → 全景场景内容：实测可视化</h1><p>仅展示本次实际运行的条件：'
        +esc(', '.join(config.get('modes',MODES)))+'。state也包含对象描述。'
        '三维图只有已有观测；新增清单尚未成为G_full。</p>'
        '<p class="warn">名称精确匹配失败不等于Adapter不支持该语义；列表条目数不等于有效补全数量。'
        '格式通过不代表内容或位置正确。</p><a href="all_cases.jpg">整组静态长图</a> · '
        '<a href="objects.csv">逐对象原文与审阅CSV</a>')
    (output/'index.html').write_text(page('场景语义补全实测对照',intro+''.join(cards)))
    with (output/'objects.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=['case_id','mode','object_id','name','reason','placement','review'])
        writer.writeheader();writer.writerows(rows)
    if any(sha(Path(p))!=h for p,h in tracked.items()):
        raise ValueError('Source artifacts changed during visualization')
    write(output/'visualization.json',dict(source=str(source),source_sha256=tracked,cases=len(prepared),
        list_entries_by_mode={m:sum(len(a.get(m,{}).get('additions',[])) for _,_,a,_ in prepared) for m in MODES},
        observed_geometry_only=True,new_geometry_invented=False,source_unchanged=True,
        notes=None if not notes_path else str(notes_path.resolve())))
    print('CPU visualization written: '+str(output/'index.html'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--notes',type=Path)
    args=parser.parse_args()
    render(args.source.resolve(),args.output,args.notes)


if __name__=='__main__':
    main()
