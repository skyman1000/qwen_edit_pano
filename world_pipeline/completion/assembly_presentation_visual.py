"""Read-only, category-labelled world views and exact camera-to-ERP projection."""
from html import escape
import itertools
import math
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

from ..common import read,write,validate_graph
from .assembly_taxonomy import vocabulary
from .assembly_pilot_visual import shared_bounds
from .semantic_layout_visual import erp_uv

OBSERVED='#287db5';MAPPED='#e88c13';UNMAPPED='#c3314d'
SIGNS=np.array(list(itertools.product((-1.,1.),repeat=3)))
EDGES=[(i,j) for i,j in itertools.combinations(range(8),2) if (SIGNS[i]!=SIGNS[j]).sum()==1]
DIRECTIONS={'front':([0,0,1],[0,1,0]),'right':([1,0,0],[0,1,0]),
    'back':([0,0,-1],[0,1,0]),'left':([-1,0,0],[0,1,0]),
    'up':([0,1,0],[0,0,-1]),'down':([0,-1,0],[0,0,1])}
LEGEND='Blue: observed | Orange: generated, mapped | Red: generated, unmapped'


def font(size=15):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def label_rows(state):
    """Display policy only. Never modify category IDs or saved world metadata."""
    cats=vocabulary()['categories'];rows=[];counts={True:0,False:0}
    for obj in state['objects']:
        visible=obj['local_evidence']=='visible';counts[visible]+=1
        code=('O' if visible else 'N')+f'{counts[visible]:02d}'
        cid=obj['category_id']['value'] if obj['category_id']['valid'] else None
        mapped=str(cid) in cats;canonical=cats.get(str(cid));raw=obj.get('raw_category') or obj.get('description') or 'unknown'
        name=canonical if mapped else raw
        coarse={'objects','object','misc','furniture','lighting','everyday belonging','everyday_belonging','unknown'}
        specific=obj.get('description') if str(raw).lower() in coarse else raw
        specific=specific or raw
        if canonical in coarse and specific!=canonical:name=specific+' ['+canonical+']'
        elif not mapped and str(raw).lower() in coarse:name=specific
        name=str(name).replace('_',' ')
        rows.append(dict(id=obj['track_id'],code=code,label=code+' '+name,category=canonical,
            raw_category=raw,description=obj.get('description'),mapped=mapped,visible=visible,color=OBSERVED if visible else MAPPED if mapped else UNMAPPED))
    return rows


def corners(obj):
    return np.asarray(obj['center_local_m']['value'])+SIGNS*np.asarray(obj['size_aabb_local_m']['value'])/2


def put_label(draw,anchor,text,color,occupied,width,height,size=14):
    """Deterministic short leaders; collisions scored rather than hiding IDs."""
    f=font(size);b=draw.textbbox((0,0),text,font=f);tw=b[2]+6;th=b[3]-b[1]+6
    candidates=[]
    for dy in (0,-th,th,-2*th,2*th,-3*th,3*th,-4*th,4*th):
        for dx in (4,-tw-4,20,-tw-20):
            x=float(np.clip(anchor[0]+dx,1,max(1,width-tw-1)));y=float(np.clip(anchor[1]+dy,26,max(26,height-th-22)))
            box=(x,y,x+tw,y+th)
            overlap=sum(max(0,min(box[2],q[2])-max(box[0],q[0]))*max(0,min(box[3],q[3])-max(box[1],q[1])) for q in occupied)
            candidates.append((overlap*100+math.hypot(dx,dy),box))
    _,box=min(candidates,key=lambda x:x[0]);occupied.append(box)
    ax,ay=anchor;tx,ty=box[:2]
    draw.line((ax,ay,tx+min(tw/2,8),ty+th/2),fill=color,width=1)
    draw.rectangle(box,fill='white');draw.text((tx+3,ty+1-b[1]),text,fill=color,font=f)


def basis(direction):
    f,u=(np.asarray(v,dtype=float) for v in DIRECTIONS[direction]);r=np.cross(u,f);return np.array([r,np.cross(f,r),f])


def line_in_canvas(p,q,width,height):
    """Liang-Barsky clipping avoids huge near-plane image coordinates."""
    p=np.asarray(p);d=np.asarray(q)-p;t0=0.;t1=1.
    for a,b in zip((-d[0],d[0],-d[1],d[1]),(p[0],width-1-p[0],p[1],height-1-p[1])):
        if abs(a)<1e-15:
            if b<0:return None
        elif a<0:t0=max(t0,b/a)
        else:t1=min(t1,b/a)
        if t0>t1:return None
    return p+t0*d,p+t1*d


def project_edges(obj,direction,hfov,width,height):
    points=corners(obj)@basis(direction).T;th=math.tan(math.radians(hfov)/2);tv=th*height/width
    def uv(p):return np.array([(0.5+p[0]/(2*p[2]*th))*(width-1),(0.5-p[1]/(2*p[2]*tv))*(height-1)])
    segments=[]
    for i,j in EDGES:
        p=points[i].copy();q=points[j].copy();near=1e-4
        if max(p[2],q[2])<near:continue
        if p[2]<near:p=p+(q-p)*((near-p[2])/(q[2]-p[2]))
        if q[2]<near:q=q+(p-q)*((near-q[2])/(p[2]-q[2]))
        clipped=line_in_canvas(uv(p),uv(q),width,height)
        if clipped is not None:segments.append(clipped)
    return segments


def perspective(state,labels,image,hfov,direction='front'):
    canvas=image.copy();draw=ImageDraw.Draw(canvas);occupied=[];shown=[]
    for obj,row in zip(state['objects'],labels):
        segments=project_edges(obj,direction,hfov,*canvas.size)
        for p,q in segments:draw.line((tuple(p),tuple(q)),fill=row['color'],width=3)
        if segments:
            anchor=np.mean([v for s in segments for v in s],axis=0)
            put_label(draw,anchor,row['label'],row['color'],occupied,*canvas.size);shown.append(row['id'])
    draw.rectangle((0,0,canvas.width,25),fill='white');draw.text((8,4),direction.upper()+' | '+LEGEND,fill='black',font=font(12))
    return canvas,shown


def erp_background(image,hfov,width=1600,height=800):
    """Project only measured RGB rays. Unknown directions remain grey."""
    if width!=2*height:raise ValueError('ERP must have exact 2:1 dimensions')
    a=(np.arange(width)+.5)/width*2*np.pi-np.pi;e=np.pi/2-(np.arange(height)+.5)/height*np.pi
    x=np.cos(e[:,None])*np.sin(a[None,:]);y=np.broadcast_to(np.sin(e[:,None]),(height,width));z=np.cos(e[:,None])*np.cos(a[None,:])
    th=math.tan(math.radians(hfov)/2);tv=th*image.height/image.width
    mask=(z>0)&(np.abs(x)<=th*z)&(np.abs(y)<=tv*z)
    rgb=np.full((height,width,3),232,dtype=np.uint8);source=np.asarray(image)
    u=(.5+x[mask]/(2*z[mask]*th))*(image.width-1);v=(.5-y[mask]/(2*z[mask]*tv))*(image.height-1)
    x0=np.floor(u).astype(int);y0=np.floor(v).astype(int);x1=np.minimum(x0+1,image.width-1);y1=np.minimum(y0+1,image.height-1)
    wx=(u-x0)[:,None];wy=(v-y0)[:,None]
    rgb[mask]=np.clip((1-wy)*((1-wx)*source[y0,x0]+wx*source[y0,x1])+wy*((1-wx)*source[y1,x0]+wx*source[y1,x1]),0,255).astype(np.uint8)
    return Image.fromarray(rgb),mask


def erp_boxes(state,labels,canvas):
    draw=ImageDraw.Draw(canvas);w,h=canvas.size;occupied=[]
    for deg in range(-180,181,45):
        u=(.5+deg/360)*w;draw.line((u,25,u,h),fill='#c8cdd4');draw.text((max(1,min(w-60,u+2)),h-20),str(deg)+' deg',font=font(12),fill='#4c5667')
    for obj,row in zip(state['objects'],labels):
        points=corners(obj)
        for i,j in EDGES:
            uv=erp_uv(np.linspace(points[i],points[j],65),w,h)
            for p,q in zip(uv[:-1],uv[1:]):
                if abs(q[0]-p[0])<w/2:draw.line((tuple(p),tuple(q)),fill=row['color'],width=2)
        c=obj['center_local_m']['value']
        if np.linalg.norm(c)>1e-8:put_label(draw,erp_uv([c],w,h)[0],row['label'],row['color'],occupied,w,h)
    draw.rectangle((0,0,w,25),fill='white');draw.text((8,4),'2:1 ERP world-state projection | '+LEGEND,fill='black',font=font(13))
    return canvas


def plan(state,labels,bounds,hfov,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(11,10));ax.set_aspect('equal');ax.set_xlim(bounds['x']);ax.set_ylim(bounds['z']);ax.grid(alpha=.2)
    ax.set(xlabel='Camera X / m',ylabel='Camera Z / m (forward +)',title='World-state XZ plan; walls are unknown\n'+LEGEND)
    for obj,row in zip(state['objects'],labels):
        c=np.asarray(obj['center_local_m']['value']);s=np.asarray(obj['size_aabb_local_m']['value'])
        ax.add_patch(Rectangle((c[0]-s[0]/2,c[2]-s[2]/2),s[0],s[2],facecolor=row['color'],edgecolor=row['color'],alpha=.16))
    t=math.tan(math.radians(hfov)/2)
    for sign in (-1,1):ax.plot([0,sign*t],[0,1],color='gray')
    ax.scatter([0],[0],c='black',s=25);ax.text(0,0,'camera');fig.tight_layout();fig.set_dpi(170);fig.canvas.draw()
    anchors=[ax.transData.transform([o['center_local_m']['value'][0],o['center_local_m']['value'][2]]) for o in state['objects']]
    fig.savefig(path,dpi=170);plt.close(fig)
    with Image.open(path) as im:canvas=im.convert('RGB')
    draw=ImageDraw.Draw(canvas);occupied=[]
    for anchor,row in zip(anchors,labels):put_label(draw,(anchor[0],canvas.height-anchor[1]),row['label'],row['color'],occupied,*canvas.size,size=16)
    canvas.save(path)


def scene3d(state,labels,bounds,hfov,image_size,output):
    import plotly.graph_objects as go
    fig=go.Figure();shown=set();text_indices=[]
    for obj,row in zip(state['objects'],labels):
        pts=corners(obj);lines=[]
        for i,j in EDGES:lines.extend([pts[i].tolist(),pts[j].tolist(),[None,None,None]])
        x,y,z=zip(*lines);role='observed' if row['visible'] else 'mapped generated' if row['mapped'] else 'unmapped generated'
        hover=escape(row['label'])+'<br>ID='+escape(row['id'])+'<br>'+escape(str(obj.get('description','')))
        fig.add_trace(go.Scatter3d(x=x,y=z,z=y,mode='lines',line=dict(color=row['color'],width=4),name=role,legendgroup=role,showlegend=role not in shown,hovertemplate=hover+'<extra></extra>'));shown.add(role)
    for role,col in (('observed',OBSERVED),('mapped generated',MAPPED),('unmapped generated',UNMAPPED)):
        records=[(o,r) for o,r in zip(state['objects'],labels) if ('observed' if r['visible'] else 'mapped generated' if r['mapped'] else 'unmapped generated')==role]
        if not records:continue
        c=np.array([o['center_local_m']['value'] for o,r in records]);text_indices.append(len(fig.data))
        fig.add_trace(go.Scatter3d(x=c[:,0],y=c[:,2],z=c[:,1],mode='text',text=[r['label'] for o,r in records],textfont=dict(color=col,size=11),name='labels',showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode='markers+text',text=['camera'],marker=dict(color='black',size=4),showlegend=False))
    fig.update_layout(title=LEGEND,scene=dict(xaxis_title='X right / m',yaxis_title='Z forward / m',zaxis_title='Y image-up / m',
        xaxis_range=bounds['x'],yaxis_range=bounds['z'],zaxis_range=bounds['y'],aspectmode='data'),margin=dict(l=0,r=0,b=0,t=55),
        updatemenus=[dict(type='buttons',buttons=[dict(label='Labels ON',method='restyle',args=[{'visible':True},text_indices]),dict(label='Labels OFF',method='restyle',args=[{'visible':False},text_indices])])])
    fig.write_html(str(output/'scene_3d.html'),include_plotlyjs=True,auto_open=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure=plt.figure(figsize=(12,10));ax=figure.add_subplot(111,projection='3d')
    for obj,row in zip(state['objects'],labels):
        pts=corners(obj)
        for i,j in EDGES:ax.plot(pts[[i,j],0],pts[[i,j],2],pts[[i,j],1],color=row['color'],lw=.8,alpha=.8)
        c=obj['center_local_m']['value'];ax.text(c[0],c[2],c[1],row['label'],fontsize=6,color=row['color'])
    ax.set(xlim=bounds['x'],ylim=bounds['z'],zlim=bounds['y'],xlabel='X right / m',ylabel='Z forward / m',zlabel='Y image-up / m',title=LEGEND)
    ax.set_box_aspect([bounds[k][1]-bounds[k][0] for k in ('x','z','y')]);ax.view_init(24,-65);figure.tight_layout();figure.savefig(output/'scene_3d.png',dpi=160);plt.close(figure)


def render_world(state,image,hfov,output,bounds=None):
    state=validate_graph(state);output=Path(output);output.mkdir(parents=True,exist_ok=False);image=image.convert('RGB');image.save(output/'local_rgb.png')
    labels=label_rows(state);write(output/'labels.json',labels);bounds=bounds or shared_bounds([state])
    plan(state,labels,bounds,hfov,output/'layout_xz.png')
    local,shown=perspective(state,labels,image,hfov);local.save(output/'local_projection.png')
    tiles=Image.new('RGB',(1800,1200),'white');direction_ids={}
    for j,d in enumerate(DIRECTIONS):
        tile,ids=perspective(state,labels,Image.new('RGB',(600,600),'#f4f6f8'),90,d);tile.save(output/('projection_'+d+'.png'));tiles.paste(tile,((j%3)*600,(j//3)*600));direction_ids[d]=ids
    tiles.save(output/'direction_projections.png');scene3d(state,labels,bounds,hfov,image.size,output)
    background,mask=erp_background(image,hfov);background.save(output/'erp_input_projection.png')
    erp_boxes(state,labels,background.copy()).save(output/'erp_overlay.png')
    erp_boxes(state,labels,Image.new('RGB',background.size,'#fafafa')).save(output/'erp_state_map.png')
    names=['local_rgb.png','layout_xz.png','local_projection.png','direction_projections.png','scene_3d.png','erp_overlay.png']
    overview=Image.new('RGB',(2400,2400),'white');draw=ImageDraw.Draw(overview)
    for j,name in enumerate(names):
        with Image.open(output/name) as im:tile=ImageOps.contain(im.convert('RGB'),(1180,750))
        x=(j%2)*1200;y=(j//2)*800;draw.text((x+12,y+8),name,font=font(20),fill='black');overview.paste(tile,(x+(1200-tile.width)//2,y+42+(750-tile.height)//2))
    overview.save(output/'overview.png')
    rows=''.join('<tr style="color:'+r['color']+'"><td>'+escape(r['code'])+'</td><td>'+escape(r['id'])+'</td><td>'+escape(str(r['category']))+'</td><td>'+escape(str(r['raw_category']))+'</td></tr>' for r in labels)
    page='<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;margin:24px}img{max-width:100%}td{padding:8px;border-bottom:1px solid #ddd}iframe{width:100%;height:720px;border:0}</style><h1>局部观测与完整世界状态</h1><p>蓝：观测；橙：可映射新增；红：未映射新增。编号和类别在各图中一致。灰色ERP区域未观测；这里显示状态投影，未生成隐藏区域RGB。</p><a href="labels.json">完整标签表</a><p><a href="scene_3d.html">单独打开交互三维图（可开关标签）</a></p>'
    for name in names+['erp_state_map.png','erp_input_projection.png']:page+='<h2>'+name+'</h2><img loading="lazy" src="'+name+'">'
    page+='<iframe loading="lazy" src="scene_3d.html"></iframe><table><tr><th>编号</th><th>原ID</th><th>数据集类别</th><th>原始类别</th></tr>'+rows+'</table>'
    (output/'index.html').write_text(page)
    checks=dict(labels=len(labels),palette=[OBSERVED,MAPPED,UNMAPPED],erp_dimensions=list(background.size),erp_observed_pixels=int(mask.sum()),
        projected_ids_in_original_view=shown,direction_projected_ids=direction_ids,full_state_modified=False)
    write(output/'visual_validation.json',checks);return checks
