"""Camera-relative coverage diagnostics, without a uniform-occupancy objective."""
import argparse
from html import escape
import json
import math
from pathlib import Path

from ..common import read,write
from . import assembly_panorama as model
from .assembly_mechanism_visual import report as probability_report


def report(root,baseline_root=None,out=None):
    root=Path(root).resolve();page=probability_report(root,baseline_root,out);dest=page.parent;audit=read(dest/'review.json');rows=[]
    for item in audit['worlds']:
        path=Path(item['source']);full=read(path);sem=read(path.parent/'semantic_world.json');contracts={o['id']:o for o in sem['objects']};objects=[]
        for obj in full['objects']:
            oid=obj['track_id']
            if oid not in contracts:continue
            x,y,z=obj['center_local_m']['value'];sz=obj['size_aabb_local_m']['value'][2]
            objects.append(dict(id=oid,azimuth_deg=math.degrees(math.atan2(x,z)),rear_half=z<0,full_rear_AABB=z+sz/2<0,
                root=contracts[oid]['attachment']['host'] is None))
        rows.append(dict(branch=item['branch'],world=item['world'],status=item['status'],objects=objects,
            added_records=len(objects),rear_half_records=sum(o['rear_half'] for o in objects),central_rear_records=sum(abs(o['azimuth_deg'])>135 for o in objects),
            roots=sum(o['root'] for o in objects),rear_roots=sum(o['root'] and o['rear_half'] for o in objects),
            spatial_program=sem.get('spatial_program'),functional_links=sem.get('functional_links',[])))
    domain=read(root/'shared/domain.json') if (root/'shared/domain.json').is_file() else {}
    expected={}
    for exp in ('end_to_end','direct_inventory'):
        for b in ('qwen','api'):
            path=root/exp/b/'completion/causal_model.json'
            if path.exists():
                law=model.marginals(read(path));expected[exp+'/'+b]=dict(any_declared_region_presence=law['any_declared_region_presence'],channel_region_joint_presence=law['channel_region_joint_presence'])
    audit['panoramic_spatial_review']=dict(worlds=rows,model_region_events=expected,spatial_support=domain.get('spatial_support'),
        rear_program_proposed=domain.get('rear_program_proposed'),
        scope='Angular counts are descriptive, not uniform coverage targets. No record in a sector does not establish empty space. Region probabilities are program events, not calibrated physical completeness.')
    write(dest/'review.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for b,ax in zip(('qwen','api'),axes):
        versions=sorted({r['branch'].split('_end_to_end_')[0] for r in rows if '_end_to_end_' in r['branch']})
        for version in versions:
            vals=[o['azimuth_deg'] for r in rows if r['branch']==version+'_end_to_end_'+b for o in r['objects'] if o['root']]
            if vals:ax.hist(vals,bins=np.arange(-180,181,15),histtype='step',linewidth=2,label=version+' / placement roots')
        hfov=domain.get('hfov',read(root/'context/experiment.json')['hfov'])
        ax.axvspan(-hfov/2,hfov/2,alpha=.12,color='blue',label='input sector')
        ax.set(xlim=(-180,180),xlabel='original-camera azimuth (degrees)',ylabel='exported placement roots',title=b)
        ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'panoramic_angles.png',dpi=160);plt.close(fig)
    section='<section><h2>360° 空间状态核对</h2><p>重点检查相机后方与输入视野边缘聚集。没有要求每个方位必须有家具；无物体记录的区域仍可能未知。中央后方定义为 |方位角|&gt;135°。</p>'
    section+='<p>后方程序是否被提出：'+escape(str(domain.get('rear_program_proposed')))+'。未提出不等于后方为空。</p><img src="panoramic_angles.png">'
    section+='<table><tr><th>分支/世界</th><th>新增记录</th><th>后半空间</th><th>中央后方</th><th>根对象</th><th>后方根对象</th></tr>'
    for r in rows:
        section+='<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in [r['branch']+'/'+r['world'],r['added_records'],r['rear_half_records'],r['central_rear_records'],r['roots'],r['rear_roots']])+'</tr>'
    section+='</table><details><summary>空间状态概率与功能服务关系</summary><pre>'+escape(json.dumps(audit['panoramic_spatial_review'],ensure_ascii=False,indent=2))+'</pre></details></section>'
    page.write_text(page.read_text()+section);return page


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline-root',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();print(report(a.root,a.baseline_root,a.out))


if __name__=='__main__':main()
