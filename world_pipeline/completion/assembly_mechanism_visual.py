"""v8 probability-source audit plus existing equal-axis XZ/3D/ERP gallery."""
import argparse
from collections import Counter
from html import escape
import itertools
import json
import math
from pathlib import Path

from ..common import read,write,sha
from . import assembly_mechanism as scm
from .assembly_activation_visual import report as geometry_report


def inventory_law(m,max_states=50000):
    if math.prod(len(v) for v in m['slots'].values())>max_states:return None
    hs=[h['id'] for h in m['profiles']];q=dict(zip(hs,m['nodes'][0]['cpt']['root']));result={}
    for h in hs:
        if not q[h]:continue
        for j,p in m['activation_rows'][h].items():
            if not p:continue
            options=[]
            for i,r in enumerate(m['slots']):
                on=m['joint_components'][j][i];row=[]
                if on<1:row.append(('none',1-on))
                row.extend((s,on*v) for s,v in scm._variants(m,h,j,r).items() if on*v>0)
                options.append(row)
            for values in itertools.product(*options):
                key=json.dumps([v for v,pv in values],separators=(',',':'))
                result[key]=result.get(key,0)+q[h]*p*math.prod(pv for v,pv in values)
    if abs(sum(result.values())-1)>1e-9:raise ValueError('Inventory enumeration is not normalized')
    return result


def _tv(a,b):return sum(abs(a.get(k,0)-b.get(k,0)) for k in set(a)|set(b))/2


def report(root,baseline_root=None,out=None):
    root=Path(root).resolve();page=geometry_report(root,baseline_root,out);dest=page.parent
    budget_matched=read(root/'call_budget.json').get('direct_and_functional_budgets_matched',False) if (root/'call_budget.json').exists() else False
    budget_scope=('Initial joint-selection requests/output limits matched; actual repairs and usage separately reported.' if budget_matched else
        'Functional/direct total request budgets differ.')
    audit=read(dest/'review.json');details={};sources={};plots=[]
    for backend in ('qwen','api'):
        folder=root/'end_to_end'/backend/'completion';path=folder/'causal_model.json'
        if not path.is_file():continue
        m=read(path);sources[str(path)]=sha(path);law=scm.marginals(m);uniform=scm.uniform_reference_ablation(m)
        result=dict(reference_vs_uniform_count_TV=_tv(law['hidden_count_probabilities'],uniform['hidden_count_probabilities']),
            conditional_uncertainty=scm.uncertainty(m),training_performed=False,causal_gain_validated=False,
            prior_ablation_scope='Same raw H/variant/activation replies and D; probability-only control, not a quality test')
        direct_path=root/'direct_inventory'/backend/'completion/causal_model.json'
        if direct_path.is_file():
            d=read(direct_path);sources[str(direct_path)]=sha(direct_path)
            if m['slots']!=d['slots'] or set(m['component_registry'])!=set(d['component_registry']):raise ValueError('Direct control candidate domain differs')
            original=scm.marginals(d);delta=0.
            for h in d['profiles']:
                after=scm.marginals(d,h['id'])
                delta=max(delta,max(abs(after['object_presence'][k]-v) for k,v in original['object_presence'].items()))
            if delta>1e-9:raise ValueError('Direct inventory still depends on H')
            first,second=inventory_law(m),inventory_law(d)
            result['direct_control']=dict(max_do_H_instance_probability_delta=delta,same_candidate_domain=True,
                joint_inventory_TV=_tv(first,second) if first is not None and second is not None else None,
                exact_inventory_enumeration=first is not None and second is not None,
                scope='Inventory-stage comparison only. H remains in metric context; '+budget_scope)
        table=m['empirical_tables'];q=law['profile_probabilities'];k0=table['reference']['valid_draws'];eta=table['reference']['concentration'];alpha=table['concentration'];channels=list(m['slots'])
        decomposition=[]
        for i,c in enumerate(channels):
            reference_on=sum(n for a,n in table['reference']['counts'].items() if a[i]=='1')
            conditional=sum(q[h]*sum(n for a,n in row['counts'].items() if a[i]=='1')/(row['valid_draws']+alpha) for h,row in table['activation'].items())
            shrink=sum(q[h]*alpha/(row['valid_draws']+alpha) for h,row in table['activation'].items())
            ref=shrink*reference_on/(k0+eta);uniform_part=shrink*eta/(2*(k0+eta))
            if abs(conditional+ref+uniform_part-law['activation_probabilities'][c])>1e-9:raise ValueError('Probability source decomposition differs')
            decomposition.append(dict(channel=c,conditional_vote_contribution=conditional,reference_vote_contribution=ref,
                uniform_tail_contribution=uniform_part,activation_probability=law['activation_probabilities'][c],
                no_conditional_on_votes_in_recognition_support=all(not any(n and a[i]=='1' for a,n in row['counts'].items()) for h,row in table['activation'].items() if q[h]),
                no_reference_on_votes=reference_on==0))
        result['activation_probability_sources']=decomposition
        calls=Counter()
        for p in (root/'mechanism_calls'/backend).glob('*.request.json'):calls[p.name.split('_')[0]]+=1
        result['actual_mechanism_requests_including_repairs']=dict(calls)
        result['planned_budget']=read(root/'call_budget.json');details[backend]=result;plots.append((backend,law,uniform,decomposition))
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,len(plots),figsize=(8*len(plots),4.6),squeeze=False)
        for (backend,law,uniform,rows),ax in zip(plots,axes[0]):
            x=list(range(len(rows)));data=[r['conditional_vote_contribution'] for r in rows];ref=[r['reference_vote_contribution'] for r in rows];tail=[r['uniform_tail_contribution'] for r in rows]
            ax.bar(x,data,label='H-conditional votes');ax.bar(x,ref,bottom=data,label='D-only reference votes')
            ax.bar(x,tail,bottom=[a+b for a,b in zip(data,ref)],label='Uniform tail')
            ax.set_xticks(x,[r['channel'].replace('observed_extension','ext').replace('new_group','new') for r in rows],rotation=60,ha='right',fontsize=8)
            ax.set(ylim=(0,1),ylabel='P(channel active)',title=backend+' / probability sources');ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(dest/'probability_sources.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,len(plots),figsize=(7*len(plots),4),squeeze=False)
        for (backend,law,uniform,rows),ax in zip(plots,axes[0]):
            for label,v in [('reference shrinkage',law),('uniform base / same frozen draws',uniform)]:
                pairs=sorted((int(n),p) for n,p in v['hidden_count_probabilities'].items());ax.plot([n for n,p in pairs],[p for n,p in pairs],label=label)
            ax.set(title=backend+' / probability-only base ablation',xlabel='Added object records',ylabel='Probability');ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(dest/'reference_base_ablation.png',dpi=150);plt.close(fig)
    audit['v8_probability_diagnostics']=details;audit['source_sha256'].update(sources);write(dest/'review.json',audit)
    section=('<section><h2>v8 免训练机制核对</h2><p>参考机制不是新的图像证据；概率尚未完成真实世界校准。'
        '直接对照不接收 H，但保留相同 q(H) 作数值布局上下文。'+escape(budget_scope)+' 不据此直接宣布因果收益。</p>'
        '<p>固定原始票数的均匀基分布对照只检验数值机制变化，没有重新生成几何。</p>')
    if plots:section+='<img src="probability_sources.png"><img src="reference_base_ablation.png">'
    for backend,d in details.items():
        section+='<details><summary>'+backend+' / 机制、预算与不确定性</summary><pre>'+escape(json.dumps(d,ensure_ascii=False,indent=2))+'</pre></details>'
    page.write_text(page.read_text()+section+'</section>')
    return page


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--baseline-root',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();print(report(a.root,a.baseline_root,a.out))


if __name__=='__main__':main()
