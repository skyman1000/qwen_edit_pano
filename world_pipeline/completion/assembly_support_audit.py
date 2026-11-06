"""CPU audit of candidate support collapse; never changes samples or old outputs.

Also includes a deliberately small count/visibility prototype. Its example
parameters are synthetic and are NOT installed as an indoor-world prior.
"""
import argparse
from html import escape
import json
import math
from pathlib import Path

from ..common import read,write,sha,new_output
from . import assembly_functional as f


def empty_probability(model,use_evidence=True):
    """Exact P(no hidden instances) in the CURRENT finite functional model."""
    nodes={n['id']:n for n in model['nodes']};total=0.
    for row in f.role_posterior(model,use_evidence)['rows']:
        for zi,z in enumerate(nodes['Z']['states']):
            pempty=1.
            for bid,need in model['need_mechanisms'].items():
                r=nodes[need['requirement_node']]
                pr=r['cpt'][z+'|'+row['roles'][need['role']]][1]
                choices={s['key'] for s in need['strategies'] if s['object_keys']}
                pb=sum(prob for state,prob in zip(nodes[bid]['states'],nodes[bid]['cpt']['1']) if state in choices)
                pempty*=1-pr*pb
            total+=row['probability']*nodes['Z']['cpt']['root'][zi]*pempty
    return min(1.,max(0.,total))


def hidden_count_posterior(prior,observed_count,detection_probability):
    """Finite P(Nhidden|Nobs) under an explicit iid Binomial detection model.

    A mathematical prototype, not geometry-aware visibility or learned prior.
    Missing reports may be used here ONLY if the detection process is known.
    """
    if type(observed_count) is not int or observed_count<0:
        raise ValueError('observed_count must be a nonnegative integer')
    if isinstance(detection_probability,bool) or not isinstance(detection_probability,(int,float)) or not 0<=detection_probability<=1:
        raise ValueError('Detection probability must be in [0,1]')
    if not prior or any(type(n) is not int or n<0 or isinstance(p,bool) or
                        not isinstance(p,(int,float)) or not math.isfinite(p) or p<0 for n,p in prior.items()):
        raise ValueError('Count prior needs nonnegative counts and finite nonnegative masses')
    mass=sum(prior.values())
    if mass<=0:raise ValueError('Count prior has zero mass')
    q=detection_probability;weights={}
    for n,p in prior.items():
        if n<observed_count:continue
        likelihood=math.comb(n,observed_count)*q**observed_count*(1-q)**(n-observed_count)
        weights[n-observed_count]=p/mass*likelihood
    evidence=sum(weights.values())
    if evidence<=0:raise ValueError('Observed count has zero probability under this count/visibility model')
    return dict(hidden_count_probabilities={n:p/evidence for n,p in weights.items()},
        evidence_probability=evidence,scope='synthetic_finite_count_binomial_prototype_not_photo_posterior')


def inspect(root,backend='qwen'):
    root=Path(root).resolve();folder=root/backend/'completion'
    model=read(folder/'causal_model.json');summary=read(folder/'summary.json')
    programs=model.get('scene_program_states',{});needs=model.get('need_mechanisms',{})
    domain=len(model['component_registry']);audit=read(folder/'probability_audit.json') if (folder/'probability_audit.json').is_file() else None
    records=[dict(node=bid,role=n['role'],strategies=len(n['strategies']),
        nonempty_strategies=sum(bool(s['object_keys']) for s in n['strategies'])) for bid,n in needs.items()]
    proposed=summary.get('proposed_sample_counts',[])
    is_functional='functional_parameters' in model
    result=dict(root=str(root),backend=backend,status=summary['status'],candidate_instances=domain,
        observed_sha256=sha(root/backend/'observer3d/G_obs.json'),RGB_sha256=sha(root/'input.png'),
        hypotheses=len(model['hypotheses']),proposed_counts=proposed,
        exported_counts=[w['new_instances'] for w in summary.get('worlds',[])],
        layout_requests=len(list((folder/'calls').glob('layout_*.request.json'))),
        empty_registry=domain==0,candidate_support_degenerate=domain==0,
        hypothesis_functions={h['id']:h['functions'] for h in model['hypotheses']},requirements=records,
        programs={fid:dict(hypothesis=p['hypothesis']['id'],candidate_instances=sum(len(g['objects']) for g in p['groups']))
            for fid,p in programs.items()},
        source_sha256={name:sha(folder/name) for name in ('summary.json','causal_model.json','structural_samples.json')},
        causal_gain_validated=False)
    if audit:
        result.update(prior_roles=audit['prior']['role_presence'],posterior_roles=audit['posterior']['role_presence'],
            expected_hidden=audit['posterior']['expected_hidden_instances'])
    if is_functional:
        prior=f.marginals(model,False);posterior=f.marginals(model)
        result.update(prior_empty_probability=empty_probability(model,False),posterior_empty_probability=empty_probability(model),
            object_probability_changes={oid:posterior['object_presence'][oid]-prior['object_presence'][oid]
                for oid in model['component_registry']},
            roles_without_requirement_path=sorted(set(f.ROLES)-{n['role'] for n in needs.values()}))
    return result


def build(roots,output,backend='qwen'):
    records=[inspect(root,backend) for root in roots]
    if len({r['observed_sha256'] for r in records})!=1 or len({r['RGB_sha256'] for r in records})!=1:
        raise ValueError('Comparison requires the same frozen RGB and G_obs')
    out=new_output(output)
    write(out/'diagnosis.json',records)
    rows=[]
    for r in records:
        rows.append('<tr><td>'+escape(Path(r['root']).name)+'</td><td>'+str(r['candidate_instances'])+'</td><td>'+
            escape(str(r['proposed_counts']))+'</td><td>'+str(r['layout_requests'])+'</td><td>'+
            str(r.get('posterior_empty_probability','n/a'))+'</td></tr>')
    html='''<!doctype html><meta charset="utf-8"><title>零新增：候选支持域审计</title>
<style>body{font:17px sans-serif;max-width:1120px;margin:32px auto}td,th{padding:9px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}.error{color:#b33228}meter{width:220px}</style>
<h1>零新增发生在哪一层</h1>
<p>以下是实际运行文件的只读审计。候选目录为空时，任意功能后验都无法产生新实例。没有修改旧结果，没有重新调用模型。</p>
<table><tr><th>方法目录</th><th>候选实例</th><th>结构样本新增数</th><th>布局请求数（含修复）</th><th>P(零新增)</th></tr>'''+''.join(rows)+'''</table>
<p><a href="diagnosis.json">原始文件哈希与完整数值</a></p>
<h2>为什么“已有功能满足”不等于“隐藏库存为零”</h2>
<p class="error">下方是简化数学原型，参数为手写示例，不是当前照片推断，不会用于实际场景采样。</p>
<p>完整数量先验 P(N=0,1,2,3)=[0.25,0.45,0.25,0.05]，已观测到 1 个实例。调整每个实例被报告的概率：</p>
<input id="q" type="range" min="1" max="100" value="30"><span id="qvalue"></span>
<p id="answer"></p><table id="counts"></table>
<p>这里假设独立同分布的 Binomial 检出过程。真实相机中的遮挡与功能组合会产生相关性，需要几何观测模型替代。</p>
<h2>实际功能概率与候选结构</h2><pre>'''+escape(json.dumps(records,ensure_ascii=False,indent=2))+'''</pre>
<script>
function update(){let q=+document.querySelector('#q').value/100,prior=[.25,.45,.25,.05],weights=[0,0,0,0];
for(let n=1;n<=3;n++)weights[n-1]=prior[n]*n*q*Math.pow(1-q,n-1);
let total=weights.reduce((a,b)=>a+b,0);weights=weights.map(w=>w/total);
document.querySelector('#qvalue').textContent=q.toFixed(2);
document.querySelector('#answer').textContent='P(还有隐藏实例 | 已观测1个) = '+(1-weights[0]).toFixed(4);
document.querySelector('#counts').innerHTML='<tr><th>隐藏数量</th><th>后验概率</th></tr>'+weights.slice(0,3).map((w,i)=>'<tr><td>'+i+'</td><td>'+w.toFixed(4)+' <meter min="0" max="1" value="'+w+'"></meter></td></tr>').join('');}
document.querySelector('#q').oninput=update;update();</script>'''
    (out/'index.html').write_text(html);return out/'index.html'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--backend',choices=['api','qwen'],default='qwen')
    a=p.parse_args();print(build(a.roots,a.output,a.backend))


if __name__=='__main__':main()
