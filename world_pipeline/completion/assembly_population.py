"""Finite full-inventory SCM with functional count priors and censored reports.

Image-conditioned proposals define a fixed truncated domain, NOT a globally
unbiased image posterior. Numerical mechanisms are explicit, replaceable priors.
No deficits, object quotas, deduplication of draws, or retry-until-nonempty.
"""
from copy import deepcopy
from html import escape
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np

from ..common import read,write,sha
from . import assembly_delta as delta, assembly_inventory as inventory
from .world_scm import validate,forward,key

PROTOCOL=delta.PROTOCOL
INTERFACE='whole_inventory_count_censoring_v3'
MODEL_VARIANT='functional_population_scm_v3'
ROLES=('social','reading','work','dining','storage','background')
# Specific nouns precede broad substrings (table lamp != table; bookcase != book).
PATTERNS=dict(work_tools=r'\b(laptop|computer|keyboard|monitor|printer)\b|电脑|键盘|显示器|打印机',
    lighting=r'\b(lamp|light|chandelier)\b|灯',
    storage=r'\b(wardrobe|cabinet|bookcase|bookshelf|dresser|shelving|shelf)\b|衣柜|书柜|书架|储物柜|置物架',
    beds=r'\b(bed|mattress)\b|床|床垫',media=r'\b(television|tv|projector|speaker|remote)\b|电视|投影仪|音箱|遥控器',
    books=r'\b(book|books|magazine|magazines)\b|书籍|杂志',
    tableware=r'\b(mug|cup|tray|plate|bowl|kettle|cutlery)\b|杯|托盘|盘子|碗|水壶|餐具',
    textiles=r'\b(pillow|cushion|blanket|rug|curtain)\b|抱枕|枕头|毯|地毯|窗帘',
    plants_decor=r'\b(plant|vase|sculpture|painting|picture|frame|artwork)\b|绿植|花瓶|雕塑|绘画|相框|装饰画',
    seating=r'\b(sofa|couch|chair|armchair|ottoman|stool|bench)\b|沙发|椅|脚凳|凳',
    tables=r'\b(table|desk|workbench)\b|桌|工作台',
    architecture=r'\b(door|window|fireplace|radiator)\b|门|窗|壁炉|暖气')
FAMILIES=tuple(PATTERNS)+('other',)
DEFAULTS=dict(version=1,provenance='explicit_reference_v3_NOT_FITTED_OR_CALIBRATED',
    role_priors=dict(zip(ROLES,[.40,.30,.25,.25,.55,.50])),
    base_means=dict(work_tools=.25,lighting=1.,storage=1.,beds=.20,media=.40,books=1.,
        tableware=1.,textiles=1.5,plants_decor=1.,seating=1.,tables=1.,architecture=1.,other=1.),
    role_loadings=dict(social=dict(seating=1.5,tables=.6,media=.8,textiles=1.),
        reading=dict(books=3.,lighting=.5,seating=.4),work=dict(work_tools=1.7,tables=.7,storage=.4),
        dining=dict(tables=.5,seating=1.5,tableware=2.),storage=dict(storage=1.5,books=1.,other=.5),
        background=dict(plants_decor=1.5,textiles=.5,other=1.)),
    dispersion=3.,population_scales=[.6,1.,1.8],scale_weights=[.25,.50,.25],
    report_probabilities=[.25,.55,.85],report_weights=[.25,.50,.25])


def parameters(value=None):
    p=deepcopy(DEFAULTS if value is None else value)
    if set(p)!=set(DEFAULTS) or p['version']!=1:raise ValueError('Use the complete population parameter version-1 schema')
    def positive(v):return type(v) in (int,float) and math.isfinite(v) and v>0
    if set(p['role_priors'])!=set(ROLES) or any(not positive(v) or v>=1 for v in p['role_priors'].values()):
        raise ValueError('All functional role priors must be in (0,1)')
    if set(p['base_means'])!=set(FAMILIES) or not all(positive(v) for v in p['base_means'].values()):
        raise ValueError('Every count family requires a positive base mean')
    if set(p['role_loadings'])!=set(ROLES):raise ValueError('All role mechanisms are required')
    for row in p['role_loadings'].values():
        if not set(row)<=set(FAMILIES) or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in row.values()):
            raise ValueError('Count loadings must be finite and nonnegative')
    if not positive(p['dispersion']):raise ValueError('dispersion must be positive')
    for states,weights in [('population_scales','scale_weights'),('report_probabilities','report_weights')]:
        if not isinstance(p[states],list) or not p[states] or len(p[states])!=len(p[weights]):
            raise ValueError('Latent states and weights differ')
        if not all(positive(v) for v in p[states]+p[weights]):raise ValueError('Latent state values and weights must be positive')
        if len(set(p[states]))!=len(p[states]):raise ValueError('Duplicate latent states')
    if any(v>=1 for v in p['report_probabilities']):raise ValueError('Report probabilities must be below one')
    if not isinstance(p['provenance'],str):raise ValueError('Parameter provenance required')
    return p


def family(description):
    description=description.replace('_',' ')
    if re.search(r'\b(tv stand|television stand|media console)\b|电视柜',description,re.I):return 'tables'
    return next((name for name,pattern in PATTERNS.items() if re.search(pattern,description,re.I)),'other')


def counts(objects,observed=False):
    result={k:0 for k in FAMILIES};bindings=[]
    for obj in objects:
        if observed:
            noun=obj.get('description','')+' '+obj.get('category','');name=family(noun);source='observed_description_and_category'
        else:
            noun=obj['category'];name=family(noun);source='category'
            # A real cached program used furniture for desk/chair/cabinet and
            # lighting for a desk lamp. Preserve raw labels but use concrete
            # instance names/variants for auditable family classification.
            if name=='other':
                for field in ('name','variant'):
                    candidate=obj.get(field,'');mapped=family(candidate)
                    if mapped!='other':noun=candidate;name=mapped;source=field;break
        result[name]+=1
        bindings.append(dict(id=obj.get('id',obj.get('key')),noun=noun,family=name,
            raw_category=obj.get('category'),noun_source=source))
    return result,bindings


def hypotheses_prompt(case,grounding):
    return ('Infer 1..2 distinct COMPLETE surrounding-world configurations compatible with this local indoor image. '
        'The photograph samples one region of a larger existing scene. Reason about possible unphotographed zones '
        'and ordinary possessions as uncertain hypotheses. A visible activity already being supported does not '
        'establish inventory completeness. Do not infer a resident biography. Use concrete function names, '
        'never copy the words coexisting activities. Return {"hypotheses":[{"description":"specific whole-world '
        'configuration","functions":["specific use"],"assumptions":["explicit uncertainty"]}]}. '
        'No object list or coordinates.\nOBSERVED REPORTS: '+json.dumps(grounding,ensure_ascii=False)+
        '\nFIXED OBSERVED INSTANCES: '+json.dumps(case['context'],ensure_ascii=False))


hypotheses_answer=delta.hypotheses_answer
furniture_answer=delta.furniture_answer
semantic_diagnostics=delta.semantic_diagnostics
active_world=inventory.active_world


def furniture_prompt(case,hypothesis,channel='surrounding'):
    focus=('Consider complete alternative surrounding organizations, including unphotographed activity or storage zones '
        'and their joint belongings.' if channel=='surrounding' else
        'Consider existing ordinary possessions, display/storage and secondary surrounding areas that are not '
        'required to satisfy the visible seating activity. Any new item must have its host included or a valid fixed host.')
    return (focus+' Return WHOLE hidden-world deltas, not lists of unmet requirements. '
        'Do not merely redescribe the photographed arrangement. All proposals are uncertain possibilities; '
        'an empty delta is permitted. The executor separately keeps an observed-only candidate, so there is no '
        'need to spend every alternative on it. There is no minimum count, mandatory class, or new functional goal. '
        'Do not infer occupations or personal history.\n'+delta.furniture_prompt(case,hypothesis))


def propose_programs(case,hypotheses,board,ask,out):
    result={}
    # Fixed budget, including empty responses. Never keep calling until nonempty.
    for z in hypotheses:
        for channel in ('surrounding','belongings'):
            variants=ask('furniture_'+z['id']+'_'+channel,board,furniture_prompt(case,z,channel),
                lambda v,z=z:furniture_answer(v,case,z))
            for variant in variants:
                fid='f'+str(len(result));variant['proposal_source']=dict(hypothesis=z['id'],channel=channel)
                result[fid]=variant
            write(out/'furniture_mechanisms.json',result);write(out/'scene_program_mechanisms.json',result)
    return result


def load_joint_domain(case,source,backend):
    """Revalidate preserved raw joint/baseline proposals, never old geometry."""
    source=Path(source).resolve();folder=source/backend/'completion'
    config=read(source/'experiment.json')
    if config.get('completion_model') not in ('joint','baseline'):
        raise ValueError('Candidate source must be an explicit joint/baseline experiment')
    if read(folder/'causal_model.json').get('response_interface')!=delta.INTERFACE:
        raise ValueError('Saved joint candidate interface differs')
    old=read(folder/'case.json')
    if sha(source/'input.png')!=sha(Path(case['image'])) or sha(source/backend/'observer3d/G_obs.json')!=sha(Path(case['state'])):
        raise ValueError('Cached candidate RGB/G_obs differs')
    if any(old.get(k)!=case.get(k) for k in ('hfov','image_size','context','observed_support_domains')):
        raise ValueError('Cached candidate camera/context/support domains differ')
    hypotheses=read(folder/'hypotheses.json');programs={};sources={}
    for z in hypotheses:
        path=folder/'calls'/('furniture_'+z['id']+'_repair.json')
        if not path.is_file():path=folder/'calls'/('furniture_'+z['id']+'.json')
        before=sha(path)
        variants=delta.furniture_answer(read(path),case,z)
        if sha(path)!=before:raise RuntimeError('Raw candidate source changed during validation')
        sources[str(path)]=before
        for variant in variants:
            variant['proposal_source']=dict(channel='revalidated_saved_joint',raw_file=str(path),raw_sha256=before)
            programs['f'+str(len(programs))]=variant
    return hypotheses,programs,dict(source=str(source),raw_sha256=sources,
        current_parser_revalidated=True,old_geometry_reused=False,old_probabilities_reused=False,
        scope='fixed_domain_probability_comparison_not_new_candidate_generation')


def build_model(hypotheses,programs,unused_contents=None,*,signature_fields=()):
    # Reuse joint inventory identity/host mechanisms, not the former R deficit gate.
    programs=deepcopy(programs)
    znull=dict(id='z_null',description='Observed-only finite candidate.',functions=[],
        assumptions=['No additional instances in this candidate; not a proof about unseen space.'])
    null=dict(description=znull['description'],hypothesis=znull,groups=[],relations=[],relation_bindings=[],
        inventory_scope='all_scale_hidden_delta',proposal_source=dict(channel='explicit_null'))
    # All empty proposals represent one inventory state, rather than multiplying
    # its mass by the number of empty paraphrases or camera-visible explanations.
    unique={};bindings={};seen={}
    for fid,program in programs.items():
        if not any(g['objects'] for g in program['groups']):
            bindings[fid]='f_null';continue
        signature=dict(groups=[dict(function=g['function'],observed_refs=sorted(g.get('observed_refs',[])),
            **{k:([dict(source=a,target=b,kind=c) for a,b,c in sorted({(v['source'],v['target'],v['kind']) for v in g[k]})]
                if k=='functional_links' else g[k]) for k in signature_fields if k in g},
            objects=[{k:o[k] for k in ('key','category','variant','placement_hint','attachment','affordances')}
                for o in sorted(g['objects'],key=lambda o:o['key'])]) for g in program['groups']],
            relations=[{k:r[k] for k in ('source','target','kind')} for r in program['relations']])
        digest=hashlib.sha256(json.dumps(signature,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if digest not in seen:seen[digest]=fid;unique[fid]=program
        bindings[fid]=seen[digest]
    programs=unique
    programs['f_null']=null
    active={p['hypothesis']['id'] for p in programs.values()}
    model=delta.build_model([h for h in hypotheses if h['id'] in active]+[znull],programs)
    model['proposed_scene_hypotheses']=deepcopy(hypotheses)
    model['proposal_bindings']=bindings
    model.update(model_variant=MODEL_VARIANT,response_interface=INTERFACE)
    return model


def _log_nb(n,mean,dispersion):
    r=dispersion
    return math.lgamma(n+r)-math.lgamma(r)-math.lgamma(n+1)+r*math.log(r/(r+mean))+n*math.log(mean/(r+mean))


def _mean(p,h,l,k,enabled=True):
    return p['population_scales'][l]*(p['base_means'][k]+sum(
        int(h[r])*p['role_loadings'][r].get(k,0.) for r in ROLES) if enabled else p['base_means'][k])


def _log_count_ratio(n,p,h,l,k,enabled=True):
    """Proper background-NB / functional-Poisson mixture, relative to NB.

    A same-shape NB mean ratio is an unbounded linear reward in n. The mixture
    instead has a finite preferred range and returns to 1-gamma in the tail,
    preserving background inventory without rewarding unlimited duplicates.
    """
    base=_mean(p,h,l,k,False);mean=_mean(p,h,l,k,enabled)
    if mean<=base:return 0.
    gamma=(mean-base)/mean
    log_poisson=n*math.log(mean)-mean-math.lgamma(n+1)
    return float(np.logaddexp(math.log1p(-gamma),math.log(gamma)+log_poisson-_log_nb(n,base,p['dispersion'])))


def _logsum(values):
    m=max(values);return m+math.log(sum(math.exp(v-m) for v in values))


def prepare_model(base,case,config,out):
    model=deepcopy(base);p=parameters(config.get('population_parameters'))
    policy=config.get('population_sampling','posterior')
    if policy not in ('posterior','prior','reference','no_function'):raise ValueError('Unknown population sampling law')
    enabled=policy!='no_function';observed,bindings=counts(case['context'],True)
    states={};vectors={};program_n={};program_counts={}
    for fid,program in model['furniture_states'].items():
        objects=[o for g in program['groups'] for o in g['objects']];hidden,_=counts(objects)
        total={k:observed[k]+hidden[k] for k in FAMILIES};vector=tuple(total[k] for k in FAMILIES)
        if vector not in vectors:
            nid='n'+str(len(vectors));vectors[vector]=nid;states[nid]=total
        program_n[fid]=vectors[vector];program_counts[fid]=dict(full=total,hidden=hidden)
    nodes=[]
    for r in ROLES:
        v=p['role_priors'][r];nodes.append(dict(id='H_'+r,states=['0','1'],parents=[],cpt={'root':[1-v,v]}))
    lw=np.asarray(p['scale_weights'],float);lw/=lw.sum();qw=np.asarray(p['report_weights'],float);qw/=qw.sum()
    lnames=[str(i) for i in range(len(lw))];qnames=[str(i) for i in range(len(qw))]
    nodes.append(dict(id='L_population',states=lnames,parents=[],cpt={'root':lw.tolist()}))
    parents=['H_'+r for r in ROLES]+['L_population'];n_cpt={}
    for values in itertools.product(['0','1'],repeat=len(ROLES)):
        h=dict(zip(ROLES,values))
        for l in range(len(lw)):
            # The image-proposed programs are a finite reference population,
            # not point samples from an evaluable global proposal density.
            # Raw full-count PMFs on these sparse representatives spuriously
            # favor sparse/null worlds. Use EXPLICIT functional density ratios
            # relative to the background law, with a uniform count-state base.
            # Turning off functional loadings then preserves the base inventory
            # support, rather than declaring all ordinary objects unnecessary.
            logs=[sum(_log_count_ratio(state[k],p,h,l,k,enabled) for k in FAMILIES)
                for state in states.values()]
            logmass=_logsum(logs)
            row=key(list(values)+[str(l)]);n_cpt[row]=[math.exp(v-logmass) for v in logs]
    nodes.append(dict(id='N_population',states=list(states),parents=parents,cpt=n_cpt))
    fstates=list(model['furniture_states']);f_cpt={}
    for nid in states:
        matches=[f for f in fstates if program_n[f]==nid]
        f_cpt[nid]=[float(f in matches)/len(matches) for f in fstates]
    nodes.append(dict(id='F',states=fstates,parents=['N_population'],cpt=f_cpt))
    nodes.append(dict(id='Q_report',states=qnames,parents=[],cpt={'root':qw.tolist()}))
    o_cpt={}
    for nid,full in states.items():
        for qi,q in enumerate(p['report_probabilities']):
            loglikelihood=0.
            for k,n in observed.items():
                if n==0:continue  # Missing categories are UNKNOWN, not negative reports.
                nfull=full[k]
                loglikelihood+=math.lgamma(nfull+1)-math.lgamma(n+1)-math.lgamma(nfull-n+1)+n*math.log(q)+(nfull-n)*math.log1p(-q)
            likelihood=math.exp(loglikelihood);o_cpt[key([nid,str(qi)])]=[likelihood,1-likelihood]
    nodes.append(dict(id='O_positive_counts',states=['report','other'],parents=['N_population','Q_report'],cpt=o_cpt))
    nodes.extend(n for n in model['nodes'] if n['id'] not in ('Z','F'))
    model.update(nodes=nodes,population_parameters=p,population_sampling=policy,functional_loadings_enabled=enabled,
        observed_counts=observed,observed_family_bindings=bindings,count_states=states,program_count_state=program_n,
        program_counts=program_counts,scene_program_states=deepcopy(model['furniture_states']),
        generation_factorization='H,L -> full N -> joint F -> E -> S; N,Q -> observed counts',
        distribution_scope='explicit_finite_count_model_conditional_on_image_proposed_domain_not_global_RGB_posterior',
        sampling_policy='iid_'+policy+'_duplicates_retained',cue_likelihood_defined=False,count_likelihood_defined=True,
        RGB_likelihood_defined=False,calibrated_posterior_defined=False,real_causal_structure_identified=False,
        candidate_support=dict(nonempty_programs=sum(any(g['objects'] for g in f['groups']) for f in model['furniture_states'].values()),
            candidate_instances=len(model['component_registry']),all_nonempty_proposals_missing=not bool(model['component_registry'])),
        likelihood_scope='coarse_independent_binomial_reports_positive_families_only_geometry_conditioned_later',
        prior_scope='finite_reference_count_distribution_times_functional_mixture_density_ratios',
        metric_mechanisms_learned=False,resident_history_inferred=False)
    validate(model)
    if out is not None:
        write(out/'population_parameters.json',p);write(out/'count_observations.json',dict(counts=observed,bindings=bindings,
            unreported_families='unknown',false_positives='not_modeled_in_this_frozen_observation_pilot'))
        audit=probability_audit(model);write(out/'probability_audit.json',audit);probability_page(model,audit,out/'probability_audit.html')
    return model


def posterior(model,use_reports=True):
    p=model['population_parameters'];nodes={n['id']:n for n in model['nodes']};rows=[]
    for values in itertools.product(['0','1'],repeat=len(ROLES)):
        h=dict(zip(ROLES,values));hp=math.prod(p['role_priors'][r] if h[r]=='1' else 1-p['role_priors'][r] for r in ROLES)
        for li,l in enumerate(nodes['L_population']['states']):
            context=key(list(values)+[l])
            for ni,n in enumerate(nodes['N_population']['states']):
                for qi,q in enumerate(nodes['Q_report']['states']):
                    weight=hp*nodes['L_population']['cpt']['root'][li]*nodes['N_population']['cpt'][context][ni]*nodes['Q_report']['cpt']['root'][qi]
                    if use_reports:weight*=nodes['O_positive_counts']['cpt'][key([n,q])][0]
                    rows.append(dict(roles=h,scale=l,count_state=n,report_mode=q,joint_mass=weight))
    total=sum(r['joint_mass'] for r in rows)
    if total<=0:raise ValueError('Population evidence has zero mass')
    for row in rows:row['probability']=row['joint_mass']/total
    return dict(rows=rows,evidence_mass=total,conditioned_on_reports=use_reports)


def marginals(model,use_reports=True,reference=False):
    post=posterior(model,use_reports);byid={n['id']:n for n in model['nodes']}
    row_weights=[row['probability'] for row in post['rows']]
    programs={fid:0. for fid in model['furniture_states']}
    for row in post['rows']:
        for fid,pb in zip(byid['F']['states'],byid['F']['cpt'][row['count_state']]):programs[fid]+=row['probability']*pb
    if reference:
        programs={fid:1/len(programs) for fid in programs}
        actual_n={n:0. for n in model['count_states']};reference_n={n:0. for n in actual_n}
        for row in post['rows']:actual_n[row['count_state']]+=row['probability']
        for fid,prob in programs.items():reference_n[model['program_count_state'][fid]]+=prob
        row_weights=[row['probability']*reference_n[row['count_state']]/actual_n[row['count_state']] for row in post['rows']]
    objects={oid:programs[obj['furniture_state']] for oid,obj in model['component_registry'].items()}
    category_counts={}
    for oid,prob in objects.items():
        cat=model['component_registry'][oid]['category'];category_counts[cat]=category_counts.get(cat,0.)+prob
    count_pmf={}
    for fid,prob in programs.items():
        n=sum(model['program_counts'][fid]['hidden'].values());count_pmf[str(n)]=count_pmf.get(str(n),0.)+prob
    return dict(program_probabilities=programs,object_presence=objects,category_expected_counts=category_counts,
        hidden_count_probabilities=count_pmf,empty_probability=count_pmf.get('0',0.),
        expected_hidden_instances=sum(objects.values()),
        role_presence={r:sum(weight for row,weight in zip(post['rows'],row_weights) if row['roles'][r]=='1') for r in ROLES})


def sample_worlds(model,seed=3407,count=3):
    if type(count) is not int or count<1:raise ValueError('count must be positive')
    policy=model['population_sampling'];use=policy!='prior';post=posterior(model,use);rng=np.random.default_rng(seed)
    byid={n['id']:n for n in model['nodes']};law=marginals(model,use,policy=='reference');worlds=[]
    weights=np.asarray([r['probability'] for r in post['rows']]);all_f=list(model['furniture_states'])
    for trial in range(count):
        fid=None;candidate_weights=weights
        if policy=='reference':
            fid=all_f[int(rng.integers(len(all_f)))];nid=model['program_count_state'][fid]
            candidate_weights=np.asarray([w if r['count_state']==nid else 0. for r,w in zip(post['rows'],weights)])
            candidate_weights/=candidate_weights.sum()
        row=post['rows'][int(rng.choice(len(post['rows']),p=candidate_weights))]
        wanted={**{'H_'+r:row['roles'][r] for r in ROLES},'L_population':row['scale'],
            'N_population':row['count_state'],'Q_report':row['report_mode']}
        if use:wanted['O_positive_counts']='report'
        if fid is not None:wanted['F']=fid
        noise={};state={}
        for node in model['nodes']:
            ps=node['cpt'][key([state[p] for p in node['parents']])];u=float(rng.random());nid=node['id']
            if nid in wanted:
                i=node['states'].index(wanted[nid]);u=float(sum(ps[:i])+ps[i]*u)
                u=min(u,float(np.nextafter(1.,0.)))
            noise[nid]=u;index=min(int(np.searchsorted(np.cumsum(ps),u,side='right')),len(ps)-1);state[nid]=node['states'][index]
        if any(state[n]!=v for n,v in wanted.items()):raise RuntimeError('Population inverse-CDF abduction failed')
        world=active_world(model,state)
        for obj in world['objects']:obj['presence_probability']=law['object_presence'][obj['id']]
        world.update(noise=noise,seed=seed,trial=trial,functional_state=row['roles'],
            population_count_state=row['count_state'],sampling_distribution=policy,
            probability_scope=model['distribution_scope'],program_probability=law['program_probabilities'][state['F']])
        worlds.append(world)
    return worlds


def intervention_diagnostics(model,world):
    result=inventory.intervention_diagnostics(model,world);rows=[]
    for r in ROLES:
        if world['structural_state']['H_'+r]!='1':continue
        do={'H_'+r:'0'};after=active_world(model,forward(model,world['noise'],do))
        rows.append(dict(do=do,before_program=world['structural_state']['F'],after_program=after['structural_state']['F'],
            after_inventory=[o['id'] for o in after['objects']],same_exogenous_noise=True,model_called=False,
            factual_report_not_reconditioned=True,geometry_not_resolved=True))
    result['functional_queries']=rows;result['scope']='assumed_finite_population_SCM_not_identified_real_room_counterfactual'
    return result


def probability_audit(model):
    prior=marginals(model,False);post=marginals(model,True);reference=marginals(model,True,True)
    return dict(prior=prior,posterior=post,reference=reference,
        sampling_policy=model['population_sampling'],candidate_support=model['candidate_support'],
        probabilities_calibrated=False,parameter_provenance=model['population_parameters']['provenance'],
        causal_effect_validated=False,positive_reports_only=True,
        empty_state_is_explicit=True,duplicate_count_vectors_share_count_mass=True,
        unknown_proposal_density_corrected=False)


def probability_page(model,audit,path):
    rows=[]
    for fid,program in model['furniture_states'].items():
        count=sum(model['program_counts'][fid]['hidden'].values())
        rows.append('<tr><td>'+escape(fid+' '+program['description'])+'</td><td>'+str(count)+'</td>'+''.join(
            '<td>'+f'{audit[k]["program_probabilities"][fid]:.5f}'+'</td>' for k in ('reference','prior','posterior'))+'</tr>')
    notice=('<p style="color:#b33228">所有提案均为空：候选支持域退化，不能据此判断真实世界没有隐藏物体。</p>'
        if model['candidate_support']['all_nonempty_proposals_missing'] else '')
    path.write_text('<!doctype html><meta charset="utf-8"><title>完整库存概率</title>'
        '<style>body{font:16px sans-serif;max-width:1150px;margin:30px auto}td,th{padding:9px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'
        '<h1>完整库存：功能计数先验与局部报告更新</h1>'+notice+'<p>实际采样分支：'+escape(model['population_sampling'])+
        '。参数来源见记录，概率尚未完成校准验证。重复和零新增样本保留，没有数量下限。布局成功集不等于原采样分布。</p>'
        '<p><a href="probability_audit.json">完整概率</a> · <a href="population_parameters.json">参数</a> · '
        '<a href="count_observations.json">观测及类别绑定</a></p>'
        '<table><tr><th>联合库存程序</th><th>新增数</th><th>均匀参考</th><th>计数先验</th><th>计数报告后验</th></tr>'+
        ''.join(rows)+'</table><h2>隐藏数量分布</h2><pre>'+escape(json.dumps({k:audit[k]['hidden_count_probabilities']
            for k in ('reference','prior','posterior')},ensure_ascii=False,indent=2))+'</pre>')
