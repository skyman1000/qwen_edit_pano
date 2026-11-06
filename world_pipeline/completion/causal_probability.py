"""Finite causal Bayesian graphs with explicit conditional mechanisms.

Numerical inference, not LLM Boolean gates. Direction/causal sufficiency must be
justified externally; fitting observational CPTs does not identify a causal DAG.
No default scene graph or fabricated production probabilities are supplied.
Individual counterfactuals require an additional structural-noise model, which
cannot be recovered uniquely from these CPTs and is deliberately not claimed.
"""
from copy import deepcopy
import itertools
import math


class CausalGraph:
    def __init__(self,spec,max_assignments=1000000):
        self.spec=deepcopy(spec)
        self.nodes={};self.order=[]
        for node in self.spec['nodes']:
            name=node['id'];states=node['states']
            if (not isinstance(name,str) or not name or name in self.nodes or
                    not isinstance(states,list) or not states or
                    any(not isinstance(s,str) or not s for s in states) or len(set(states))!=len(states)):
                raise ValueError('Need distinct node IDs and nonempty named states')
            self.nodes[name]=node
        pending=set(self.nodes)
        while pending:
            ready=sorted(n for n in pending if set(self.nodes[n]['parents'])<=set(self.order))
            if not ready:raise ValueError('Unknown parents or directed cycle')
            self.order.extend(ready);pending.difference_update(ready)
        total=math.prod(len(self.nodes[n]['states']) for n in self.order)
        if not 1<=total<=max_assignments:
            raise ValueError('Exact enumeration budget exceeded; use a scalable inference backend')
        self.tables={}
        for n in self.order:
            node=self.nodes[n];parents=node['parents']
            if len(set(parents))!=len(parents):raise ValueError('Duplicate parents')
            expected=set(itertools.product(*(self.nodes[p]['states'] for p in parents)))
            table={}
            for row in node['cpt']:
                key=tuple(row['when']);probs=row['probabilities']
                if key not in expected or key in table or set(probs)!=set(node['states']):
                    raise ValueError('Missing/duplicate/invalid conditional probability row: '+n)
                if (any(type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1
                        for p in probs.values()) or not math.isclose(sum(probs.values()),1.,abs_tol=1e-9)):
                    raise ValueError('CPT probabilities must be finite and normalized: '+n)
                table[key]=probs
            if set(table)!=expected:raise ValueError('Incomplete CPT: '+n)
            self.tables[n]=table

    def _assignments(self,evidence,interventions):
        for assignments in (evidence,interventions):
            for n,v in assignments.items():
                if n not in self.nodes or v not in self.nodes[n]['states']:
                    raise ValueError('Unknown query node/state: '+str(n))
        # Conditioning after intervention is a different, explicit operation.
        if any(n in evidence and evidence[n]!=v for n,v in interventions.items()):
            raise ValueError('Evidence contradicts intervention')
        states=[self.nodes[n]['states'] for n in self.order]
        for values in itertools.product(*states):
            assignment=dict(zip(self.order,values))
            if any(assignment[n]!=v for n,v in {**evidence,**interventions}.items()):continue
            mass=1.
            for n in self.order:
                if n in interventions:continue  # do() replaces the mechanism, cutting incoming edges.
                key=tuple(assignment[p] for p in self.nodes[n]['parents'])
                mass*=self.tables[n][key][assignment[n]]
            if mass:yield assignment,mass

    def query(self,target,evidence=None,interventions=None):
        if target not in self.nodes:raise ValueError('Unknown target')
        probs={s:0. for s in self.nodes[target]['states']};total=0.
        for assignment,mass in self._assignments(evidence or {},interventions or {}):
            probs[assignment[target]]+=mass;total+=mass
        if total<=0:raise ValueError('Evidence has zero probability under this declared model')
        return dict(probabilities={s:p/total for s,p in probs.items()},
                    evidence_probability_in_regime=total,
                    query_kind='conditional_interventional' if interventions else 'observational_posterior',
                    identified_from_data=False,
                    assumption='Externally specified causal DAG and adequate confounder coverage',
                    parameter_source=self.spec.get('parameter_source','unspecified'))

    def counterfactual(self,*args,**kwargs):
        raise NotImplementedError('Individual counterfactuals require structural equations and '
                                  'cross-world noise coupling; CPTs alone do not specify these')


def fit_cpts(nodes,records,alpha=1.):
    """Fit parameters of a GIVEN graph to complete training annotations only.

    Missing context/activity labels are an error, not invented by the learner.
    Dirichlet smoothing is a declared statistical prior, not causal evidence.
    Annotations/splits/source hashes must be bound by the calling data pipeline.
    """
    if not math.isfinite(alpha) or alpha<=0:raise ValueError('Need positive smoothing prior')
    records=list(records)
    if not records:raise ValueError('Need training records')
    result=deepcopy(nodes)
    by_id={n['id']:n for n in result}
    for n in result:
        if set(n['parents'])-by_id.keys():raise ValueError('Unknown parents')
        combinations=list(itertools.product(*(by_id[p]['states'] for p in n['parents'])))
        counts={key:{s:alpha for s in n['states']} for key in combinations}
        support={key:0 for key in combinations}
        for row in records:
            if any(p not in row for p in [n['id']]+n['parents']):
                raise ValueError('Missing training annotation; cannot fit latent states as if observed')
            key=tuple(row[p] for p in n['parents']);value=row[n['id']]
            if key not in counts or value not in n['states']:raise ValueError('Invalid training state')
            counts[key][value]+=1;support[key]+=1
        n['cpt']=[dict(when=list(key),probabilities={s:v/sum(values.values()) for s,v in values.items()},
                       training_support=support[key]) for key,values in counts.items()]
    spec=dict(nodes=result,parameter_source='training_counts_on_declared_graph',
              smoothing_alpha=alpha,identified_from_data=False)
    CausalGraph(spec)
    return spec
