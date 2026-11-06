"""Same small SCM; export coverage of whole programs, explicitly not IID."""
from copy import deepcopy
import numpy as np

from ...common import write
from ..simple_world import model as base
from ..world_scm import forward

PROTOCOL=base.PROTOCOL
INTERFACE=MODEL_VARIANT='scoped_joint_hypothesis_host_scm_v2'
build_model=base.build_model
load_domain=base.load_domain
semantic_diagnostics=base.semantic_diagnostics
intervention_diagnostics=base.intervention_diagnostics


def prepare_model(model,case,config,out):
    m=base.prepare_model(model,case,config,out)
    m.update(model_variant=MODEL_VARIANT,response_interface=INTERFACE,
        sampling_policy='program_stratified_exports_not_IID_or_posterior_ranking',
        export_design='Each accepted whole program once before repeating programs for geometry variants; bounded output budget',
        proposal_scope='global_hidden_areas_then_host_complete_instantiation; unresolved is not absence')
    if out: write(out/'export_design.json',dict(policy=m['sampling_policy'],design=m['export_design'],
        reference_law=m['distribution_scope'],posterior_ranked=False))
    return m


def active_world(model,state):
    w=base.active_world(model,state); p=model['furniture_states'][state['F']]
    for k in ('scope_ledger','scope_complete_within_hypothesis','unresolved_areas','unmodelled_space'):
        w[k]=deepcopy(p[k])
    return w


def sample_worlds(model,seed=3407,count=2):
    rng=np.random.default_rng(seed); programs=list(model['furniture_states']); worlds=[]
    for index in range(count):
        fid=programs[index % len(programs)]; z=model['furniture_states'][fid]['hypothesis']['id']
        noise={n['id']:float(rng.random()) for n in model['nodes']}; state={}
        # Conditional inverse-CDF noise retains an executable factual state.
        # This is a designed export schedule, not IID sampling or a probability
        # estimate. Same-noise interventions remain model-relative diagnostics.
        for node in model['nodes'][:2]:
            wanted=z if node['id']=='Z' else fid
            ps=node['cpt']['root' if not node['parents'] else state['Z']]
            j=node['states'].index(wanted)
            if ps[j]<=0: raise ValueError('Export design selected an impossible program/hypothesis pair')
            noise[node['id']]=min(float(sum(ps[:j])+ps[j]*rng.random()),float(np.nextafter(1.,0.)))
            state[node['id']]=wanted
        world=active_world(model,forward(model,noise))
        world.update(noise=noise,seed=seed,trial=index,export_design=model['sampling_policy'],
            geometry_variant=index>=len(programs))
        worlds.append(world)
    return worlds
