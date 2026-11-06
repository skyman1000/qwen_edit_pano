"""New experiment entry; original Observer implementations and frozen models."""
import argparse
import hashlib
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys

from PIL import Image

from ..common import ROOT,new_output,read,write,sha,validate_graph
from .assembly_delta import PROTOCOL,INTERFACE,MODEL_VARIANT
from .assembly_model import LEGACY_PROTOCOL
from .fresh_pipeline import DEFAULT_IMAGE,DEFAULT_MODEL,preflight,provider,sources
from .world_generation import recipe


def code_hashes():
    files={**sources(),**{str(p.resolve()):sha(p) for p in Path(__file__).parent.glob('assembly_*.py')}}
    for name in ('assembly_compare.sh','functional_completion.sh','population_completion.sh','sdscene_compare.sh','modular_completion.sh','activation_completion.sh','realization_completion.sh','mechanism_completion.sh','panorama_completion.sh','assembly_set_completion.sh','holistic_completion.sh','compositional_completion.sh','organized_completion.sh','organized_range_repair.sh'):
        p=ROOT/'qwen_edit_pano/scripts'/name;files[str(p)]=sha(p)
    return files


def model_info(name):
    if name=='needs':from . import assembly_needs as model
    elif name=='functional':from . import assembly_functional as model
    elif name=='population':from . import assembly_population as model
    elif name=='sdscene':from . import assembly_sdscene as model
    elif name=='modular':from . import assembly_modular as model
    elif name=='activation':from . import assembly_activation as model
    elif name=='realization':from . import assembly_realization as model
    elif name=='mechanism':from . import assembly_mechanism as model
    elif name=='panorama':from . import assembly_panorama as model
    elif name=='assemblyset':from . import assembly_set as model
    elif name=='holistic':from . import assembly_holistic as model
    elif name=='compositional':from . import assembly_compositional as model
    elif name=='baseline':from . import assembly_baseline as model
    elif name=='joint':from . import assembly_delta as model
    else:raise ValueError('Unknown completion model: '+str(name))
    return dict(completion_model=name,model_variant=model.MODEL_VARIANT,response_interface=model.INTERFACE)


def initialize(output,image,hfov,local_model,api_model,seed=3407,observations_from=None,shared_observer='qwen',completion_model='joint',worlds=3):
    if type(worlds) is not int or worlds<1:raise ValueError('worlds must be positive')
    preflight(image,hfov,local_model)
    frozen=None
    if observations_from:
        source=observations_from.resolve();config=read(source/'experiment.json')
        if config['hfov']!=hfov:raise ValueError('Frozen-observation HFOV differs')
        with Image.open(image) as im:a=im.convert('RGB')
        with Image.open(source/'input.png') as im:b=im.convert('RGB')
        if a.size!=b.size or a.tobytes()!=b.tobytes():raise ValueError('Frozen-observation image differs')
        for folder,file in [('observer2d','G_2d.json'),('observer2d','grounding.json'),('observer3d','G_obs.json')]:
            if not (source/shared_observer/folder/file).is_file():raise ValueError('Missing completed shared observation: '+str(source/shared_observer/folder/file))
        state=validate_graph(read(source/shared_observer/'observer3d/G_obs.json'))
        if state.get('stage')!='prediction' or state.get('scope')!='observed':raise ValueError('Shared observations must be predicted G_obs, not scene GT')
        frozen=dict(source=str(source),backend=shared_observer,state_sha256=sha(source/shared_observer/'observer3d/G_obs.json'),
            input_sha256={name:sha(source/shared_observer/name) for name in ('observer2d/G_2d.json','observer2d/grounding.json','observer3d/G_obs.json')})
    out=new_output(output)
    with Image.open(image) as im:rgb=im.convert('RGB');rgb.save(out/'input.png');size=list(rgb.size)
    config=dict(protocol=PROTOCOL,response_interface=INTERFACE,model_variant=MODEL_VARIANT,id=image.parent.name,image_source=str(image.resolve()),image_sha256=sha(out/'input.png'),
        image_size=size,hfov=hfov,local_model=str(local_model.resolve()),api_model=api_model,
        api_base=os.environ.get('SILICONFLOW_API_BASE','https://api.siliconflow.cn/v1'),generation=recipe(),max_tokens=8192,
        seed=seed,worlds=worlds,experiment_kind='fixed_observation_completion_comparison' if frozen else 'independent_RGB_end_to_end_comparison',
        frozen_observations=frozen,scene_gt_used=False,adapter_connected=False,
        sampling_law='explicit_uniform_reference; not_calibrated_posterior',code_sha256=code_hashes(),created=datetime.now().isoformat())
    config.update(model_info(completion_model));write(out/'experiment.json',config)
    for backend in ('api','qwen'):
        (out/backend).mkdir()
        if frozen:
            for name in ('observer2d','observer3d'):shutil.copytree(Path(frozen['source'])/shared_observer/name,out/backend/name)
            write(out/backend/'observation_source.json',frozen)
    return out


def check(root,strict_code=True):
    if not (root/'experiment.json').is_file():raise ValueError('Experiment not initialized: '+str(root))
    config=read(root/'experiment.json')
    if config['protocol'] not in (PROTOCOL,LEGACY_PROTOCOL):raise ValueError('Expected an assembly SCM experiment')
    if strict_code and config['protocol']!=PROTOCOL:raise ValueError('Legacy completion is preserved. Use retry/retry2 to create a NEW experiment with current mechanisms.')
    changed=[p for p,h in config['code_sha256'].items() if not Path(p).is_file() or sha(p)!=h]
    if strict_code and changed:raise ValueError('Sources changed after experiment initialization: '+str(changed))
    config['source_changes_since_init']=changed
    if sha(root/'input.png')!=config['image_sha256']:raise ValueError('Experiment image changed')
    if config.get('frozen_observations'):
        hashes=config['frozen_observations']['input_sha256']
        if any(sha(root/b/name)!=digest for b in ('api','qwen') for name,digest in hashes.items()):
            raise ValueError('Shared observed states changed')
    return config


def restart(output,source,completion_model=None,seed=None,worlds=None):
    """Fork both completed observers; regenerate ALL semantic mechanisms/layouts.

    The two original observation branches and experimental design stay intact.
    This deliberately does not replay old mechanisms after a prompt/interface
    correction. No model inference occurs here; every reused file is hashed.
    """
    from .fresh_observer import load_case
    source=source.resolve();config=check(source,strict_code=False);manifests={}
    selected=completion_model or config.get('completion_model','joint')
    config.pop('semantic_mechanisms_from',None)
    config.pop('candidate_programs_from',None)
    selected_info=model_info(selected)
    if worlds is not None and (type(worlds) is not int or worlds<1):raise ValueError('worlds must be positive')
    for backend in ('api','qwen'):
        load_case(source,backend)
        state=validate_graph(read(source/backend/'observer3d/G_obs.json'))
        if state.get('stage')!='prediction' or state.get('scope')!='observed':
            raise ValueError('Restart requires predicted observed states, not GT: '+backend)
        for name in ('observer2d/G_2d.json','observer2d/grounding.json','observer3d/G_obs.json'):
            if not (source/backend/name).is_file():raise ValueError('Missing saved observation: '+str(source/backend/name))
        manifests[backend]={str(p.relative_to(source/backend)):sha(p)
            for folder in ('observer2d','observer3d') for p in sorted((source/backend/folder).rglob('*')) if p.is_file()}
    preflight(source/'input.png',config['hfov'],Path(config['local_model']))
    out=new_output(output);shutil.copyfile(source/'input.png',out/'input.png')
    reused=dict(source=str(source),source_experiment_sha256=sha(source/'experiment.json'),
        observer_file_sha256=manifests,semantic_mechanisms_reused=False,layouts_reused=False,
        source_changes_at_restart=config.pop('source_changes_since_init'))
    config.update(protocol=PROTOCOL,response_interface=INTERFACE,model_variant=MODEL_VARIANT,observation_reuse=reused,code_sha256=code_hashes(),created=datetime.now().isoformat())
    config.update(selected_info)
    if seed is not None:config['seed']=seed
    if worlds is not None:config['worlds']=worlds
    for backend in ('api','qwen'):
        (out/backend).mkdir()
        for folder in ('observer2d','observer3d'):shutil.copytree(source/backend/folder,out/backend/folder)
        for name,digest in manifests[backend].items():
            if sha(out/backend/name)!=digest:raise ValueError('Reused observation changed during copy: '+name)
        write(out/backend/'observation_source.json',dict(source=str(source/backend),file_sha256=manifests[backend]))
    write(out/'experiment.json',config);check(out);return out


class SeededProvider:
    """Keep existing JSON repair policy, record experiment-specific local seeds."""
    def __init__(self,inner,seed):self.inner=inner;self.seed=seed
    def __getattr__(self,name):return getattr(self.inner,name)
    def ask(self,image,prompt,path,**kwargs):
        kwargs['seed']=self.seed+kwargs.get('seed',3407)-3407
        return self.inner.ask(image,prompt,path,**kwargs)


def stage_seed(seed,name):
    return int(hashlib.sha256((str(seed)+':'+name).encode()).hexdigest()[:8],16)%(2**31-1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['preflight','init','restart','observer2d','observer3d','world','review'])
    p.add_argument('--root',type=Path);p.add_argument('--backend',choices=['api','qwen'])
    p.add_argument('--source-root',type=Path)
    p.add_argument('--image',type=Path,default=Path(os.environ.get('CAUPANO_IMAGE',DEFAULT_IMAGE)))
    p.add_argument('--hfov',type=float,default=float(os.environ.get('CAUPANO_HFOV','90')))
    p.add_argument('--local-model',type=Path,default=Path(os.environ.get('CAUPANO_QWEN_MODEL',DEFAULT_MODEL)))
    p.add_argument('--api-model',default=os.environ.get('SILICONFLOW_MODEL','deepseek-ai/DeepSeek-V4.1-Flash'))
    p.add_argument('--seed',type=int);p.add_argument('--worlds',type=int)
    p.add_argument('--completion-model',choices=['joint','baseline','needs','functional','population','sdscene','modular','activation','realization','mechanism','panorama','assemblyset','holistic','compositional'])
    p.add_argument('--functional-parameters',type=Path,help='Complete explicit numerical parameter JSON; snapshotted into new experiment')
    p.add_argument('--functional-sampling',choices=['prior','posterior'],default='posterior')
    p.add_argument('--population-parameters',type=Path)
    p.add_argument('--population-sampling',choices=['posterior','prior','reference','no_function'],default='posterior')
    p.add_argument('--candidate-programs-from',type=Path,help='Revalidate saved joint/baseline raw proposals for population model')
    p.add_argument('--semantic-mechanisms-from',type=Path,help='Freeze functional candidate grammar for probability ablation')
    p.add_argument('--observations-from',type=Path);p.add_argument('--other-root',type=Path)
    p.add_argument('--shared-observer',choices=['api','qwen'],default='qwen');a=p.parse_args()
    try:
        if a.semantic_mechanisms_from and a.candidate_programs_from:raise ValueError('Choose one candidate reuse source')
        if a.action=='preflight':print(preflight(a.image,a.hfov,a.local_model));return
        if a.action in ('init','restart'):
            out=a.root or ROOT/'qwen_edit_pano/outputs'/('assembly_rgb_compare_'+datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}')
            if a.action=='restart':
                if a.source_root is None:p.error('--source-root required for restart')
                result=restart(out,a.source_root,a.completion_model,a.seed,a.worlds)
            else:result=initialize(out,a.image,a.hfov,a.local_model,a.api_model,a.seed if a.seed is not None else 3407,
                a.observations_from,a.shared_observer,a.completion_model or 'joint',a.worlds if a.worlds is not None else 3)
            if a.functional_parameters or a.population_parameters or a.completion_model in ('functional','population') or a.semantic_mechanisms_from or a.candidate_programs_from:
                updated=read(result/'experiment.json')
                if updated['completion_model']=='functional':
                    from .assembly_functional import parameters
                    if a.population_parameters:raise ValueError('Population parameters require population model')
                    if a.functional_parameters:updated['functional_parameters']=parameters(read(a.functional_parameters))
                    updated['functional_sampling']=a.functional_sampling
                elif updated['completion_model']=='population':
                    from .assembly_population import parameters
                    if a.functional_parameters:raise ValueError('Use --population-parameters for population model')
                    if a.population_parameters:updated['population_parameters']=parameters(read(a.population_parameters))
                    updated['population_sampling']=a.population_sampling
                    if a.candidate_programs_from:updated['candidate_programs_from']=str(a.candidate_programs_from.resolve())
                else:raise ValueError('Probability options require functional or population model')
                if a.candidate_programs_from and updated['completion_model']!='population':raise ValueError('Joint candidate reuse requires population model')
                if a.semantic_mechanisms_from:updated['semantic_mechanisms_from']=str(a.semantic_mechanisms_from.resolve())
                write(result/'experiment.json',updated)
            print(result)
            return
        if a.root is None:p.error('--root required')
        root=a.root.resolve();config=check(root,strict_code=a.action!='review')
        if a.action=='review':
            from .assembly_review import review,suite
            print(suite(root,a.other_root.resolve()) if a.other_root else review(root));return
        if a.backend is None:p.error('--backend required')
        name={'observer2d':'observer2d','observer3d':'observer3d','world':'completion'}[a.action]
        if (root/a.backend/name).exists():raise ValueError('Stage already exists, original preserved: '+str(root/a.backend/name))
        if a.action=='observer3d':
            from .fresh_wilddet import run,DEFAULT_REPO,DEFAULT_CHECKPOINT
            summary=run(root,a.backend,DEFAULT_REPO,DEFAULT_CHECKPOINT,check_experiment=check)
        else:
            if a.action=='world':
                from .fresh_observer import load_case
                load_case(root,a.backend)
            q=SeededProvider(provider(config,a.backend),config['seed'])
            if a.action=='observer2d':
                from .fresh_observer import observe
                summary=observe(q,root,a.backend,config['max_tokens'])
            else:
                from .assembly_world import run
                summary=run(q,root,a.backend)
        print(summary,flush=True)
        # A completed batch may contain blocked hypotheses as well as usable
        # numeric candidates. Keep partial status/diagnostics visible; do not
        # report a Slurm process failure merely because an alternative failed.
        if summary.get('status') in ('failed','no_world_exported','empty_candidate_domain_no_completion'):sys.exit(1)
    except (ValueError,FileNotFoundError,FileExistsError) as exc:p.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
