"""Explicit stages for a completely fresh API/Qwen RGB-to-world comparison."""
import argparse
from datetime import datetime
from pathlib import Path
import os
import sys

from PIL import Image

from ..common import ROOT,new_output,read,sha,write
from .fresh_model import PROTOCOL
from .world_generation import recipe
from .qwen_upstream import DEFAULT_MODEL
from .box_observer import DEFAULT_CHECKPOINT,DEFAULT_REPO,intrinsics

DEFAULT_IMAGE=ROOT/'qwen_edit_pano/data/box_world_val6_v1/X7HyMhZNoso_d0468b580cd146778bae5e486bf3e50e/local.png'


def sources():
    directory=Path(__file__).parent
    files=list(directory.glob('fresh_*.py'))
    files.extend(directory/(name+'.py') for name in ('box_observer','world_generation','world_api','world_scm',
        'world_scm_runner','world_scm_spec','scene_program','scene_program_world','scene_program_geometry',
        'scm_joint_geometry','scm_typed_layout','scene_completion','off_fov_completion','conditional_world_geometry',
        'frozen_causal_layout','semantic_layout_visual','batch_world_review','category_bridge','geometry','qwen_upstream',
        'qwen_upstream_core','box_layout'))
    files.extend([directory.parent/'common.py',ROOT/'qwen_edit_pano/scripts/fresh_compare.sh'])
    return {str(p.resolve()):sha(p) for p in files}


def preflight(image,hfov,local_model):
    if not image.is_file():raise ValueError('Missing local RGB: '+str(image))
    with Image.open(image) as im:size=im.size
    intrinsics(*size,hfov)
    if not (local_model/'config.json').is_file() or not list(local_model.glob('*.safetensors')):
        raise ValueError('Missing local Qwen config or shards: '+str(local_model))
    if not DEFAULT_CHECKPOINT.is_file():raise ValueError('Missing WildDet3D checkpoint: '+str(DEFAULT_CHECKPOINT))
    import scipy,transformers,torch
    from .category_bridge import official_labels
    official_labels()
    return dict(status='preflight_passed',image=str(image.resolve()),size=size,hfov=hfov,
        qwen=str(local_model.resolve()),wilddet_checkpoint=str(DEFAULT_CHECKPOINT),
        model_loaded=False,gpu_task_submitted=False,api_called=False,
        versions=dict(scipy=scipy.__version__,transformers=transformers.__version__,torch=torch.__version__))


def initialize(output,image,hfov,local_model,api_model,max_tokens=8192):
    preflight(image,hfov,local_model)
    out=new_output(output)
    with Image.open(image) as im:rgb=im.convert('RGB');rgb.save(out/'input.png');size=list(rgb.size)
    config=dict(protocol=PROTOCOL,id=image.parent.name,image_source=str(image.resolve()),image_source_sha256=sha(image),
        image_sha256=sha(out/'input.png'),image_size=size,hfov=hfov,
        local_model=str(local_model.resolve()),api_model=api_model,
        api_base=os.environ.get('SILICONFLOW_API_BASE','https://api.siliconflow.cn/v1'),
        generation=recipe(),max_tokens=max_tokens,seed=3407,
        budgets=dict(configurations=3,groups=4,max_new=24),
        layout_search=dict(region_keep=3,beam_width=4,rounds=8,relation_weight=1.),
        independent_observers=True,old_domain_reused=False,old_semantic_model_reused=False,
        shared_model_structure=True,shared_numerical_CPT=False,scene_gt_used=False,
        code_sha256=sources(),created=datetime.now().isoformat())
    write(out/'experiment.json',config)
    for backend in ('api','qwen'):(out/backend).mkdir()
    return out


def check(root,strict_code=True):
    p=root/'experiment.json'
    if not p.is_file():raise ValueError('Experiment not initialized: '+str(p)+'. Use init/run, not a guessed resume folder.')
    config=read(p)
    if config['protocol']!=PROTOCOL:raise ValueError('Wrong experiment protocol')
    changed=[path for path,digest in config['code_sha256'].items() if not Path(path).is_file() or sha(path)!=digest]
    if strict_code and changed:raise ValueError('Source changed since init: '+', '.join(changed))
    config['source_changes_since_init']=changed
    if sha(root/'input.png')!=config['image_sha256']:raise ValueError('Experiment RGB changed')
    return config


def provider(config,backend):
    if backend=='api':
        from .world_api import SiliconFlowJSON
        return SiliconFlowJSON(model=config['api_model'],base=config['api_base'],generation_profile='matched_v1',
            thinking='off',max_calls=64,timeout=600)
    from .world_generation import MatchedLocalJSON
    return MatchedLocalJSON(config['local_model'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preflight','init','observer2d','world','review'])
    p.add_argument('--root',type=Path);p.add_argument('--backend',choices=['api','qwen'])
    p.add_argument('--image',type=Path,default=Path(os.environ.get('CAUPANO_IMAGE',DEFAULT_IMAGE)))
    p.add_argument('--hfov',type=float,default=float(os.environ.get('CAUPANO_HFOV','90')))
    p.add_argument('--local-model',type=Path,default=Path(os.environ.get('CAUPANO_QWEN_MODEL',DEFAULT_MODEL)))
    p.add_argument('--api-model',default=os.environ.get('SILICONFLOW_MODEL','deepseek-ai/DeepSeek-V4.1-Flash'))
    p.add_argument('--max-tokens',type=int,default=8192)
    a=p.parse_args()
    try:
        if a.action=='preflight':print(preflight(a.image,a.hfov,a.local_model));return
        if a.action=='init':
            output=a.root or ROOT/'qwen_edit_pano/outputs'/('fresh_rgb_compare_'+datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}')
            print(initialize(output,a.image,a.hfov,a.local_model,a.api_model,a.max_tokens));return
        if a.root is None:p.error('--root required')
        root=a.root.resolve();config=check(root,strict_code=a.action!='review')
        if a.action=='review':
            from .fresh_review import review
            print(review(root));return
        if a.backend is None:p.error('--backend required')
        dest=root/a.backend/('observer2d' if a.action=='observer2d' else 'completion')
        if dest.exists():raise ValueError('Stage output already exists; preserved without overwrite: '+str(dest))
        # Check prerequisites before loading 8B parameters or contacting a provider.
        if a.action=='world':
            from .fresh_observer import load_case
            load_case(root,a.backend)
        q=provider(config,a.backend)
        if a.action=='observer2d':
            from .fresh_observer import observe
            summary=observe(q,root,a.backend,config['max_tokens'])
        else:
            from .fresh_world import run
            summary=run(q,root,a.backend,config['max_tokens'])
        print(summary,flush=True)
        if summary.get('status') in ('failed','no_world_exported') or summary.get('failures'):sys.exit(1)
    except (ValueError,FileNotFoundError,FileExistsError) as exc:
        p.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
