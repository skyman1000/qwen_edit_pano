"""Fixed living-room GT interventions for the current learned-ID World branch.

CPU --preflight validates inputs/weights. GPU generation is explicitly user-run.
Original pack, GT, checkpoints and old Fourier probes are never modified.
"""
import argparse
import copy
import json
import time
from pathlib import Path

from ..build_polished_world import ROOT, read, atomic_json, validate_objects
from ..common import new_output
from ..object_world.data import load_pack, tensorize, target_image

SID = 'X7HyMhZNoso_d0468b580cd146778bae5e486bf3e50e'
OLD_PLAN = ROOT/'qwen_edit_pano/data/fourier_living_state_probe_v1/plan.json'
DEFAULT_CHECKPOINT = ROOT/'qwen_edit_pano/outputs/object_world_id_A_dual_full_v1/run-1792820440973994739/checkpoint-step010000'
VERSION = 'spatial_world_living_state_probe_v1'
LEGACY_CASES = ['baseline','correct','shuffled','remove_tv','move_tv','half_tv','mirror_tv','add_plant']
EXPANDED_CASES = LEGACY_CASES + ['picture_tv','remove_hidden_chair','hidden_chair_to_plant',
    'hidden_chair_to_cabinet','hidden_chair_to_stool','remove_dining_table','dining_table_to_bed',
    'dining_table_to_sofa','add_floor_plant','add_floor_chair','add_floor_cabinet']


def legacy_variants(row):
    """Keep the old TV interventions, plus a small synthetic plant addition."""
    byid = {o['instance_id']:o for o in row['objects']}
    tv = byid['360']
    if tv['category_id'] != 22 or not tv['center_valid'] or not tv['size_valid']:
        raise ValueError('Expected the original TV360 in the fixed validation scene')
    cases = [('correct','Unmodified validation GT',copy.deepcopy(row))]
    for name,description in [
        ('remove_tv','Remove only TV360'),
        ('move_tv','TV360 center x minus 1 metre; other fields fixed'),
        ('half_tv','TV360 AABB full lengths x/y halved; center/depth fixed'),
        ('mirror_tv','TV360 category 22 -> mirror21; geometry fixed'),
        ('add_plant','Add a 0.35 x 0.60 x 0.35 m plant above table354/tray361; conflicts with unchanged visible Local')]:
        edited = copy.deepcopy(row)
        obj = next(o for o in edited['objects'] if o['instance_id'] == '360')
        if name == 'remove_tv':
            edited['objects'] = [o for o in edited['objects'] if o['instance_id'] != '360']
        elif name == 'move_tv': obj['center_local_m'][0] -= 1.
        elif name == 'half_tv':
            obj['size_aabb_local_m'][0] *= .5; obj['size_aabb_local_m'][1] *= .5
        elif name == 'mirror_tv': obj['category_id'] = 21
        else:
            table,tray = byid['354'],byid['361']
            if table['category_id'] != 5 or not all(o['center_valid'] and o['size_valid'] for o in (table,tray)):
                raise ValueError('Addition requires the original table354/tray361 boxes')
            top = max(o['center_local_m'][1]+o['size_aabb_local_m'][1]/2 for o in (table,tray))
            iid = str(max(int(o['instance_id']) for o in edited['objects'])+1)
            edited['objects'].append(dict(instance_id=iid,category_id=14,
                center_local_m=[table['center_local_m'][0],top+.02+.30,table['center_local_m'][2]],
                size_aabb_local_m=[.35,.60,.35],center_valid=True,size_valid=True))
        cases.append((name,description,edited))
    return cases


def floor_addition(row,category):
    """Shared, visually inspected floor area; ground height is an AABB proxy."""
    table=next(o for o in row['objects'] if o['instance_id']=='335')
    if table['category_id']!=5 or not table['center_valid'] or not table['size_valid']:
        raise ValueError('Expected original dining table335')
    # Behind/left of the camera, in front of the TV wall. Same box across all
    # categories isolates the category intervention. No mesh collision claim.
    floor=table['center_local_m'][1]-table['size_aabb_local_m'][1]/2
    return dict(instance_id=str(max(int(o['instance_id']) for o in row['objects'])+1),
        category_id=category,center_local_m=[-1.8,floor+.55,-.20],
        size_aabb_local_m=[.85,1.10,.75],center_valid=True,size_valid=True)


def variants(row,suite='expanded'):
    cases=legacy_variants(row)
    if suite=='legacy':return cases
    if suite!='expanded':raise ValueError('Unknown probe suite')
    byid={o['instance_id']:o for o in row['objects']}
    if byid['328']['category_id']!=3 or byid['335']['category_id']!=5:
        raise ValueError('Expected original hidden dining chair328 and table335')
    for iid in ('328','335'):
        if not byid[iid]['center_valid'] or not byid[iid]['size_valid'] or byid[iid]['center_local_m'][2]>=0:
            raise ValueError('Expanded substitutions must use valid objects behind the Local camera')
    edits=[
        ('picture_tv','TV360 -> picture6; same center and full AABB','360',6),
        ('remove_hidden_chair','Remove only hidden dining chair328; table/other chairs fixed','328',None),
        ('hidden_chair_to_plant','Hidden chair328 -> plant14; geometry fixed','328',14),
        ('hidden_chair_to_cabinet','Hidden chair328 -> cabinet7; geometry fixed','328',7),
        ('hidden_chair_to_stool','Hidden chair328 -> stool19; geometry fixed','328',19),
        ('remove_dining_table','Remove dining table335; all chairs fixed (stress test)','335',None),
        ('dining_table_to_bed','Dining table335 -> bed11; same large box (scene-change stress test)','335',11),
        ('dining_table_to_sofa','Dining table335 -> sofa10; same large box','335',10)]
    for name,description,iid,category in edits:
        edited=copy.deepcopy(row)
        if category is None:edited['objects']=[o for o in edited['objects'] if o['instance_id']!=iid]
        else:next(o for o in edited['objects'] if o['instance_id']==iid)['category_id']=category
        cases.append((name,description,edited))
    for name,category,label in [('add_floor_plant',14,'plant'),('add_floor_chair',3,'chair'),('add_floor_cabinet',7,'cabinet')]:
        edited=copy.deepcopy(row);edited['objects'].append(floor_addition(row,category))
        cases.append((name,f'Add {label}{category} at the shared floor box; retain every GT object',edited))
    return cases


def changes(original,edited):
    a = {o['instance_id']:o for o in original['objects']}
    b = {o['instance_id']:o for o in edited['objects']}
    return dict(removed=[a[i] for i in a.keys()-b.keys()],added=[b[i] for i in b.keys()-a.keys()],
        modified=[dict(instance_id=i,before=a[i],after=b[i]) for i in sorted(a.keys()&b.keys()) if a[i] != b[i]])


def checkpoint_config(path,plan=None):
    from .runtime import VERSION as MODEL_VERSION,LEGACY_VERSION
    config,complete = read(path/'config.json'),read(path/'COMPLETE.json')
    if config['version'] not in (MODEL_VERSION,LEGACY_VERSION) or complete['version'] != config['version']:
        raise ValueError('Use a current learned-ID World checkpoint, not legacy Fourier/semantic weights')
    if not (path/'world.safetensors').is_file(): raise FileNotFoundError(path/'world.safetensors')
    if plan is not None and config['pack_plan'] != plan:
        raise ValueError('Checkpoint and current training pack disagree')
    return config,complete['step']


def write_views(output,original,names):
    from ..id_binding_world.state_probe_review import write_review
    write_review(output,original,names)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reference-checkpoint',type=Path)
    p.add_argument('--no-reference',action='store_true',help='Omit the matched step1000 image')
    p.add_argument('--output',type=Path)
    p.add_argument('--suite',choices=['expanded','legacy'],default='expanded')
    p.add_argument('--steps',type=int,default=28)
    p.add_argument('--cfg',type=float,default=4.)
    p.add_argument('--seed',type=int,help='Default: the exact seed of the old single-image probe')
    p.add_argument('--cases',nargs='+',choices=EXPANDED_CASES,help='Optional subset; default runs the selected suite')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--review-only',action='store_true',help='Rebuild visualizations on CPU after all selected images exist')
    args = p.parse_args()
    if args.output is None:args.output=ROOT/'qwen_edit_pano/outputs/object_world_spatial_probe_v1'/(args.checkpoint.parent.name+'-'+args.checkpoint.parent.parent.name+'-'+args.checkpoint.name)
    args.cases=args.cases or ['baseline','correct','move_tv','mirror_tv','remove_tv']
    if args.suite=='legacy' and any(c not in LEGACY_CASES for c in args.cases):p.error('Expanded cases require --suite expanded')
    if args.steps < 1 or args.cfg < 1 or len(set(args.cases)) != len(args.cases):p.error('Invalid generation settings')
    config,step = checkpoint_config(args.checkpoint)
    if args.reference_checkpoint is None:args.reference_checkpoint=Path(config.get('initial_checkpoint',DEFAULT_CHECKPOINT))
    plan,records = load_pack(config['pack'])
    if config['pack_plan'] != plan:raise ValueError('Pack changed since training')
    original = next(r for r in records if r['id'] == SID)
    if original['split'] != 'val':raise ValueError('The fixed scene must remain held out')
    old = read(OLD_PLAN)
    if old['id'] != SID:raise ValueError('Old probe ID differs')
    seed = old['expected_sample_seed'] if args.seed is None else args.seed
    donor = next(r for r in records if r['id'] == old['shuffled_id'])
    if donor['split'] != 'val' or donor['scene_id'] == original['scene_id']:raise ValueError('Expected an independent held-out donor')
    available = {name:(description,row) for name,description,row in variants(original,args.suite)}
    available['shuffled'] = ('Foreign GT from the SAME donor as the old probe',donor)
    available['baseline'] = ('Panorama-finetuned Qwen epoch015 scale0.5; World disabled',None)
    cases = []
    reference = None
    if not args.no_reference:
        reference,ref_step = checkpoint_config(args.reference_checkpoint,plan)
        if any(reference[k] != config[k] for k in ('blocks','width','pano_lora_scale','category_encoding')):
            raise ValueError('Reference architecture or Panorama scale differs')
        cases.append((f'correct_step{ref_step:06d}',f'Matched original GT at reference step {ref_step}',original,args.reference_checkpoint))
    for name in args.cases:cases.append((name,*available[name],args.checkpoint))
    for _,_,row,_ in cases:
        if row is not None:
            validate_objects(row['objects'],plan['names']);tensorize(row,plan['max_objects'])
    # Validate World weights on CPU even for preflight, without loading Qwen.
    import torch
    from safetensors.torch import load_file
    from .runtime import WorldBranch,load_world,VERSION as SPATIAL_VERSION
    torch.set_num_threads(4)
    snapshot=Path(plan['base_snapshot'])
    hidden_config=read(snapshot/'transformer/config.json')
    hidden=hidden_config['num_attention_heads']*hidden_config['attention_head_dim']
    cpu_world=WorldBranch(hidden,config['blocks'],config['width'],plan['height'],plan['padding_columns']).eval()
    for path in {path for _,_,_,path in cases}:
        state=load_file(str(path/'world.safetensors'),device='cpu')
        if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite checkpoint')
        load_world(cpu_world,state,read(path/'config.json'))
    del cpu_world,state
    manifest = dict(version=VERSION,suite=args.suite,checkpoint=str(args.checkpoint.resolve()),checkpoint_step=step,
        reference_checkpoint=str(args.reference_checkpoint.resolve()) if reference else None,
        id=SID,split='val',seed=seed,steps=args.steps,cfg=args.cfg,height=plan['height'],width=2*plan['height'],
        base_checkpoint=plan['base_checkpoint'],pano_lora_scale=config['pano_lora_scale'],
        category_encoding=config['category_encoding'],category_names=plan['names'],world_cfg='same state for positive and negative branches',
        world_version=config['version'],spatial_mode=config.get('spatial_mode','legacy'),routing_policy=config.get('routing_policy'),
        cases=[dict(name=name,description=description,checkpoint=str(path.resolve()),
                    object_count=len(row['objects']) if row else 0,
                    changes=changes(original,row) if row is not None and row is not donor else None)
               for name,description,row,path in cases],
        note='GT target/region masks never enter generation. Synthetic edits have no edited GT panorama; addition uses camera-AABB approximate support, not a physical layout guarantee. Token omission is not an explicit image deletion command. No hashes computed.')
    # Store only fields consumed by the encoder. No stale training region cache
    # or reused visibility metadata is represented as supervision for edited GT.
    states = {name:dict(id=row['id'],frame=row['frame'],objects=row['objects']) for name,_,row,_ in cases if row is not None}
    print(json.dumps(dict(stage='preflight',id=SID,checkpoint_step=step,seed=seed,
        generated_panoramas=len(cases),denoising_steps=len(cases)*args.steps,
        pano_lora_scale=config['pano_lora_scale'],cases=[(n,len(r['objects']) if r else 0) for n,_,r,_ in cases]),ensure_ascii=False),flush=True)
    if args.preflight:return
    if args.review_only and not args.output.exists():raise ValueError('Review requires an existing generated output')
    if args.output.exists():
        if read(args.output/'run_config.json') != manifest or read(args.output/'input_states.json') != states:
            raise ValueError('Existing output belongs to different settings/states; choose a NEW output')
        output=args.output.resolve()
    else:
        output=new_output(args.output);atomic_json(output/'run_config.json',manifest);atomic_json(output/'input_states.json',states)
    from PIL import Image
    pending=[]
    for name,description,row,path in cases:
        image_path=output/(name+'.png');meta_path=output/(name+'.json')
        if image_path.is_file() and meta_path.is_file():
            with Image.open(image_path) as saved:
                if saved.size != (2*plan['height'],plan['height']):raise ValueError('Saved image dimensions differ')
                saved.verify()
            print('Reusing completed case:',name,flush=True)
        else:pending.append((name,description,row,path))
    started=time.monotonic()
    if args.review_only and pending:raise ValueError('Missing generated cases: '+', '.join(c[0] for c in pending))
    if pending:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('Allocate one BF16 CUDA GPU')
        from diffusers import AutoencoderKLQwenImage
        from qwen_edit_pano.paired_pipeline import PairedPanoPipeline
        from qwen_edit_pano.paired_data import local_image
        from qwen_edit_pano.inference_state import enforce_inference_eval
        from ..scaled_world.binding import apply_scale
        from ..inference_diagnostics import RelativeResidualRecorder
        vae=AutoencoderKLQwenImage.from_pretrained(str(snapshot),subfolder='vae',torch_dtype=torch.float32,local_files_only=True)
        pipe=PairedPanoPipeline.from_pretrained(str(snapshot),vae=vae,torch_dtype=torch.bfloat16,local_files_only=True)
        pipe.vae.to(dtype=torch.float32)
        pipe.load_lora_weights(plan['base_checkpoint'],weight_name='pytorch_lora_weights.safetensors',
                               adapter_name='pano',local_files_only=True,use_safetensors=True)
        atomic_json(output/'pano_scale_check.json',apply_scale(pipe.transformer,config['pano_lora_scale'],'pano'))
        pipe.enable_pano(plan['padding_columns'])
        for module in (pipe.transformer,pipe.vae,pipe.text_encoder):
            enforce_inference_eval(module);module.requires_grad_(False)
        world=WorldBranch(hidden,config['blocks'],config['width'],plan['height'],plan['padding_columns']).cuda().eval().requires_grad_(False)
        world.attach(pipe.transformer);recorder=RelativeResidualRecorder(world);pipe.enable_model_cpu_offload()
        local=local_image(original);local.save(output/'local.png')
        loaded=None
        for index,(name,description,row,path) in enumerate(pending,1):
            world.clear_context()
            if row is not None and path != loaded:
                load_world(world,load_file(str(path/'world.safetensors')),read(path/'config.json'));loaded=path
            current_config=read(path/'config.json');world.enabled=current_config['version']==SPATIAL_VERSION and current_config['spatial_mode']=='spatial'
            recorder.reset();image_started=time.monotonic()
            with torch.inference_mode():
                world.set_world(None if row is None else tuple(t.cuda() for t in tensorize(row,plan['max_objects'])))
                image=pipe(image=local,prompt=plan['prompt'],negative_prompt=' ',height=plan['height'],width=2*plan['height'],
                    num_inference_steps=args.steps,true_cfg_scale=args.cfg,
                    generator=torch.Generator(device='cpu').manual_seed(seed)).images[0]
            if image.size != (2*plan['height'],plan['height']):raise ValueError('Unexpected generated ERP shape')
            tmp=output/(name+'.tmp.png');image.save(tmp);tmp.replace(output/(name+'.png'))
            atomic_json(output/(name+'.json'),dict(seed=seed,checkpoint=str(path.resolve()),
                generation_seconds=time.monotonic()-image_started,relative_residual=recorder.summary(),spatial=world.spatial_diagnostics() if world.enabled and row is not None else {},routing=world.routing_meta))
            world.clear_context()
            print(dict(completed=index,total_pending=len(pending),case=name,
                elapsed_seconds=time.monotonic()-started),flush=True)
    # Ground truth is read AFTER generation, solely for the output review.
    target_image(original).save(output/'gt_erp.png')
    write_views(output,original,[n for n,_,_,_ in cases])
    atomic_json(output/'COMPLETE.json',dict(version=VERSION,cases=[n for n,_,_,_ in cases],checkpoint_step=step,
        note='Generation completed. Object-control success still requires inspecting the images.'))


if __name__ == '__main__':main()
