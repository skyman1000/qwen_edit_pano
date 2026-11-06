"""Protocol-checked baseline reuse, no image/model hashes or GPU work."""
from PIL import Image
from ..build_polished_world import read


def check_baselines(directory,config,selected,panel,steps,cfg,seed,split):
    if panel is None:raise ValueError('Baseline reuse requires the fixed panel')
    previous=read(directory/'run_config.json');complete=read(directory/'COMPLETE.json')
    for key,value in dict(steps=steps,cfg=cfg,seed=seed,split=split).items():
        if previous[key]!=value:raise ValueError('Baseline protocol differs: '+key)
    if (previous['ids']!=[r['id'] for r in selected] or complete['samples']!=len(selected)
            or 'baseline' not in previous['modes'] or 'baseline' not in complete['modes']):
        raise ValueError('Baseline image panel is incomplete/different')
    if previous['config']['pano_lora_scale']!=config['pano_lora_scale']:
        raise ValueError('Frozen Panorama scale differs')
    for key in ('base_snapshot','base_checkpoint','height','padding_columns','prompt'):
        if previous['config']['pack_plan'][key]!=config['pack_plan'][key]:
            raise ValueError('Frozen baseline binding differs: '+key)
    if read(directory/'diagnostic_panel.json')!=panel:raise ValueError('Baseline panel identities differ')
    for row,entry in zip(selected,panel['samples']):
        meta=read(directory/row['id']/'baseline.json')
        if meta['seed']!=entry['seed'] or meta['world_source'] is not None or meta['oracle']:
            raise ValueError('Baseline was not generated with World disabled')
        with Image.open(directory/row['id']/'baseline.png') as image:
            height=config['pack_plan']['height']
            if image.size!=(2*height,height):raise ValueError('Baseline resolution differs')
    return dict(source=str(directory.resolve()),images=len(selected),protocol_checked=True,
                file_content_hashes_computed=False)
