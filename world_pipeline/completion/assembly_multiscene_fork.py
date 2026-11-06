"""Fork a failed single-output suite, preserving successful scenes and CPTs."""
from pathlib import Path
import shutil

from ..common import read,write,sha


def json_hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in root.rglob('*.json') if p.is_file()}


def check_hashes(root,expected):
    changed=[k for k,h in expected.items() if not (root/k).is_file() or sha(root/k)!=h]
    if changed:raise ValueError('Preserved artifacts changed: '+str(changed[:12]))


def rebase(value,source,dest):
    if isinstance(value,str):return str(dest)+value[len(str(source)):] if value==str(source) or value.startswith(str(source)+'/') else value
    if isinstance(value,list):return [rebase(v,source,dest) for v in value]
    if isinstance(value,dict):return {rebase(k,source,dest):rebase(v,source,dest) for k,v in value.items()}
    return value


def fork_suite(root,source,inputs,*,numeric_recovery=False):
    from .assembly_multiscene import initialize_suite
    from .assembly_experiment import code_hashes
    source=Path(source).resolve();root=Path(root).resolve();s=read(source/'suite.json')
    if s.get('mode')!='single' or s.get('worlds')!=1:raise ValueError('Recovery expects the original single-output suite')
    if root==source or source in root.parents:raise ValueError('Use a new output outside the original suite')
    permitted={'assembly_multiscene.py','assembly_multiscene_fork.py','assembly_multiscene_report.py','assembly_organized_repair.py','organized_multiscene.sh'}
    if numeric_recovery:
        # Explicitly opted-in solver recovery, with no new semantic draws.
        permitted|={'assembly_spatial.py','assembly_coordinate_transport.py','assembly_io_recovery.py'}
    changed=[p for p,h in s['code_sha256'].items() if not Path(p).is_file() or sha(p)!=h]
    if any(Path(p).name not in permitted for p in changed):raise ValueError('SCM, provider or physical compiler changed; cannot certify fixed-mechanism recovery: '+str(changed))
    before=json_hashes(source)
    initialize_suite(root,inputs,s['seed'],1,s['draws'],s['local_model'],mode='single')
    new=read(root/'suite.json');lookup={r['id']:r for r in new['scenes']};history=root/'recovery_history';history.mkdir()
    # A recovery may itself be forked; retain its old history in each scene.
    write(history/'source_suite.json',s);(history/'phase_records').mkdir()
    copied=[];reset=[];reused_worlds=[]
    for r in s['scenes']:
        sid=r['id']
        if sid not in lookup or lookup[sid]['image_sha256']!=r['image_sha256']:raise ValueError('Old scenes must be retained with identical RGB: '+sid)
        src=source/'scenes'/sid;dst=root/'scenes'/sid
        if any(p.is_symlink() for p in src.rglob('*')):raise ValueError('Recovery requires ordinary copied files, not symlinks')
        shutil.copytree(src,dst,dirs_exist_ok=True)
        for p in dst.rglob('experiment.json'):
            config=rebase(read(p),source,root);config['code_sha256']=code_hashes();config['numeric_request_interface']='typed_coordinates_lossless_envelopes_v3';write(p,config)
        failed=dst/'observation/qwen/observer2d'
        if failed.exists() and not (failed/'G_2d.json').exists():
            archive=dst/'recovery_history/observer2d_original_failure';archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(failed),str(archive));reset.append(sid+'/observer2d')
        for p in (source/'phase_records').glob(sid+'_*.json'):
            record=read(p);shutil.copyfile(p,history/'phase_records'/p.name)
            if record['status'] in ('complete','partial') and not (record['phase']=='worlds' and record['status']!='complete'):
                record=rebase(record,source,root);record['reused_from']=str(p);write(root/'phase_records'/p.name,record)
        copied.append(sid)
        for p in dst.rglob('G_full.json'):reused_worlds.append(str(p.relative_to(root)))
    new.update(recovery_source=str(source),recovery_scope='Reuse complete worlds and accepted mechanisms; repair failed interfaces; generate only previously unrun images. Not fresh paired capability evidence.')
    write(root/'suite.json',new)
    protected={}
    for sid in copied:
        for p in (root/'scenes'/sid).rglob('*.json'):
            if p.name=='experiment.json':continue
            if p.name=='summary.json' and p.parent.name=='completion' and (p.parent/'summary.json').exists() and read(p).get('failures'):continue
            protected[str(p.relative_to(root))]=sha(p)
    manifest=dict(source_root=str(source),source_json_sha256=before,protected_copy_json_sha256=protected,
        source_code_changes=changed,reused_scenes=copied,new_scenes=[r['id'] for r in new['scenes'] if r['id'] not in copied],
        reset_failed_interfaces=reset,reused_worlds=reused_worlds,probabilities_resampled_for_reused_scenes=False,source_modified=False)
    write(root/'recovery_manifest.json',manifest);check_hashes(source,before);check_hashes(root,protected)
    return dict(root=str(root),inputs=len(new['scenes']),reused_complete_worlds=len(reused_worlds),reset_interfaces=reset,new_scenes=manifest['new_scenes'])


def verify_recovery(root):
    manifest=read(root/'recovery_manifest.json')
    check_hashes(Path(manifest['source_root']),manifest['source_json_sha256']);check_hashes(root,manifest['protected_copy_json_sha256'])
    return manifest
