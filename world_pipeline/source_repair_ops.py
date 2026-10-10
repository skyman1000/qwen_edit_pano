"""Versioned source quarantine and sensor-fusion repair. Raw assets stay intact.

This changes QC scope to retained observations, not camera poses or GT geometry.
All output remains a candidate requiring downstream projection/visual review.
"""
import copy
import csv
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
from .common import ROOT, read, sha
from .checkpoint_io import atomic_json

PROFILE = 'source_quarantine_refusion_v1'


def link_files(source, output, exclude=()):
    output.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name not in exclude:
            dest = output/path.name
            if dest.exists() or dest.is_symlink():
                raise ValueError(f'Repair output already exists: {dest}')
            dest.symlink_to(path.resolve(), target_is_directory=path.is_dir())


def label_conflicts(region):
    """Identify exact linked instances; never choose a replacement category."""
    from qwen_pano.caupano.data.matterport.semseg_json import load_semseg_json
    from qwen_pano.caupano.data.matterport.camera_parser import member_index
    index = read(region/'region_index.json')
    mapping_path = ROOT/'benchmark_assets/Matterport3D_metadata/official/category_mapping.tsv'
    if sha(mapping_path) != index['category_mapping_sha256']:
        raise ValueError('Vocabulary changed')
    with mapping_path.open() as f:
        mapping = {int(r['index']):r for r in csv.DictReader(f,delimiter='\t')}
    house = read(Path(index['house_directory'])/'house.json')
    with ZipFile(Path(index['scan_dir'])/'house_segmentations.zip') as z:
        name = member_index(z)[index['scan_id']+'.semseg.json']
        groups = load_semseg_json(z.read(name),name,[])['segGroups']
    by_segments = {}
    for g in groups:
        key = frozenset(g['segments'])
        if key in by_segments:
            raise ValueError('Ambiguous house segment linkage')
        by_segments[key] = g
    house_ids = {o['id'] for o in house['objects']}
    result = []
    with ZipFile(Path(index['scan_dir'])/'region_segmentations.zip') as z:
        for row in index['regions']:
            groups = load_semseg_json(z.read(row['semseg_member']),row['semseg_member'],[])['segGroups']
            with np.load(region/row['array_file'],allow_pickle=False) as a:
                for g in groups:
                    mask = a['face_instance_local']==int(g['objectId'])
                    raw = [int(x) for x in np.unique(a['face_category_mapping_id'][mask]) if x>0]
                    bad = [x for x in raw if mapping[x]['raw_category'].replace('#',' ') != g['label'].replace('#',' ')]
                    if not bad:
                        continue
                    key = frozenset(row['region_id']*1000000+s for s in g['segments'])
                    target = by_segments.get(key)
                    if target is None or target['id'] not in house_ids:
                        raise ValueError('Cannot quarantine an unlinked conflict')
                    result.append(dict(region_id=row['region_id'],local_instance_id=int(g['objectId']),
                        house_object_id=target['id'],instance_id=target['id']+1,
                        region_label=g['label'],house_label_index=target.get('label_index'),
                        mesh_mapping_ids=raw,mesh_labels=[mapping[x]['raw_category'] for x in raw],
                        affected_faces=int(mask.sum()),array_sha256=sha(region/row['array_file']),
                        validation_error=f"{row['region_id']}/object {g['objectId']}: mesh category differs from semseg label"))
    return result


def accept_quarantined_region(region, conflicts):
    # The ORIGINAL validator has just run on this directory; only the exact
    # isolated lexical mismatches may be resolved by removing their supervision.
    report = read(region/'region_validation.json')
    expected = [c['validation_error'] for c in conflicts]
    if not expected or sorted(report['errors']) != sorted(expected):
        raise ValueError('Other region failures remain; quarantine cannot bypass them')
    original = copy.deepcopy(report)
    atomic_json(region/'region_validation.original.json',original)
    audit = dict(profile=PROFILE,conflicts=conflicts,
        raw_geometry_and_numeric_labels_modified=False,
        resolution='Keep geometry as occluders; remove these instances from downstream category supervision',
        original_validation_sha256=sha(region/'region_validation.original.json'))
    atomic_json(region/'semantic_quarantine.json',audit)
    report.update(status='PASS',errors=[],ready_for_global_object_linkage=True,
        ready_for_world_supervision=False,semantic_quarantine=audit,
        scope='Region structure valid; listed lexical category conflicts NOT resolved, excluded from training')
    atomic_json(region/'region_validation.json',report)


def choose_observations(checks):
    good = [r for r in checks if r['count']>0 and r['median_relative'] is not None and
            np.isfinite(r['median_relative']) and r['median_relative']<=.10]
    rejected = [r for r in checks if r not in good]
    yaw = {int(r['observation_id'].rsplit('_',1)[-1]) for r in good}
    supported = len(good)>=9 and yaw==set(range(6))
    return good,rejected,supported


def refusion(index, source, output):
    from qwen_pano.caupano.data.build_geometry_pilot import image_from_zip,camera_points,depth_stats
    geo = read(source/'geometry_report.json')
    allowed = {'source_camera_depth_mesh_disagreement','sensor_coverage_below_0.25',
               'sensor_mesh_median_relative_above_0.10'}
    if (geo['status']!='BUILT_FOR_REVIEW' or not all(geo['checks'].values()) or
            set(geo['review_flags'])-allowed or geo['height']!=512 or geo['erp_sampling']!='pixel_centers'):
        raise ValueError('Not eligible for observation-only repair')
    good,rejected,supported = choose_observations(geo['source_camera_depth_checks'])
    if not supported:
        raise ValueError('Insufficient reliable observations: need >=9 and all six yaw groups; no relaxation')
    sample = next(r for r in map(json.loads,(index/'panorama_index.jsonl').read_text().splitlines())
                  if r['sample_id']==geo['sample_id'])
    observations = {r['observation_id']:r for r in map(json.loads,Path(sample['observations_path']).read_text().splitlines())}
    known = {r['observation_id'] for r in geo['source_camera_depth_checks']}
    if known!=set(sample['observation_ids']):
        raise ValueError('Observation report/sample mismatch')
    retained = {r['observation_id'] for r in good}
    h,w = geo['height'],geo['width']; basis=np.asarray(geo['erp_to_world_basis_candidate']);origin=np.asarray(geo['origin_world_candidate'])
    sensor=np.full(h*w,np.inf,dtype='float32'); rgb=np.zeros((h*w,3),dtype='uint8')
    for oi,oid in enumerate(sample['observation_ids']):
        if oid not in retained:
            continue
        depth=image_from_zip(sample['depth_source'][oi]).astype('float64')/4000.
        color=image_from_zip(sample['perspective_rgb_source'][oi],rgb=True)
        if color.shape[:2]!=depth.shape:
            raise ValueError('Color/depth dimensions differ')
        directions=camera_points(observations[oid],depth.shape)
        pose=np.asarray(observations[oid]['undistorted']['camera_to_world']); valid=depth>0
        points=(directions[valid]*depth[valid,None])@pose[:3,:3].T+pose[:3,3]
        local=(points-origin)@basis;rho=np.linalg.norm(local,axis=-1);nonzero=rho>1e-6
        local,rho=local[nonzero],rho[nonzero]
        lon=np.arctan2(local[:,0],local[:,2]);lat=np.arcsin(np.clip(local[:,1]/rho,-1,1))
        u=np.floor(w*(lon/(2*np.pi)+.5)).astype('int64')%w
        v=np.clip(np.floor(h*(.5-lat/np.pi)).astype('int64'),0,h-1)
        pixels=v*w+u;order=np.lexsort((rho,pixels));_,first=np.unique(pixels[order],return_index=True)
        chosen=order[first];bins=pixels[chosen];closer=rho[chosen]<sensor[bins];chosen,bins=chosen[closer],bins[closer]
        sensor[bins]=rho[chosen];rgb[bins]=color[valid][nonzero][chosen]
    valid=np.isfinite(sensor).reshape(h,w); sensor=np.where(np.isfinite(sensor),sensor,0).reshape(h,w)
    mesh=np.load(source/'depth_mesh_erp.npy',allow_pickle=False)
    meshvalid=np.load(source/'depth_mesh_valid.npy',allow_pickle=False)
    stats=depth_stats(sensor,mesh,valid&meshvalid)
    flags=[]
    if not stats['count'] or stats['median_relative']>.10:
        flags.append('sensor_mesh_median_relative_above_0.10')
    if valid.mean()<.25:
        flags.append('sensor_coverage_below_0.25')
    exclude={'geometry_report.json','depth_sensor_erp.npy','depth_sensor_valid.npy',
             'rgb_sensor_projection.png','preview.jpg','preview.png','geometry_overview.jpg'}
    link_files(source,output,exclude)
    np.save(output/'depth_sensor_erp.npy',sensor,allow_pickle=False)
    np.save(output/'depth_sensor_valid.npy',valid,allow_pickle=False)
    Image.fromarray(rgb.reshape(h,w,3)).save(output/'rgb_sensor_projection.png')
    audit=dict(profile=PROFILE,original_report=str((source/'geometry_report.json').resolve()),
        original_report_sha256=sha(source/'geometry_report.json'),original_review_flags=geo['review_flags'],
        retained_observation_ids=sorted(retained),quarantined_observations=rejected,
        acceptance='>=9 retained, all 6 yaw groups, unchanged 0.25 coverage and 0.10 depth thresholds',
        limitation='Retained-view depth evidence only; not proof quarantined views or all objects are geometrically correct',
        before=dict(sensor_coverage=geo['sensor_coverage'],sensor_mesh_depth=geo['sensor_mesh_depth']),
        after=dict(sensor_coverage=float(valid.mean()),sensor_mesh_depth=stats),
        poses_mesh_labels_instances_modified=False,ready_for_training=False)
    atomic_json(output/'repair_audit.json',audit)
    repaired=copy.deepcopy(geo)
    repaired.update(review_flags=flags,sensor_coverage=float(valid.mean()),sensor_mesh_depth=stats,
        repair=audit,depth_validation_scope='retained_source_observations_only')
    # Preserve ALL original per-camera checks, including failed/unknown ones.
    repaired['checks']['nonempty_sensor']=bool(valid.any())
    repaired['checks']['overlapping_depth']=bool((valid&meshvalid).any())
    atomic_json(output/'geometry_report.json',repaired)
    if flags:
        raise ValueError('Refusion still fails coverage/depth checks; diagnostic output retained')


def repaired_world(index, geometry, output):
    from .lean_world import build
    build(index,geometry,output)
    # The lean builder commits before quarantine is applied. Hide that marker
    # until the complete repaired catalog is committed, including interruptions.
    report=read(output/'build_report.json')
    (output/'build_report.json').unlink()
    sample=next(r for r in map(json.loads,(index/'panorama_index.jsonl').read_text().splitlines())
                if r['sample_id']==read(geometry/'geometry_report.json')['sample_id'])
    quarantine=Path(sample['region_index_path']).parent/'semantic_quarantine.json'
    if quarantine.exists():
        audit=read(quarantine); ids={r['instance_id'] for r in audit['conflicts']}
        catalog=read(output/'objects.json')
        if ids-{o['id'] for o in catalog['objects']}:
            raise ValueError('Quarantined instances missing from catalog')
        for obj in catalog['objects']:
            if obj['id'] in ids:
                obj['supervision_mask']['category']=False
        catalog['semantic_quarantine']=audit
        atomic_json(output/'objects.json',catalog)
        state=read(output/'world_state.json');state['semantic_quarantine']=audit
        atomic_json(output/'world_state.json',state)
    report['repair_profile']=PROFILE
    atomic_json(output/'build_report.json',report)
