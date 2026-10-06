"""Trace audited GT object categories back to raw .house lines and pinned TSV (CPU)."""
import argparse
import csv
import hashlib
import zipfile
from pathlib import Path

from .common import ROOT, read, sha, write, new_output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit_manifest = read(args.audit/'audit_manifest.json')
    for path, digest in audit_manifest['source_hashes'].items():
        if sha(path) != digest:
            raise ValueError(f'Audited source changed: {path}')
    cases = [r for r in read(args.audit/'review_cases.json') if r['direction']=='gt_to_pred']
    tsvpath = ROOT/'benchmark_assets/Matterport3D_metadata/official/category_mapping.tsv'
    with tsvpath.open() as f:
        categories = {int(r['index']):r for r in csv.DictReader(f,delimiter='\t')}
    scenes, evidence, hashes = {}, [], {str(tsvpath):sha(tsvpath)}
    for case in cases:
        scene = case['sample_id'].split('_',1)[0]
        if scene not in scenes:
            housepath=ROOT/f'qwen_pano/outputs/caupano/house_pilot/{scene}/house.json'
            linkpath=ROOT/f'qwen_pano/outputs/caupano/object_link_pilot/{scene}/object_links.json'
            house, links = read(housepath), read(linkpath)
            if links['category_mapping_sha256'] != sha(tsvpath):
                raise ValueError('Category TSV mismatch')
            archive=ROOT/f'benchmark_assets/Matterport3D_raw/v1/scans/{scene}/house_segmentations.zip'
            with zipfile.ZipFile(archive) as z:
                raw=z.read(house['source_member'])
            hashes[str(housepath)]=sha(housepath); hashes[str(linkpath)]=sha(linkpath)
            hashes[f'{archive}!{house["source_member"]}']=hashlib.sha256(raw).hexdigest()
            scenes[scene]=(house,links,raw.decode().splitlines(),str(archive))
        house,links,lines,archive=scenes[scene]
        oid=int(case['source_house_object_id'])
        obj=next(o for o in house['objects'] if o['id']==oid)
        cat=next(c for c in house['categories'] if c['id']==obj['category_id'])
        canonical=categories[cat['category_mapping_id']]
        original_object=lines[obj['source_line']-1]
        original_category=lines[cat['source_line']-1]
        ot,ct=original_object.split(),original_category.split()
        if ot[0]!='O' or int(ot[1])!=oid or int(ot[3])!=cat['id']:
            raise ValueError('Raw house object record mismatch')
        if ct[0]!='C' or int(ct[1])!=cat['id'] or int(ct[2])!=cat['category_mapping_id']:
            raise ValueError('Raw house category record mismatch')
        linked=[r for r in links['links'] if r['house_object_id']==oid]
        agrees=(int(canonical['mpcat40index'])==case['category_id'] and
                all(str(r['instance_id'])==case['object_id'] and
                    r['method']=='exact_segment_set' for r in linked) and bool(linked))
        evidence.append(dict(sample_id=case['sample_id'],gt_id=case['object_id'],
            house_object_id=oid,source_category_name=cat['category_mapping_name'],
            tsv_raw_category=canonical['raw_category'],tsv_category=canonical['mpcat40'],
            gt_category=case['category'],candidate_prediction=case['candidate_category'],
            mapping_and_link_consistent=agrees,raw_object_line=original_object,
            raw_category_line=original_category,house_zip=archive,
            house_member=house['source_member'],region_links=linked,
            interpretation='Source consistency only; not visual correctness or bbox completeness certification'))
    out=new_output(args.output)
    write(out/'source_evidence.json',evidence)
    write(out/'SUMMARY.json',dict(samples=len({e['sample_id'] for e in evidence}),
        checked_objects=len(evidence),consistent=sum(e['mapping_and_link_consistent'] for e in evidence),
        source_hashes=hashes,audit_cases_sha256=sha(args.audit/'review_cases.json'),
        script_sha256=sha(__file__),ready_for_training=False))
    print(f'{len(evidence)} objects traced; {sum(e["mapping_and_link_consistent"] for e in evidence)} consistent; {out}')


if __name__=='__main__':
    main()
