"""Review offscreen semantic hypotheses separately from sampled geometry; CPU only."""
import argparse
from collections import Counter
import csv
from copy import deepcopy
import html
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from ..common import read, rows, write
from .geometry import CATEGORY_IDS, SIGNS
from .hybrid import local_box, outside_view, matches_region
from .qwen_upstream_review import match_metrics


def offscreen_reference(state, hfov, image_size):
    # GT hidden also includes in-frame occlusion. Only provably outside AABBs here.
    result = deepcopy(state)
    result['objects'] = []
    for obj in state['objects']:
        if not all(obj[key]['valid'] for key in ('center_local_m', 'size_aabb_local_m')):
            continue
        corners = np.asarray(obj['center_local_m']['value'])+SIGNS*np.asarray(obj['size_aabb_local_m']['value'])/2
        if outside_view(corners, hfov, image_size):
            result['objects'].append(obj)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    out = args.run/'review'; out.mkdir(parents=True, exist_ok=True)
    refs = {r['id']: r for r in rows(args.bundle/'references.jsonl')}
    config = read(args.run/'completion/config.json') if (args.run/'completion/config.json').exists() else None
    sections, records, manual = [], [], []
    for entry in rows(args.bundle/'inputs.jsonl'):
        sid = entry['id']; directory = args.run/'proposals'/sid
        with Image.open(entry['local_rgb']) as image:
            image_size = image.size
        image_name = sid+'.png'; shutil.copyfile(entry['local_rgb'], out/image_name)
        parts = [f'<h2>{html.escape(sid)}</h2><img width="420" src="{image_name}">']
        status = read(directory/'status.json') if (directory/'status.json').exists() else dict(status='missing')
        rec = dict(id=sid, building=entry['building'], proposal_status=status)
        if status['status'] != 'completed':
            parts.append('<pre>'+html.escape(json.dumps(status, ensure_ascii=False))+'</pre>')
            records.append(rec); sections.append('\n'.join(parts)); continue
        proposal = read(directory/'proposal.json')
        parts.append('<p>Visible description: '+html.escape(proposal['description'])+'</p>')
        parts.append('<table><tr><th>Candidate</th><th>Visible cue / reason</th><th>Hypothesized region</th><th>Relation</th></tr>')
        for i, item in enumerate(proposal['candidates']):
            parts.append('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in
                [item['category'], item.get('evidence', ''), item.get('region', 'unspecified'), item['relation']])+'</tr>')
            manual.append(dict(id=sid, candidate=i, category=item['category'],
                plausible_offscreen='', duplicates_visible_instance='', conflicts_with_visible_image='', comment=''))
        parts.append('</table>')
        gt = read(refs[sid]['gt_hidden'])
        reference = offscreen_reference(gt, entry['hfov_deg'], image_size)
        pred_cats = Counter(CATEGORY_IDS[item['category']] for item in proposal['candidates'])
        gt_cats = Counter(o['category_id']['value'] for o in reference['objects'])
        rec.update(proposed=len(proposal['candidates']), gt_hidden_total=len(gt['objects']),
            gt_definitely_offscreen=len(reference['objects']),
            semantic_category_count_overlap=sum((pred_cats & gt_cats).values()))
        for condition in ['gt', 'pred']:
            d = args.run/'completion'/condition/sid/'seed0'
            if not (d/'status.json').exists():
                continue
            st = read(d/'status.json'); details = dict(status=st)
            if st['status'] == 'completed':
                audit = read(d/'sampling_audit.json')
                sampled = [a for a in audit if 'native' in a]
                accepted = [a for a in sampled if a['accepted']]
                outside = 0; sector = 0
                for a in sampled:
                    _, corners = local_box(a['native'], config['camera_height'])
                    outside += outside_view(corners, entry['hfov_deg'], image_size)
                    sector += matches_region(corners, proposal['candidates'][a['candidate']].get('region', 'unspecified'),
                                             entry['hfov_deg'], image_size)
                details.update(raw_samples=len(sampled), raw_outside_view=int(outside),
                    raw_region_compatible=int(sector), accepted=len(accepted),
                    rejection_reasons=dict(Counter(a['reason'] for a in sampled if not a['accepted'])),
                    gt_offscreen_reference_metrics=match_metrics(read(d/'G_hidden.json'), reference))
            rec[condition] = details
        parts.append('<pre>'+html.escape(json.dumps(rec, ensure_ascii=False, indent=2))+'</pre>')
        records.append(rec); sections.append('\n'.join(parts))
    note = ('Semantic hypotheses need human image review. GT is one possible world, not a list of all plausible objects. '
            'GT reference includes only hidden objects whose full camera AABB is outside the view: a conservative subset, '
            'not occluded objects inside the frame. Unspecified region imposes no directional constraint. '
            'Low accepted count is a geometry failure, not proof that no offscreen object exists. '
            'Post-filter outside-view success is enforced, so inspect raw sample counts too.')
    write(out/'offscreen_metrics.json', dict(note=note, records=records))
    template = out/'semantic_review.csv'
    if not template.exists():
        with template.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['id', 'candidate', 'category', 'plausible_offscreen',
                'duplicates_visible_instance', 'conflicts_with_visible_image', 'comment'])
            writer.writeheader(); writer.writerows(manual)
    (out/'offscreen.html').write_text('<!doctype html><meta charset="utf-8"><title>Offscreen hypotheses</title>'
        '<style>body{font-family:sans-serif;margin:24px}td,th{border:1px solid #aaa;padding:8px}pre{white-space:pre-wrap}</style>'
        '<h1>Local RGB → offscreen hypotheses → candidate geometry</h1><p>'+html.escape(note)+'</p>'
        '<p>Manual review: semantic_review.csv. If generated, <a href="review.html">layout plots</a> '
        'use the older broad GT-hidden reference; offscreen-only reference metrics are on this page.</p>'+'\n'.join(sections))
    print('Offscreen report:', out/'offscreen.html')


if __name__ == '__main__':
    main()
