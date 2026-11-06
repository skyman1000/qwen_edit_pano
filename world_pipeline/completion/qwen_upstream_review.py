"""CPU review: instance matching, hidden GT references and paired layout plots."""
import argparse
from collections import defaultdict
import html
import itertools
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from ..common import read, rows, write
from .geometry import CANONICAL_CATEGORIES


def view_diagnostics(state, hfov_deg, image_size):
    """Centre projection only: this does not establish occlusion or full-box visibility."""
    tx = np.tan(np.deg2rad(hfov_deg)/2)
    ty = tx*image_size[1]/image_size[0]
    centers = [o['center_local_m']['value'] for o in state['objects']
               if o['center_local_m']['valid']]
    inside = sum(z > 0 and abs(x) <= z*tx and abs(y) <= z*ty for x, y, z in centers)
    return dict(centers_in_input_fov=int(inside), centers_tested=len(centers),
                note='Centre-only frustum test, not visibility or occlusion verification')


def iou_boxes(lo1, hi1, lo2, hi2):
    inter = np.maximum(0., np.minimum(hi1, hi2)-np.maximum(lo1, lo2)).prod()
    union = (hi1-lo1).prod()+(hi2-lo2).prod()-inter
    return float(inter/union) if union > 0 else 0.


def pair_iou(a, b, use_2d=False):
    if use_2d:
        x, y = a['bbox2d_visible_xyxy_norm']['value'], b['bbox2d_visible_xyxy_norm']['value']
        if x is None or y is None:
            return 0.
        return iou_boxes(np.array(x[:2]), np.array(x[2:]), np.array(y[:2]), np.array(y[2:]))
    if any(o[k]['value'] is None for o in (a, b) for k in ('center_local_m', 'size_aabb_local_m')):
        return 0.
    ca, cb = np.array(a['center_local_m']['value']), np.array(b['center_local_m']['value'])
    sa, sb = np.array(a['size_aabb_local_m']['value']), np.array(b['size_aabb_local_m']['value'])
    return iou_boxes(ca-sa/2, ca+sa/2, cb-sb/2, cb+sb/2)


def match_metrics(pred, gt, use_2d=False):
    """Same category, IoU>=.25; maximize cardinality before IoU. No scale alignment."""
    pred, gt = pred['objects'], gt['objects']
    score = np.zeros((len(pred), len(gt)))
    for i, a in enumerate(pred):
        for j, b in enumerate(gt):
            if a['category_id']['value'] == b['category_id']['value']:
                score[i, j] = pair_iou(a, b, use_2d)
    valid = score >= .25
    ii, jj = linear_sum_assignment(-(valid*(min(score.shape, default=0)+1)+score))
    pairs = [(int(i), int(j)) for i, j in zip(ii, jj) if valid[i, j]]
    center_errors, size_errors = [], []
    for i, j in pairs:
        if any(o[k]['value'] is None for o in (pred[i], gt[j]) for k in ('center_local_m', 'size_aabb_local_m')):
            continue
        center_errors.append(float(np.linalg.norm(np.array(pred[i]['center_local_m']['value'])-
                                                   np.array(gt[j]['center_local_m']['value']))))
        size_errors.append(float(np.abs(np.array(pred[i]['size_aabb_local_m']['value'])-
                                         np.array(gt[j]['size_aabb_local_m']['value'])).mean()))
    return dict(pred_count=len(pred), gt_count=len(gt), matches=len(pairs), pairs=pairs,
                valid_3d_count=sum(o['center_local_m']['valid'] and o['size_aabb_local_m']['valid'] for o in pred),
                precision=len(pairs)/len(pred) if pred else None,
                recall=len(pairs)/len(gt) if gt else None,
                matched_center_errors_m=center_errors, matched_size_mae_m=size_errors,
                match_rule='same_category_2d_iou_0.25' if use_2d else 'same_category_aabb3d_iou_0.25')


def diagnostics(hidden, observed):
    duplicates, overlaps = [], []
    for i, h in enumerate(hidden['objects']):
        for j, o in enumerate(observed['objects']):
            score = pair_iou(h, o)
            if score > .5:
                overlaps.append([i, j, score])
            if score > .5 and h['category_id']['value'] == o['category_id']['value']:
                duplicates.append([i, j, score])
    return dict(possible_observed_duplicates=duplicates, high_aabb_overlap_pairs=overlaps,
                note='Review flags only: AABBs and support/containment are not collision geometry.')


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        if row['status'] in ('completed', 'partial'):
            groups[(row['kind'], 'all')].append(row)
            groups[(row['kind'], row['building'])].append(row)
    result = []
    for (kind, building), group in sorted(groups.items()):
        ms = [r['metrics'] for r in group]
        pred = sum(m['pred_count'] for m in ms); gt = sum(m['gt_count'] for m in ms)
        matched = sum(m['matches'] for m in ms)
        errors = [e for m in ms for e in m['matched_center_errors_m']]
        sizes = [e for m in ms for e in m['matched_size_mae_m']]
        result.append(dict(kind=kind, building=building, successful_records=len(ms), partial_records=sum(r['status']=='partial' for r in group),
                           valid_3d_count=sum(m['valid_3d_count'] for m in ms),
                           pred_count=pred, gt_count=gt, matches=matched,
                           precision=matched/pred if pred else None,
                           recall=matched/gt if gt else None,
                           matched_center_error_m=float(np.mean(errors)) if errors else None,
                           matched_size_mae_m=float(np.mean(sizes)) if sizes else None))
    return result


def layout(path, states, bounds=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for state, color, name in states:
        for i, obj in enumerate(state['objects']):
            if obj['center_local_m']['value'] is None or obj['size_aabb_local_m']['value'] is None:
                continue
            c, s = np.array(obj['center_local_m']['value']), np.array(obj['size_aabb_local_m']['value'])
            for ax, indices in zip(axes, ([0, 2], [0, 1])):
                lo, span = (c-s/2)[indices], s[indices]
                ax.add_patch(Rectangle(lo, *span, fill=False, edgecolor=color,
                                      linestyle='--' if name.startswith('GT hidden') else '-',
                                      alpha=.65, label=name if i == 0 else None))
                ax.text(c[indices[0]], c[indices[1]], CANONICAL_CATEGORIES[obj['category_id']['value']],
                        fontsize=5, color=color, alpha=.8)
    for ax, ylabel in zip(axes, ('local z (forward)', 'local y (image up)')):
        ax.plot(0, 0, 'k^'); ax.relim(); ax.autoscale_view(); ax.margins(.1)
        ax.set_aspect('equal'); ax.set_xlabel('local x (image right)'); ax.set_ylabel(ylabel)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    if bounds is not None:
        lo, hi = bounds
        for ax, indices in zip(axes, ([0, 2], [0, 1])):
            ax.set_xlim(lo[indices[0]], hi[indices[0]])
            ax.set_ylim(lo[indices[1]], hi[indices[1]])
    fig.suptitle('Camera-axis AABBs, metres; not gravity-aligned floor plans')
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def projection(image, boxes, hfov, output):
    """Draw available in-front OBB edges; no occlusion or hidden-depth claims."""
    draw = ImageDraw.Draw(image)
    w, h = image.size; f = w/(2*np.tan(np.deg2rad(hfov)/2))
    for obj in boxes:
        c = np.array(obj['corners_local'])
        for i, j in itertools.combinations(range(8), 2):
            # SIGNS order: cube edges differ in one binary coordinate.
            if (i ^ j) not in (1, 2, 4) or min(c[i, 2], c[j, 2]) <= .05:
                continue
            pts = [(float(w/2+f*c[k, 0]/c[k, 2]), float(h/2-f*c[k, 1]/c[k, 2])) for k in (i, j)]
            draw.line(pts, fill='orange', width=2)
    image.save(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--observer', type=Path, required=True)
    p.add_argument('--completion', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline-observer', type=Path)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    refs = {r['id']: r for r in rows(a.bundle/'references.jsonl')}
    conf = read(a.completion/'config.json') if a.completion and (a.completion/'config.json').exists() else None
    records, sections, diversity = [], [], []
    for entry in rows(a.bundle/'inputs.jsonl'):
        sid = entry['id']; od = a.observer/sid
        gtobs, gthid = read(refs[sid]['gt_obs']), read(refs[sid]['gt_hidden'])
        # One shared extent for this scene, including all saved candidate seeds.
        extent_objects = gtobs['objects']+gthid['objects']
        if (od/'G_obs.json').exists():
            extent_objects += read(od/'G_obs.json')['objects']
        if conf:
            for condition in conf['conditions']:
                for seed in conf['seeds']:
                    cp = a.completion/condition/sid/f'seed{seed}'/'G_full.json'
                    if cp.exists():
                        extent_objects += read(cp)['objects']
        points = [np.zeros(3)]
        for obj in extent_objects:
            if obj['center_local_m']['value'] is None or obj['size_aabb_local_m']['value'] is None:
                continue
            c, s = np.array(obj['center_local_m']['value']), np.array(obj['size_aabb_local_m']['value'])
            points.extend((c-s/2, c+s/2))
        lo, hi = np.min(points, axis=0), np.max(points, axis=0)
        margin = np.maximum((hi-lo)*.05, .5)
        bounds = lo-margin, hi+margin
        media = a.output/sid; media.mkdir(exist_ok=True)
        im = Image.open(entry['local_rgb']).convert('RGB'); im.thumbnail((640, 640)); im.save(media/'local.png')
        parts = [f'<h2>{html.escape(sid)}</h2><img width="360" src="{sid}/local.png">']
        observed = None
        status = read(od/'status.json') if (od/'status.json').exists() else dict(status='missing')
        rec = dict(id=sid, building=entry['building'], kind='observer_3d', **status)
        if status['status'] in ('completed', 'partial'):
            observed = read(od/'G_obs.json')
            rec['metrics'] = match_metrics(observed, gtobs)
            records.append(dict(id=sid, building=entry['building'], kind='observer_2d',
                                status=status['status'], metrics=match_metrics(observed, gtobs, True)))
            for name in ('instances.png',):
                Image.open(od/name).save(media/name)
                parts.append(f'<img width="360" src="{sid}/{name}">')
            projection(im.copy(), read(od/'boxes.json'), entry['hfov_deg'], media/'projection.png')
            parts.append(f'<img width="360" src="{sid}/projection.png">')
            layout(media/'observed.png', [(gtobs, 'green', 'GT observed'), (observed, 'blue', 'Pred observed')], bounds)
            parts.append(f'<img width="750" src="{sid}/observed.png">')
        records.append(rec)
        parts.append('<pre>Observer: '+html.escape(str(rec))+'</pre>')
        if a.baseline_observer:
            bp = a.baseline_observer/sid
            bs = read(bp/'status.json') if (bp/'status.json').exists() else dict(status='missing')
            br = dict(id=sid, building=entry['building'], kind='baseline_observer_3d', **bs)
            if bs['status'] in ('completed', 'empty', 'partial'):
                br['status'] = 'completed'
                br['original_status'] = bs['status']
                br['metrics'] = match_metrics(read(bp/'G_obs.json'), gtobs)
            records.append(br)
        if conf:
            for condition in conf['conditions']:
                worlds = []
                for seed in conf['seeds']:
                    d = a.completion/condition/sid/f'seed{seed}'
                    st = read(d/'status.json') if (d/'status.json').exists() else dict(status='missing')
                    r = dict(id=sid, building=entry['building'], kind='hidden_'+condition, **st)
                    r['seed'] = seed
                    if (d/'hidden_plan.txt').exists():
                        parts.append('<h3>Hypothetical hidden plan</h3><pre>'+html.escape(
                            (d/'hidden_plan.txt').read_text())+'</pre>')
                    if st['status'] == 'completed':
                        gh, go, gf = (read(d/f'G_{key}.json') for key in ('hidden', 'obs', 'full'))
                        # Compare against source context, not just another generated file.
                        expected_obs = gtobs if condition == 'gt' else observed
                        r['observed_preserved'] = (expected_obs is not None and go == expected_obs
                            and gf['objects'][:len(go['objects'])] == go['objects'])
                        r['metrics'] = match_metrics(gh, gthid)
                        r['diagnostics'] = diagnostics(gh, go)
                        r['diagnostics'].update(view_diagnostics(gh, entry['hfov_deg'], im.size))
                        worlds.append(gh)
                        name = f'{condition}_{seed}.png'
                        layout(media/name, [(go, 'blue', 'Input observed'), (gh, 'orange', 'Proposed hidden'),
                                            (gthid, 'green', 'GT hidden reference')], bounds)
                        parts.append(f'<h3>{condition} seed {seed}</h3><img width="750" src="{sid}/{name}">')
                    records.append(r)
                    parts.append('<pre>'+html.escape(str(r))+'</pre>')
                similarities = []
                for first, second in itertools.combinations(worlds, 2):
                    m = match_metrics(first, second)
                    n = m['pred_count']+m['gt_count']
                    similarities.append(2*m['matches']/n if n else 1.)
                diversity.append(dict(id=sid, condition=condition, pairwise_same_class_iou_match_f1=similarities))
        sections.append('\n'.join(parts))
    summary = summarize(records)
    failed = [r for r in records if r['status'] != 'completed']
    write(a.output/'metrics.json', dict(summary=summary, records=records, failures=failed, diversity=diversity,
          note='GT is one reference, not the only plausible world. Aggregates include completed and partial outputs; inspect unresolved queries and geometry coverage. '
               '3D matched errors are conditional on IoU matching; 2D matched errors better isolate depth/size errors. '
               'Candidate matching similarity measures repetition, not plausibility or calibrated uncertainty.'))
    text = ('<!doctype html><meta charset="utf-8"><title>Qwen upstream review</title>'
            '<style>body{font-family:sans-serif;margin:24px}pre{white-space:pre-wrap}img{vertical-align:top}</style>'
            '<h1>Qwen observed + hidden state review</h1>'
            '<p>Blue: observed input. Orange: hidden proposal. Dashed green: one GT reference. '
            'Inspect image conflicts manually; AABB overlap is not physical collision. '
            'Geometry uses camera axes, not a gravity-aligned floor plan.</p>'
            f'<p>Partial/failed/missing records: {len(failed)}. Aggregates include partial records, with valid_3d_count reported.</p>'
            '<pre>'+html.escape(str(summary))+'</pre>'+'\n'.join(sections))
    (a.output/'review.html').write_text(text)
    print(f'Review: {a.output / "review.html"}\nMetrics: {a.output / "metrics.json"}')


if __name__ == '__main__':
    main()
