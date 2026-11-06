"""Manual local-Qwen pipeline: observations -> global SCM -> audited worlds.

Run stages in separate processes so WildDet3D and Qwen never occupy the GPU
together. A finite attempt budget is preserved across resume, including failures.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
from html import escape
import json
from pathlib import Path
import re
import shutil
import sys
import traceback

from PIL import Image, ImageDraw

from ..common import ROOT, new_output, read, rows, sha, write
from . import global_world_model as model, global_world_geometry as geometry
from . import global_world_observation as observation_contract
from . import global_world_proposals as proposals
from .qwen_upstream import DEFAULT_MODEL

DEFAULT_INPUTS = ROOT/'qwen_edit_pano/data/observer3d_indoor_val5_inputs.jsonl'


def seed_for(seed, name):
    return int.from_bytes(hashlib.sha256((str(seed)+'|'+name).encode()).digest()[:4], 'big')


def source_hashes():
    folder = Path(__file__).parent
    names = ('global_world', 'global_world_model', 'global_world_geometry', 'global_world_observation',
             'global_world_proposals', 'global_world_feasibility', 'world_scm',
             'world_generation', 'qwen_upstream', 'fresh_observer', 'fresh_wilddet', 'box_observer',
             'assembly_io_recovery', 'assembly_observer_recovery', 'assembly_experiment', 'world_scm_runner', 'category_bridge', 'geometry',
             'assembly_presentation_visual')
    paths = [folder/(n+'.py') for n in names] + [folder.parent/'common.py',
        ROOT/'qwen_edit_pano/scripts/global_world.sh']
    return {str(p.resolve()): sha(p) for p in paths}


def prepare(root, inputs=None, image=None, hfov=90., local_model=DEFAULT_MODEL,
            worlds=6, hypotheses=3, alternatives=2, draws=4, seed=3407,
            maximum_objects=48, observations_from=None, solver_starts=3, solver_iterations=180,
            recover_from=None):
    if not (1 <= worlds <= 32 and 1 <= hypotheses <= 6 and 1 <= alternatives <= 4
            and 1 <= draws <= 32 and 1 <= maximum_objects <= 64
            and 1 <= solver_starts <= 8 and 1 <= solver_iterations <= 1000):
        raise ValueError('Configuration exceeds the bounded computation budgets')
    if inputs is not None and image is not None:
        raise ValueError('Choose an image or an input manifest')
    if observations_from and recover_from:
        raise ValueError('Choose immutable metric observations or explicit 2D recovery, not both')
    local_model = Path(local_model).resolve()
    if not (local_model/'config.json').is_file() or not list(local_model.glob('*.safetensors')):
        raise ValueError('Missing local Qwen config/shards: '+str(local_model))
    recovery_suite = read(Path(recover_from)/'suite.json') if recover_from else None
    if recovery_suite and recovery_suite.get('kind') not in ('global_conditional_program_scm_v1', 'global_conditional_program_scm_v2', model.PROTOCOL):
        raise ValueError('Recovery source must be a global-world run, not scene GT')
    if recovery_suite and image is None and inputs is None:
        records = [dict(id=s['id'], local_rgb=s['source_image'], hfov_degrees=s['hfov'], split=s.get('split','unspecified'))
                   for s in recovery_suite['scenes']]
    else:
        records = [dict(id=Path(image).stem, local_rgb=str(image), hfov_degrees=hfov)] if image else rows(inputs or DEFAULT_INPUTS)
    if not records:raise ValueError('Empty input manifest')
    clean = []
    for r in records:
        sid = r['id'];path = Path(r['local_rgb']).resolve()
        if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', sid):
            raise ValueError('Scene IDs must be safe unique identifiers')
        fov = r.get('hfov_degrees', r.get('hfov_deg', hfov))
        if not model.number(fov) or not 1 <= fov <= 160:raise ValueError('HFOV must be in [1,160] degrees')
        with Image.open(path) as im:size = list(im.size)
        if r.get('local_sha256') and sha(path) != r['local_sha256']:
            raise ValueError('Input hash mismatch: '+sid)
        clean.append(dict(id=sid, source_image=str(path), source_sha256=sha(path), hfov=fov,
            image_size=size, split=r.get('split', 'unspecified')))
    if len({s['id'] for s in clean}) != len(clean):raise ValueError('Duplicate scene IDs')
    imported = {}
    if observations_from:
        from .local_observation import visible_state, bind_observations
        source = Path(observations_from).resolve();suite = read(source/'suite.json')
        for s in clean:
            old = next((v for v in suite['scenes'] if v['id'] == s['id']), None)
            if old is None or old['hfov'] != s['hfov']:raise ValueError('Reused observation ID/HFOV mismatch')
            obs = source/'scenes'/s['id']/'observation'
            with Image.open(obs/'input.png') as a, Image.open(s['source_image']) as b:
                if a.size != b.size or a.convert('RGB').tobytes() != b.convert('RGB').tobytes():
                    raise ValueError('Reused observation RGB mismatch')
            two = visible_state(read(obs/'qwen/observer2d/G_2d.json'))
            three = visible_state(read(obs/'qwen/observer3d/G_obs.json'), metric=True)
            bind_observations(two, three)
            imported[s['id']] = obs
    recovery = {}
    if recovery_suite:
        recovery_source = Path(recover_from).resolve()
        for s in clean:
            old = next((v for v in recovery_suite['scenes'] if v['id'] == s['id']), None)
            if old is None or old['hfov'] != s['hfov']:raise ValueError('Recovery scene/HFOV mismatch')
            source = recovery_source/'scenes'/s['id']/'observation'
            if sha(source/'input.png') != old['image_sha256']:raise ValueError('Recovery image changed since source preparation')
            with Image.open(source/'input.png') as a, Image.open(s['source_image']) as b:
                if a.size != b.size or a.convert('RGB').tobytes() != b.convert('RGB').tobytes():
                    raise ValueError('Recovery RGB pixels mismatch')
            recovery[s['id']] = source
    root = new_output(root)
    config = dict(kind=model.PROTOCOL, scenes=clean, local_model=str(local_model),
        local_model_config_sha256=sha(local_model/'config.json'), worlds=worlds,
        hypotheses=hypotheses, alternatives=alternatives, draws=draws, seed=seed,
        maximum_objects=maximum_objects, solver_starts=solver_starts, solver_iterations=solver_iterations,
        code_sha256=source_hashes(), API_used=False, training_performed=False, scene_gt_used=False,
        observer_interface='original_v13_seeded_Qwen_2D_then_official_WildDet3D',
        observation_post_review=False,category_mapping='existing_official_category_bridge_only',
        sampling_policy='fixed_budget_IID_on_explicit_executable_proxy_compatible_domain; repeats_empty_draws_and_all_candidate_failures_retained',
        source_observation_hashes={}, created=datetime.now().isoformat())
    config['recovery'] = dict(source=str(Path(recover_from).resolve()),
        policy='reuse_reviewed_2D_with_explicit_semantic_normalization; rerun_ALL_3D; independently_recognize_failed_drafts') if recover_from else None
    for s in clean:
        obs = root/'scenes'/s['id']/'observation';obs.mkdir(parents=True)
        with Image.open(s['source_image']) as im:im.convert('RGB').save(obs/'input.png')
        s['image_sha256'] = sha(obs/'input.png')
        write(obs/'experiment.json', dict(protocol=model.PROTOCOL, id=s['id'], hfov=s['hfov'],
            image_size=s['image_size'], local_model=str(local_model), max_tokens=8192, split=s['split'],seed=seed))
        if s['id'] in imported:
            for name in ('observer2d', 'observer3d'):
                src = imported[s['id']]/'qwen'/name;dst = obs/'qwen'/name
                shutil.copytree(src, dst)
                for file in src.rglob('*'):
                    if file.is_file():config['source_observation_hashes'][str(file)] = sha(file)
            write(obs/'imported_observation.json', dict(source=str(imported[s['id']]),
                observations_reused=True, new_model_inference=False, quality_not_certified=True))
            freeze_observations(obs)
        if s['id'] in recovery:
            source = recovery[s['id']]
            copied = []
            if (source/'draft').exists():
                shutil.copytree(source/'draft', obs/'draft')
                copied.extend(p for p in (source/'draft').rglob('*') if p.is_file())
            review = source/'review_calls/whole_image_grounding.call.json'
            if review.exists():
                copied.append(review)
                (obs/'recovery').mkdir(exist_ok=True)
                shutil.copyfile(review, obs/'recovery/previous_review.call.json')
                record = read(review)
                if record.get('status') == 'accepted':
                    clean_inventory = observation_contract.inventory_answer(record['raw'])
                    write_inventory(obs, clean_inventory, dict(previous_valid_2d_reused=True,
                        source=str(review), source_sha256=sha(review), metric_geometry_reused=False))
            copied.append(source/'input.png')
            for file in copied:config['source_observation_hashes'][str(file)] = sha(file)
            write(obs/'recovery_source.json', dict(source=str(source), source_protocol=recovery_suite['kind'],
                reviewed_2d_reused=(obs/'qwen/observer2d/G_2d.json').exists(), metric_geometry_reused=False))
    write(root/'suite.json', config)
    if recover_from:
        write(root/'recovery_plan.json', dict(source=str(Path(recover_from).resolve()),
            source_results_modified=False, model_called=False,
            scenes=[dict(id=s['id'], **read(root/'scenes'/s['id']/'observation/recovery_source.json')) for s in clean],
            downstream_policy='ALL WildDet3D geometry recomputed after specific semantic prompts; no old G_obs frozen as new output'))
    return root


def verify(root):
    root = Path(root).resolve();config = read(root/'suite.json')
    if config['kind'] != model.PROTOCOL:raise ValueError('Wrong run kind')
    changed = [p for p, h in config['code_sha256'].items() if not Path(p).is_file() or sha(p) != h]
    if changed:raise ValueError('Source changed since prepare; use a new run: '+str(changed))
    if sha(Path(config['local_model'])/'config.json') != config['local_model_config_sha256']:
        raise ValueError('Local model config changed')
    for s in config['scenes']:
        if sha(s['source_image']) != s['source_sha256'] or sha(root/'scenes'/s['id']/'observation/input.png') != s['image_sha256']:
            raise ValueError('Input image changed: '+s['id'])
        binding = root/'scenes'/s['id']/'observation/observation_binding.json'
        if binding.exists():
            for name, digest in read(binding).items():
                if sha(binding.parent/name) != digest:raise ValueError('Frozen observation changed: '+name)
    for p, h in config['source_observation_hashes'].items():
        if not Path(p).is_file() or sha(p) != h:raise ValueError('Reused source observation changed: '+p)
    return config


def check_observation(root):
    verify(Path(root).parents[2])
    return read(Path(root)/'experiment.json')


def freeze_observations(obs):
    names = ('qwen/observer2d/G_2d.json', 'qwen/observer2d/grounding.json', 'qwen/observer3d/G_obs.json')
    hashes = {name: sha(obs/name) for name in names}
    target = obs/'observation_binding.json'
    if target.exists() and read(target) != hashes:raise ValueError('Frozen 2D/3D observations changed')
    if not target.exists():write(target, hashes)


class Calls:
    """Two bounded schema attempts, journaled before execution; never retry a rejected world."""
    def __init__(self, provider, out, seed):
        self.provider = provider;self.out = Path(out);self.seed = seed
        self.out.mkdir(parents=True, exist_ok=True)

    def ask(self, name, image, prompt, validator, tokens=8192):
        fingerprint = hashlib.sha256((prompt+'|'+str(tokens)).encode() +
            (image.convert('RGB').tobytes() if image is not None else b'')).hexdigest()
        journal = self.out/(name+'.call.json')
        if journal.exists():
            saved = read(journal)
            if saved['fingerprint'] != fingerprint:raise ValueError('Cached question changed: '+name)
            if saved['status'] == 'accepted':return validator(saved['raw'])
            if saved['status'] == 'failed':raise ValueError('Preserved failed request '+name+': '+saved['error'])
        else:
            saved = dict(fingerprint=fingerprint, status='pending', attempts=[], name=name)
            write(journal, saved)
        errors = []
        for attempt in range(2):
            path = self.out/(name+f'.try{attempt}.txt')
            current_prompt = prompt + ('\nSCHEMA CORRECTION: '+errors[-1][:1200] if errors else '')
            recorded = next((a for a in saved['attempts'] if a['attempt'] == attempt), None)
            try:
                if recorded is not None:
                    if recorded.get('error'):
                        errors.append(recorded['error']);continue
                    # If interrupted after generation, recover only the stored reply.
                    if path.with_suffix('.json').is_file():raw = read(path.with_suffix('.json'))
                    elif path.is_file():
                        from .geometry import parse_json
                        raw = parse_json(path.read_text())
                    else:raise RuntimeError('Interrupted request has no saved reply; no stochastic retry')
                else:
                    recorded = dict(attempt=attempt, seed=seed_for(self.seed, name+'|'+str(attempt)))
                    saved['attempts'].append(recorded);write(journal, saved)
                    print(json.dumps(dict(event='local_qwen_request', stage=name, attempt=attempt,
                        max_tokens=tokens), ensure_ascii=False), flush=True)
                    raw = self.provider().ask(image, current_prompt, path, seed=recorded['seed'],
                        max_tokens=tokens, sample=True, profile='completion')
                answer = validator(raw)
                saved.update(status='accepted', raw=raw);write(journal, saved)
                return answer
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                error = type(exc).__name__+': '+str(exc);errors.append(error)
                recorded['error'] = error;write(journal, saved)
            except Exception as exc:
                saved.update(status='failed', error=type(exc).__name__+': '+str(exc));write(journal, saved)
                raise
        saved.update(status='failed', error='; '.join(errors));write(journal, saved)
        raise ValueError(saved['error'])


def write_inventory(obs, clean, provenance=None):
    config = read(obs/'experiment.json')
    state, mapping = observation_contract.export_inventory(clean, config)
    dest = obs/'qwen/observer2d';dest.mkdir(parents=True, exist_ok=False)
    write(dest/'G_2d.json', state);write(dest/'labels.json', mapping)
    write(dest/'grounding.json', dict(instances=[dict(id=o['track_id'], variant=r['name'], on_floor=r['on_floor'])
        for o, r in zip(state['objects'], clean['objects'])], evidence_description=clean['evidence_description']))
    write(dest/'semantic_audit.json', observation_contract.quality_audit(clean))
    if provenance:write(dest/'provenance.json', provenance)
    with Image.open(obs/'input.png') as im:image = im.convert('RGB')
    draw = ImageDraw.Draw(image)
    for o in state['objects']:
        b = o['bbox2d_visible_xyxy_norm']['value'];box = [v*k for v, k in zip(b, image.size*2)]
        draw.rectangle(box, outline='#287db5', width=3)
        draw.text(tuple(box[:2]), o['track_id']+' '+o['raw_category'], fill='red')
    image.save(dest/'boxes2d.png')
    write(dest/'summary.json', dict(status='completed', objects=len(state['objects']),
        whole_image_review=True, API_used=False, quality_certified=False,
        specific_semantics_preserved=True, provenance=provenance))


def observe2d(obs, provider, seed):
    """Use exactly the existing v13 observer, including its stage-seed wrapper.

    Category mapping remains in its original exporter; raw nouns, IDs and boxes
    are never replaced by a second global-world inventory request.
    """
    from .assembly_io_recovery import observe_chunked
    from .assembly_experiment import SeededProvider,stage_seed
    config=read(obs/'experiment.json')
    summary=obs/'qwen/observer2d/summary.json'
    if summary.exists():return read(summary)
    return observe_chunked(SeededProvider(provider(),stage_seed(seed,'shared_observer')),
        obs,'qwen',config.get('max_tokens',8192))


def observe2d_review_v2(obs, provider, seed):
    """Historical v2 recovery only; never used for fresh production observations."""
    from .assembly_io_recovery import observe_chunked
    draft = obs/'draft'
    if not draft.exists():
        draft.mkdir();shutil.copyfile(obs/'input.png', draft/'input.png')
        shutil.copyfile(obs/'experiment.json', draft/'experiment.json')
    # A saved failure is not repeated until it becomes successful.
    summary = draft/'qwen/observer2d/summary.json'
    if not summary.exists():
        try:observe_chunked(provider(), draft)
        except (ValueError, TypeError, KeyError) as exc:
            # Recognition/schema failures are recoverable by independent grounding.
            # Runtime/device failures still propagate; no endless retries.
            write(obs/'draft_failure.json', dict(error=type(exc).__name__+': '+str(exc),
                action='independent_whole_image_grounding', model_output_fabricated=False))
    draft_context = dict(status=read(summary) if summary.exists() else 'no_completed_draft')
    inventory_path = draft/'qwen/observer2d/G_2d.json'
    if inventory_path.exists():draft_context['inventory'] = read(inventory_path)
    grounding_path = draft/'qwen/observer2d/grounding.json'
    if grounding_path.exists():draft_context['evidence'] = read(grounding_path)
    elif (draft/'qwen/observer2d').exists():
        draft_context['failed_replies'] = []
        for path in sorted((draft/'qwen/observer2d').glob('chunk_*.json')):
            if path.name.count('.') == 1:
                draft_context['failed_replies'].append(dict(file=path.name, reply=read(path)))
    calls = Calls(provider, obs/'review_calls', seed)
    prompt = observation_contract.review_prompt(draft_context)
    with Image.open(obs/'input.png') as im:image = im.convert('RGB')
    clean = calls.ask('whole_image_grounding', image, prompt, observation_contract.inventory_answer)
    write_inventory(obs, clean, dict(draft_status=draft_context['status'], independent_review=True,
        previous_valid_2d_reused=False, metric_geometry_reused=False))


def choices(calls, name, values, packet, count, seed, hypothesis=None):
    votes = [];failures = []
    for index in range(count):
        rng = __import__('random').Random(seed_for(seed, name+'|'+str(index)))
        order = list(values);rng.shuffle(order)
        aliases = {'option_'+str(i): v for i, v in enumerate(order)}
        options = [dict(choice=a, state={k:item for k,item in v.items() if k not in (
            'initial_placement','compiled_layout','geometry_audit','visibility_audit',
            'proposal_rejections','inventory_rejections','support_relation_evidence')}) for a,v in aliases.items()]
        def validate(value):
            choice = value.get('choice') if isinstance(value, dict) else None
            if not isinstance(choice, str) or choice not in aliases:raise ValueError('Unknown choice alias')
            return aliases[choice]['id']
        try:
            votes.append(calls.ask(name+'_'+str(index), None,
                model.choice_prompt(name, options, packet, hypothesis), validate, tokens=256))
        except Exception as exc:failures.append(dict(draw=index, error=str(exc)))
    return dict(counts=dict(Counter(votes)), votes=votes, failures=failures,
        valid_draws=len(votes), designed_encodings=count)


def make_board(state, image, hfov, path):
    from .assembly_presentation_visual import label_rows, perspective, plan, shared_bounds
    from PIL import ImageOps
    labels = label_rows(state);overlay, _ = perspective(state, labels, image, hfov)
    plan_path = path.with_name('layout_xz.png');plan(state, labels, shared_bounds([state]), hfov, plan_path)
    board = Image.new('RGB', (1536, 1080), 'white');draw = ImageDraw.Draw(board)
    for im, box, title in [(image, (0, 0, 768, 540), 'ORIGINAL RGB'),
                           (overlay, (768, 0, 1536, 540), 'FROZEN + PROPOSED BOX PROJECTION')]:
        draw.text((box[0]+8, box[1]+8), title, fill='black')
        tile = ImageOps.contain(im, (748, 500));board.paste(tile, (box[0]+10, box[1]+30))
    with Image.open(plan_path) as im:tile = ImageOps.contain(im.convert('RGB'), (1500, 510))
    board.paste(tile, ((1536-tile.width)//2, 560));board.save(path)
    return board


def complete_scene(scene, config, provider):
    obs = scene/'observation';dest = scene/'completion';dest.mkdir(parents=True, exist_ok=True)
    freeze_observations(obs)
    observed_path = obs/'qwen/observer3d/G_obs.json';observed = read(observed_path)
    packet = model.observation_packet(observed, read(obs/'qwen/observer2d/grounding.json'))
    source_hash = sha(observed_path)
    if (dest/'G_obs.json').exists() and sha(dest/'G_obs.json') != source_hash:
        raise ValueError('Frozen observation changed during resume')
    shutil.copyfile(observed_path, dest/'G_obs.json')
    calls = Calls(provider, dest/'calls', seed_for(config['seed'], observed['id']))
    with Image.open(obs/'input.png') as im:image = im.convert('RGB')
    camera = dict(hfov=read(obs/'experiment.json')['hfov'], image_size=list(image.size))
    hypotheses = calls.ask('global_hypotheses', image,
        model.hypotheses_prompt(packet, camera, config['hypotheses']),
        lambda v: model.hypotheses_answer(v, packet, config['hypotheses']))
    write(dest/'hypotheses.json', hypotheses)
    programs = {};proposal_failures = []
    for h in hypotheses:
        programs[h['id']] = []
        try:
            inventories = calls.ask('inventory_'+h['id'], image,
                proposals.inventory_prompt(h, packet, config['alternatives'], config['maximum_objects']),
                lambda v,h=h: proposals.inventory_answer(v,h,packet,config['alternatives'],config['maximum_objects']))
            write(dest/(h['id']+'_inventories.json'), inventories)
            for inventory in inventories:
                try:
                    p = proposals.materialize(calls,image,h,inventory,packet,dest,camera)
                    candidate = dest/'program_candidates'/p['id'];candidate.mkdir(parents=True,exist_ok=True)
                    write(candidate/'joint_program.json',p)
                    layout,numeric = geometry.solve_layout(h,p,packet,p['initial_placement'],
                        seed_for(config['seed'],observed['id']+'|program|'+p['id']),
                        starts=config['solver_starts'],maxiter=config['solver_iterations'],camera=camera)
                    write(candidate/'geometry_audit.json',numeric)
                    if any('aabb_size_m' not in row for row in layout.values()):
                        raise ValueError('Frozen dimensions do not fit hypothesis envelope; candidate excluded without resizing')
                    state = model.export_graph(observed,h,p,layout)
                    write(candidate/'G_full_candidate.json',state)
                    visibility = geometry.visibility_audit(state,camera['hfov'],list(image.size))
                    write(candidate/'visibility_audit.json',visibility)
                    print(json.dumps(dict(event='candidate_feasibility',scene=observed['id'],program=p['id'],
                        geometry_passed=numeric['passed'],visibility_passed=visibility['passed'],
                        geometry_issues=numeric['issues'],visibility_issues=visibility['issues']),ensure_ascii=False),flush=True)
                    make_board(state,image,camera['hfov'],candidate/'review_board.png')
                    if not numeric['passed'] or not visibility['passed']:
                        proposal_failures.append(dict(hypothesis=h['id'],program=p['id'],stage='candidate_feasibility',
                            geometry_issues=numeric['issues'],visibility_issues=visibility['issues']))
                        continue
                    p.update(compiled_layout=layout,geometry_audit=numeric,visibility_audit=visibility)
                    programs[h['id']].append(p)
                except Exception as exc:
                    proposal_failures.append(dict(hypothesis=h['id'],program=inventory['id'],error=str(exc)))
        except Exception as exc:
            proposal_failures.append(dict(hypothesis=h['id'], error=str(exc)))
    write(dest/'domain.json', dict(hypotheses=hypotheses, programs=programs, failures=proposal_failures,
        scope='image_specific_finite_executable_proxy_compatible_joint_program_domain; not_causal_identification',
        domain_filter='fixed_inventory_dimensions_hosts_regions_envelope; no_rejection_repair_or_resampling'))
    recognition = choices(calls, 'recognition', hypotheses, packet, config['draws'], config['seed'])
    conditional = {}
    for h in hypotheses:
        if programs[h['id']]:
            conditional[h['id']] = choices(calls, 'realization_'+h['id'], programs[h['id']], packet,
                config['draws'], config['seed'], h)
    write(dest/'mechanisms.json', dict(recognition=recognition, conditional=conditional,
        estimator='empirical_alias_permuted_choices_with_total_Dirichlet_mass_1',
        calibrated=False, invalid_draws_retained=True))
    scm = model.compile_model(hypotheses, programs, recognition['counts'],
        {h: v['counts'] for h, v in conditional.items()})
    write(dest/'model.json', scm)
    samples = model.sample_worlds(scm, config['worlds'], seed_for(config['seed'], observed['id']+'|worlds'))
    write(dest/'samples.json', samples)
    results = []
    for sample in samples:
        out = dest/('world_'+str(sample['index']).zfill(2));out.mkdir(exist_ok=True)
        if (out/'result.json').exists():results.append(read(out/'result.json'));continue
        write(out/'draw.json', sample)
        write(out/'causal_diagnostics.json', model.intervention_diagnostics(scm, sample))
        result = dict(index=sample['index'], hypothesis=sample['hypothesis'], program=sample['program'],
            status='failed', failures=[], observation_sha256=source_hash)
        try:
            if sample['program'] not in scm['programs']:
                raise ValueError('Selected unavailable program mechanism; failed sample retained')
            h = next(h for h in hypotheses if h['id'] == sample['hypothesis']);p = scm['programs'][sample['program']]
            write(out/'joint_program.json', p)
            placement = deepcopy(p['initial_placement'])
            write(out/'initial_placement.json', placement)
            layout,numeric = deepcopy(p['compiled_layout']),deepcopy(p['geometry_audit'])
            write(out/'geometry_audit.json', numeric)
            if any('aabb_size_m' not in row for row in layout.values()):
                raise ValueError('Frozen dimensions cannot fit envelope; proposal retained')
            state = model.export_graph(observed, h, p, layout);write(out/'G_full_candidate.json', state)
            visibility = geometry.visibility_audit(state, camera['hfov'], list(image.size))
            write(out/'visibility_audit.json', visibility)
            board = make_board(state, image, camera['hfov'], out/'review_board.png')
            if not numeric['passed']:result['failures'].append(dict(stage='geometry', issues=numeric['issues']))
            if not visibility['passed']:result['failures'].append(dict(stage='visibility', issues=visibility['issues']))
            if not result['failures']:
                semantic_program = {k:v for k,v in p.items() if k not in (
                    'initial_placement','compiled_layout','geometry_audit','visibility_audit')}
                prompt = ('Independently audit this proposed complete existing world against the ORIGINAL photograph '
                    'and frozen observations. The board also shows approximate boxes/top view, not hidden RGB. '
                    'Assess functional realizations, dimensions, room assumptions, shared instance identity, '
                    'and whether hidden additions contradict visible image regions. Do not reward more objects '
                    'or rear coverage; sparse/empty hidden inventory is legal. '
                    'Return {"compatible":true,"issues":[],"unresolved_assumptions":["..."]}. '
                    'Use compatible=false for concrete contradictions; list specific IDs/reasons. '
                    'Do not propose edits. Passing is an uncertain VLM judgment, not certified correctness. '
                    '\nHYPOTHESIS: '+json.dumps(h)+'\nPROGRAM: '+json.dumps(semantic_program)+
                    '\nSOLVED_LAYOUT: '+json.dumps(layout)+'\nOBSERVATIONS: '+json.dumps(packet))
                def validate(value):
                    if not isinstance(value, dict) or type(value.get('compatible')) is not bool:
                        raise ValueError('Explicit compatible boolean required')
                    model.strings(value.get('issues'), 'issues');model.strings(value.get('unresolved_assumptions'), 'unresolved_assumptions')
                    if value['compatible'] and value['issues']:raise ValueError('Compatible judgment contradicts issues')
                    if not value['compatible'] and not value['issues']:raise ValueError('Rejected world needs a concrete reason')
                    return value
                # Repeated IID draws of the exact same state share one audit;
                # never repeat a critic until a favorable verdict appears.
                critic = calls.ask('critic_'+p['id'], board, prompt, validate, tokens=2048)
                write(out/'qwen_compatibility.json', critic)
                if not critic['compatible']:result['failures'].append(dict(stage='qwen_compatibility', issues=critic['issues']))
            if not result['failures']:
                if sha(observed_path) != source_hash:raise ValueError('Observation mutated during completion')
                write(out/'G_full.json', state);result['status'] = 'accepted_proxy_compatible'
                result['new_objects'] = len(p['objects'])
            else:result['status'] = 'rejected'
        except Exception as exc:
            result['failures'].append(dict(stage='execution', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc()))
        write(out/'result.json', result);results.append(result)
    summary = dict(status='completed' if any(r['status'] == 'accepted_proxy_compatible' for r in results) else 'partial_or_failed',
        results=results, attempts=len(samples), accepted=sum(r['status'] == 'accepted_proxy_compatible' for r in results),
        rejected=sum(r['status']=='rejected' for r in results), failed=sum(r['status']=='failed' for r in results),
        all_draws_accepted=all(r['status']=='accepted_proxy_compatible' for r in results),
        mechanism_coverage=scm['mechanism_coverage'],
        distinct_sampled_programs=len({s['program'] for s in samples}),
        distinct_accepted_programs=len({r['program'] for r in results if r['status']=='accepted_proxy_compatible'}),
        proposal_failures=proposal_failures, observations_unchanged=sha(observed_path) == source_hash,
        probability_scope='law_conditioned_on_executable_proxy_domain; accepted_subset_is_selected_not_calibrated_posterior',
        API_used=False, training_performed=False, RGB_panorama_generated=False)
    write(dest/'summary.json', summary)
    return summary


def run_phase(root, phase, provider_factory=None):
    root = Path(root).resolve();config = verify(root);engine = None;results = []
    def provider():
        nonlocal engine
        if engine is None:
            if provider_factory:engine = provider_factory()
            else:
                from .world_generation import MatchedLocalJSON
                engine = MatchedLocalJSON(config['local_model'])
        return engine
    for s in config['scenes']:
        scene = root/'scenes'/s['id'];obs = scene/'observation'
        marker = scene/(phase+'_result.json')
        if marker.exists():results.append(read(marker));continue
        row = dict(scene=s['id'], phase=phase, status='failed')
        try:
            if phase == 'observer2d':
                target = obs/'qwen/observer2d/summary.json'
                if not target.exists():
                    if (obs/'recovery_source.json').exists():
                        observe2d_review_v2(obs,provider,seed_for(config['seed'],s['id']))
                    else:observe2d(obs, provider, config['seed'])
                if read(target)['status'] != 'completed':raise ValueError('Observer2D failed')
            elif phase == 'observer3d':
                if not (obs/'qwen/observer2d/G_2d.json').is_file():raise ValueError('Missing 2D observation')
                target = obs/'qwen/observer3d/summary.json'
                if not target.exists():
                    from .fresh_wilddet import run
                    from .box_observer import DEFAULT_REPO, DEFAULT_CHECKPOINT
                    run(obs, 'qwen', DEFAULT_REPO, DEFAULT_CHECKPOINT, check_experiment=check_observation)
                if read(target)['status'] != 'completed':raise ValueError('Observer3D failed')
                freeze_observations(obs)
            else:
                if not (obs/'qwen/observer3d/G_obs.json').is_file():raise ValueError('Missing frozen metric observation')
                row['result'] = complete_scene(scene, config, provider)
                if row['result']['status'] != 'completed':row['status'] = 'partial_or_failed'
            if row['status'] != 'partial_or_failed':row['status'] = 'completed'
        except Exception as exc:
            row.update(error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
        write(marker, row);results.append(row)
        progress = dict(scene=s['id'],phase=phase,status=row['status'])
        if row.get('result'):
            progress.update({k:row['result'][k] for k in ('accepted','attempts','rejected','failed','distinct_accepted_programs')})
        if row.get('error'):progress['error'] = row['error'][:800]
        print(json.dumps(progress, ensure_ascii=False), flush=True)
    return results


def render_snapshot(state_path, image_path, hfov, destination, notice):
    """Reuse project views; an interrupted render never becomes a complete page."""
    destination = Path(destination)
    candidates = [destination] + sorted(destination.parent.glob(destination.name+'_render_*'))
    for candidate in candidates:
        if (candidate/'index.html').is_file():return candidate, None
    index = 1
    while destination.with_name(destination.name+f'_render_{index:03d}').exists():index += 1
    staging = destination.with_name(destination.name+f'_render_{index:03d}')
    try:
        from .assembly_presentation_visual import render_world
        with Image.open(image_path) as im:render_world(read(state_path), im, hfov, staging)
        page = staging/'index.html'
        if not page.is_file():raise ValueError('Renderer did not finish index.html')
        banner = '<div style="padding:14px;background:#fff0cb">'+escape(notice)+'</div>'
        html = page.read_text()
        page.write_text(html.replace('<h1>', banner+'<h1>', 1) if '<h1>' in html else html+banner)
        if not destination.exists():staging.rename(destination);staging = destination
        return staging, None
    except Exception as exc:
        error = destination.with_name(destination.name+'_render_error.json')
        write(error, dict(status='render_failed', source=str(state_path), error=type(exc).__name__+': '+str(exc),
            incomplete_render_preserved=str(staging), model_called=False))
        return None, error


def report(root):
    root = Path(root).resolve();config = verify(root);cards = [];records = []
    for s in config['scenes']:
        scene = root/'scenes'/s['id'];obs = scene/'observation';comp = scene/'completion'
        record = dict(scene=s['id'], observed=None, worlds=[], phases={})
        if (obs/'qwen/observer3d/G_obs.json').exists():record['observed'] = len(read(obs/'qwen/observer3d/G_obs.json')['objects'])
        base = 'scenes/'+s['id']
        card = '<article><h2>'+escape(s['id'])+'</h2><div class="grid"><div><p>原局部 RGB</p><img src="'+base+'/observation/input.png"></div>'
        boxes = obs/'qwen/observer2d/boxes2d.png'
        if boxes.exists():card += '<div><p>可见实例二维框</p><img src="'+base+'/observation/qwen/observer2d/boxes2d.png"></div>'
        card += '</div>'
        for phase in ('observer2d', 'observer3d', 'complete'):
            p = scene/(phase+'_result.json')
            record['phases'][phase] = read(p)['status'] if p.exists() else 'not_run_or_interrupted'
            card += '<p>'+phase+': '+escape(record['phases'][phase])
            if p.exists():card += ' <a href="'+base+'/'+p.name+'">记录</a>'
            card += '</p>'
        observed_state = obs/'qwen/observer3d/G_obs.json'
        if observed_state.exists():
            views, error = render_snapshot(observed_state, obs/'input.png', s['hfov'], obs/'views',
                '仅可见观测 G_obs；周围未知空间留空。这里没有隐藏补全。')
            if views:card += '<p><a href="'+str((views/'index.html').relative_to(root))+'">仅观测：重投影 / 平面 / 6方向 / ERP / 交互3D</a></p>'
            if error:card += '<p><a href="'+str(error.relative_to(root))+'">观测可视化错误</a></p>'
        for name, label in (('hypotheses.json', '全局假设'), ('domain.json','候选域与排除记录'),
                            ('model.json', '模型与依赖'), ('summary.json', '全部采样及失败汇总')):
            if (comp/name).exists():card += '<a href="'+base+'/completion/'+name+'">'+label+'</a> · '
        for candidate in sorted((comp/'program_candidates').glob('*')):
            if not candidate.is_dir():continue
            url = str(candidate.relative_to(root));audit = candidate/'geometry_audit.json'
            numeric = read(audit) if audit.exists() else {}
            va = candidate/'visibility_audit.json';visibility = read(va) if va.exists() else {}
            card += '<details><summary>采样前候选 '+escape(candidate.name)+'（未获语义验收）：几何 '
            card += str(numeric.get('passed','未完成'))+'，可见性 '+str(visibility.get('passed','未完成'))+'</summary>'
            for name,label in (('joint_program.json','库存和固定尺寸'),('geometry_audit.json','几何约束'),
                               ('visibility_audit.json','原相机检查'),('G_full_candidate.json','候选状态')):
                if (candidate/name).exists():card += '<a href="'+url+'/'+name+'">'+label+'</a> · '
            if (candidate/'review_board.png').exists():card += '<img loading="lazy" class="board" src="'+url+'/review_board.png">'
            card += '</details>'
        world_results = (read(comp/'summary.json')['results'] if (comp/'summary.json').exists() else
            [read(p) for p in sorted(comp.glob('world_*/result.json'))])
        if (comp/'summary.json').exists():
            summary = read(comp/'summary.json')
            card += '<p><strong>通过 '+str(summary['accepted'])+' / '+str(summary['attempts'])+' 次固定预算采样</strong>；'
            card += '拒绝 '+str(summary.get('rejected',0))+'，执行失败 '+str(summary.get('failed',0))+'。</p>'
            card += '<p>不同候选程序：采到 '+str(summary.get('distinct_sampled_programs','未统计'))
            card += '，通过 '+str(summary.get('distinct_accepted_programs','未统计'))+'；相同程序的重复采样不算新假设。</p>'
            coverage = summary.get('mechanism_coverage', {})
            if coverage.get('excluded_hypotheses'):
                card += '<p>候选机制不完整：排除了 '+escape(', '.join(coverage['excluded_hypotheses']))
                card += '；其原始估计概率质量为 '+format(coverage['excluded_elicited_mass'],'.3f')
                card += '。采样显式条件于可执行候选域，不代表完整后验。</p>'
        if world_results:
            for r in world_results:
                out = comp/('world_'+str(r['index']).zfill(2));url = base+'/completion/'+out.name
                record['worlds'].append(r)
                accepted = r['status'] == 'accepted_proxy_compatible'
                card += '<div class="world"><h3>World '+str(r['index'])+': '+escape(r['status'])+'</h3><p><a href="'+url+'/result.json">结果/失败</a>'
                if (out/'review_board.png').exists():card += ' · <a href="'+url+'/review_board.png">原图/重投影/平面图</a>'
                state_path = out/'G_full.json' if accepted else out/'G_full_candidate.json'
                if state_path.exists():
                    label = '已通过世界' if accepted else '未通过候选（仅诊断）'
                    views, error = render_snapshot(state_path, obs/'input.png', s['hfov'], out/'views', label+'；框代理与 ERP 状态投影，不是生成的隐藏 RGB。')
                    card += ' · <a href="'+url+'/'+state_path.name+'">'+label+' JSON</a>'
                    if views:card += ' · <a href="'+str((views/'index.html').relative_to(root))+'">'+label+'：完整状态可视化 / 交互3D</a>'
                    if error:card += ' · <a href="'+str(error.relative_to(root))+'">可视化错误</a>'
                    state = read(state_path)
                    card += '</p><details><summary>假设、功能与依赖关系</summary><pre>'+escape(json.dumps(dict(
                        hypothesis=state.get('hypothesis'), activities=state.get('activities'), relations=state.get('relations'),
                        supports=[dict(id=o['track_id'], support=o['support']) for o in state['objects'] if 'support' in o]),
                        ensure_ascii=False, indent=2))+'</pre></details>'
                else:card += '</p>'
                if r.get('failures'):card += '<details><summary>失败原因</summary><pre>'+escape(json.dumps(r['failures'], ensure_ascii=False, indent=2))+'</pre></details>'
                if (out/'review_board.png').exists():card += '<img loading="lazy" class="board" src="'+url+'/review_board.png">'
                card += '</div>'
        records.append(record);cards.append(card+'</article>')
    write(root/'results.json', dict(protocol=model.PROTOCOL, records=records, API_used=False,
        semantic_quality_certified=False, RGB_panorama_generated=False))
    page = '<!doctype html><meta charset="utf-8"><title>Global world completion</title><style>body{font:16px/1.6 system-ui;margin:24px}article{border-top:1px solid #bbb;padding:14px}img{max-width:100%}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.grid img{max-height:540px}.world{border:1px solid #ddd;padding:12px;margin:16px 0}.board{width:100%;max-width:1100px}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>'
    page += '<h1>局部 RGB → 多假设完整世界状态</h1><p>仅本地 Qwen3-VL + WildDet3D。蓝色为保留观测；新增为假设。通过仅表示框代理约束、可见性检查与 Qwen 审查通过，不是真值或物理认证。ERP 是状态投影，没有生成隐藏 RGB。所有失败和重复采样均保留。</p><p><a href="suite.json">配置与源码哈希</a> · <a href="results.json">全部结果</a></p>'
    (root/'index.html').write_text(page+''.join(cards))
    return root/'index.html'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'check', 'observer2d', 'observer3d', 'complete', 'report'))
    p.add_argument('--root', type=Path, required=True)
    source = p.add_mutually_exclusive_group();source.add_argument('--inputs', type=Path);source.add_argument('--image', type=Path)
    p.add_argument('--hfov', type=float, default=90.);p.add_argument('--local-model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--worlds', type=int, default=6);p.add_argument('--hypotheses', type=int, default=3)
    p.add_argument('--alternatives', type=int, default=2);p.add_argument('--draws', type=int, default=4)
    p.add_argument('--maximum-objects', type=int, default=48);p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--solver-starts', type=int, default=3);p.add_argument('--solver-iterations', type=int, default=180)
    p.add_argument('--observations-from', type=Path)
    p.add_argument('--recover-from', type=Path, help='Create a NEW run from saved 2D reviews/draft failures; never reuse old 3D')
    a = p.parse_args()
    try:
        if a.action == 'prepare':
            prepare(a.root, a.inputs, a.image, a.hfov, a.local_model, a.worlds, a.hypotheses,
                a.alternatives, a.draws, a.seed, a.maximum_objects, a.observations_from,
                a.solver_starts, a.solver_iterations, a.recover_from)
            print(str(a.root));return
        if a.action == 'check':verify(a.root);print('Input/source bindings verified; no inference');return
        if a.action == 'report':print(report(a.root));return
        results = run_phase(a.root, a.action)
        if any(r['status'] != 'completed' for r in results):sys.exit(1)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:p.exit(2, str(exc)+'\n')


if __name__ == '__main__':main()
