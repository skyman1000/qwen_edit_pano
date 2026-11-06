"""CPU contract and integration tests; no actual Qwen/WildDet3D/API execution."""
from copy import deepcopy
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ..common import field, read, sha, write
from .geometry import graph
from . import global_world as pipeline, global_world_model as model, global_world_geometry as geometry
from .world_scm import forward


def observation():
    obj = dict(track_id='obs_000', raw_category='cabinet', description='opaque cabinet',
        category_id=field(7, 'model_prediction'), center_local_m=field([0., -1., 3.], 'model_prediction'),
        size_aabb_local_m=field([1., 1., 1.], 'model_prediction'),
        bbox2d_visible_xyxy_norm=field([.4, .55, .6, .9], 'model_prediction'),
        in_primary_room=field(None, reason='unknown'), local_evidence='visible')
    g = graph('test_scene', 'test', 'observed', [obj], 'test-vocabulary')
    g['scene_gt_used'] = False
    return g


def hypothesis():
    return dict(id='h0', description='mixed use', assumptions=['level-camera approximation'],
        space=dict(extra_clearance_m=dict(left=4., right=4., rear=5., front=3., below=0., above=1.)),
        functions=['storage'], room=dict(x_min=-5., x_max=5., z_min=-5., z_max=6., floor_y=-1.5, ceiling_y=1.5))


def program(empty=False):
    return dict(id='h0_b0', hypothesis='h0', description='joint optional inventory', assumptions=[],
        objects=[] if empty else [dict(id='n0', category='cabinet', variant='opaque storage cabinet',
            size_m=[1., 1., 1.], support=dict(kind='floor', host=None), visibility='outside_fov',
            opaque=True, reason='a distinct storage instance')],
        activities=[dict(function='storage', uses=['obs_000'] if empty else ['obs_000', 'n0'], realization='shared storage')],
        relations=[], access_zones=[])


def packet(g=None):
    return model.observation_packet(g or observation(), dict(instances=[], evidence_description='one cabinet'))


def parsed_program(p):
    return model.programs_answer(dict(programs=[p]), hypothesis(), packet(), 2, 48)[0]


class GlobalModelTests(unittest.TestCase):
    def test_empty_additions_and_shared_observed_instances_are_legal(self):
        p = parsed_program(program(True))
        p['activities'].append(dict(function='work', uses=['obs_000'], realization='shared existing surface'))
        p = parsed_program(p)
        m = model.compile_model([hypothesis()], {'h0': [p]}, {'h0': 4}, {'h0': {p['id']: 4}})
        state = model.export_graph(observation(), hypothesis(), p, {})
        self.assertEqual(state['objects'], observation()['objects'])
        self.assertEqual(len(model.sample_worlds(m, 4, 8)), 4)

    def test_unknown_labels_are_preserved_without_catchall(self):
        p = program();p['objects'][0]['category'] = 'never_seen_fixture_label_123'
        pos = {'n0': dict(center_m=[0, -1, -3], yaw_deg=0, aabb_size_m=[1, 1, 1])}
        g = model.export_graph(observation(), hypothesis(), p, pos)
        self.assertIsNone(g['objects'][-1]['category_id']['value'])
        self.assertEqual(g['objects'][-1]['raw_category'], 'never_seen_fixture_label_123')

    def test_cycles_duplicate_ids_and_bad_references_rejected(self):
        p = program();q = deepcopy(p['objects'][0]);q['id'] = 'n1';p['objects'].append(q)
        p['objects'][0]['support'] = dict(kind='top', host='n1')
        p['objects'][1]['support'] = dict(kind='top', host='n0')
        with self.assertRaisesRegex(ValueError, 'cycle'):parsed_program(p)
        p = program();p['objects'] *= 2
        with self.assertRaises(ValueError):parsed_program(p)
        p = program();p['activities'][0]['uses'] = ['missing']
        with self.assertRaises(ValueError):parsed_program(p)

    def test_relative_space_compiler_contains_observations_without_absolute_guessing(self):
        h = hypothesis();h['room']['z_max'] = 2  # Old numeric example is not consumed.
        parsed = model.hypotheses_answer(dict(hypotheses=[h]), packet(), 3)[0]
        self.assertGreaterEqual(parsed['room']['z_max'], 3.5)
        self.assertTrue(parsed['spatial_audit']['numeric_containment_compiled'])
        self.assertFalse(parsed['spatial_audit']['walls_measured'])
        h['space']['extra_clearance_m']['left'] = -1
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            model.hypotheses_answer(dict(hypotheses=[h]), packet(), 3)

    def test_host_intervention_changes_installation_not_child_ownership(self):
        p = program();child = deepcopy(p['objects'][0]);child.update(id='n1', size_m=[.1, .1, .1], category='book')
        child['support'] = dict(kind='top', host='n0');p['objects'].append(child)
        p = parsed_program(p)
        m = model.compile_model([hypothesis()], {'h0': [p]}, {'h0': 4}, {'h0': {p['id']: 4}})
        noise = {n['id']: .3 for n in m['nodes']}
        factual = forward(m, noise)
        after = forward(m, noise, {'E_h0_b0__n0': '0'})
        self.assertEqual(factual['S_h0_b0__n1'], '1')
        self.assertEqual(after['E_h0_b0__n1'], '1')
        self.assertEqual(after['S_h0_b0__n1'], '0')

    def test_global_hypothesis_selects_joint_program_and_operational_failure_is_not_a_world(self):
        h0 = hypothesis();h1 = deepcopy(h0);h1['id'] = 'h1'
        p0 = parsed_program(program());p1 = deepcopy(p0);p1.update(id='h1_b0', hypothesis='h1')
        m = model.compile_model([h0, h1], {'h0': [p0], 'h1': [p1]}, {'h0': 2, 'h1': 2},
            {'h0': {p0['id']: 4}, 'h1': {p1['id']: 4}})
        noise = {n['id']: .3 for n in m['nodes']}
        self.assertEqual(forward(m, noise, {'H': 'h0'})['B'], p0['id'])
        self.assertEqual(forward(m, noise, {'H': 'h1'})['B'], p1['id'])
        with self.assertRaisesRegex(ValueError,'No executable'):
            model.compile_model([h0], {'h0': []}, {'h0': 4}, {})
        with self.assertRaises(ValueError):model.compile_model([h0], {'h0': [p0]}, {}, {'h0': {p0['id']: 4}})

    def test_observation_rejects_hidden_gt_or_incomplete_metric_state(self):
        for change in ('hidden', 'gt', 'missing'):
            g = observation()
            if change == 'hidden':g['scope'] = 'full'
            elif change == 'gt':g['scene_gt_used'] = True
            else:g['objects'][0]['center_local_m'] = field(None, reason='missing')
            with self.assertRaises(ValueError):packet(g)


class GeometryTests(unittest.TestCase):
    def test_joint_collision_solver_moves_centres_preserving_sizes_yaw_and_observations(self):
        p = program();other = deepcopy(p['objects'][0]);other['id'] = 'n1';p['objects'].append(other)
        initial = {i: dict(center_m=[0, -1, -3], yaw_deg=0.) for i in ('n0', 'n1')}
        original = deepcopy(initial);obs = packet();before = deepcopy(obs)
        layout, audit = geometry.solve_layout(hypothesis(), p, obs, initial, 11, starts=3, maxiter=120)
        self.assertTrue(audit['passed'], audit)
        self.assertGreaterEqual(max(abs(np.array(layout['n0']['center_m'])-layout['n1']['center_m'])), .97)
        self.assertEqual(initial, original);self.assertEqual(obs, before)
        self.assertEqual(layout['n0']['aabb_size_m'], [1, 1, 1])
        self.assertEqual(layout['n0']['yaw_deg'], 0.)

    def test_top_contact_and_inside_containment(self):
        p = program();p['objects'][0].update(size_m=[.2, .2, .2], support=dict(kind='top', host='obs_000'))
        initial = {'n0': dict(center_m=[0, 1, 3], yaw_deg=90.)}
        layout, audit = geometry.solve_layout(hypothesis(), p, packet(), initial, 1)
        self.assertTrue(audit['passed'], audit)
        self.assertAlmostEqual(layout['n0']['center_m'][1], -.4, delta=.031)
        p['objects'][0]['support']['kind'] = 'inside'
        initial['n0']['center_m'] = [0, -1, 3]
        _, audit = geometry.solve_layout(hypothesis(), p, packet(), initial, 1)
        self.assertTrue(audit['passed'], audit)

    def test_impossible_frozen_dimensions_are_failure_not_shrunk(self):
        p = program();p['objects'][0]['size_m'] = [20., 1., 1.]
        initial = {'n0': dict(center_m=[0, -1, -3], yaw_deg=0.)}
        _, audit = geometry.solve_layout(hypothesis(), p, packet(), initial, 1)
        self.assertFalse(audit['passed']);self.assertEqual(p['objects'][0]['size_m'][0], 20.)

    def test_yaw_rotated_dimensions_and_placement_identity_contract(self):
        self.assertTrue(np.allclose(geometry.aabb_size([2, 1, .5], 90), [.5, 1, 2]))
        with self.assertRaises(ValueError):geometry.placement_answer(dict(placements=[]), program())

    def test_access_zone_conflict_is_a_real_constraint(self):
        p = program();p['objects'][0]['support'] = dict(kind='free', host=None)
        # Zone is too large for the envelope irrespective of placement.
        p['access_zones'] = [dict(owner='n0', offset_m=[0, 0, 0], size_m=[30, 1, 1], purpose='fixture')]
        _, audit = geometry.solve_layout(hypothesis(), p, packet(),
            {'n0': dict(center_m=[0, -1, -3], yaw_deg=0.)}, 1, starts=1, maxiter=20)
        self.assertFalse(audit['passed'])
        self.assertTrue(any('access_' in v for v in audit['issues']))

    def state_at(self, centre, visibility='occluded', raw='cabinet'):
        g = observation();g['objects'][0]['center_local_m']['value'] = [0., 0., 3.]
        g['objects'][0]['size_aabb_local_m']['value'] = [1., 2., .5]
        g['objects'][0]['raw_category'] = raw;g['objects'][0]['description'] = raw
        p = program();p['objects'][0]['size_m'] = [.3, .3, .3];p['objects'][0]['visibility'] = visibility
        return model.export_graph(g, hypothesis(), p,
            {'n0': dict(center_m=centre, yaw_deg=0, aabb_size_m=[.3, .3, .3])})

    def test_occlusion_behind_opaque_box_and_transparent_counterexample(self):
        a = geometry.visibility_audit(self.state_at([0, 0, 4]), 90, [1024, 1024])
        self.assertTrue(a['passed'], a)
        b = geometry.visibility_audit(self.state_at([0, 0, 4], raw='glass window'), 90, [1024, 1024])
        self.assertFalse(b['passed'], b)

    def test_visible_addition_cannot_be_called_hidden(self):
        a = geometry.visibility_audit(self.state_at([0, 0, 2]), 90, [1024, 1024])
        self.assertFalse(a['passed'])
        b = geometry.visibility_audit(self.state_at([0, 0, -3], 'outside_fov'), 90, [1024, 1024])
        self.assertTrue(b['passed'], b)

    def test_near_plane_and_parallel_rays(self):
        self.assertTrue(geometry.intersects_frustum([0, 0, .01], [1, 1, 1], 90, [1024, 1024]))
        hits = geometry.ray_hits([[0, 0, 3], [2, 0, 3]], [[1, 1, 1], [1, 1, 1]], np.array([[0., 0., 1.]]))
        self.assertAlmostEqual(hits[0, 0], 2.5);self.assertTrue(np.isinf(hits[1, 0]))


class FixtureProvider:
    def __init__(self, reject=False):self.calls = [];self.reject = reject
    def ask(self, image, prompt, path, **kwargs):
        self.calls.append(dict(prompt=prompt, image=image is not None, path=str(path)))
        if prompt.startswith('Propose alternative GLOBAL'):
            value = dict(hypotheses=[hypothesis()])
        elif prompt.startswith('Infer JOINT inventories'):
            value = dict(programs=[program()])
        elif prompt.startswith('Estimate intrinsic FULL'):
            value = dict(objects=[dict(id='n0',size_m=[1,1,1],support='floor',opaque=True)])
        elif prompt.startswith('Plan relative placement'):
            value = dict(objects=[dict(id='n0',region='rear',yaw_deg=0,anchor='camera',offset_xz_m=[0,-3])])
        elif prompt.startswith('Specify ONLY necessary'):
            value = dict(relations=[],access_zones=[])
        elif '\nOPTIONS: ' in prompt:
            options = json.loads(prompt.split('\nOPTIONS: ')[1].split('\nSCHEMA CORRECTION:')[0])
            value = dict(choice=options[0]['choice'])
        elif prompt.startswith('Jointly place ALL'):
            value = dict(placements=[dict(id='n0', center_m=[0, -1, -3], yaw_deg=0.)])
        elif prompt.startswith('Independently audit'):
            value = dict(compatible=not self.reject, issues=['fixture semantic contradiction'] if self.reject else [], unresolved_assumptions=['unknown hidden room'])
        else:raise AssertionError(prompt[:100])
        write(Path(path).with_suffix('.json'), value);Path(path).write_text(json.dumps(value))
        return value


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base = Path(self.tmp.name)

    def scene(self):
        scene = self.base/'scenes/test_scene';obs = scene/'observation';obs.mkdir(parents=True)
        Image.new('RGB', (128, 128), 'white').save(obs/'input.png')
        write(obs/'experiment.json', dict(id='test_scene', hfov=90, image_size=[128, 128]))
        for name in ('observer2d', 'observer3d'):(obs/'qwen'/name).mkdir(parents=True)
        g = observation();write(obs/'qwen/observer2d/G_2d.json', g);write(obs/'qwen/observer3d/G_obs.json', g)
        write(obs/'qwen/observer2d/grounding.json', dict(instances=[], evidence_description='one cabinet'))
        return scene

    def config(self):
        return dict(seed=3, worlds=2, hypotheses=2, alternatives=2, draws=2, maximum_objects=48, solver_starts=1, solver_iterations=30)

    @staticmethod
    def board(state, image, hfov, path):
        im = Image.new('RGB', (128, 128), 'white');im.save(path);return im

    def test_pipeline_frozen_observations_multiple_worlds_and_resume_no_new_calls(self):
        scene = self.scene();p = FixtureProvider();original = sha(scene/'observation/qwen/observer3d/G_obs.json')
        with patch.object(pipeline, 'make_board', self.board):
            summary = pipeline.complete_scene(scene, self.config(), lambda: p)
            self.assertEqual(summary['accepted'], 2, summary)
            self.assertEqual(sum(c['prompt'].startswith('Independently audit') for c in p.calls),1)
            count = len(p.calls)
            summary2 = pipeline.complete_scene(scene, self.config(), lambda: p)
        self.assertEqual(summary, summary2);self.assertEqual(len(p.calls), count)
        for f in (scene/'completion').glob('world_*/G_full.json'):
            self.assertEqual(read(f)['objects'][:1], observation()['objects'])
        self.assertEqual(sha(scene/'completion/G_obs.json'), original)
        self.assertEqual(sha(scene/'observation/qwen/observer3d/G_obs.json'), original)

    def test_rejection_retains_candidate_and_does_not_resample(self):
        scene = self.scene();p = FixtureProvider(reject=True)
        with patch.object(pipeline, 'make_board', self.board):summary = pipeline.complete_scene(scene, self.config(), lambda: p)
        self.assertEqual(summary['accepted'], 0)
        self.assertEqual(len(list((scene/'completion').glob('world_*/G_full_candidate.json'))), 2)
        self.assertFalse(list((scene/'completion').glob('world_*/G_full.json')))
        self.assertEqual(len(read(scene/'completion/samples.json')), 2)

    def test_all_infeasible_proposals_stop_before_sampling_and_keep_failure_evidence(self):
        scene = self.scene()
        class Oversized(FixtureProvider):
            def ask(self,image,prompt,path,**kwargs):
                value = super().ask(image,prompt,path,**kwargs)
                if prompt.startswith('Estimate intrinsic FULL'):
                    value['objects'][0]['size_m']=[100,1,1]
                    write(Path(path).with_suffix('.json'),value);Path(path).write_text(json.dumps(value))
                return value
        provider = Oversized()
        with patch.object(pipeline,'make_board',self.board):
            with self.assertRaisesRegex(ValueError,'No executable hypothesis'):
                pipeline.complete_scene(scene,self.config(),lambda:provider)
        comp = scene/'completion'
        self.assertEqual(read(comp/'domain.json')['programs'],{'h0':[]})
        self.assertFalse((comp/'samples.json').exists())
        self.assertTrue(read(comp/'domain.json')['failures'])
        self.assertFalse(read(comp/'program_candidates/h0_b0/geometry_audit.json')['passed'])
        self.assertEqual(read(comp/'program_candidates/h0_b0/joint_program.json')['objects'][0]['size_m'],[100,1,1])

    def test_frozen_observation_tampering_is_detected(self):
        scene = self.scene();obs = scene/'observation';pipeline.freeze_observations(obs)
        g = read(obs/'qwen/observer3d/G_obs.json');g['objects'][0]['description'] = 'changed'
        write(obs/'qwen/observer3d/G_obs.json', g)
        with self.assertRaises(ValueError):pipeline.freeze_observations(obs)

    def test_request_failure_and_schema_repair_have_fixed_budget(self):
        class Bad:
            count = 0
            def ask(self, image, prompt, path, **kw):self.count += 1;return {}
        p = Bad();calls = pipeline.Calls(lambda: p, self.base/'calls', 1)
        def invalid(v):raise ValueError('bad schema')
        for _ in range(2):
            with self.assertRaises(ValueError):calls.ask('bad', None, 'question', invalid)
        self.assertEqual(p.count, 2)

    def test_default_observation_uses_original_stage_seed_and_does_not_replace_inventory(self):
        obs = self.base/'observation';obs.mkdir();Image.new('RGB', (128, 128), 'white').save(obs/'input.png')
        write(obs/'experiment.json', dict(id='test_scene', hfov=90, image_size=[128, 128]))
        class Observer:
            calls = 0
            def ask(self, image, prompt, path, **kw):
                self.calls += 1
                self.seed=kw['seed']
                if 'DRAFT_CONTEXT:' in prompt:raise AssertionError('No new whole-image replacement review')
                value = dict(objects=[dict(name='storage cabinet', category='cabinet',
                    bbox_1000=[100, 100, 500, 700], on_floor=True)], evidence_description='one cabinet')
                if 'DRAFT_CONTEXT:' in prompt:value['review_status'] = 'complete'
                else:value['done'] = True
                write(Path(path).with_suffix('.json'), value);Path(path).write_text(json.dumps(value))
                return value
        p = Observer();pipeline.observe2d(obs, lambda: p, 3)
        self.assertEqual(p.calls, 1)
        from .assembly_experiment import stage_seed
        self.assertEqual(p.seed,stage_seed(3,'shared_observer'))
        self.assertEqual(len(read(obs/'qwen/observer2d/G_2d.json')['objects']), 1)
        self.assertEqual(read(obs/'qwen/observer2d/summary.json')['request_interface'],'incremental_visible_chunks_lossless_v2')
        self.assertFalse((obs/'draft').exists());self.assertFalse((obs/'review_calls').exists())

    def test_imported_observation_is_bound_to_same_pixels_and_never_recomputed(self):
        source = self.base/'source';src = source/'scenes/test_scene/observation';src.mkdir(parents=True)
        image = self.base/'test_scene.png';Image.new('RGB', (128, 128), 'white').save(image)
        shutil = __import__('shutil');shutil.copyfile(image, src/'input.png')
        write(source/'suite.json', dict(scenes=[dict(id='test_scene', hfov=90)]))
        for name in ('observer2d', 'observer3d'):
            (src/'qwen'/name).mkdir(parents=True)
            write(src/'qwen'/name/'summary.json', dict(status='completed'))
        g = observation();write(src/'qwen/observer2d/G_2d.json', g);write(src/'qwen/observer3d/G_obs.json', g)
        write(src/'qwen/observer2d/grounding.json', dict(instances=[], evidence_description='cabinet'))
        local = self.base/'qwen';local.mkdir();(local/'config.json').write_text('{}');(local/'one.safetensors').write_text('fixture')
        root = self.base/'imported'
        with patch.object(pipeline, 'new_output', side_effect=lambda p: (Path(p).mkdir(parents=True), Path(p))[1]):
            pipeline.prepare(root, image=image, local_model=local, observations_from=source)
        pipeline.verify(root)
        for phase in ('observer2d', 'observer3d'):
            result = pipeline.run_phase(root, phase, lambda: self.fail('must not load model'))
            self.assertEqual(result[0]['status'], 'completed')
        self.assertEqual(sha(src/'qwen/observer3d/G_obs.json'), sha(root/'scenes/test_scene/observation/qwen/observer3d/G_obs.json'))

    def test_interrupted_saved_response_recovered_without_provider(self):
        calls = pipeline.Calls(lambda: self.fail('provider must not be called'), self.base/'calls', 1)
        prompt = 'question';fingerprint = __import__('hashlib').sha256((prompt+'|8192').encode()).hexdigest()
        write(calls.out/'saved.call.json', dict(fingerprint=fingerprint, status='pending', attempts=[dict(attempt=0, seed=1)]))
        write(calls.out/'saved.try0.json', dict(ok=True))
        self.assertEqual(calls.ask('saved', None, prompt, lambda v: v), dict(ok=True))

    def test_prepare_and_run_phase_are_local_only_and_report_links_exist(self):
        image = self.base/'test_scene.png';Image.new('RGB', (128, 128), 'white').save(image)
        local = self.base/'qwen';local.mkdir();(local/'config.json').write_text('{}');(local/'one.safetensors').write_text('fixture')
        root = self.base/'run'
        with patch.object(pipeline, 'new_output', side_effect=lambda p: (Path(p).mkdir(parents=True), Path(p))[1]):
            pipeline.prepare(root, image=image, local_model=local, worlds=2, hypotheses=2, alternatives=2, draws=2)
        self.assertFalse(read(root/'suite.json')['API_used']);pipeline.verify(root)
        scene = root/'scenes/test_scene';obs = scene/'observation'
        for name in ('observer2d', 'observer3d'):(obs/'qwen'/name).mkdir(parents=True)
        g = observation()
        write(obs/'qwen/observer2d/G_2d.json', g);write(obs/'qwen/observer3d/G_obs.json', g)
        write(obs/'qwen/observer2d/grounding.json', dict(instances=[], evidence_description='one cabinet'))
        for name in ('observer2d', 'observer3d'):write(obs/'qwen'/name/'summary.json', dict(status='completed'))
        for phase in ('observer2d', 'observer3d'):pipeline.run_phase(root, phase, lambda: self.fail('observer already exists'))
        p = FixtureProvider()
        with patch.object(pipeline, 'make_board', self.board):
            results = pipeline.run_phase(root, 'complete', lambda: p)
        self.assertEqual(results[0]['status'], 'completed', results)
        def render(state, image, hfov, out):out.mkdir();(out/'index.html').write_text('fixture view')
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_presentation_visual.render_world', render):pipeline.report(root)
        broken = []
        class Links(HTMLParser):
            def handle_starttag(self, tag, attrs):
                for k, v in attrs:
                    if k in ('href', 'src') and not (root/v).exists():broken.append(v)
        Links().feed((root/'index.html').read_text());self.assertEqual(broken, [])

    def test_report_renders_rejected_candidates_without_publishing_full_worlds(self):
        scene = self.scene();provider = FixtureProvider(reject=True)
        with patch.object(pipeline, 'make_board', self.board):
            pipeline.complete_scene(scene, self.config(), lambda: provider)
        original = read(scene/'completion/summary.json')
        def render(state, image, hfov, out):out.mkdir();(out/'index.html').write_text('fixture view')
        with patch.object(pipeline, 'verify', return_value=dict(scenes=[dict(id='test_scene', hfov=90)])), \
             patch('qwen_edit_pano.world_pipeline.completion.assembly_presentation_visual.render_world', render):
            pipeline.report(self.base)
        page = (self.base/'index.html').read_text()
        self.assertIn('未通过候选（仅诊断）', page)
        self.assertTrue((scene/'observation/views/index.html').exists())
        self.assertEqual(len(list((scene/'completion').glob('world_*/views/index.html'))), 2)
        self.assertFalse(list((scene/'completion').glob('world_*/G_full.json')))
        self.assertEqual(read(scene/'completion/summary.json'), original)

    def test_incomplete_render_is_preserved_and_report_can_resume(self):
        dest = self.base/'views';state_path = self.base/'state.json';write(state_path, observation())
        image = self.base/'input.png';Image.new('RGB', (128, 128), 'white').save(image)
        def partial(state, im, hfov, out):out.mkdir();raise RuntimeError('fixture interruption')
        def complete(state, im, hfov, out):out.mkdir();(out/'index.html').write_text('finished')
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_presentation_visual.render_world', partial):
            views, error = pipeline.render_snapshot(state_path, image, 90, dest, 'fixture')
        self.assertIsNone(views);self.assertTrue(error.exists())
        self.assertTrue((self.base/'views_render_001').exists())
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_presentation_visual.render_world', complete):
            views, error = pipeline.render_snapshot(state_path, image, 90, dest, 'fixture')
        self.assertEqual(views, dest);self.assertIsNone(error)
        self.assertTrue((self.base/'views_render_001').exists())

    def test_local_srun_worker_does_not_submit_nested_jobs_and_still_reports_partial(self):
        fake = self.base/'python_fixture'
        fake.write_text('#!'+sys.executable+'\nimport json,os,sys\nfrom pathlib import Path\na=sys.argv[1:]\naction=a[a.index("-m")+2]\nroot=Path(a[a.index("--root")+1])\nwith open(os.environ["GW_FIXTURE_TRACE"],"a") as f:f.write(action+"\\n")\nif action=="prepare":root.mkdir(parents=True)\nif action=="report":(root/"index.html").write_text("fixture")\nif action=="complete" and os.environ.get("GW_FIXTURE_PARTIAL")=="1":sys.exit(1)\n')
        fake.chmod(0o755)
        srun = self.base/'srun';srun.write_text('#!/bin/sh\nexit 77\n');srun.chmod(0o755)
        from ..common import ROOT
        for partial in (False, True):
            root = self.base/('run_partial' if partial else 'run_ok');trace = self.base/(root.name+'.log')
            env = dict(os.environ, CAUPANO_QWEN_PYTHON=str(fake), CAUPANO_WILDDET_PYTHON=str(fake),
                GW_FIXTURE_TRACE=str(trace), GW_FIXTURE_PARTIAL=str(int(partial)), PATH=str(self.base)+os.pathsep+os.environ['PATH'])
            result = subprocess.run(['bash', str(ROOT/'qwen_edit_pano/scripts/global_world.sh'), 'local', str(root)], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, int(partial), result.stderr+result.stdout)
            self.assertEqual(trace.read_text().splitlines(), ['prepare', 'observer2d', 'observer3d', 'complete', 'report'])
            self.assertTrue((root/'index.html').exists())


if __name__ == '__main__':unittest.main()
