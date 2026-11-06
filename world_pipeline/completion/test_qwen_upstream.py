"""CPU contracts and fake-model workflow; no downloads, GPU or model weights."""
import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from ..common import read, write, write_rows
from .geometry import graph
from .qwen_upstream_core import (assemble, compact_observations, hidden_prompt,
                                 instances_from_response, observed_object, grounded_instances, observed_without_3d)
from .qwen_upstream_review import match_metrics
from . import qwen_upstream as pipeline


INSTANCE = dict(id='obs_0000', category='chair', bbox_2d=[200, 200, 800, 800], description='center chair')
BOX = {'bbox_3d': [1., .5, 3., 2., 1., .5, 0., .5, 0.]}
PROPOSAL = {'hidden_objects': [dict(category='table', center_local_m=[-1., 0., -2.],
                                   size_aabb_local_m=[2., 1., 1.]) ]}


def observed():
    obj, _ = observed_object(INSTANCE, BOX)
    return graph('scene', 'val', 'observed', [obj], 'test-vocab')


class UpstreamTests(unittest.TestCase):
    def test_view_diagnostic_and_ambiguous_instances(self):
        from .qwen_upstream_review import view_diagnostics
        state = observed()
        self.assertEqual(view_diagnostics(state, 90, (100, 100))['centers_in_input_fov'], 1)
        state['objects'][0]['center_local_m']['value'][2] = -3
        self.assertEqual(view_diagnostics(state, 90, (100, 100))['centers_in_input_fov'], 0)
        with self.assertRaisesRegex(ValueError, 'ambiguous identity'):
            pipeline.observed_object(INSTANCE, [BOX, BOX])

    def test_generated_presence_penalty(self):
        import torch
        tokens = torch.tensor([[1, 2, 3, 3], [2, 3, 1, 1]])
        scores = torch.zeros(2, 5)
        actual = pipeline.GeneratedPresencePenalty(2, 1.5)(tokens, scores)
        self.assertEqual(actual.tolist(), [[0., 0., 0., -1.5, 0.], [0., -1.5, 0., 0., 0.]])
        self.assertEqual(float(scores.sum()), 0.)

    def test_single_retry_and_unknown_geometry(self):
        class Model:
            calls = 0
            def ask(self, *args, **kwargs):
                self.calls += 1
                return {'bbox_3d': [.15, .07]}
        model = Model()
        with self.assertRaises(ValueError):
            pipeline.observe_query(model, None, 'prompt', Path('/tmp/not_written.txt'),
                                   lambda r: observed_object(INSTANCE, r), 512)
        self.assertEqual(model.calls, 2)
        obj = observed_without_3d(INSTANCE, 'incomplete_model_response')
        state = graph('scene', 'val', 'observed', [obj], 'test-vocab')
        from ..oracle_model import tensorize
        _, geometry, mask = tensorize(state)
        self.assertEqual(geometry[0, 1].tolist(), [0.]*8)
        self.assertTrue(mask[0, 1])
        self.assertIsNone(obj['center_local_m']['value'])
        self.assertEqual(match_metrics(state, observed())['matches'], 0)
        self.assertEqual(match_metrics(state, observed(), True)['matches'], 1)
        self.assertEqual(match_metrics(state, observed(), True)['matched_center_errors_m'], [])
    def test_observer_generation_profile_and_cache(self):
        import torch
        from types import SimpleNamespace
        class Batch(dict):
            def to(self, device):
                return self
            @property
            def input_ids(self):
                return self['input_ids']
        message_log, encoding_log = [], []
        class Processor:
            def apply_chat_template(self, *args, **kwargs):
                message_log.append(deepcopy(args[0]))
                return 'prompt'
            def __call__(self, **kwargs):
                encoding_log.append(kwargs)
                return Batch(input_ids=torch.tensor([[1]]))
            def batch_decode(self, *args, **kwargs):
                return ['{"boxes":[]}']
        calls = []
        def generate(**kwargs):
            calls.append(kwargs)
            return torch.tensor([[1, 2, 3]])
        model = pipeline.Qwen.__new__(pipeline.Qwen)
        model.torch, model.processor = torch, Processor()
        model.model = SimpleNamespace(device='cpu', generate=generate)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'query.txt'
            for _ in range(2):
                self.assertEqual(model.ask(None, 'test', path, profile='observer'), {'boxes': []})
            self.assertEqual(len(calls), 1)
            self.assertNotIn('images', encoding_log[-1])
            self.assertEqual([c['type'] for c in message_log[-1][-1]['content']], ['text'])
            self.assertTrue(calls[0]['do_sample'])
            self.assertEqual((calls[0]['temperature'], calls[0]['top_p'], calls[0]['top_k']), (.7, .8, 20))
            text = model.ask(None, 'text request', Path(tmp)/'plan.txt',
                             profile='official', response_format='text')
            self.assertEqual(text, '{"boxes":[]}')
            self.assertEqual(len(calls[-1]['logits_processor']), 1)
            examples=[dict(role='user',content='example room'),dict(role='assistant',content='example layout')]
            example_path=Path(tmp)/'few_shot.txt'
            model.ask(None,'current room',example_path,profile='observer',examples=examples)
            model.ask(None,'current room',example_path,profile='observer',examples=examples)
            self.assertEqual(len(calls),3)
            self.assertEqual([m['role'] for m in message_log[-1]],['user','assistant','user'])
            self.assertEqual(message_log[-1][:2],examples)
            examples[1]['content']='different example layout'
            model.ask(None,'current room',example_path,profile='observer',examples=examples)
            self.assertEqual(len(calls),4)  # Different demonstrations invalidate the response cache.
            from PIL import Image
            image = Image.new('RGB', (32, 32))
            model.ask(image, 'current room', example_path, profile='observer', examples=examples)
            self.assertEqual(len(calls), 5)  # A text answer is not reused for an image request.
            self.assertEqual(encoding_log[-1]['images'], [image])
            self.assertEqual([c['type'] for c in message_log[-1][-1]['content']], ['image', 'text'])
            def fail(**kwargs):
                raise RuntimeError('simulated interrupted generation')
            model.model.generate = fail
            with self.assertRaises(RuntimeError):
                model.ask(None, 'changed prompt', path, profile='observer')
            self.assertFalse(path.with_suffix('.json').exists())

    def test_real_missing_category_and_runaway_regressions(self):
        evidence = read(Path(__file__).parent/'fixtures/observer_run001_failures.json')
        # Both are windows in the actual pool response; the second omitted category.
        response = evidence['window_query_without_repeated_category']
        self.assertNotIn('category', response[1])
        result, audit = grounded_instances('window', response, offset=3)
        self.assertEqual([x['category'] for x in result], ['window', 'window'])
        self.assertEqual([x['id'] for x in result], ['obs_0003', 'obs_0004'])
        self.assertEqual(result[1]['bbox_2d'], response[1]['bbox_2d'])
        repeated = evidence['repeated_box']
        result, audit = grounded_instances('objects', [repeated]*3)
        self.assertEqual(len(result), 1)
        self.assertEqual(audit['exact_duplicates'], [1, 2])
        with self.assertRaisesRegex(ValueError, '0..1000'):
            grounded_instances('window', [evidence['out_of_range_box']])
        with self.assertRaises(ValueError):
            grounded_instances('objects', [repeated]*33)
        empty, audit = grounded_instances('chair', {'boxes': []})
        self.assertEqual(empty, [])

    def test_observer_rotated_full_aabb_and_frame(self):
        obj, box = observed_object(INSTANCE, BOX)
        np.testing.assert_allclose(obj['center_local_m']['value'], [1., -.5, 3.])
        np.testing.assert_allclose(obj['size_aabb_local_m']['value'], [.5, 1., 2.], atol=1e-12)
        np.testing.assert_allclose(np.ptp(box['corners_local'], axis=0), [.5, 1., 2.], atol=1e-12)
        self.assertEqual(obj['bbox2d_visible_xyxy_norm']['value'], [.2, .2, .8, .8])

    def test_instances_same_class_stay_distinct(self):
        item = dict(category='appliance', bbox_2d=[1, 2, 100, 200])
        result = instances_from_response({'instances': [item, item]})
        self.assertEqual([x['id'] for x in result], ['obs_0000', 'obs_0001'])
        self.assertEqual(result[0]['category'], 'appliances')
        with self.assertRaises(ValueError):
            instances_from_response({'instances': [dict(category='chair', bbox_2d=[2, 3, 1, 4])]})

    def test_observed_preserved_and_no_gt_prompt_leak(self):
        obs = observed(); obs['room'] = {'secret': 'HIDDEN_GT_SENTINEL'}
        obs['objects'][0]['gt_house_object_id'] = 'HIDDEN_GT_SENTINEL'
        original = deepcopy(obs)
        hidden, full = assemble(obs, PROPOSAL, 160, 64)
        self.assertEqual(obs, original)
        self.assertEqual(full['objects'][:1], obs['objects'])
        full['objects'][0]['center_local_m']['value'][0] = 99
        self.assertEqual(obs, original)
        self.assertNotIn('HIDDEN_GT_SENTINEL', hidden_prompt(obs, 'visible chair', '', 64))
        self.assertEqual(compact_observations(obs)[0]['id'], 'obs_0000')
        self.assertEqual(hidden['objects'][0]['local_evidence'], 'unknown')
        with self.assertRaises(ValueError):
            assemble(obs, PROPOSAL, 1, 64)

    def test_tensorize_and_new_fourier_contract(self):
        import torch
        from ..oracle_model import tensorize
        from ..fourier_world.runtime import GeometryFeatures
        hidden, full = assemble(observed(), PROPOSAL, 160, 64)
        category, geometry, mask = tensorize(full, max_objects=160)
        self.assertEqual(tuple(geometry.shape), (1, 161, 8))
        np.testing.assert_allclose(geometry[0, 2].numpy(), [-1, 0, -2, 2, 1, 1, 1, 1])
        self.assertEqual(int(mask.sum()), 3)
        self.assertEqual(int(category[0, 2]), 5)
        features = GeometryFeatures()(geometry)
        self.assertEqual(tuple(features.shape), (1, 161, 392))
        self.assertTrue(torch.isfinite(features).all())

    def test_metrics_empty_and_duplicate_predictions(self):
        obs = observed(); empty = dict(obs, objects=[])
        self.assertEqual(match_metrics(empty, obs)['recall'], 0.)
        self.assertIsNone(match_metrics(empty, empty)['recall'])
        duplicate = dict(obs, objects=obs['objects']*2)
        result = match_metrics(duplicate, obs)
        self.assertEqual(result['matches'], 1)
        self.assertEqual(result['precision'], .5)
        shifted = deepcopy(obs)
        shifted['objects'][0]['center_local_m']['value'][2] += 2
        # Matching by visual region permits measuring depth error independently of 3D IoU.
        self.assertEqual(match_metrics(shifted, obs, True)['matched_center_errors_m'], [2.])

    def test_empty_hidden_is_valid_but_invalid_hidden_is_not(self):
        obs = observed()
        hidden, full = assemble(obs, {'hidden_objects': []}, 160, 64)
        self.assertEqual(hidden['objects'], [])
        self.assertEqual(full['objects'], obs['objects'])
        invalid = deepcopy(PROPOSAL)
        invalid['hidden_objects'][0]['size_aabb_local_m'][0] = 0
        with self.assertRaises(ValueError):
            assemble(obs, invalid, 160, 64)
        with self.assertRaises(ValueError):
            compact_observations(full)

    def test_fake_model_end_to_end_and_resume(self):
        from . import qwen_upstream_review as review
        class FakeQwen:
            calls = 0
            def __init__(self, model):
                pass
            def ask(self, image, prompt, path, **options):
                FakeQwen.calls += 1
                if path.stem == 'categories':
                    self_check = options['profile'] == 'observer' and options['max_tokens'] <= 768
                    if not self_check:
                        raise AssertionError('Expected bounded observer category request')
                    return dict(description='A visible chair.', categories=['chair'])
                if path.stem == 'ground_chair':
                    if options['max_tokens'] > 1536 or options['profile'] != 'observer':
                        raise AssertionError('Expected bounded 2D grounding')
                    return dict(boxes=[INSTANCE['bbox_2d']])
                if path.stem == 'proposal':
                    self_check = 'GT_HIDDEN_SECRET' not in prompt
                    if not self_check:
                        raise AssertionError('GT leaked')
                    write(path.with_suffix('.request.json'), dict(prompt=prompt))
                    return deepcopy(PROPOSAL)
                if options['max_tokens'] > 512 or options['profile'] != 'observer':
                    raise AssertionError('Expected single-instance 3D grounding')
                return deepcopy(BOX)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); oracle = root/'oracle'; oracle.mkdir()
            full = observed(); full['scope'] = 'full'; full['room'] = {'type': {'value': 'b'}}
            hid, _ = assemble(observed(), PROPOSAL, 160, 64)
            hid['objects'][0]['local_evidence'] = 'unobserved'
            hid['objects'][0]['secret'] = 'GT_HIDDEN_SECRET'
            full['objects'] += hid['objects']
            write(root/'full.json', full)
            buf = io.BytesIO(); Image.new('RGB', (64, 64), 'gray').save(buf, format='PNG')
            archive_path = root/'image.zip'
            with zipfile.ZipFile(archive_path, 'w') as z:
                z.writestr('face2.png', buf.getvalue())
            write_rows(oracle/'samples.jsonl', [dict(id='scene', split='val', graph=str(root/'full.json'),
                paired=dict(scene_id='building', local_condition=dict(archive=str(archive_path), member='face2.png')))])
            write(oracle/'BUNDLE.json', dict(max_objects=160))
            bundle = root/'bundle'
            pipeline.prepare(argparse.Namespace(bundle=bundle, oracle=oracle, selection_seed=0, per_building=3))
            obsargs = argparse.Namespace(action='observe', bundle=bundle, output=root/'observer',
                                         model=root/'fake', max_new_tokens=8192)
            compargs = argparse.Namespace(action='complete', bundle=bundle, output=root/'completion',
                observer=obsargs.output, model=root/'fake', max_new_tokens=8192, max_hidden=64,
                conditions=['gt', 'pred'], seeds=[0, 1])
            with patch.object(pipeline, 'Qwen', FakeQwen):
                pipeline.observe(obsargs); pipeline.complete(compargs)
                calls = FakeQwen.calls
                pipeline.observe(obsargs); pipeline.complete(compargs)
                self.assertEqual(FakeQwen.calls, calls)
            with patch('sys.argv', ['review', '--bundle', str(bundle), '--observer', str(obsargs.output),
                                   '--completion', str(compargs.output), '--output', str(root/'review')]):
                review.main()
            metrics = read(root/'review/metrics.json')
            self.assertEqual(metrics['failures'], [])
            completed = [r for r in metrics['records'] if r['kind'].startswith('hidden')]
            self.assertEqual(len(completed), 4)
            self.assertTrue(all(r['observed_preserved'] for r in completed))
            self.assertTrue((root/'review/review.html').exists())
            class SimpleQwen:
                calls = 0
                def __init__(self, model):
                    pass
                def ask(self, image, prompt, path, **options):
                    SimpleQwen.calls += 1
                    if options['profile'] != 'official' or 'GT_HIDDEN_SECRET' in prompt:
                        raise AssertionError('Wrong profile or leaked GT')
                    if path.stem == 'hidden_plan':
                        if options['response_format'] != 'text':
                            raise AssertionError('Plan must be text')
                        plan = 'Hypothesis: a table behind the camera.'
                        path.write_text(plan)
                        return plan
                    if path.stem == 'proposal':
                        if 'Hypothesis: a table behind the camera.' not in prompt:
                            raise AssertionError('Missing plan')
                        return deepcopy(PROPOSAL)
                    if not path.stem.startswith('obs_'):
                        raise AssertionError('2D discovery should be reused')
                    if 'red rectangle' not in prompt or json.dumps(INSTANCE['bbox_2d']) in prompt:
                        raise AssertionError('Marked query must omit numeric 2D coordinates')
                    if image.getpixel((12, 12)) != (255, 0, 0):
                        raise AssertionError('Target must be marked on the original full image')
                    result = deepcopy(BOX)
                    result['bbox_3d'][0] = .9
                    return result
            simpleobs = deepcopy(obsargs)
            simpleobs.simple = True
            simpleobs.mark_target = True
            simpleobs.reuse_instances = obsargs.output
            simpleobs.output = root/'simple_observer'
            simplecomp = deepcopy(compargs)
            simplecomp.simple = True
            simplecomp.observer = simpleobs.output
            simplecomp.output = root/'simple_completion'
            simplecomp.conditions = ['pred']; simplecomp.seeds = [0]
            with patch.object(pipeline, 'Qwen', SimpleQwen):
                pipeline.observe(simpleobs); pipeline.complete(simplecomp)
                calls = SimpleQwen.calls
                pipeline.observe(simpleobs); pipeline.complete(simplecomp)
                self.assertEqual(SimpleQwen.calls, calls)
            newobs = read(simpleobs.output/'scene/G_obs.json')
            self.assertAlmostEqual(newobs['objects'][0]['center_local_m']['value'][0], .9)
            newfull = read(simplecomp.output/'pred/scene/seed0/G_full.json')
            self.assertEqual(newfull['objects'][:1], newobs['objects'])
            self.assertEqual(len(newfull['objects']), 2)
            with patch('sys.argv', ['review', '--bundle', str(bundle), '--observer', str(simpleobs.output),
                                   '--completion', str(simplecomp.output), '--output', str(root/'simple_review')]):
                review.main()
            self.assertIn('Hypothesis: a table behind the camera.',
                          (root/'simple_review/review.html').read_text())
            class BrokenQwen(FakeQwen):
                def ask(self, image, prompt, path, **options):
                    return {'hidden_objects': [{'category': 'chair'}]}
            badargs = deepcopy(compargs); badargs.output = root/'bad_completion'
            with patch.object(pipeline, 'Qwen', BrokenQwen), self.assertRaises(SystemExit):
                pipeline.complete(badargs)
            with patch('sys.argv', ['review', '--bundle', str(bundle), '--observer', str(obsargs.output),
                                   '--completion', str(badargs.output), '--output', str(root/'bad_review')]):
                review.main()
            bad = read(root/'bad_review/metrics.json')
            self.assertEqual(len(bad['failures']), 4)
            self.assertFalse(any('metrics' in r for r in bad['failures']))
            self.assertFalse(list(badargs.output.glob('*/*/*/G_full.json')))
            class Missing3D(FakeQwen):
                def ask(self, image, prompt, path, **options):
                    if path.stem.startswith('obs_'):
                        return {'bbox_3d': [.15, .07]}
                    return super().ask(image, prompt, path, **options)
            partialargs = deepcopy(obsargs); partialargs.output = root/'partial_observer'
            with patch.object(pipeline, 'Qwen', Missing3D):
                pipeline.observe(partialargs)
            self.assertEqual(read(partialargs.output/'scene/status.json')['status'], 'partial')
            pc = deepcopy(compargs); pc.observer = partialargs.output; pc.output = root/'partial_completion'
            pc.conditions = ['pred']; pc.seeds = [0]
            with patch.object(pipeline, 'Qwen', FakeQwen):
                pipeline.complete(pc)
            with patch('sys.argv', ['review', '--bundle', str(bundle), '--observer', str(pc.observer),
                                   '--completion', str(pc.output), '--output', str(root/'partial_review')]):
                review.main()
            pm = read(root/'partial_review/metrics.json')
            self.assertEqual(pm['records'][0]['metrics']['valid_3d_count'], 0)
            self.assertTrue(next(r for r in pm['records'] if r['kind']=='hidden_pred')['observed_preserved'])


if __name__ == '__main__':
    unittest.main()
