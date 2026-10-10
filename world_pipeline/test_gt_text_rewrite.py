"""CPU-only integrity tests; no models, CUDA, or project data needed."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from .gt_text_rewrite import VERSION, parse_caption, apply_rewrites, text_sha


class RewriteTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_caption('```json\n{"caption":"A bed behind the camera."}\n```'),
                         'A bed behind the camera.')
        for raw in ('{"caption":""}', '{"caption":"a","caption":"b"}',
                    '{"caption":"a","reason":"b"}', '{"caption":null}', '{"caption":'):
            with self.assertRaises((ValueError, TypeError)):
                parse_caption(raw)

    def fixture(self):
        cfg = {k: 1 for k in ('checkpoint', 'protected_files', 'ids', 'split', 'seeds',
               'shuffle_seed', 'shuffle_mapping', 'text_version', 'text_format', 'steps',
               'cfg', 'negative_prompt', 'offload', 'max_text_tokens', 'max_encoder_tokens')}
        cfg.update(pano={'prompt': 'Task.'}, code={})
        pool = {sid: dict(graph_sha256=sid, structured_text=f'facts {sid}') for sid in ('a','b')}
        prompts = {'a': {mode: dict(prompt='Task.\n\n'+pool[sid]['structured_text'],
                                   source_id=sid, source_graph_sha256=sid)
                         for mode, sid in [('correct','a'),('shuffled','b')]}}
        bundle = dict(version=VERSION, reference_config=copy.deepcopy(cfg),
                      reference_prompts=copy.deepcopy(prompts), reference_run='/unused',
                      sources={sid: dict(source_graph_sha256=sid, source_text=item['structured_text'],
                                        source_text_sha256=text_sha(item['structured_text']), caption=f'caption {sid}')
                               for sid, item in pool.items()})
        return cfg, pool, prompts, bundle

    def test_sources_and_rejections(self):
        for mutation in ('none', 'seed', 'facts', 'empty', 'mapping'):
            cfg, pool, prompts, bundle = self.fixture()
            if mutation == 'seed':
                cfg['seeds'] = 2
            elif mutation == 'facts':
                bundle['sources']['a']['source_text'] = 'changed'
            elif mutation == 'empty':
                bundle['sources']['b']['caption'] = None
            elif mutation == 'mapping':
                prompts['a']['shuffled']['source_id'] = 'a'
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'rewrites.json'
                path.write_text(json.dumps(bundle))
                if mutation != 'none':
                    with self.assertRaises(ValueError, msg=mutation):
                        apply_rewrites(cfg, pool, prompts, path)
                else:
                    apply_rewrites(cfg, pool, prompts, path)
                    self.assertEqual(prompts['a']['llm_correct']['prompt'], 'Task.\n\ncaption a')
                    self.assertEqual(prompts['a']['llm_shuffled']['prompt'], 'Task.\n\ncaption b')


if __name__ == '__main__':
    unittest.main()
