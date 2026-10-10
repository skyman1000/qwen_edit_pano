"""CPU regression tests for data recovery; no network, GPU or mesh rendering."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from .checkpoint_io import (atomic_json, archive_partial, commit_sample, load_sample,
                            output_lock)
from .common import ROOT, read
from .expand_gt_sources import Builder, paired_selection
from . import prepare_gt


class RecoveryTests(unittest.TestCase):
    def test_download_scan_list_is_scoped_without_network(self):
        import download_mp_py3 as downloader
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'scans.txt'
            path.write_text('a\nb\n')
            argv = ['download_mp_py3.py', '-o', tmp, '--scan-list', str(path), '--type', 'house_segmentations']
            with patch('sys.argv', argv), patch('builtins.input', return_value=''), \
                    patch.object(downloader, 'get_release_scans', return_value=['a','b','c']), \
                    patch.object(downloader, 'download_release') as download:
                downloader.main()
                self.assertEqual(download.call_args.args[0], ['a','b'])
                self.assertEqual(download.call_args.args[2], ['house_segmentations'])
            path.write_text('a\na\n')
            with patch('sys.argv', argv), patch('builtins.input', return_value=''), \
                    patch.object(downloader, 'get_release_scans', return_value=['a','b','c']), \
                    patch.object(downloader, 'download_release') as download:
                with self.assertRaises(SystemExit):
                    downloader.main()
                download.assert_not_called()

    def test_commit_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'data.txt').write_text('complete')
            commit_sample(path, {'id': 'x'}, {'id': 'x'})
            self.assertEqual(load_sample(path)[0]['id'], 'x')
            (path/'data.txt').write_text('changed')
            with self.assertRaises(ValueError):
                load_sample(path)

    def test_partial_preserved_and_link_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sample'
            path.mkdir()
            (path/'partial').write_text('do not delete')
            archive_partial(path)
            saved = next(Path(tmp).glob('sample.interrupted-*'))
            self.assertEqual((saved/'partial').read_text(), 'do not delete')
            path.symlink_to(saved, target_is_directory=True)
            with self.assertRaises(ValueError):
                archive_partial(path)

    def test_writer_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with output_lock(tmp):
                with self.assertRaises(ValueError):
                    with output_lock(tmp):
                        pass

    def test_export_interruption_resume_preserves_completed_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = argparse.Namespace(pairs=ROOT/'qwen_edit_pano/data/paired_full_v1',
                                      min_score=.65, min_margin=.03, resume=False, threads=1)
            selected = [dict(id='h_a', scene_id='h', split='train'), dict(id='h_b', scene_id='h', split='val')]
            counts = {'train': dict(paired_rows=1, world_identity_matches=1, selected=1),
                      'val': dict(paired_rows=1, world_identity_matches=1, selected=1)}
            def export(row, output, scene, args, mapping_hash):
                directory = output/'samples'/row['id']
                directory.mkdir(parents=True)
                (directory/'payload').write_text(row['id'])
                if row['id'] == 'h_b' and not args.resume:
                    raise KeyboardInterrupt('simulated kill during sample')
                return (dict(id=row['id'], split=row['split'], directory='samples/'+row['id'], review_flags=[]),
                        dict(id=row['id']))
            scene = argparse.Namespace(source_sha256={})
            with patch.object(prepare_gt, 'MeshScene', return_value=scene), patch.object(prepare_gt, 'export_one', side_effect=export) as render:
                with self.assertRaises(KeyboardInterrupt):
                    prepare_gt.run_export(args, out, selected, counts, 'fixture', {})
                atomic_json(out/'review_decisions.json', {'h_a': dict(decision='approved', notes='keep')})
                args.resume = True
                prepare_gt.run_export(args, out, selected, counts, 'fixture', {})
                self.assertEqual(render.call_count, 3)  # a once, interrupted b twice
            self.assertEqual(read(out/'EXPORT_COMPLETE.json')['exported'], 2)
            self.assertEqual(read(out/'review_decisions.json')['h_a']['decision'], 'approved')
            self.assertEqual(len(list((out/'samples').glob('h_b.interrupted-*'))), 1)

    def test_legacy_migration_without_rendering(self):
        original = ROOT/'qwen_edit_pano/data/gt_world_pilot_v1'
        manifest = json.loads((original/'gt_manifest.jsonl').read_text().splitlines()[0])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            directory = out/manifest['directory']
            shutil.copytree(original/manifest['directory'], directory)
            (directory/'DATA_COMPLETE.json').unlink(missing_ok=True)
            shutil.copy(original/'contract.json', out/'contract.json')
            (out/'scene_sources').mkdir()
            (out/'gt_manifest.jsonl').write_text(json.dumps(manifest)+'\n')
            row = read(directory/'provenance.json')['paired_row']
            prepare_gt.validate_resume(out, read(out/'contract.json'), [row])
            recovered, observer = load_sample(directory)
            self.assertEqual(recovered, manifest)
            self.assertEqual(observer['id'], row['id'])
            # Relocated packages must accept known pre-move orchestration hashes
            # while still rejecting arbitrary implementation changes.
            marker = read(out/'resume_contract.json')
            marker['implementation'] = {
                k.replace('/qwen_edit_pano/world_pipeline/', '/research/world_pipeline/'):
                    ('c7208df32165f94df039bd58acf9b3d5e7d0fcf219f542bcad1957b89f536bdb'
                     if k.endswith('/prepare_gt.py') else v)
                for k, v in marker['implementation'].items()}
            atomic_json(out/'resume_contract.json', marker)
            prepare_gt.validate_resume(out, read(out/'contract.json'), [row])
            self.assertTrue((out/'resume_contract.before_relocation.json').is_file())
            bad = read(out/'resume_contract.json')
            bad['implementation'][str(Path(prepare_gt.__file__).resolve())] = '0'*64
            atomic_json(out/'resume_contract.json', bad)
            with self.assertRaises(ValueError):
                prepare_gt.validate_resume(out, read(out/'contract.json'), [row])
            changed = read(out/'contract.json')
            changed['thresholds']['ncc'] = 0.9
            with self.assertRaises(ValueError):
                prepare_gt.validate_resume(out, changed, [row])

    def test_real_legacy_stage_reuse_and_resume(self):
        old = ROOT/'qwen_pano/outputs/caupano'
        rows = paired_selection(ROOT/'qwen_edit_pano/data/paired_full_v1')
        row = next(r for r in rows if (old/'world_state_pilot'/r['scene_id']/r['source_view_id']/'build_report.json').is_file())
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(work_root=Path(tmp), reuse_root=old, threads=1)
            builder = Builder(args)
            index = old/'canonical_pilot'/row['scene_id']
            with patch.object(builder, 'run', side_effect=AssertionError('Should reuse, not execute')):
                builder.sample(row, index)
                builder.sample(row, index)
            self.assertEqual(len(list((Path(tmp)/'checkpoints').glob('*.json'))), 3)
            self.assertTrue((Path(tmp)/'world_state_pilot'/row['scene_id']/row['source_view_id']).is_symlink())

    def test_partial_stage_rebuilt_and_then_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, old, index = root/'work', root/'old', root/'index'
            work.mkdir(); old.mkdir(); index.mkdir()
            row = dict(id='h_u', scene_id='h', source_view_id='u')
            (index/'panorama_index.jsonl').write_text(json.dumps({'sample_id':'h_u'})+'\n')
            partial = work/'erp_alignment_pilot/h/u'
            partial.mkdir(parents=True)
            (partial/'unfinished').write_text('old partial')
            builder = Builder(argparse.Namespace(work_root=work, reuse_root=old, threads=1))
            def run(module, arguments, label):
                dest = Path(arguments[arguments.index('--output')+1])
                dest.mkdir(parents=True)
                name, status = {'tools.erp_alignment_pilot': ('alignment_report.json','CANDIDATE_REQUIRES_VISUAL_REVIEW'),
                                'data.build_geometry_pilot': ('geometry_report.json','BUILT_FOR_REVIEW'),
                                'data.world_state_builder': ('build_report.json','BUILT')}[module]
                atomic_json(dest/name, dict(sample_id='h_u', status=status, height=512, checks={'fixture': True}))
            with patch.object(builder, 'run', side_effect=run) as mock:
                builder.sample(row, index)
                self.assertEqual(mock.call_count, 3)
                builder.sample(row, index)
                self.assertEqual(mock.call_count, 3)
            self.assertEqual(len(list(partial.parent.glob('u.interrupted-*'))), 1)


if __name__ == '__main__':
    unittest.main()
