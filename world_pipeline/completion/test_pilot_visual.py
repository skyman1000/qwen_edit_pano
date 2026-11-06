"""Causal graph semantics and actual saved-state review, without model inference."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest

from ..common import read,write,sha
from . import assembly_pilot_visual as visual
from .scene_causal_diagram import specification,validate


class VisualGraphTests(unittest.TestCase):
    def test_camera_changes_representation_and_visibility_without_changing_the_room(self):
        spec=specification();validate(spec);parents={n['id']:set(n['parents']) for n in spec['nodes']}
        def ancestors(node):
            result=set(parents[node])
            for p in parents[node]:result.update(ancestors(p))
            return result
        self.assertIn('K',ancestors('O'));self.assertIn('K',ancestors('G'))
        self.assertNotIn('K',ancestors('W'))
        self.assertNotIn('O',ancestors('H'));self.assertNotIn('A',ancestors('R'))
        self.assertFalse(spec['calibrated_posterior_implemented'])

    def test_contact_proxy_reports_tiny_contact_without_certifying_physics(self):
        def box(id,center,size):return dict(track_id=id,center_local_m=dict(value=center),size_aabb_local_m=dict(value=size))
        state=dict(objects=[box('table',[0,0,0],[1,1,1]),box('mug',[.59999,.6,0],[.2,.2,.2])])
        semantic=dict(objects=[dict(id='mug',attachment=dict(kind='top',host='table'))])
        result=visual.contact_proxies(state,semantic)[0]
        self.assertGreater(result['child_footprint_on_host_AABB_fraction'],0)
        self.assertLess(result['child_footprint_on_host_AABB_fraction'],.001)
        self.assertIn('not_surface_or_stability',result['scope'])

    def test_gallery_retains_real_states_shows_failed_needs_and_checks_frozen_observations(self):
        source=Path(__file__).resolve().parents[2]/'outputs/causal_completion_pilot_20261024_220629_380006_3407'
        if not source.is_dir():self.skipTest('Real completed pilot unavailable in this checkout')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);write(root/'pilot.json',read(source/'pilot.json'))
            hashes={}
            for method in ['baseline','joint','needs']:
                target=root/method;target.mkdir()
                shutil.copyfile(source/method/'input.png',target/'input.png')
                shutil.copyfile(source/method/'experiment.json',target/'experiment.json')
                obs=target/'qwen/observer3d';obs.mkdir(parents=True)
                shutil.copyfile(source/method/'qwen/observer3d/G_obs.json',obs/'G_obs.json')
                dest=target/'qwen/completion';dest.mkdir(parents=True)
                summary=read(source/method/'qwen/completion/summary.json')
                summary['worlds']=summary.get('worlds',[])[:1]
                write(dest/'summary.json',summary)
                for record in summary['worlds']:
                    folder=dest/record['folder'];folder.mkdir()
                    for name in ('G_full.json','semantic_world.json','numeric_review.json'):
                        original=source/method/'qwen/completion'/record['folder']/name
                        shutil.copyfile(original,folder/name)
                    hashes[folder/'G_full.json']=sha(folder/'G_full.json')
            page=visual.build(root)
            self.assertTrue(page.is_file());self.assertIn('没有实际三维补全结果',page.read_text())
            review=read(page.parent/'review.json');self.assertEqual(len(review['worlds']),2)
            self.assertEqual(review['branch_status']['needs'],'failed')
            self.assertTrue((page.parent/'synchronized_3d.html').is_file())
            for path,digest in hashes.items():self.assertEqual(sha(path),digest)
            changed=deepcopy(read(root/'needs/qwen/observer3d/G_obs.json'))
            changed['objects'][0]['description']='Different frozen evidence'
            write(root/'needs/qwen/observer3d/G_obs.json',changed)
            with self.assertRaisesRegex(ValueError,'same RGB and G_obs'):
                visual.build(root)


if __name__=='__main__':unittest.main()
