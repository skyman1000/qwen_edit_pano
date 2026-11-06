"""Cross-scene saved-failure regressions and typed-request invariants; CPU only."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from ..common import ROOT,read
from . import global_world_model as model,global_world_geometry as geometry,global_world_proposals as proposals
from .test_global_world import observation,hypothesis,program,packet

SOURCE = ROOT/'qwen_edit_pano/outputs/global_world_srun_002'


class TypedProposalTests(unittest.TestCase):
    def test_missing_prose_does_not_delete_a_valid_world_but_a_bad_host_does(self):
        p = program();p['objects'][0]['reason'] = None
        parsed = model.programs_answer(dict(programs=[p]),hypothesis(),packet(),2,48)
        self.assertIsNone(parsed[0]['objects'][0]['reason'])
        bad = deepcopy(p);bad['objects'][0]['support']['host'] = 'obs_000'
        values = model.programs_answer(dict(programs=[p,bad]),hypothesis(),packet(),2,48)
        self.assertEqual(len(values),1);self.assertEqual(len(values[0]['proposal_rejections']),1)
        with self.assertRaisesRegex(ValueError,'Invalid support host'):
            model.programs_answer(dict(programs=[bad]),hypothesis(),packet(),2,48)

    def test_operational_failure_is_explicitly_excluded_not_sampled_as_a_world(self):
        h0 = hypothesis();h1 = deepcopy(h0);h1['id']='h1';p = program()
        m = model.compile_model([h0,h1],{'h0':[p],'h1':[]},{'h1':4},{'h0':{p['id']:4}})
        self.assertEqual(m['mechanism_coverage']['excluded_hypotheses'],['h1'])
        self.assertGreater(m['mechanism_coverage']['excluded_elicited_mass'],.8)
        self.assertTrue(all(s['hypothesis']=='h0' and s['program']==p['id'] for s in model.sample_worlds(m,32,3)))
        self.assertFalse(any(s.startswith('failed_') for n in m['nodes'] for s in n['states']))

    def test_typed_rows_accept_only_lossless_identity_and_preserve_independent_rows(self):
        check = lambda row: model.vector(row.get('center'),3,'center')
        r = proposals.typed_rows(dict(objects={'n1':dict(center=[1,2,3])}),{'n1'},check)
        self.assertEqual(r['rows'],{'n1':[1,2,3]})
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            proposals.typed_rows(dict(objects={'n1':dict(id='n0',center=[1,2,3])}),{'n1'},check)
        r = proposals.typed_rows(dict(objects=[dict(id='n0',center=[1,2,3]),dict(id='n1',center=None)]),{'n0','n1'},check)
        self.assertEqual(r['rows'],{'n0':[1,2,3]});self.assertEqual(len(r['errors']),1)

    def test_bounded_missing_row_request_keeps_exact_previously_accepted_values(self):
        class Calls:
            requested=[]
            def ask(self,name,image,prompt,validator,**kwargs):
                self.requested.append(prompt)
                value = (dict(objects=[dict(id='n0',center=[7,8,9]),dict(id='n1',center=None)])
                    if len(self.requested)==1 else dict(objects={'n1':dict(center=[1,2,3])}))
                return validator(value)
        calls = Calls()
        with tempfile.TemporaryDirectory() as tmp:
            rows = proposals.fill_rows(calls,'test',None,[dict(id='n0'),dict(id='n1')],
                lambda missing,accepted:str((missing,accepted)),
                lambda row:model.vector(row.get('center'),3,'center'),Path(tmp))
        self.assertEqual(rows,dict(n0=[7,8,9],n1=[1,2,3]))
        self.assertIn("(['n1'],",calls.requested[1]);self.assertEqual(len(calls.requested),2)

    def test_original_camera_region_is_hard_even_when_another_hidden_sector_is_feasible(self):
        h = hypothesis();h['room']['z_min']=0.
        p = program();p['objects'][0]['camera_region']='rear'
        before = deepcopy(p)
        _,audit = geometry.solve_layout(h,p,packet(),{'n0':dict(center_m=[0,-1,3],yaw_deg=0)},3,
            starts=1,maxiter=20,camera=dict(hfov=90,image_size=[1024,1024]))
        self.assertFalse(audit['passed']);self.assertIn('n0:camera_region',audit['issues']);self.assertEqual(p,before)

    def test_grounded_operation_zone_avoids_owner_height_aliasing(self):
        p = program();p['objects'][0]['size_m']=[.4,.4,.4]
        p['objects'][0]['camera_region']='rear'
        p['access_zones']=[dict(owner='n0',base='floor',offset_xz_m=[0,1],size_m=[.6,1.6,.6],purpose='standing access')]
        layout,audit = geometry.solve_layout(hypothesis(),p,packet(),
            {'n0':dict(center_m=[0,-1,-3],yaw_deg=0)},3,camera=dict(hfov=90,image_size=[1024,1024]))
        self.assertTrue(audit['passed'],audit)
        self.assertEqual(layout['n0']['aabb_size_m'],[.4,.4,.4])


@unittest.skipUnless(SOURCE.is_dir(),'Saved run002 not available')
class ActualRun002Tests(unittest.TestCase):
    def test_actual_five_scene_failure_partition_and_all_generation_eos(self):
        self.assertEqual(len(list(SOURCE.glob('scenes/*/observation/qwen/observer3d/G_obs.json'))),5)
        records = [read(p) for p in SOURCE.glob('scenes/*/completion/world_*/result.json')]
        self.assertEqual(len(records),30)
        unavailable = sum(any('unavailable program mechanism' in f.get('error','') for f in r['failures']) for r in records)
        self.assertEqual(unavailable,18)
        self.assertEqual(sum(r['status']=='accepted_proxy_compatible' for r in records),0)
        finishes = [read(p)['finish_reason'] for p in SOURCE.glob('scenes/*/completion/calls/*.generation.json')]
        self.assertEqual(len(finishes),93);self.assertEqual(set(finishes),{'eos'})

    def test_actual_bathroom_support_duplicate_is_lossless_and_placeholder_nouns_are_rejected(self):
        for scene in SOURCE.joinpath('scenes').iterdir():
            c = scene/'completion';pk = model.observation_packet(read(c/'G_obs.json'),
                read(scene/'observation/qwen/observer2d/grounding.json'))
            if scene.name.startswith('x8F5'):
                h = read(c/'hypotheses.json')[0];raw = read(c/'calls/programs_h0.try0.json');before=deepcopy(raw)
                parsed = model.programs_answer(raw,h,pk,2,48)
                self.assertEqual(len(parsed[0]['objects']),2);self.assertEqual(raw,before)
                self.assertEqual(parsed[0]['objects'][1]['support'],dict(kind='top',host='n0'))
                self.assertEqual(parsed[0]['support_relation_evidence'][0]['kind'],'support_top')
            if scene.name.startswith('zsNo'):
                h = read(c/'hypotheses.json')[2];raw=read(c/'calls/programs_h2.try0.json')
                with self.assertRaisesRegex(ValueError,'placeholder'):
                    proposals.inventory_answer(raw,h,pk,2,48)

    def test_all_five_actual_living_room_failures_have_feasible_hidden_layout_without_resizing(self):
        scene = next(p for p in SOURCE.joinpath('scenes').iterdir() if p.name.startswith('X7Hy'))
        c = scene/'completion';observed = read(c/'G_obs.json')
        pk = model.observation_packet(observed,read(scene/'observation/qwen/observer2d/grounding.json'))
        hypotheses = {h['id']:h for h in read(c/'hypotheses.json')};checked=0
        camera=dict(hfov=90,image_size=[1024,1024])
        for d in sorted(c.glob('world_*')):
            if not (d/'initial_placement.json').exists():continue
            p=read(d/'joint_program.json');proposal=read(d/'initial_placement.json');h=hypotheses[p['hypothesis']]
            before=deepcopy((p,observed,h,proposal))
            layout,audit=geometry.solve_layout(h,p,pk,proposal,3,camera=camera)
            self.assertTrue(audit['passed'],audit)
            state=model.export_graph(observed,h,p,layout)
            self.assertTrue(geometry.visibility_audit(state,90,[1024,1024])['passed'])
            self.assertEqual(state['objects'][:len(observed['objects'])],observed['objects'])
            self.assertEqual((p,observed,h,proposal),before)
            for o in p['objects']:
                self.assertEqual(layout[o['id']]['aabb_size_m'],geometry.aabb_size(o['size_m'],proposal[o['id']]['yaw_deg']))
            checked+=1
        self.assertEqual(checked,5)

    def test_all_six_actual_bedroom_programs_have_an_analytic_floor_zone_conflict(self):
        scene=next(p for p in SOURCE.joinpath('scenes').iterdir() if p.name.startswith('zsNo'));checked=0
        for path in scene.glob('completion/world_*/joint_program.json'):
            p=read(path);objects={o['id']:o for o in p['objects']};zone=p['access_zones'][0];owner=objects[zone['owner']]
            self.assertEqual(owner['support']['kind'],'floor')
            residual=owner['size_m'][1]/2+zone['offset_m'][1]-zone['size_m'][1]/2
            self.assertAlmostEqual(residual,-.55);checked+=1
        self.assertEqual(checked,6)


if __name__=='__main__':unittest.main()
