"""Real-response interface regressions and original-v13 CPU integration."""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from ...common import ROOT, read, write, sha
from .. import assembly_multiscene as pipeline, assembly_compositional as original
from .. import assembly_coordinate_transport as geometry
from ..test_compositional import CPUProgramProvider
from ..test_multiscene import CPUProvider
from . import contract as c, workflow as w
from .requests import Requests, RecordedFailure

SOURCE = ROOT / 'qwen_edit_pano/outputs/local_observation_fixed_002'
BASELINE = ROOT / 'qwen_edit_pano/outputs/v13_original_recovery_001'
LIVING = 'X7HyMhZNoso_2bb4931911e4412cb51f6437a82c5b6c'
BATH = 'x8F5xyUWy9e_3a127fd2c5f34d978cba253145dae7d6'
KITCHEN = '8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029'


def wire(program):
    return dict(alternatives=[dict(description=program['description'], groups=[dict(function=g['function'], reason=g['reason'],
        observed_refs=g.get('observed_refs', []), functional_links=g.get('functional_links', []),
        new_objects=[dict(key=o['key'], category=o['category'], variant=o['variant'], reason=o['distinct_instance_reason'],
            placement=o['placement_hint'], attachment=o['attachment'], affordances=o['affordances']) for o in g['objects']]) for g in program['groups']],
        relations=[dict(r, kind={'in_front_of':'farther_than', 'behind':'nearer_than'}.get(r['kind'],r['kind']),
                        **({'scope':'whole_aabb'} if r['kind'] in ('above','below') else {})) for r in program['relations']])])


class ProposalCPU(CPUProgramProvider):
    """Synthetic replies are explicit; geometry/CPT execution stays original."""
    def ask(self, image, prompt, path, **kwargs):
        path = Path(path)
        if path.stem.startswith('hypotheses'):
            value = dict(hypotheses=[dict(description='Possible hidden work arrangement', functions=['work'], assumptions=['Unseen region may support work'])])
            self.requests.append(dict(path=str(path), prompt=prompt, **kwargs))
        else:
            value = super().ask(image, prompt, path, **kwargs)
            for alt in value['alternatives']:
                for r in alt['relations']:
                    if r['kind'] in ('above', 'below'): r['scope'] = 'whole_aabb'
                g = alt['groups'][0]; objects = g['new_objects']
                g['functional_links'] = [dict(source=objects[0]['key'], target=objects[1]['key'], kind='serves', purpose='Synthetic semantic intent, not a hard spatial ordering')]
        path.write_text(json.dumps(value)); write(path.with_suffix('.json'), value)
        write(path.with_suffix('.request.json'), dict(prompt=prompt, image_delivered=image is not None, **kwargs))
        return value


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'research/audits', prefix='candidate_contract_CPU_')
        self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, CAUPANO_PROGRAM_CATALOG='', CAUPANO_LAYOUT_RGB='0'))
        self.stack.enter_context(patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight', return_value={}))

    def case(self, sid=LIVING):
        return read(BASELINE/'scenes'/sid/'completion/shared/case.json')

    def prepared(self):
        source = self.root/'source'; source.mkdir(); suite = read(SOURCE/'suite.json'); suite['scenes'] = suite['scenes'][:1]
        write(source/'suite.json', suite); sid = suite['scenes'][0]['id']
        shutil.copytree(SOURCE/'scenes'/sid/'observation', source/'scenes'/sid/'observation')
        root = self.root/'run'; w.prepare(root, source, worlds=2, draws=4)
        w.base.phase(root, 'prepare-scenes')
        return root, sid

    def test_prompt_has_one_depth_contract_and_no_legacy_instructions(self):
        case = self.case(); h = original.profile('z0', ['reading'])
        text = c.furniture_prompt(case, h, 'surrounding')
        self.assertNotIn('"in_front_of"', text); self.assertNotIn('"behind"', text)
        self.assertNotIn('behind means SMALLER', text)
        self.assertIn('farther_than is greater Z', text); self.assertIn('scope="whole_aabb"', text)
        self.assertIn('functional_links', text)

    def test_real_four_failed_replies_get_specific_non_destructive_feedback(self):
        count = 0
        for summary in BASELINE.glob('scenes/*/completion/proposals/qwen/summary.json'):
            case = read(summary.parents[2]/'shared/case.json'); hypotheses = read(summary.parent/'hypotheses.json')
            for failure in read(summary).get('failed_queries', []):
                path = summary.parent/'calls'/(failure['stage']+'_repair.json'); raw = read(path); before = sha(path)
                h = next(h for h in hypotheses if failure['stage'].startswith('furniture_'+h['id']+'_'))
                with self.assertRaises(ValueError) as error: c.furniture_answer(raw, case, h)
                self.assertTrue(any(s in str(error.exception) for s in ('Spatial kind', 'host=null', 'scope=whole_aabb', 'Actual attachment')))
                self.assertEqual(sha(path), before); count += 1
        self.assertEqual(count, 4)

    def test_real_shelf_containment_is_preserved_as_semantics_not_near(self):
        comp = BASELINE/'scenes'/BATH/'completion'; path = comp/'proposals/qwen/calls/furniture_z1_surrounding_repair.json'
        raw = read(path)
        # Explicitly state the geometric scope in the test input. No automatic
        # claim that an untyped old above edge means this in the real scene.
        for alt in raw['alternatives']:
            for r in alt['relations']:
                if r['kind'] in ('above', 'below'): r['scope'] = 'whole_aabb'
        h = next(h for h in read(comp/'proposals/qwen/hypotheses.json') if h['id']=='z1')
        parsed = c.furniture_answer(raw, self.case(BATH), h)[0]
        self.assertEqual(sum(len(g['objects']) for g in parsed['groups']), sum(len(g['new_objects']) for g in raw['alternatives'][0]['groups']))
        links = [l for g in parsed['groups'] for l in g['functional_links']]
        link = next(l for l in links if l['kind']=='contained_in')
        self.assertEqual(link['source'], 'toilet_tissue_001'); self.assertEqual(link['target'], 'cabinet_002')
        self.assertFalse(any(r['source']=='toilet_tissue_001' and r['kind']=='near' for r in parsed['relations']))

    def test_top_does_not_silently_become_inside_or_a_new_host(self):
        comp = BASELINE/'scenes'/KITCHEN/'completion'; raw = read(comp/'proposals/qwen/calls/furniture_z0_surrounding_repair.json')
        h = read(comp/'proposals/qwen/hypotheses.json')[0]
        with self.assertRaisesRegex(ValueError, 'Actual attachment'):
            c.furniture_answer(raw, self.case(KITCHEN), h)

    def test_real_failed_lighting_intent_can_use_explicit_semantics_without_geometry_change(self):
        comp = BASELINE/'scenes'/LIVING/'completion'; folder = comp/'end_to_end/qwen/completion/world_03'
        observed = read(comp/'context/qwen/observer3d/G_obs.json'); world = read(folder/'semantic_world.json')
        layout = read(folder/'execution.json')['layout']; plane = read(folder/'numeric_review.json')['declared_support_plane']
        before = deepcopy(layout); original_world = deepcopy(world)
        strict = geometry.compile_layout(self.case(), observed, world, layout, plane)
        self.assertFalse(strict['review']['geometry_contract_passed'])
        # This is an explicit corrected semantic declaration, NOT an automatic
        # inference from arbitrary English or a silent edit of the saved world.
        edge = next(r for r in world['relations'] if r['kind']=='above')
        world['relations'] = [r for r in world['relations'] if r is not edge]
        world['functional_links'] = [dict(source=edge['source'], target=edge['target'], kind='illuminates', purpose=edge['reason'])]
        solved = geometry.compile_layout(self.case(), observed, world, layout, plane)
        self.assertTrue(solved['review']['geometry_contract_passed'])
        self.assertEqual(layout, before)
        self.assertEqual(world['objects'], original_world['objects'])
        self.assertEqual(len(solved['items']), 5)
        self.assertFalse(solved['review']['unsatisfied_declared_ordering'])

    def test_bound_repairs_do_not_echo_a_loop_and_resume_has_zero_calls(self):
        image = self.root/'input.png'; Image.new('RGB',(10,10)).save(image)
        class Provider:
            def __init__(self): self.calls=[]
            def ask(self, image, prompt, path, **kw):
                self.calls.append((prompt,kw['seed']))
                path.write_text('repeating prose; '*6000)
                if len(self.calls)==1: raise ValueError('Unfinished local response reached token limit; original text preserved')
                write(path.with_suffix('.json'),{}); return {}
        provider = Provider(); req = Requests(self.root/'calls',19,image,lambda:provider)
        self.assertEqual(req.ask('a','base',lambda v:['valid']),['valid'])
        self.assertLess(len(provider.calls[1][0]),2200)
        self.assertNotEqual(provider.calls[0][1],provider.calls[1][1])
        self.assertEqual(req.ask('a','base',lambda v:[]),['valid']); self.assertEqual(len(provider.calls),2)

    def test_recorded_failure_is_not_redrawn(self):
        image = self.root/'input.png'; Image.new('RGB',(10,10)).save(image)
        class Provider:
            def __init__(self): self.calls=0
            def ask(self,image,prompt,path,**kw):
                self.calls+=1; path.write_text('{}'); write(path.with_suffix('.json'),{}); return {}
        provider=Provider(); request=Requests(self.root/'calls',91,image,lambda:provider)
        def reject(v): raise ValueError('specific unknown host h0')
        with self.assertRaisesRegex(ValueError,'specific unknown host h0'): request.ask('bad','request',reject)
        with self.assertRaisesRegex(RecordedFailure,'specific unknown host h0'): request.ask('bad','request',reject)
        self.assertEqual(provider.calls,2)

    def test_repeated_glasses_are_flagged_not_deleted(self):
        path = BASELINE/'scenes'/KITCHEN/'completion/proposals/qwen/programs.json'
        programs = read(path); before = sha(path); stats = c.coverage(programs)
        self.assertTrue(any(r['category']=='glass' and r['count']==22 for r in stats['repeated_instances_for_review']))
        self.assertEqual(sha(path),before)

    def test_original_executor_and_functional_metadata_survive_two_rounds(self):
        root, sid = self.prepared(); proposer = ProposalCPU(); downstream = CPUProvider(all_present=True)
        with patch.object(w,'provider',return_value=proposer), patch.object(pipeline,'provider',return_value=downstream):
            w.propose(root); w.base.phase(root,'pool')
            early = w.candidate_report(root)
            self.assertTrue((root/'candidates.html').exists())
            self.assertFalse((root/'scenes'/sid/'completion/shared/mechanisms/qwen.json').exists())
            w.base.phase(root,'mechanisms'); w.base.phase(root,'worlds')
        comp = root/'scenes'/sid/'completion'; domain = read(comp/'shared/domain.json')
        self.assertEqual(set(domain['proposal_provenance']),{'qwen'})
        self.assertTrue(any(g['functional_links'] for k,p in domain['programs'].items() if k!='f_null' for g in p['groups']))
        frames = list((comp/'end_to_end/qwen/completion').glob('world_*/G_full.json'))
        self.assertEqual(len(frames),2)
        self.assertTrue(any(read(f).get('functional_links') for f in frames))
        result = w.report(root)
        self.assertEqual(result['status'],'completed_needs_review',result)
        self.assertFalse(result['execution_errors'])
        with patch.object(w,'provider',side_effect=AssertionError('cached resume must not load a model')):
            w.propose(root)
            self.assertEqual(w.report(root)['counts'],result['counts'])
        self.assertTrue((root/'report'/sid/'world_01/views/scene_3d.html').exists())
        self.assertEqual(w.pinned.verify()['source_files'],118)
        first = [r['prompt'] for r in proposer.requests if 'round_00' in r['path']]
        second = [r['prompt'] for r in proposer.requests if 'round_01' in r['path']]
        self.assertTrue(first and second)
        self.assertTrue(any('FIRST_ROUND_COVERAGE' in text for text in second))
        self.assertFalse(any('FIRST_ROUND_COVERAGE' in text for text in first))

    def test_genuine_empty_proposals_do_not_create_a_fake_domain(self):
        root,sid=self.prepared()
        class Empty(ProposalCPU):
            def ask(self,image,prompt,path,**kwargs):
                if Path(path).stem.startswith('furniture_'):
                    value=dict(alternatives=[dict(description='Possible open surrounding space with no new program',groups=[],relations=[])])
                    path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value
                return super().ask(image,prompt,path,**kwargs)
        with patch.object(w,'provider',return_value=Empty()): w.propose(root)
        w.base.phase(root,'pool')
        preview=w.candidate_report(root)
        self.assertEqual(preview['status'],'candidate_domain_incomplete')
        self.assertEqual(preview['domains_ready'],0)
        self.assertFalse(preview['execution_errors'])
        self.assertFalse((root/'scenes'/sid/'completion/shared/domain.json').exists())
        result=w.report(root)
        self.assertEqual(result['counts'],{'empty_candidate_domain':2})
        self.assertEqual(result['status'],'completion_incomplete')

    def test_comparison_baseline_is_optional_and_not_a_generation_dependency(self):
        root,sid=self.prepared()
        other=self.root/'without_baseline'
        w.prepare(other,self.root/'source',rounds=1,worlds=1,baseline=self.root/'does_not_exist')
        suite,meta=w.check(other)
        self.assertFalse(meta['baseline_statistics'][sid]['comparable'])
        self.assertEqual(meta['baseline_bindings'],{})
        self.assertEqual(suite['backends'],['qwen'])

    def test_unknown_ids_cycles_and_geometric_order_remain_constraints(self):
        case = self.case(); program = next(iter(read(BASELINE/'scenes'/LIVING/'completion/proposals/qwen/programs.json').values()))
        raw = wire(program); obj = raw['alternatives'][0]['groups'][0]['new_objects'][0]
        obj['attachment'] = dict(kind='top',host='h0')
        with self.assertRaisesRegex(ValueError,'h0'): c.furniture_answer(raw,case,program['hypothesis'])
        raw=wire(program); objects=raw['alternatives'][0]['groups'][0]['new_objects']
        objects[0]['attachment']=dict(kind='top',host=objects[1]['key'])
        objects[1]['attachment']=dict(kind='top',host=objects[0]['key'])
        with self.assertRaisesRegex(ValueError,'cycle'): c.furniture_answer(raw,case,program['hypothesis'])


if __name__=='__main__': unittest.main()
