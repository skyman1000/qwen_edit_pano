"""CPU tests for semantic projection, explicit review accounting and provenance."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from ..common import read, sha, write
from .category_bridge import category
from .content_review import PROTOCOL, apply_review, load_reviewed_plan, prompt, run
from .scene_completion import map_plan, run as layout_run
from .semantic_categories import resolve_identity, resolve_noun
from .semantic_probe import state_context
from .test_semantic_probe import state


def plan(count=1):
    return dict(scene_type='living room',scene_hypothesis='A plausible surrounding seating area.',
                additions=[dict(id=f'new_{i}',name='television',object_type='television',
                                reason='Entertainment for seating.',placement='Behind the camera.') for i in range(count)])


def review(count=1):
    return dict(scene_type='living room',scene_hypothesis='A hypothetical entertainment area.',
                reviews=[dict(id=f'new_{i}',decision='retain',reason='Plausible function outside the photograph.',
                              evidence_ids=[]) for i in range(count)])


class ContentReviewTests(unittest.TestCase):
    def test_semantic_alias_uses_existing_specific_class_and_preserves_legacy_mapping(self):
        self.assertEqual(category('television')[0],'misc')
        result=resolve_noun('television')
        self.assertEqual(result['category'],'tv_monitor')
        self.assertEqual(result['legacy_category'],'misc')
        self.assertTrue(any(r['field']=='nyuClass' and r['category_id']==22 for r in result['evidence']))
        self.assertEqual(category('television')[0],'misc')
        self.assertEqual(map_plan(plan())[0]['category'],'misc')
        self.assertEqual(map_plan(plan(),'semantic_lexicon_v1')[0]['category'],'tv_monitor')

    def test_unknown_and_outside_nouns_are_not_forced_into_supported_categories(self):
        for noun in ('skyline','footbath','rug','a photograph of a television'):
            with self.subTest(noun=noun):self.assertIsNone(resolve_noun(noun)['category'])
        self.assertIsNone(resolve_identity('rug','cushion')['category'])
        self.assertTrue(resolve_identity('heater','wooden sauna heater')['identity_loss_review'])
        console=resolve_identity('console','media console')
        self.assertEqual(console['mapping_input'],'name')
        self.assertTrue(console['identity_loss_review'])

    def test_conflicting_specific_identity_and_ambiguous_alias_are_not_silently_resolved(self):
        self.assertIsNone(resolve_identity('table','bookshelf')['category'])
        sources=[dict(category_id=3),dict(category_id=5)]
        with patch('qwen_edit_pano.world_pipeline.completion.semantic_categories.semantic_aliases',
                   return_value=({'AMBIG'.lower():sources},'test_digest')):
            self.assertEqual(resolve_noun('AMBIG')['status'],'ambiguous_semantic_alias')
            self.assertIsNone(resolve_noun('AMBIG')['category'])

    def test_review_separates_relocation_from_explicit_withdrawal_and_deferral(self):
        original=plan(4);before=deepcopy(original);answer=review(4)
        answer['reviews'][1].update(decision='relocate',placement='A separate unseen wall facing the seating.')
        answer['reviews'][2]['decision']='withdraw'
        answer['reviews'][3]['decision']='defer'
        revised,withdrawn,deferred=apply_review(answer,original,dict(context=[]))
        self.assertEqual(original,before)
        self.assertEqual([o['id'] for o in revised['additions']],['new_0','new_1'])
        self.assertEqual(revised['additions'][0],original['additions'][0])
        self.assertEqual(revised['additions'][1]['name'],original['additions'][1]['name'])
        self.assertEqual(withdrawn[0]['original'],original['additions'][2])
        self.assertEqual(deferred[0]['original'],original['additions'][3])
        self.assertFalse(revised['content_review']['semantics_verified'])

    def test_review_rejects_silent_omission_identity_changes_geometry_and_unknown_evidence(self):
        for kind in ('missing','duplicate','new_id','rename','geometry','evidence','missing_location'):
            answer=review();item=answer['reviews'][0]
            if kind=='missing':answer['reviews']=[]
            elif kind=='duplicate':answer['reviews']*=2
            elif kind=='new_id':item['id']='new_foreign'
            elif kind=='rename':item['name']='chair'
            elif kind=='geometry':item['center']=[0,0,-3]
            elif kind=='evidence':item['evidence_ids']=['nonexistent']
            elif kind=='missing_location':item['decision']='relocate'
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                apply_review(answer,plan(),dict(context=[]))

    def test_harmless_repetition_of_unchanged_registry_fields_does_not_require_new_inference(self):
        answer=review()
        answer['reviews'][0].update(name='television',object_type='television',placement='Behind the camera.')
        revised,_,_=apply_review(answer,plan(),dict(context=[]))
        self.assertEqual(revised['additions'],plan()['additions'])

    def test_reviewing_observations_instead_of_additions_reports_both_id_sets(self):
        # Regression: the saved sauna answer reviewed six observations and omitted all additions.
        original=plan(3);before=deepcopy(original);answer=review(6)
        observed=[dict(id=f'observed_{i:04d}') for i in range(6)]
        for item,obj in zip(answer['reviews'],observed):item['id']=obj['id']
        with self.assertRaises(ValueError) as error:
            apply_review(answer,original,dict(context=observed))
        message=str(error.exception)
        self.assertIn('observed IDs used as review targets',message)
        self.assertIn("missing candidates=['new_0', 'new_1', 'new_2']",message)
        for obj in observed:self.assertIn(obj['id'],message)
        self.assertEqual(original,before)

    def test_duplicate_id_and_invalid_decision_have_distinct_diagnostics(self):
        duplicate=review();duplicate['reviews']*=2
        with self.assertRaisesRegex(ValueError,'Duplicate review target ID: new_0'):
            apply_review(duplicate,plan(),dict(context=[]))
        invalid=review();invalid['reviews'][0]['decision']='accept'
        with self.assertRaisesRegex(ValueError,"Invalid review decision for new_0: 'accept'"):
            apply_review(invalid,plan(),dict(context=[]))

    def test_prompt_does_not_treat_unseen_objects_as_false_or_include_case_specific_fixes(self):
        question=prompt(dict(context=[]),dict(scene_hypothesis='Generic scene.',additions=[]))
        self.assertIn('Unseen does not mean absent',question)
        self.assertIn('relocate the same instance',question)
        self.assertNotIn('television',question)
        self.assertNotIn('sauna',question)

    def test_saved_review_can_feed_layout_without_rerunning_content_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'input.png';Image.new('RGB',(64,64)).save(image)
            observed=state();state_path=root/'obs.json';write(state_path,observed)
            case=dict(id='case',image=str(image),state=str(state_path),hfov=90,
                      context=state_context(observed),source_sha256={str(image):sha(image),str(state_path):sha(state_path)})
            entry=dict(case=case,plan=plan(),source_sha256=case['source_sha256'])
            source=root/'reviews';source.mkdir()
            write(source/'config.json',dict(protocol=PROTOCOL,inputs=[entry]))
            class Reviewer:
                def ask(self,image,question,path,**kwargs):
                    answer=review();path.write_text(json.dumps(answer));write(path.with_suffix('.json'),answer)
                    write(path.with_suffix('.request.json'),dict(prompt=question));return answer
            self.assertTrue(run(Reviewer(),[entry],source))
            revised,provenance=load_reviewed_plan(source,case)
            case_with_binding=dict(case,reviewed_plan_sha256=provenance)
            class Layout:
                calls=0
                def ask(self,image,question,path,**kwargs):
                    self.calls+=1
                    return dict(objects=[dict(id='new_0',center=[0,0,-3],size=[1,.6,.1],reason='Unseen viewing area.')])
            output=root/'layout';output.mkdir();model=Layout()
            self.assertTrue(layout_run(model,[case_with_binding],output,stage='full',
                                      reviewed_plans_from=source,category_policy='semantic_lexicon_v1'))
            self.assertEqual(model.calls,1)
            full=read(output/'case/G_full.json')
            self.assertEqual(full['objects'][:len(observed['objects'])],observed['objects'])
            self.assertEqual(full['objects'][-1]['category_id']['value'],22)
            self.assertFalse(full['content_review']['semantics_verified'])
            changed=review();changed['reviews'][0]['decision']='defer'
            folder=source/'case';write(folder/'review.json',changed);write(folder/'review.txt',changed)
            revised,withdrawn,deferred=apply_review(changed,plan(),case)
            for name,value in [('semantic_plan.json',revised),('withdrawn_hypotheses.json',withdrawn),('deferred_hypotheses.json',deferred)]:
                write(folder/name,value)
            with self.assertRaisesRegex(ValueError,'Deferred hypotheses'):load_reviewed_plan(source,case)


if __name__=='__main__':unittest.main()
