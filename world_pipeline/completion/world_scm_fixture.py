"""Explicit synthetic mechanism fixture, NEVER a real model prediction."""
from copy import deepcopy
import json

from ..common import write


def semantic_object(key,name,category,kind='floor',target=None):
    return dict(key=key,name=name,category=category,variant=name,identity='new_unobserved',
        distinct_instance_reason='CPU TEST: an independent hypothetical instance, not a moved photographed object.',
        placement_hint='CPU TEST: behind original camera',support=dict(kind=kind,target=target))


def fixture_domain():
    return dict(contexts=[dict(description='CPU TEST: lounge with optional reading/storage surroundings'),
                          dict(description='CPU TEST: mixed lounge and work surroundings')],
        organizations=[dict(description='CPU TEST: lounge continuation'),dict(description='CPU TEST: lounge with secondary activity zone')],
        groups=[dict(name='CPU TEST: reading assembly',mechanism_reason='Joint side-table and lamp choice under organization.',parents=[],modes=[
            dict(description='CPU TEST: independent side table with table lamp',observed_role_ids=[],objects=[
                semantic_object('table','side table','table'),semantic_object('light','table lamp','lamp','object','table')])]),
            dict(name='CPU TEST: storage assembly',mechanism_reason='Storage choice changes with reading assembly and global organization.',parents=[0],modes=[
                dict(description='CPU TEST: bookshelf and independent floor plant',observed_role_ids=[],objects=[
                    semantic_object('shelf','bookshelf','bookshelf'),semantic_object('plant','potted plant','plant')])])])


class FixtureProvider:
    execution_kind='cpu_fixture_not_model_prediction'
    def __init__(self,case):self.case=case;self.calls=0;self.usage=[];self.requests=[]

    def ask(self,image,prompt,path,**kwargs):
        self.calls+=1;self.requests.append(dict(name=path.stem,image_used=image is not None,prompt=prompt))
        name=path.stem
        if name=='grounding':
            value=dict(instances=[dict(id=o['id'],variant=o['description'],on_floor=i==0)
                for i,o in enumerate(self.case['context'])],evidence_description='CPU TEST: visible lounge furniture; outside contents not observed.')
        elif name=='domain':value=fixture_domain()
        elif name.startswith('mechanism_'):
            payload=json.loads(prompt.split('\n')[-1]);variable=payload['variable'];rows=[]
            for given in payload['required_given_rows']:
                if variable=='C':ps=[.6,.4]
                elif variable=='S':ps=[.8,.2] if given['C']=='c0' else [.3,.7]
                elif variable=='E':ps=[.15,.85] if given['S']=='s0' else [.65,.35]
                elif variable=='g0':ps=[.4,.6] if given['S']=='s0' else [.2,.8]
                else:ps=[.15,.85] if given['g0']=='1' else [.7,.3]
                rows.append(dict(given=given,p=ps))
            value=dict(rows=rows)
        elif name.startswith('layout_'):
            selected=json.loads(prompt.split('JOINT ASSEMBLIES: ',1)[1].split('\nFROZEN OBSERVATIONS:',1)[0]);objects=[]
            for group in selected:
                for o in group['mode']['objects']:
                    dims,anchor,offset={
                        'table':([.55,.6,.55],'camera',[2.5,-1.,-3.1]),
                        'light':([.25,.45,.25],'table',[0.,0.,0.]),
                        'shelf':([1.1,1.8,.35],'camera',[-2.5,-1.,-3.2]),
                        'plant':([.35,.8,.35],'shelf',[1.5,0.,0.])}[o['key']]
                    objects.append(dict(id=group['group']+'_'+o['key'],width_m=dims[0],height_m=dims[1],depth_m=dims[2],
                        yaw_deg=0.,anchor=anchor,offset_m=offset))
            value=dict(objects=objects)
        else:raise ValueError('Unexpected fixture query: '+name)
        path.write_text(json.dumps(value));write(path.with_suffix('.json'),value)
        write(path.with_suffix('.request.json'),dict(prompt=prompt,synthetic_test=True,image_used=image is not None))
        return deepcopy(value)
