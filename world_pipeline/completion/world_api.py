"""Small SiliconFlow image/text JSON client. No SDK, implicit calls or secrets in logs."""
import base64
from copy import deepcopy
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
import uuid

from ..common import write
from .geometry import parse_json
from .world_generation import SYSTEM_PROMPT,image_bytes

DEFAULT_API_MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash'
DEFAULT_API_BASE = 'https://api.siliconflow.cn/v1'


class APIRequestError(RuntimeError):
    """Sanitized transport diagnostics; no provider body or credentials."""
    def __init__(self,category,**details):
        self.details=dict(category=category,**details)
        super().__init__('SiliconFlow '+category+'; '+json.dumps(details,ensure_ascii=False)+
            '. Request may have reached provider; no automatic retry was performed.')


def read_stream(response,state,started,timeout,stage,clock=time.monotonic):
    """Parse SSE frames; a valid completion requires both finish_reason and [DONE].

    Keep only final content and reasoning character counts, never reasoning text.
    A partial JSON prefix must not become a successful answer on disconnect.
    """
    data=[];last_progress=started
    def event():
        if not data:return
        payload='\n'.join(data);data.clear()
        if payload=='[DONE]':state['done']=True;return
        value=json.loads(payload)
        if value.get('error'):raise APIRequestError('stream_error_event')
        for name in ('id','model','usage'):
            if value.get(name) is not None:state[name]=value[name]
        for choice in value.get('choices',[]):
            if choice.get('index',0)!=0:continue
            delta=choice.get('delta') or {}
            if isinstance(delta.get('content'),str):state['parts'].append(delta['content'])
            if isinstance(delta.get('reasoning_content'),str):state['reasoning_chars']+=len(delta['reasoning_content'])
            if choice.get('finish_reason') is not None:state['finish_reason']=choice['finish_reason']
    for raw in response:
        now=clock()
        if now-started>timeout:raise TimeoutError('request elapsed deadline')
        if now-last_progress>=30:
            print(json.dumps(dict(stage=stage,event='api_stream_progress',elapsed_seconds=round(now-started,1),
                final_chars=sum(map(len,state['parts'])),reasoning_chars=state['reasoning_chars'])),flush=True)
            last_progress=now
        line=raw.decode('utf-8').rstrip('\r\n')
        if not line:event()
        elif line.startswith('data:'):data.append(line[5:].lstrip(' '))
        # Comments/heartbeats and event/id fields carry no answer text.
        if state['done']:break
    if data:event()
    if not state['done'] or state['finish_reason'] is None:
        raise APIRequestError('incomplete_stream',done=state['done'],finish_reason=state['finish_reason'])
    return dict(id=state.get('id'),model=state.get('model'),usage=state.get('usage'),
                choices=[dict(finish_reason=state['finish_reason'],message=dict(content=''.join(state['parts'])))])


def image_part(image):
    return dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+
        base64.b64encode(image_bytes(image)).decode('ascii')))


class SiliconFlowJSON:
    execution_kind='siliconflow_api'
    def __init__(self,model=DEFAULT_API_MODEL,base=DEFAULT_API_BASE,key_env='SILICONFLOW_API_KEY',
                 timeout=600,max_calls=32,thinking='auto',temperature=.4,opener=None,stream=True,generation_profile='legacy'):
        parsed=urlparse(base)
        if parsed.scheme!='https' or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('API base must be an HTTPS URL without credentials/query/fragment')
        self.key=os.environ.get(key_env,'').strip()
        if not self.key:raise ValueError(f'Set {key_env} before API execution; do not put a key in command arguments')
        self.model,self.base=model,base.rstrip('/')
        self.timeout,self.max_calls=timeout,max_calls
        self.thinking,self.temperature=thinking,temperature
        self.generation_profile=generation_profile
        if generation_profile=='matched_v1':
            if thinking=='on':raise ValueError('matched_v1 requests non-thinking mode for both backends')
            self.thinking,self.temperature='off',.4
        elif generation_profile!='legacy':raise ValueError('Unknown generation profile')
        self.stream=stream
        self.calls=0;self.usage=[];self.opener=opener or urllib.request.urlopen

    def _request(self,method,endpoint,payload=None,stream_state=None,stage=None):
        if self.calls>=self.max_calls:raise APIRequestError('request_budget_exhausted',max_calls=self.max_calls,request_started=False)
        self.calls+=1
        trace_id=uuid.uuid4().hex;started=time.monotonic()
        request=urllib.request.Request(self.base+endpoint,
            data=None if payload is None else json.dumps(payload,allow_nan=False).encode(),method=method,
            headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json','X-Trace-Id':trace_id})
        try:
            with self.opener(request,timeout=self.timeout) as response:
                result=(read_stream(response,stream_state,started,self.timeout,stage) if stream_state is not None else
                        json.loads(response.read().decode('utf-8')))
        except urllib.error.HTTPError as exc:
            # Server error bodies can echo submitted headers. Do not log them.
            raise APIRequestError(f'HTTP {exc.code}',elapsed_seconds=round(time.monotonic()-started,2),
                timeout_seconds=self.timeout,trace_id=trace_id) from None
        except (urllib.error.URLError,TimeoutError,ConnectionError,http.client.HTTPException) as exc:
            cause=exc.reason if isinstance(exc,urllib.error.URLError) else exc
            raise APIRequestError('timeout' if isinstance(cause,TimeoutError) else 'connection_error',
                exception_type=type(cause).__name__,elapsed_seconds=round(time.monotonic()-started,2),
                timeout_seconds=self.timeout,trace_id=trace_id) from None
        except APIRequestError as exc:
            details={**exc.details};category=details.pop('category')
            raise APIRequestError(category,**{**details,'elapsed_seconds':round(time.monotonic()-started,2),
                'timeout_seconds':self.timeout,'trace_id':trace_id}) from None
        except (json.JSONDecodeError,UnicodeError) as exc:
            raise APIRequestError('invalid_response_encoding',exception_type=type(exc).__name__,
                elapsed_seconds=round(time.monotonic()-started,2),trace_id=trace_id) from None
        return result

    def check_model(self):
        data=self._request('GET','/models')
        ids={x['id'] for x in data.get('data',[]) if isinstance(x,dict) and 'id' in x}
        if self.model not in ids:raise ValueError(f'Configured model not returned by /models: {self.model}')
        return dict(model=self.model,listed=True,vision_inference_tested=False)

    def ask(self,image,prompt,path,**kwargs):
        path=Path(path)
        content=[dict(type='text',text=prompt)] if image is None else [image_part(image),dict(type='text',text=prompt)]
        payload=dict(model=self.model,stream=self.stream,temperature=self.temperature,
            max_tokens=kwargs.get('max_tokens',8192),response_format=dict(type='json_object'),
            messages=[dict(role='system',content=SYSTEM_PROMPT),dict(role='user',content=content)])
        if self.generation_profile=='matched_v1':
            # Both matched backends generate text, then use the same external
            # JSON parser/validator. Do not grant only the API a JSON decoder.
            payload.update(top_p=.9,top_k=50,frequency_penalty=0.,response_format=dict(type='text'))
        if self.thinking!='auto':payload['enable_thinking']=self.thinking=='on'
        logged=deepcopy(payload)
        for part in logged['messages'][1]['content']:
            if part['type']=='image_url':
                encoded=part['image_url']['url']
                part['image_url']=dict(encoded_sha256=hashlib.sha256(encoded.encode()).hexdigest(),redacted=True)
        write(path.with_suffix('.request.json'),dict(base_url=self.base,payload=logged,provider='siliconflow',
            generation_profile=self.generation_profile,
            canonical_image_sha256=hashlib.sha256(image_bytes(image)).hexdigest() if image is not None else None))
        state=dict(parts=[],reasoning_chars=0,done=False,finish_reason=None) if self.stream else None
        try:response=self._request('POST','/chat/completions',payload,stream_state=state,stage=path.stem)
        except APIRequestError as exc:
            partial=''.join(state['parts']) if state is not None else ''
            write(path.with_suffix('.transport_error.json'),dict(**exc.details,stream=self.stream,
                final_chars=len(partial),reasoning_chars=state['reasoning_chars'] if state else None,
                response_id=state.get('id') if state else None))
            if partial:path.with_suffix('.partial.txt').write_text(partial)
            raise
        # Exclude provider reasoning traces; preserve final content and usage.
        choices=response.get('choices',[])
        if not choices:raise ValueError('API response has no choices')
        choice=choices[0];raw=choice.get('message',{}).get('content')
        write(path.with_suffix('.response.json'),dict(id=response.get('id'),model=response.get('model'),
            finish_reason=choice.get('finish_reason'),usage=response.get('usage'),content=raw))
        self.usage.append(dict(model=response.get('model'),usage=response.get('usage',{})))
        if not isinstance(raw,str) or not raw.strip():raise ValueError('API returned no final text; inspect response JSON')
        path.write_text(raw)
        if choice.get('finish_reason')!='stop':
            raise ValueError(f'API answer incomplete: finish_reason={choice.get("finish_reason")}; raw preserved')
        value=parse_json(raw)
        if not isinstance(value,dict):raise ValueError('Need a JSON object from API')
        write(path.with_suffix('.json'),value)
        return value
