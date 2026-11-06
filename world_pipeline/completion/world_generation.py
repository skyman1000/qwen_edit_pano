"""An explicit common generation recipe; never promises identical server internals."""
import hashlib
import io
from pathlib import Path

from PIL import Image

from ..common import read,write
from .geometry import parse_json

SYSTEM_PROMPT='Return one JSON object matching the requested schema. Provide concise assumptions, not hidden chain-of-thought.'
MATCHED_OPTIONS=dict(do_sample=True,temperature=.4,top_p=.9,top_k=50,repetition_penalty=1.)


def image_bytes(image):
    rgb=image.convert('RGB').copy();rgb.thumbnail((1536,1536))
    buffer=io.BytesIO();rgb.save(buffer,format='JPEG',quality=92)
    return buffer.getvalue()


def recipe():
    return dict(profile='matched_v1',system_prompt=SYSTEM_PROMPT,temperature=.4,top_p=.9,top_k=50,
        repetition_penalty=1.,frequency_penalty=0.,presence_penalty=0.,requested_API_thinking=False,
        image='same_RGB_JPEG92_max1536_bytes_before_backend_processing',
        output_format='text generation followed by the same external JSON parser and validation',
        seed_scope='local_seed_recorded; API_seed_not_documented_or_sent',
        limitations=['server_internal_decoding_not_verified','tokenizers_and_vision_encoders_differ',
                    'token_budgets_are_model_tokenizer_specific'])


class MatchedLocalJSON:
    execution_kind='local_Qwen'
    generation_profile='matched_v1'

    def __init__(self,model,engine=None):
        if engine is None:
            from .qwen_upstream import Qwen
            engine=Qwen(model)
        self.engine=engine;self.model_path=str(model)

    def ask(self,image,prompt,path,seed=3407,max_tokens=8192,**kwargs):
        path=Path(path);data=image_bytes(image) if image is not None else None
        settings=dict(prompt=prompt,system_prompt=SYSTEM_PROMPT,seed=seed,max_tokens=max_tokens,
            model=self.model_path,generation=dict(MATCHED_OPTIONS),generation_profile=self.generation_profile,
            text_only=image is None,canonical_image_sha256=hashlib.sha256(data).hexdigest() if data else None)
        request_path=path.with_suffix('.request.json');parsed_path=path.with_suffix('.json')
        if parsed_path.exists() and request_path.exists() and read(request_path)==settings:
            value=read(parsed_path)
            if isinstance(value,dict):return value
        parsed_path.unlink(missing_ok=True);write(request_path,settings)
        content=([dict(type='image')] if data is not None else [])+[dict(type='text',text=prompt)]
        messages=[dict(role='system',content=SYSTEM_PROMPT),dict(role='user',content=content)]
        engine=self.engine
        text=engine.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=dict(text=[text],return_tensors='pt')
        if data is not None:
            with Image.open(io.BytesIO(data)) as decoded:inputs['images']=[decoded.convert('RGB').copy()]
        encoded=engine.processor(**inputs).to(engine.model.device);encoded.pop('token_type_ids',None)
        engine.torch.manual_seed(seed)
        with engine.torch.inference_mode():
            tokens=engine.model.generate(**encoded,max_new_tokens=max_tokens,**MATCHED_OPTIONS)
        tokens=tokens[:,encoded.input_ids.shape[1]:]
        raw=engine.processor.batch_decode(tokens,skip_special_tokens=True)[0];path.write_text(raw)
        eos=engine.model.generation_config.eos_token_id
        eos=eos if isinstance(eos,(list,tuple)) else ([] if eos is None else [eos])
        ended=bool(tokens.shape[1] and int(tokens[0,-1]) in eos)
        finish='eos' if ended else 'length' if tokens.shape[1]>=max_tokens else 'stop'
        write(path.with_suffix('.generation.json'),dict(generated_tokens=tokens.shape[1],max_tokens=max_tokens,
            finish_reason=finish,ended_with_eos=ended,generation_profile=self.generation_profile))
        if finish=='length':raise ValueError('Unfinished local response reached token limit; original text preserved')
        value=parse_json(raw)
        if not isinstance(value,dict):raise ValueError('The top-level JSON must be an object with the named request fields')
        write(parsed_path,value);return value
