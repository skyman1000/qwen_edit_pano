"""Inference-only diagnostic policies; default training/runtime remains unchanged."""
from contextlib import contextmanager


@contextmanager
def world_cfg_policy(transformer,world,policy):
    if policy not in ('both','positive_only'):raise ValueError('Unknown World CFG policy')
    original=transformer.cache_context;sentinel=object()
    instance_original=transformer.__dict__.get('cache_context',sentinel)
    calls=dict(policy=policy,cond=0,uncond=0,disabled_uncond=0)
    @contextmanager
    def cache_context(name):
        if name not in ('cond','uncond'):raise ValueError('Unexpected Qwen CFG context: '+str(name))
        calls[name]+=1;saved=world.current
        try:
            with original(name):
                if policy=='positive_only' and name=='uncond':
                    world.current=None;calls['disabled_uncond']+=1
                yield
        finally:
            world.current=saved
    transformer.cache_context=cache_context
    try:yield calls
    finally:
        if instance_original is sentinel:del transformer.cache_context
        else:transformer.cache_context=instance_original
