"""Bounded retries of rejected HTTP 429 requests, without new model draws.

The original API codec is reused unchanged. Timeouts, partial streams and JSON
errors are never retried here. Successful wire files are published at the
logical request path so the original accepted-response cache still verifies.
"""
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import urllib.error

from ..common import ROOT,read,write
from .world_api import APIRequestError


def retry_after(value,now=None):
    if value is None:return None
    try:seconds=float(value)
    except (ValueError,TypeError):
        try:
            date=parsedate_to_datetime(value)
            if date.tzinfo is None:date=date.replace(tzinfo=timezone.utc)
            seconds=date.timestamp()-(time.time() if now is None else now)
        except (ValueError,TypeError,OverflowError):return None
    return max(0.,seconds) if math.isfinite(seconds) else None


class RateGate:
    """Cooperating processes with the same account/model share start spacing."""
    def __init__(self,path,interval=3.,clock=time.time,sleep=time.sleep):
        self.path=Path(path);self.interval=interval;self.clock=clock;self.sleep=sleep
        self.path.parent.mkdir(parents=True,exist_ok=True)

    def state(self,update=None):
        fd=os.open(self.path,os.O_RDWR|os.O_CREAT,0o600)
        with os.fdopen(fd,'r+') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            raw=f.read();state=json.loads(raw) if raw else dict(last_start=0.,not_before=0.)
            result=update(state) if update else deepcopy_state(state)
            if update:f.seek(0);f.truncate();json.dump(state,f);f.flush()
            return result

    def cooldown(self,seconds):
        self.state(lambda s:s.update(not_before=max(s['not_before'],self.clock()+seconds)))

    def wait(self,budget=300.):
        waited=0.
        while True:
            def reserve(s):
                now=self.clock();delay=max(s['last_start']+self.interval,s['not_before'])-now
                if delay<=0:s['last_start']=now
                return max(0.,delay)
            delay=self.state(reserve)
            if delay<=0:return waited
            if delay>budget-waited:raise APIRequestError('rate_limit_wait_budget_exhausted',required_wait_seconds=round(delay,2),maximum_wait_seconds=budget,request_started=False)
            step=min(30.,delay);print(json.dumps(dict(stage='api_rate_wait',seconds=round(step,2))),flush=True)
            self.sleep(step);waited+=step


def deepcopy_state(state):return dict(state)


class PacedAPI:
    execution_kind='siliconflow_api'
    def __init__(self,inner,interval=3.,retries=3,backoff=20.,wait_budget=300.,gate=None):
        if type(retries) is not int or not 0<=retries<=3:raise ValueError('HTTP429 retry budget must be 0..3')
        if any(not math.isfinite(v) or v<0 for v in (interval,backoff,wait_budget)) or wait_budget<=0:raise ValueError('Finite nonnegative pacing settings and positive wait budget required')
        self.inner=inner;self.max_calls=inner.max_calls;self.logical_calls=0;self.retries=retries;self.backoff=backoff;self.wait_budget=wait_budget
        digest=hashlib.sha256((inner.base+'|'+inner.model+'|'+inner.key).encode()).hexdigest()
        self.gate=gate or RateGate(ROOT/'qwen_edit_pano/outputs/.siliconflow_api_pacing'/(digest+'.json'),interval)
        self._opener=inner.opener;self.server_delay=None;inner.opener=self._open

    def __getattr__(self,name):return getattr(self.inner,name)

    def _open(self,*args,**kwargs):
        try:return self._opener(*args,**kwargs)
        except urllib.error.HTTPError as exc:
            if exc.code==429:self.server_delay=retry_after(exc.headers.get('Retry-After') if exc.headers else None)
            raise

    def ask(self,image,prompt,path,**kwargs):
        if self.logical_calls>=self.max_calls:raise APIRequestError('logical_request_budget_exhausted',max_calls=self.max_calls,request_started=False)
        self.logical_calls+=1;self.inner.max_calls=self.max_calls*(1+self.retries)
        path=Path(path);folder=path.parent/(path.stem+'_wire_calls');folder.mkdir(parents=True,exist_ok=False)
        events=[];waited=0.
        for attempt in range(self.retries+1):
            waited+=self.gate.wait(self.wait_budget-waited)
            wire=folder/f'wire_{attempt:02d}.txt';self.server_delay=None
            try:
                value=self.inner.ask(image,prompt,wire,**kwargs)
                self.publish(wire,path);events.append(dict(wire_attempt=attempt,status='response_received'))
                write(folder/'retry_summary.json',dict(logical_calls=self.logical_calls,events=events,total_wait_seconds=waited));return value
            except APIRequestError as exc:
                category=exc.details.get('category');events.append(dict(wire_attempt=attempt,status=category))
                write(folder/'retry_summary.json',dict(logical_calls=self.logical_calls,events=events,total_wait_seconds=waited))
                if category!='HTTP 429' or attempt==self.retries:
                    self.publish(wire,path);raise
                delay=max(self.backoff*2**attempt,self.server_delay or 0.)
                self.gate.cooldown(delay)
                print(json.dumps(dict(stage='api_http429_backoff',wire_attempt=attempt+1,retries_remaining=self.retries-attempt,delay_seconds=delay)),flush=True)
                # RateGate respects Retry-After and rejects waits beyond the
                # remaining bounded budget instead of retrying too soon.
            except ValueError:
                self.publish(wire,path);raise

    @staticmethod
    def publish(wire,path):
        for suffix in ('.txt','.json','.request.json','.response.json','.transport_error.json','.partial.txt'):
            src=wire.with_suffix(suffix)
            if src.exists():shutil.copyfile(src,path.with_suffix(suffix))


def wrap(provider):
    return PacedAPI(provider,interval=float(os.environ.get('CAUPANO_API_MIN_INTERVAL','3')),
        retries=int(os.environ.get('CAUPANO_API_429_RETRIES','3')),
        backoff=float(os.environ.get('CAUPANO_API_429_BACKOFF','20')),
        wait_budget=float(os.environ.get('CAUPANO_API_WAIT_BUDGET','300')))
