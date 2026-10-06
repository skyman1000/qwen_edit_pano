"""Atomic CPU data checkpoints. Never silently overwrite a partial sample."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time

from .common import read, sha


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_rows(path, values):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


@contextmanager
def output_lock(output):
    with (Path(output)/'.writer.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'Another writer holds {output}') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def archive_partial(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f'Refusing to modify reused source: {path}')
    if path.exists():
        dest = path.with_name(path.name + f'.interrupted-{time.time_ns()}')
        path.rename(dest)


def file_hashes(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(Path(directory).rglob('*'))
            if p.is_file() and p.name not in {'DATA_COMPLETE.json', 'DATA_COMPLETE.json.tmp'}}


def verify_files(directory, hashes):
    for name, expected in hashes.items():
        path = Path(directory)/name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Checkpoint file missing or changed: {path}')


def commit_sample(directory, manifest, observer):
    atomic_json(Path(directory)/'DATA_COMPLETE.json', dict(
        manifest=manifest, observer=observer, files=file_hashes(directory)))


def load_sample(directory):
    record = read(Path(directory)/'DATA_COMPLETE.json')
    verify_files(directory, record['files'])
    return record['manifest'], record['observer']
