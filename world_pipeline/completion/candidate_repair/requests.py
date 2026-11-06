"""Bounded correction and immutable receipts for proposal-stage requests."""
import hashlib
import json
from pathlib import Path

from PIL import Image

from ...common import read, write, sha
from ..assembly_experiment import stage_seed


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write(path, value)


class RecordedFailure(ValueError):
    pass


class Requests:
    def __init__(self, folder, seed, board, client):
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True)
        self.seed, self.board, self.client = seed, Path(board), client

    def ask(self, name, prompt, validator, budget=8192):
        receipt = self.folder / (name + '.receipt.json')
        binding = dict(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                       image_sha256=sha(self.board), seed=self.seed, max_tokens=budget, attempts=2)
        if receipt.exists():
            saved = read(receipt)
            if saved['binding'] != binding or any(sha(p) != h for p, h in saved['artifacts'].items()):
                raise ValueError('Saved proposal request binding changed: ' + str(receipt))
            if saved['status'] == 'failed': raise RecordedFailure(saved['error'])
            digest = hashlib.sha256(json.dumps(saved['value'], sort_keys=True).encode()).hexdigest()
            if digest != saved['value_sha256']: raise ValueError('Accepted proposal value changed')
            return saved['value']
        errors = []
        for attempt in range(2):
            path = self.folder / (name + f'_attempt{attempt}.txt')
            if path.exists() or path.with_suffix('.request.json').exists():
                raise RecordedFailure('Interrupted request without receipt: ' + str(path) + '; preserve it and start a new run')
            correction = ''
            if attempt:
                # Do not echo an entire looping 40KB rejected response into the
                # next request. The exact field error is the useful feedback.
                correction = ('\nCORRECT THE COMPLETE JSON RESPONSE. Previous response failed: ' + errors[-1]['error'][:1800] +
                    '\nReturn concise complete records. Preserve valid identity/host references; do not silently delete '
                    'required hosts or turn semantic relations into unrelated spatial aliases. Do not repeat long prose.')
            try:
                with Image.open(self.board) as im:
                    value = self.client().ask(im.convert('RGB'), prompt + correction, path,
                        seed=stage_seed(self.seed, name + ':attempt:' + str(attempt)),
                        max_tokens=budget, profile='completion', sample=True)
                if not isinstance(value, dict): raise ValueError('Return one JSON object with the requested fields')
                parsed = validator(value)
            except Exception as exc:
                error = dict(attempt=attempt, error=type(exc).__name__ + ': ' + str(exc))
                errors.append(error); save(path.with_suffix('.error.json'), error)
                # Runtime/transport errors are not mislabeled as schema errors.
                if not isinstance(exc, ValueError) or attempt:
                    self.finish(receipt, binding, name, dict(status='failed', error=error['error'], errors=errors))
                    raise
                continue
            digest = hashlib.sha256(json.dumps(parsed, sort_keys=True).encode()).hexdigest()
            self.finish(receipt, binding, name, dict(status='accepted', value=parsed, value_sha256=digest, errors=errors))
            return parsed

    def finish(self, receipt, binding, name, result):
        files = {str(p): sha(p) for p in self.folder.glob(name + '_attempt*') if p.is_file()}
        save(receipt, dict(result, binding=binding, artifacts=files))
