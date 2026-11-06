"""Inspect repeated complete JSON objects without repairing truncated answers."""
import json


def complete_array_prefix(text):
    """Diagnostic only: return complete dict entries before an unfinished tail."""
    start = text.find('[')
    if start < 0:
        return []
    decoder, entries, offset = json.JSONDecoder(), [], start+1
    while offset < len(text):
        while offset < len(text) and text[offset] in ' \r\n\t,':
            offset += 1
        if offset >= len(text) or text[offset] == ']':
            break
        try:
            value, offset = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            break
        if not isinstance(value, dict):
            break
        entries.append(value)
    return entries


def repeated_object_tail(text, repeats=3):
    """Detect exact duplicate records, preserving instance identity.

    Different chairs/cushions may legitimately have identical attributes. Never
    strip id/track_id here: this is a generation-loop guard, not category-level
    deduplication. Schema validators still check missing/duplicate IDs after EOS.
    """
    entries = complete_array_prefix(text)
    if len(entries) < repeats:
        return False
    signatures = [json.dumps(item, sort_keys=True, ensure_ascii=False)
                  for item in entries[-repeats:]]
    return len(set(signatures)) == 1


class RepeatedJSONObjectStop:
    """Transformers stopping-criteria callable; stops a loop as failure, not success."""
    def __init__(self, processor, prompt_length, check_every=32):
        self.processor, self.prompt_length = processor, prompt_length
        self.check_every, self.triggered = check_every, False

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        length = input_ids.shape[1]-self.prompt_length
        if length and length % self.check_every == 0:
            text = self.processor.batch_decode(input_ids[:, self.prompt_length:], skip_special_tokens=True)[0]
            self.triggered = repeated_object_tail(text)
        return torch.full((input_ids.shape[0],), self.triggered, dtype=torch.bool, device=input_ids.device)
