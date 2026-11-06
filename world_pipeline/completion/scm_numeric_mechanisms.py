"""Numeric answer events; retain aligned full-sequence scoring including EOS.

The background/configuration text mechanisms are unchanged for an isolated
comparison. Counts are JSON integers, without a count-dependent prose suffix.
No probabilities from the old forced-prefix diagnostic are reused.
"""
from .scm_typed_queries import ContinuationMechanismScorer


class NumericCountScorer(ContinuationMechanismScorer):
    count_event_contract='json_integer_v1'

    def score(self,description,parents,states):
        numeric=all(s['id'].isdigit() for s in states)
        if numeric and any(s['description']!=str(int(s['id'])) for s in states):
            raise ValueError('Numeric mechanism needs canonical JSON integer answer events')
        answer=super().score(description,parents,states)
        answer['count_event_contract']=self.count_event_contract if numeric else 'unchanged_text_v1'
        return answer
