"""Check resume settings; training source hashes are provenance, not a lock."""


def config_differences(previous, current, *, inference=False):
    # Editing inference, diagnostics or comments must not invalidate training
    # checkpoints. Keep hashes in saved configs for provenance only. Model,
    # data/cache identity and training settings still have to match.
    # Existing inference output reuse remains strict, including LoRA scale.
    ignored = set() if inference else {'source_hashes'}
    missing = object()
    return sorted(k for k in (previous.keys() | current.keys()) - ignored
                  if previous.get(k, missing) != current.get(k, missing))


def compatible_config(previous, current, *, inference=False):
    return not config_differences(previous, current, inference=inference)
