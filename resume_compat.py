"""No cross-model or source-changing resumes."""
def compatible_config(previous, current, *, inference=False):
    return previous == current
