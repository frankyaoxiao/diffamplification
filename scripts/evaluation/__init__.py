"""Evaluation package."""

# Re-export canonical CoT evaluator path for convenience
try:
    from .cot_amp.amplify_cot import main as cot_amp_main  # noqa: F401
except Exception:
    pass


