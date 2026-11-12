"""Standalone coregistration module scaffold."""

from .compat_runner import run_full_coregistration
from .contracts import MeXalignRunInput, MeXalignRunResult, EvaluationResult, RunInput
from .pipeline import evaluate_registration

__all__ = [
    "MeXalignRunInput",
    "MeXalignRunResult",
    "EvaluationResult",
    "RunInput",
    "evaluate_registration",
    "run_full_coregistration",
]
