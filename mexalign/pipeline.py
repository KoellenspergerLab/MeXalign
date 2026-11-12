from pathlib import Path

from .contracts import EvaluationResult, RunInput
from .io_masks import read_binary_mask
from .metrics import dice_score, iou_score


def evaluate_registration(run_input: RunInput) -> EvaluationResult:
    fixed = read_binary_mask(Path(run_input.fixed_mask_path))
    registered = read_binary_mask(Path(run_input.registered_mask_path))

    return EvaluationResult(
        dice=dice_score(fixed, registered),
        iou=iou_score(fixed, registered),
        metadata={
            "fixed_reference_path": str(run_input.fixed_mask_path),
            "fixed_mask_path": str(run_input.fixed_mask_path),
            "registered_mask_path": str(run_input.registered_mask_path),
            "moving_mask_path": str(run_input.moving_mask_path),
            "metrics": "dice,iou",
        },
    )
