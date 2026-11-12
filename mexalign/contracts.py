from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RunInput:
    fixed_mask_path: Path
    moving_mask_path: Path
    registered_mask_path: Path


@dataclass(frozen=True)
class EvaluationResult:
    dice: float
    iou: float
    metadata: Dict[str, str]


@dataclass(frozen=True)
class MeXalignRunInput:
    directory: Path
    fixed_mask_filename: str = "ICP_mask.png"
    moving_mask_filename: str = "F_mask.png"
    fixed_eval_filename: str = "ICP_DNA.tiff"
    script_variant: str = "rotation"  # rotation | gradient_turned | core
    selected_moving_channels: Optional[List[str]] = None
    overrides: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MeXalignRunResult:
    output_dir: Path
    registered_mask_path: Path
    evaluation: EvaluationResult
