from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict

from .core import CoregistrationPipeline as MeXalignPipeline
from .core import PipelineConfig as MeXalignPipelineConfig
from .contracts import MeXalignRunInput, MeXalignRunResult, RunInput
from .pipeline import evaluate_registration


_SCRIPT_MODULES = {
    "rotation": "mexalign.legacy.MeXalign_legacy_rotation",
    "gradient_turned": "mexalign.legacy.MeXalign_legacy_gradient_turned",
}
_SCRIPT_FILES = {
    "rotation": "MeXalign_legacy_rotation.py",
    "gradient_turned": "MeXalign_legacy_gradient_turned.py",
}


def run_full_coregistration(run_input: MeXalignRunInput) -> MeXalignRunResult:
    cfg, worker = build_mexalign_worker(run_input)
    logs = []
    _attach_log_collectors(worker, logs)
    worker.run()

    output_dir, registered_mask = build_output_paths(cfg)

    if not registered_mask.exists():
        recent = "\n".join(logs[-20:]) if logs else "No pipeline logs were captured."
        raise RuntimeError(
            "Coregistration did not produce registered_mask.tif.\n"
            f"Variant: {run_input.script_variant}\n"
            f"Output dir: {output_dir}\n"
            f"Recent logs:\n{recent}"
        )

    evaluation = evaluate_coregistration_output(cfg, run_input, registered_mask)

    return MeXalignRunResult(
        output_dir=output_dir,
        registered_mask_path=registered_mask,
        evaluation=evaluation,
    )


def build_mexalign_worker(run_input: MeXalignRunInput) -> tuple[Any, Any]:
    module = _load_script_module(run_input.script_variant)
    cfg = module.PipelineConfig()
    cfg.directory = str(run_input.directory)
    cfg.fixed_mask_filename = run_input.fixed_mask_filename
    cfg.moving_mask_filename = run_input.moving_mask_filename
    cfg.selected_moving_channels = run_input.selected_moving_channels
    _apply_overrides(cfg, run_input.overrides)
    worker = module.CoregistrationPipeline(cfg)
    return cfg, worker


def build_output_paths(cfg: Any) -> tuple[Path, Path]:
    output_dir = Path(cfg.directory) / cfg.output_subdir
    registered_mask = output_dir / "registered_mask.tif"
    return output_dir, registered_mask


def evaluate_coregistration_output(cfg: Any, run_input: MeXalignRunInput, registered_mask: Path):
    """
    Evaluate coregistration quality by comparing the two cell segmentation masks
    on the same pixel grid.

    Reference  : cfg.fixed_mask_filename  (e.g. ICP_mask.png)
                 — the cell segmentation of the FIXED (reference) tissue.
    Registered : registered_mask.tif
                 — the MOVING tissue's cell segmentation after being warped into
                   fixed-tissue space.

    Dice and IoU therefore answer:
        "After alignment, how well do the cell outlines of the two tissues overlap?"

    NOTE: the fixed_eval_filename field (e.g. ICP_DNA.tiff) is intentionally NOT
    used here.  A fluorescence signal binarised at >0 covers ~90 % of the image
    area, making Dice / IoU essentially insensitive to alignment quality.
    """
    fixed_mask_path = Path(cfg.directory) / cfg.fixed_mask_filename
    if not fixed_mask_path.exists():
        raise FileNotFoundError(
            f"Fixed mask not found for evaluation: {fixed_mask_path}. "
            "Set cfg.fixed_mask_filename to the segmentation mask of the reference tissue."
        )
    return evaluate_registration(
        RunInput(
            fixed_mask_path=fixed_mask_path,
            moving_mask_path=Path(cfg.directory) / cfg.moving_mask_filename,
            registered_mask_path=registered_mask,
        )
    )


def _load_script_module(variant: str):
    if variant.lower() == "mexalign":
        class _ModularModule:
            PipelineConfig = MeXalignPipelineConfig
            CoregistrationPipeline = MeXalignPipeline

        return _ModularModule

    variant_key = variant.lower()
    module_name = _SCRIPT_MODULES.get(variant_key)
    if not module_name:
        supported = ", ".join(sorted(["mexalign", *_SCRIPT_MODULES.keys()]))
        raise ValueError(f"Unknown script_variant '{variant}'. Supported: {supported}")

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        script_name = _SCRIPT_FILES.get(variant_key)
        if not script_name:
            raise

    root = Path(__file__).resolve().parent.parent
    script_path = root / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Expected script file not found: {script_path}")

    file_module_name = f"mexalign_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(file_module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_overrides(cfg: Any, overrides: Dict[str, object]) -> None:
    for key, value in overrides.items():
        _set_cfg_key(cfg, key, value)


def _set_cfg_key(cfg: Any, dotted_key: str, value: object) -> None:
    if not dotted_key:
        raise ValueError("Override key must be non-empty.")

    parts = dotted_key.split(".")
    target = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise AttributeError(f"Unknown override path: {dotted_key}")
        target = getattr(target, part)

    final_key = parts[-1]
    if not hasattr(target, final_key):
        raise AttributeError(f"Unknown override key: {dotted_key}")
    setattr(target, final_key, value)


def _attach_log_collectors(worker: Any, logs: list[str]) -> None:
    def _append(msg: object) -> None:
        logs.append(str(msg))

    if hasattr(worker, "message"):
        worker.message.connect(_append)
    if hasattr(worker, "error"):
        worker.error.connect(lambda msg: logs.append(f"ERROR: {msg}"))


def resolve_eval_reference(case_dir: Path, eval_filename: str) -> Path:
    eval_path = case_dir / eval_filename
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation reference not found: {eval_path}. "
            "Expected original DNA signal image (e.g., ICP_DNA.tiff)."
        )
    return eval_path
