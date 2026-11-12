import argparse
import json
from pathlib import Path

from .compat_runner import run_full_coregistration
from .contracts import MeXalignRunInput


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full legacy coregistration behavior through the modular interface."
    )
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--fixed-mask-filename", default="ICP_mask.png")
    parser.add_argument("--moving-mask-filename", default="F_mask.png")
    parser.add_argument(
        "--fixed-eval-filename",
        default="ICP_DNA.tiff",
        help="Reference image for evaluation (will be binarized), e.g. ICP_DNA.tiff",
    )
    parser.add_argument(
        "--script-variant",
        default="rotation",
        choices=["rotation", "gradient_turned", "mexalign"],
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Override config values via dotted path, e.g. "
            "'coarse_config.iters=150' or 'rotation_enabled=true'"
        ),
    )
    args = parser.parse_args()

    overrides = _parse_overrides(args.override)
    result = run_full_coregistration(
        MeXalignRunInput(
            directory=args.directory,
            fixed_mask_filename=args.fixed_mask_filename,
            moving_mask_filename=args.moving_mask_filename,
            fixed_eval_filename=args.fixed_eval_filename,
            script_variant=args.script_variant,
            overrides=overrides,
        )
    )

    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "registered_mask_path": str(result.registered_mask_path),
                "evaluation": {
                    "dice": result.evaluation.dice,
                    "iou": result.evaluation.iou,
                    "metadata": result.evaluation.metadata,
                },
            },
            indent=2,
        )
    )


def _parse_overrides(override_items: list[str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for item in override_items:
        if "=" not in item:
            raise ValueError(f"Invalid --override '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        parsed[key] = _coerce_value(value.strip())
    return parsed


def _coerce_value(raw: str) -> object:
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


if __name__ == "__main__":
    main()
