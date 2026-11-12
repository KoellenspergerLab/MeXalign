import argparse
import json
from pathlib import Path

from .contracts import RunInput
from .pipeline import evaluate_registration


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate coregistration with Dice and IoU.")
    parser.add_argument("--fixed-mask", required=True, type=Path)
    parser.add_argument("--moving-mask", required=True, type=Path)
    parser.add_argument("--registered-mask", required=True, type=Path)
    args = parser.parse_args()

    result = evaluate_registration(
        RunInput(
            fixed_mask_path=args.fixed_mask,
            moving_mask_path=args.moving_mask,
            registered_mask_path=args.registered_mask,
        )
    )
    print(
        json.dumps(
            {
                "dice": result.dice,
                "iou": result.iou,
                "metadata": result.metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
