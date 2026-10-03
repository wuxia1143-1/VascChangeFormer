from __future__ import annotations

import argparse
import json

from .derive_v40_decision_ablations import derive_decision_mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--mode", choices=("no_calibration", "no_tail", "central"), required=True
    )
    parser.add_argument("--copy-fold-artifacts", action="store_true")
    args = parser.parse_args()
    summary = derive_decision_mode(
        args.source,
        args.output,
        args.model_name,
        args.mode,
        copy_fold_artifacts=args.copy_fold_artifacts,
    )
    print(json.dumps(summary["pooled_change_metrics"], indent=2))


if __name__ == "__main__":
    main()
