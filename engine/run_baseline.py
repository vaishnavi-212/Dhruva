#!/usr/bin/env python3
"""Entry point. Establishes the 'before' picture on IO-VNBD.

    python run_baseline.py data/your_drive.csv

Produces results/position_plot.png -- the plot SIH26168 requires in the
proposal -- plus results/drift_curve.png and results/summary.json.
"""
import sys
from harness import dataset, outage, baselines, evaluate


def main(path: str, hz: float = 10.0):
    trace = dataset.load(path, hz=hz)

    outs = outage.by_distance(trace, targets_m=(50, 200, 500, 1000, 2000))
    outs = outage.snap_to_fixes(trace, outs)
    outs = outage.moving_only(trace, outs)
    if not outs:
        print("No usable blackout windows — is the vehicle moving in this trace?")
        return

    evaluate.run(trace, baselines.REGISTRY, outs, outdir="results")

    print("\nNext: write your model with the same signature as "
          "harness/baselines.py:naive_integration, add it to the predictors "
          "dict, and re-run. The gap you open over naive_integration is the "
          "entire contribution.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 10.0)
