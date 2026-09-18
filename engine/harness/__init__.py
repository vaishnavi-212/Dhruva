"""Evaluation harness for SIH26168 — Intelligent Dead Reckoning.

Built around ISRO's stated benchmark: positional drift < 10% of distance
travelled during a GNSS blackout.
"""
__all__ = ["dataset", "outage", "metrics", "baselines", "plots", "evaluate"]
