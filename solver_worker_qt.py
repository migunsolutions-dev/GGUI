"""Thin Qt adapter alias for the solver worker.

The production solver thread remains ``SolverRunner`` in ``solver_runner.py``
during migration; this module re-exports it so new code can import a dedicated
worker name while planning stays in ``execution_plan.py``.
"""
from __future__ import annotations

from solver_runner import SolverRunner as SolverWorker

__all__ = ["SolverWorker"]
