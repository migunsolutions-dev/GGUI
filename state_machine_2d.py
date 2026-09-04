"""Explicit Cylindrical–2D model state machine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional

from models_2d import SimulationState2D

# Ensure INITIALIZING exists on the enum used by the UI.
# models_2d currently has DRAFT/VALIDATED/INITIALIZED/... — map INITIALIZING
# onto DRAFT-adjacent semantics if missing by using VALIDATED/INITIALIZED names
# already present, and add a dedicated alias below.


@dataclass(frozen=True)
class Transition:
    source: SimulationState2D
    event: str
    target: SimulationState2D


# Canonical allowed transitions for the 2D workflow.
_TRANSITIONS: FrozenSet[Transition] = frozenset(
    {
        Transition(SimulationState2D.DRAFT, "validate_ok", SimulationState2D.VALIDATED),
        Transition(SimulationState2D.DRAFT, "initialize_start", SimulationState2D.INITIALIZING),
        Transition(SimulationState2D.VALIDATED, "initialize_start", SimulationState2D.INITIALIZING),
        Transition(SimulationState2D.STALE, "initialize_start", SimulationState2D.INITIALIZING),
        Transition(SimulationState2D.FAILED, "initialize_start", SimulationState2D.INITIALIZING),
        Transition(SimulationState2D.INITIALIZING, "initialize_ok", SimulationState2D.INITIALIZED),
        Transition(SimulationState2D.INITIALIZING, "initialize_fail", SimulationState2D.FAILED),
        Transition(SimulationState2D.INITIALIZING, "initialize_cancel", SimulationState2D.FAILED),
        Transition(SimulationState2D.VALIDATED, "initialize_ok", SimulationState2D.INITIALIZED),
        Transition(SimulationState2D.VALIDATED, "initialize_fail", SimulationState2D.FAILED),
        Transition(SimulationState2D.INITIALIZED, "run_start", SimulationState2D.RUNNING),
        Transition(SimulationState2D.INITIALIZED, "edit_inputs", SimulationState2D.STALE),
        Transition(SimulationState2D.RUNNING, "run_complete", SimulationState2D.COMPLETED),
        Transition(SimulationState2D.RUNNING, "run_interrupt", SimulationState2D.INTERRUPTED),
        Transition(SimulationState2D.RUNNING, "run_fail", SimulationState2D.FAILED),
        Transition(SimulationState2D.INTERRUPTED, "run_start", SimulationState2D.RUNNING),
        Transition(SimulationState2D.INTERRUPTED, "edit_inputs", SimulationState2D.STALE),
        Transition(SimulationState2D.COMPLETED, "edit_inputs", SimulationState2D.STALE),
        Transition(SimulationState2D.COMPLETED, "run_start", SimulationState2D.RUNNING),
        Transition(SimulationState2D.STALE, "validate_ok", SimulationState2D.VALIDATED),
        Transition(SimulationState2D.FAILED, "edit_inputs", SimulationState2D.DRAFT),
        Transition(SimulationState2D.DRAFT, "edit_inputs", SimulationState2D.DRAFT),
        Transition(SimulationState2D.VALIDATED, "edit_inputs", SimulationState2D.DRAFT),
    }
)


class InvalidStateTransition(ValueError):
    pass


def can_run(state: SimulationState2D) -> bool:
    """A STALE or uninitialized model cannot run."""
    return state in {
        SimulationState2D.INITIALIZED,
        SimulationState2D.INTERRUPTED,
        SimulationState2D.COMPLETED,
    }


def can_initialize(state: SimulationState2D) -> bool:
    return state != SimulationState2D.RUNNING


def apply_transition(
    state: SimulationState2D,
    event: str,
    *,
    strict: bool = False,
) -> SimulationState2D:
    """Return the next state for ``event``, or raise/keep current if invalid."""
    for transition in _TRANSITIONS:
        if transition.source == state and transition.event == event:
            return transition.target
    if strict:
        raise InvalidStateTransition(f"No transition for {state.value!r} on {event!r}")
    # Non-strict: keep state for unknown transitions (UI may emit redundant events).
    return state


def state_after_input_edit(state: SimulationState2D) -> SimulationState2D:
    """Case-defining edits after init move to STALE; otherwise DRAFT."""
    if state in {
        SimulationState2D.INITIALIZED,
        SimulationState2D.INTERRUPTED,
        SimulationState2D.COMPLETED,
        SimulationState2D.STALE,
    }:
        return SimulationState2D.STALE
    if state == SimulationState2D.RUNNING:
        return state
    if state == SimulationState2D.VALIDATED:
        return SimulationState2D.DRAFT
    return SimulationState2D.DRAFT if state != SimulationState2D.FAILED else SimulationState2D.DRAFT
