"""
Manual tail-call optimization for Python.

CPython has no native TCO: a "simple" tail-recursive function can
still raise `RecursionError` on large inputs (long sibling chains,
many merge iterations, long sentence lists, deep trees, ...). This
module provides the classic functional-language workaround -- a
*tail_call_optimized*:

  * A tail call doesn't recurse directly; it returns a `TailCall`, a
    deferred "call this with no arguments next".
  * `tail_call_optimized(step)` wraps `step` in a driver that repeatedly
    unwraps `TailCall`s in a flat loop until a real value comes back.

Each "hop" the driver takes is one ordinary Python call/return — O(1)
stack depth — so the *loop*, not the call stack, grows with recursion
depth. The `while` loop inside `tail_call_optimized()` is the one deliberate
exception to this project's "no explicit loops" style: it is the
tail_call_optimized harness itself, not application logic, and it is the only
way to bridge Python's non-tail-call-optimized call stack into a flat
iteration. Every recursive function built on top of it is still
written as ordinary self-recursion.

IMPORTANT — the self-recursion trap: a tail-call optimized function must call
its own *undecorated* body from its `TailCall` thunks, never the
public, already-wrapped name. Calling the wrapped name would start a
*new*, nested tail_call_optimized driver on every hop (adding a real stack
frame each time) instead of being unwound by the *same* outer loop.
The convention used throughout this codebase is::

    def _foo_step(...):                       # raw, self-recursive body
        ...
        return TailCall(lambda: _foo_step(...))  # calls itself, not `foo`

    foo = tail_call_optimized(_foo_step)                # public, driver-wrapped name

Usage::

    def _countdown_step(n):
        return "done" if n <= 0 else TailCall(lambda: _countdown_step(n - 1))

    countdown = tail_call_optimized(_countdown_step)
    countdown(10**6)  # O(1) Python stack frames, not RecursionError
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class TailCall:
    """A deferred tail call: `thunk` is a zero-argument callable that
    produces the next step -- another `TailCall`, or a final value."""

    thunk: Callable[[], Any]


def tail_call_optimized(step: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wrap a `TailCall`-returning, self-recursive `step` function in a
    driver that flattens its recursion into a single loop, giving it
    O(1) Python call-stack growth regardless of how many hops it takes.
    """

    @wraps(step)
    def driver(*args: Any, **kwargs: Any) -> Any:
        current = step(*args, **kwargs)
        while isinstance(current, TailCall):
            current = current.thunk()
        return current

    return driver


@dataclass(frozen=True)
class AsyncTailCall:
    """An asynchronous deferred tail call: `thunk` produces an awaitable
    that resolves to the next step -- another `AsyncTailCall` or a final value."""

    thunk: Callable[[], Awaitable[Any]]


def async_tail_call_optimized(
    step: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """
    Async equivalent of `tail_call_optimized`. Wraps an `AsyncTailCall`-returning
    async `step` function in a driver that flattens its recursion into an async
    `while` loop.
    """

    @wraps(step)
    async def driver(*args: Any, **kwargs: Any) -> Any:
        current = await step(*args, **kwargs)
        while isinstance(current, AsyncTailCall):
            current = await current.thunk()
        return current

    return driver


def depth_first_search(
    get_children: Callable[[T], list[T]], transform: Callable[[T], R]
) -> Callable[[list[T], list[R]], list[R]]:
    """
    Build a tail-recursive, pre-order depth-first
    traversal: ``traverse(pending, done)`` visits every node reachable
    from `pending` via `get_children` (depth-first, left-to-right,
    parent before children -- the same order plain recursion would
    visit them in) and returns ``done + [transform(node) for node in
    that order]``.

    Shared by every "flatten/walk/render a recursive tree" function in
    this project (models.py, main.py) so none of them are bounded by
    Python's call-stack depth.
    """

    def _step(pending: list[T], done: list[R]) -> Any:
        if not pending:
            return done
        head, *rest = pending
        return TailCall(
            lambda: _step(
                [*get_children(head), *rest], [*done, transform(head)]
            )
        )

    return tail_call_optimized(_step)
