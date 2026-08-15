"""Request-scoped authority for authenticated API-server runs.

The API server validates signed run-context headers before allocating an
agent.  Only the resulting name and opaque context identifier cross into the
agent worker thread; signing material never does.  Plugins read the bound
value through :func:`current_trusted_api_run_context` while handling a tool
call.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TrustedApiRunContext:
    """Validated authority for one API-server run.

    ``context_id`` is intentionally omitted from ``repr`` because it is an
    opaque bearer used only to bind a plugin call to its private adapter.
    """

    name: str
    context_id: str = field(repr=False)
    toolsets: tuple[str, ...]


_CURRENT_TRUSTED_API_RUN_CONTEXT: ContextVar[TrustedApiRunContext | None] = ContextVar(
    "current_trusted_api_run_context", default=None
)


def bind_trusted_api_run_context(
    context: TrustedApiRunContext,
) -> Token[TrustedApiRunContext | None]:
    """Bind *context* to the current execution context."""

    return _CURRENT_TRUSTED_API_RUN_CONTEXT.set(context)


def reset_trusted_api_run_context(
    token: Token[TrustedApiRunContext | None],
) -> None:
    """Restore the binding that preceded *token*."""

    _CURRENT_TRUSTED_API_RUN_CONTEXT.reset(token)


def current_trusted_api_run_context() -> TrustedApiRunContext | None:
    """Return the validated run context visible to the current tool call."""

    return _CURRENT_TRUSTED_API_RUN_CONTEXT.get()
