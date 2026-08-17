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
import json
import os
import sqlite3
import time
from contextlib import closing
from typing import Any

from hermes_constants import get_hermes_home


RUN_CONTEXT_AUDIENCE = "hermes-api-server/v1/runs"
RUN_CONTEXT_CLAIM_VERSION = "v1"
RUN_CONTEXT_CLOCK_SKEW_SECONDS = 5
_MAX_DURABLE_CLAIMS = 10_000


class RunContextReplayStoreUnavailable(RuntimeError):
    """The durable one-shot fence could not safely accept a claim."""


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


def canonical_request_digest(body: Any) -> str:
    """Return the SHA-256 of the canonical JSON request body.

    The gateway authenticates the parsed JSON value, rather than incidental
    whitespace or key order on the wire.  JSON's strict encoder gives callers
    one unambiguous representation and rejects non-finite values.
    """
    import hashlib

    canonical = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def canonical_signature_payload(
    *, context_name: str, context_id: str, expires_at: int, body_digest: str
) -> bytes:
    """Canonical bytes covered by a signed run-context claim."""
    return (
        f"{RUN_CONTEXT_CLAIM_VERSION}\n{context_name}\n{context_id}\n"
        f"{RUN_CONTEXT_AUDIENCE}\n{expires_at}\n{body_digest}"
    ).encode("utf-8")


def _claim_db_path():
    return get_hermes_home() / "state.db"


def durable_claim_store_available() -> bool:
    """Conservatively report whether the configured durable claim path is usable."""
    path = _claim_db_path()
    parent = path.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        return False
    return not path.exists() or (path.is_file() and os.access(path, os.W_OK))


def claim_once_durably(*, claim_digest: str, expires_at: int) -> bool:
    """Atomically claim an unexpired signed request across gateway restarts.

    We never evict a live claim to make room: reaching the bounded limit is a
    temporary availability failure, not permission to replay an accepted
    operation.  Only SHA-256 claim digests enter persistent storage.
    """
    now = time.time()
    path = _claim_db_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=10)) as conn:
            conn.execute("PRAGMA busy_timeout = 10000")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS api_run_context_claims (
                    claim_digest TEXT PRIMARY KEY,
                    expires_at INTEGER NOT NULL,
                    claimed_at REAL NOT NULL
                )"""
            )
            with conn:
                # The verifier admits a claim until ``expires_at + skew``.
                # Retain it through that same window or an accepted claim at
                # the boundary could be replayed after a restart.
                conn.execute(
                    "DELETE FROM api_run_context_claims WHERE expires_at + ? < ?",
                    (RUN_CONTEXT_CLOCK_SKEW_SECONDS, now),
                )
                row = conn.execute(
                    "SELECT 1 FROM api_run_context_claims WHERE claim_digest = ?",
                    (claim_digest,),
                ).fetchone()
                if row is not None:
                    return False
                count = conn.execute(
                    "SELECT COUNT(*) FROM api_run_context_claims"
                ).fetchone()[0]
                if count >= _MAX_DURABLE_CLAIMS:
                    raise RunContextReplayStoreUnavailable(
                        "signed run-context claim store is full"
                    )
                conn.execute(
                    "INSERT INTO api_run_context_claims (claim_digest, expires_at, claimed_at) VALUES (?, ?, ?)",
                    (claim_digest, expires_at, now),
                )
                return True
    except RunContextReplayStoreUnavailable:
        raise
    except (OSError, sqlite3.Error) as exc:
        raise RunContextReplayStoreUnavailable(
            "signed run-context claim store unavailable"
        ) from exc
