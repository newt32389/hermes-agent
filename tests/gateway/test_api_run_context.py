"""Behavior contracts for request-scoped signed API-run authority."""

from concurrent.futures import ThreadPoolExecutor

import pytest

import gateway.api_run_context as api_run_context
from gateway.api_run_context import (
    RunContextReplayStoreUnavailable,
    TrustedApiRunContext,
    bind_trusted_api_run_context,
    current_trusted_api_run_context,
    reset_trusted_api_run_context,
)


def test_binding_is_scoped_and_opaque_id_is_not_represented():
    context = TrustedApiRunContext(
        name="bounded_context",
        context_id="opaque_context_identifier_1234",
        toolsets=("bounded_tools",),
    )

    assert "opaque_context_identifier_1234" not in repr(context)
    assert current_trusted_api_run_context() is None

    token = bind_trusted_api_run_context(context)
    try:
        assert current_trusted_api_run_context() is context
    finally:
        reset_trusted_api_run_context(token)

    assert current_trusted_api_run_context() is None


def test_binding_does_not_leak_to_an_unbound_worker_thread():
    context = TrustedApiRunContext(
        name="bounded_context",
        context_id="opaque_context_identifier_1234",
        toolsets=("bounded_tools",),
    )
    token = bind_trusted_api_run_context(context)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            observed = executor.submit(current_trusted_api_run_context).result()
    finally:
        reset_trusted_api_run_context(token)

    assert observed is None


def test_durable_claim_store_fails_closed_at_capacity_without_evicting_live_claims(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(api_run_context, "_MAX_DURABLE_CLAIMS", 1)

    assert api_run_context.claim_once_durably(
        claim_digest="a" * 64, expires_at=9_999_999_999
    )
    with pytest.raises(RunContextReplayStoreUnavailable):
        api_run_context.claim_once_durably(
            claim_digest="b" * 64, expires_at=9_999_999_999
        )
    assert not api_run_context.claim_once_durably(
        claim_digest="a" * 64, expires_at=9_999_999_999
    )
