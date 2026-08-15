"""Behavior contracts for request-scoped signed API-run authority."""

from concurrent.futures import ThreadPoolExecutor

from gateway.api_run_context import (
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
