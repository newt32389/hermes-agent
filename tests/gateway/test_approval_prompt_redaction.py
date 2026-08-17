"""Regression test for approval prompt credential redaction (issue #48456).

When Tirith flags a command for containing a credential-shaped pattern, the
gateway approval prompt must redact the credential from the command text
before sending it to the chat platform. Without this fix, the raw command
(with the credential in plaintext) is sent verbatim to Telegram/Discord/etc.,
undoing Tirith's redaction one layer up.

The redaction is wired through the module-level ``_redact_approval_command``
seam. These tests bind that seam -- the production wiring -- not just the
underlying ``redact_sensitive_text`` helper, so they fail if the redaction
call is removed from either approval path.

Credential fixtures are built at runtime from a benign prefix + a run of
``X`` characters (the same trick tests/agent/test_redact.py uses): they match
the redactor regexes so the assertions stay meaningful, but contain no real
or real-looking key, so secret scanners do not flag this file.
"""

import asyncio
import threading

import pytest

from gateway.run import _redact_approval_command

# Synthetic, scanner-safe credential fixtures. Each matches its redactor
# regex (ghp_/sk-/JWT) but is unmistakably fake -- a run of X's, never a
# real or real-format key.
_FAKE_GHP = "ghp_" + "X" * 36
_FAKE_OPENAI = "sk-proj-" + "X" * 40
_FAKE_JWT = "eyJ" + "X" * 20 + "." + "eyJ" + "X" * 24 + "." + "X" * 30


class TestRedactApprovalCommand:
    """Contract for the approval-prompt redaction seam used by the gateway."""

    def test_redacts_github_pat(self):
        raw = "curl -H 'Authorization: token " + _FAKE_GHP + "' https://api.github.com/user"
        out = _redact_approval_command(raw)
        assert _FAKE_GHP not in out
        # command structure preserved so the operator can still judge the action
        assert "curl" in out
        assert "github.com" in out

    def test_redacts_openai_key(self):
        raw = "export OPENAI_API_KEY=" + _FAKE_OPENAI + " && python s.py"
        out = _redact_approval_command(raw)
        assert _FAKE_OPENAI not in out
        assert "python s.py" in out

    def test_redacts_bearer_token(self):
        raw = "curl -H 'Authorization: Bearer " + _FAKE_JWT + "' https://api.example.com"
        out = _redact_approval_command(raw)
        assert _FAKE_JWT not in out


    def test_forces_redaction_even_when_disabled(self, monkeypatch):
        """force=True must redact even if security.redact_secrets is off -- the
        approval prompt is a hard secret-egress boundary regardless of config."""
        raw = "curl -H 'Authorization: token " + _FAKE_GHP + "' https://api.github.com"
        # With redaction globally disabled, the seam must STILL redact (force=True).
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False, raising=False)
        out = _redact_approval_command(raw)
        assert _FAKE_GHP not in out


class TestApprovalCommandWiring:
    """Guard the production wiring on BOTH approval-notify transports:
    The chat-platform seam is source-level coverage. The API path has a
    behavior-level signed-context test below because its final delivery is
    intentionally mediated by the active-run fence rather than a direct queue
    write."""

    def _assert_redacts_then_uses(self, module, func_name: str, sink_substr: str):
        """Parse `module`'s full AST, locate the (possibly nested) function
        `func_name`, and assert it contains an assignment
        `<x> = _redact_approval_command(...)` whose result is then used by a
        statement matching `sink_substr` on a LATER line. Walking the real AST
        (not a source slice) is refactor-robust and rejects discarded-result
        calls (the call must be an assignment, not a bare expression)."""
        import ast
        import inspect

        source = inspect.getsource(module)
        tree = ast.parse(source)
        target_fn = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                target_fn = node
                break
        assert target_fn is not None, f"function {func_name} not found in {module.__name__}"

        redact_line = None
        for node in ast.walk(target_fn):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fn = node.value.func
                if isinstance(fn, ast.Name) and fn.id == "_redact_approval_command":
                    redact_line = node.lineno
        assert redact_line is not None, (
            f"{func_name} must assign the result of _redact_approval_command(...) "
            "(a discarded-result call would still leak the raw command)"
        )

        sink_line = None
        for node in ast.walk(target_fn):
            seg = ast.get_source_segment(source, node)
            if seg and sink_substr in seg and getattr(node, "lineno", 0) > redact_line:
                sink_line = node.lineno
                break
        assert sink_line is not None, (
            f"`{sink_substr}` sink not found after the redaction in {func_name}"
        )

    def test_chat_platform_path_redacts_before_send(self):
        import gateway.run as run

        self._assert_redacts_then_uses(run, "_approval_notify_sync", "send_exec_approval")

    @pytest.mark.asyncio
    async def test_signed_api_approval_event_is_redacted_and_active_fenced(
        self, monkeypatch
    ):
        """A signed approval notification is redacted and cannot outlive SSE.

        This drives the actual nested API callback rather than constraining its
        implementation detail.  The first notification proves delivery and
        exact-ID redaction; after the stream is detached, a second notification
        must not be enqueued into the retired queue.
        """
        from unittest.mock import MagicMock, patch

        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from gateway.config import PlatformConfig
        from gateway.platforms.api_server import (
            APIServerAdapter,
            cors_middleware,
            security_headers_middleware,
        )
        from tools import approval as approval_mod

        context_name = "fitness_mobile_plan_generation"
        context_id = "dispatch_abcdefghijklmnopqrstuvwxyz012345"
        secret_name = "TEST_HERMES_RUN_CONTEXT_SIGNING_SECRET"
        secret = "s" * 64
        monkeypatch.setenv(secret_name, secret)
        config = {
            "gateway": {
                "api_server": {
                    "trusted_run_contexts": {
                        context_name: {
                            "signing_secret_env": secret_name,
                            "toolset_mode": "replace",
                            "toolsets": [context_name],
                        }
                    }
                }
            }
        }
        payload = {"input": "hello"}
        import hashlib
        import hmac
        import json
        import time

        expires_at = int(time.time()) + 60
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        signature = hmac.new(
            secret.encode(),
            (
                f"v1\n{context_name}\n{context_id}\nhermes-api-server/v1/runs\n"
                f"{expires_at}\n{digest}"
            ).encode(),
            hashlib.sha256,
        ).hexdigest()
        headers = {
            "X-Hermes-Run-Context": context_name,
            "X-Hermes-Run-Context-Id": context_id,
            "X-Hermes-Run-Context-Signature": signature,
            "X-Hermes-Run-Context-Claim": f"v1.{expires_at}.{digest}",
        }
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
        app = web.Application(
            middlewares=[mw for mw in (cors_middleware, security_headers_middleware) if mw]
        )
        app.router.add_post("/v1/runs", adapter._handle_runs)
        first_sent = threading.Event()
        release_second = threading.Event()
        agent = MagicMock()

        def _run(*, task_id, **_kwargs):
            callback = approval_mod._gateway_notify_cbs[task_id]
            callback({"command": f"echo {context_id} {_FAKE_GHP}"})
            first_sent.set()
            release_second.wait(timeout=3)
            callback({"command": f"echo {context_id} second"})
            return {"final_response": "done"}

        agent.run_conversation.side_effect = _run
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
        async with TestClient(TestServer(app)) as client:
            with (
                patch("gateway.run._load_gateway_config", return_value=config),
                patch.object(adapter, "_create_agent", return_value=agent),
            ):
                response = await client.post("/v1/runs", json=payload, headers=headers)
                run_id = (await response.json())["run_id"]
                assert first_sent.wait(timeout=3)
                queue = adapter._run_streams[run_id]
                for _ in range(40):
                    if not queue.empty():
                        break
                    await asyncio.sleep(0.05)
                event = queue.get_nowait()
                assert event["event"] == "approval.request"
                assert context_id not in str(event)
                assert _FAKE_GHP not in str(event)
                assert "[redacted run context]" in str(event)
                adapter._run_streams.pop(run_id)
                release_second.set()
                for _ in range(40):
                    if adapter._run_statuses.get(run_id, {}).get("status") == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert queue.empty()


class TestApprovalTextFallbackContract:
    def test_smart_deny_only_advertises_one_operation(self):
        from gateway.run import _format_exec_approval_fallback

        text = _format_exec_approval_fallback(
            "rm -rf /", "dangerous deletion", "/",
            allow_permanent=False, smart_denied=True,
        )
        assert "owner override" in text.lower()
        assert "one operation" in text.lower()
        assert "`/approve`" in text
        assert "approve session" not in text
        assert "approve always" not in text
