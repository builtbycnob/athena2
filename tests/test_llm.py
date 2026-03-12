# tests/test_llm.py
import threading
import pytest
import json
import httpx
from unittest.mock import patch, MagicMock
from athena.agents.llm import invoke_llm, parse_json_response, GenerationResult, reset_stats
from athena.agents.errors import JSONTruncatedError, JSONMalformedError, NonJSONOutputError
import athena.agents.llm as llm_mod


class TestParseJsonResponseWithRepair:
    """Test the new parse pipeline with repair."""

    def test_parses_clean_json(self):
        result = parse_json_response('{"key": "value"}', finish_reason="stop")
        assert result.data == {"key": "value"}
        assert result.applied_fixes == []
        assert not result.was_truncated

    def test_repairs_truncated_json(self):
        result = parse_json_response(
            '{"key": "value", "other": "trunc',
            finish_reason="length",
        )
        assert result.data["key"] == "value"
        assert result.was_truncated
        assert "repair_truncated" in result.applied_fixes or "json_repair_library" in result.applied_fixes

    def test_fixes_trailing_comma(self):
        result = parse_json_response('{"key": "value",}', finish_reason="stop")
        assert result.data == {"key": "value"}
        assert "trailing_commas" in result.applied_fixes

    def test_strips_thinking_blocks(self):
        raw = '<think>\nReasoning here\n</think>\n{"key": "value"}'
        result = parse_json_response(raw, finish_reason="stop")
        assert result.data == {"key": "value"}

    def test_raises_classified_error_on_non_json(self):
        with pytest.raises(NonJSONOutputError):
            parse_json_response(
                "The user wants me to act as a lawyer",
                finish_reason="stop",
                prompt_tokens=5000,
                output_tokens=2000,
            )

    def test_truncated_garbage_still_parses_with_library_repair(self):
        """json_repair library is permissive — even garbage produces valid JSON."""
        result = parse_json_response(
            "completely broken {{{ not json at all",
            finish_reason="length",
            prompt_tokens=5000,
            output_tokens=16384,
        )
        assert result.was_truncated
        assert "json_repair_library" in result.applied_fixes or "json_repair_library_raw" in result.applied_fixes


class TestInvokeLLMRefactored:
    @patch("athena.agents.llm._call_model")
    def test_returns_parsed_dict(self, mock_call):
        mock_call.return_value = ('{"test": true}', "stop", 100, 50)
        result = invoke_llm("system", "user", temperature=0.5)
        assert result == {"test": True}

    @patch("athena.agents.llm._call_model")
    def test_retries_on_unrepairable_truncation(self, mock_call):
        # json_repair is very permissive — use pure prose (no braces) to
        # trigger NonJSONOutputError → JSONTruncatedError on length finish
        mock_call.side_effect = [
            ("The model refuses to produce JSON output and instead writes prose", "length", 100, 16384),
            ('{"key": "value"}', "stop", 100, 200),
        ]
        result = invoke_llm("system", "user", temperature=0.5)
        assert result == {"key": "value"}
        assert mock_call.call_count == 2

    @patch("athena.agents.llm._call_model")
    def test_saves_failure_artifact(self, mock_call, tmp_path):
        mock_call.return_value = ("not json at all", "stop", 100, 50)
        import athena.agents.llm as llm_mod
        old_dir = llm_mod._FAILURE_DIR
        llm_mod._FAILURE_DIR = str(tmp_path)
        try:
            with pytest.raises(NonJSONOutputError):
                invoke_llm("system", "user", temperature=0.5)
            artifacts = list(tmp_path.iterdir())
            assert len(artifacts) == 1
            content = artifacts[0].read_text()
            assert "not json at all" in content
        finally:
            llm_mod._FAILURE_DIR = old_dir


def _make_omlx_response(content='{"ok": true}', finish_reason="stop",
                         prompt_tokens=100, completion_tokens=50, cached_tokens=0):
    """Helper to build a mock oMLX JSON response."""
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
        },
    }


class TestOmlxBackend:
    """Tests for the oMLX HTTP backend."""

    @patch("athena.agents.llm._OMLX_CLIENT")
    def test_successful_call(self, _):
        mock_client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _make_omlx_response('{"result": 42}', cached_tokens=80)
        resp.raise_for_status = MagicMock()
        mock_client.post.return_value = resp

        llm_mod._OMLX_CLIENT = mock_client
        old_calls = llm_mod._stats["calls"]
        try:
            text, fr, pt, ot = llm_mod._call_model_omlx("sys", "usr", 0.5)
            assert text == '{"result": 42}'
            assert fr == "stop"
            assert pt == 100
            assert ot == 50
        finally:
            llm_mod._OMLX_CLIENT = None
            llm_mod._stats["calls"] = old_calls

    @patch("athena.agents.llm._OMLX_CLIENT")
    def test_connection_retry(self, _):
        mock_client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _make_omlx_response()
        resp.raise_for_status = MagicMock()
        mock_client.post.side_effect = [httpx.ConnectError("refused"), resp]

        llm_mod._OMLX_CLIENT = mock_client
        old_calls = llm_mod._stats["calls"]
        try:
            text, fr, pt, ot = llm_mod._call_model_omlx("sys", "usr", 0.5)
            assert text == '{"ok": true}'
            assert mock_client.post.call_count == 2
        finally:
            llm_mod._OMLX_CLIENT = None
            llm_mod._stats["calls"] = old_calls

    @patch("athena.agents.llm._call_model")
    def test_truncation_retry_via_http(self, mock_call):
        """finish_reason=length from oMLX triggers the same retry logic."""
        mock_call.side_effect = [
            ("The model refuses to produce JSON output and instead writes prose", "length", 100, 16384),
            ('{"key": "value"}', "stop", 100, 200),
        ]
        result = invoke_llm("system", "user", temperature=0.5)
        assert result == {"key": "value"}
        assert mock_call.call_count == 2

    @patch("athena.agents.llm._OMLX_CLIENT")
    def test_cached_tokens_in_stats(self, _):
        mock_client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _make_omlx_response(cached_tokens=500)
        resp.raise_for_status = MagicMock()
        mock_client.post.return_value = resp

        llm_mod._OMLX_CLIENT = mock_client
        old_cached = llm_mod._stats["cached_tokens"]
        old_calls = llm_mod._stats["calls"]
        try:
            llm_mod._call_model_omlx("sys", "usr", 0.5)
            llm_mod._call_model_omlx("sys", "usr", 0.5)
            assert llm_mod._stats["cached_tokens"] - old_cached == 1000
        finally:
            llm_mod._OMLX_CLIENT = None
            llm_mod._stats["cached_tokens"] = old_cached
            llm_mod._stats["calls"] = old_calls


class TestHealthCheck:
    """Tests for _ensure_omlx health check."""

    @patch("athena.agents.llm.httpx.Client")
    def test_health_check_succeeds(self, MockClient):
        mock_client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"data": [{"id": "test-model"}]}
        resp.raise_for_status = MagicMock()
        mock_client.get.return_value = resp
        MockClient.return_value = mock_client

        llm_mod._OMLX_CLIENT = None
        try:
            client = llm_mod._ensure_omlx()
            assert client is mock_client
            mock_client.get.assert_called_once_with("/v1/models")
        finally:
            llm_mod._OMLX_CLIENT = None

    @patch("athena.agents.llm.time.sleep")
    @patch("athena.agents.llm.httpx.Client")
    def test_health_check_timeout(self, MockClient, mock_sleep):
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("refused")
        MockClient.return_value = mock_client

        # Make time.time() advance past the 30s deadline
        times = [0.0, 0.0, 31.0]
        with patch("athena.agents.llm.time.time", side_effect=times):
            llm_mod._OMLX_CLIENT = None
            try:
                with pytest.raises(ConnectionError, match="unreachable"):
                    llm_mod._ensure_omlx()
            finally:
                llm_mod._OMLX_CLIENT = None


class TestBackendDispatch:
    """Tests for the _call_model dispatcher."""

    @patch("athena.agents.llm._call_model_omlx", return_value=("t", "stop", 1, 1))
    def test_omlx_dispatch(self, mock_omlx):
        old = llm_mod._BACKEND
        llm_mod._BACKEND = "omlx"
        try:
            llm_mod._call_model("s", "u", 0.5)
            mock_omlx.assert_called_once_with("s", "u", 0.5, llm_mod._DEFAULT_MAX_TOKENS, None)
        finally:
            llm_mod._BACKEND = old

    @patch("athena.agents.llm._call_model_mlx", return_value=("t", "stop", 1, 1))
    def test_mlx_dispatch(self, mock_mlx):
        old = llm_mod._BACKEND
        llm_mod._BACKEND = "mlx"
        try:
            llm_mod._call_model("s", "u", 0.5)
            mock_mlx.assert_called_once_with("s", "u", 0.5, llm_mod._DEFAULT_MAX_TOKENS)
        finally:
            llm_mod._BACKEND = old

    def test_invalid_backend_raises(self):
        old = llm_mod._BACKEND
        llm_mod._BACKEND = "foo"
        try:
            with pytest.raises(ValueError, match="Unknown ATHENA_BACKEND"):
                llm_mod._call_model("s", "u", 0.5)
        finally:
            llm_mod._BACKEND = old


class TestThreadSafety:
    """Test thread-safety of stats and singleton init."""

    def test_concurrent_stats_updates(self):
        """N threads × M increments, verify total = N×M."""
        reset_stats()
        n_threads = 8
        n_increments = 100
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            for _ in range(n_increments):
                with llm_mod._lock:
                    llm_mod._stats["calls"] += 1
                    llm_mod._stats["total_tokens"] += 10

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert llm_mod._stats["calls"] == n_threads * n_increments
        assert llm_mod._stats["total_tokens"] == n_threads * n_increments * 10
        reset_stats()

    @patch("athena.agents.llm.httpx.Client")
    def test_concurrent_ensure_omlx(self, MockClient):
        """5 threads race _ensure_omlx(), verify single Client created."""
        mock_client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"data": [{"id": "test-model"}]}
        resp.raise_for_status = MagicMock()
        mock_client.get.return_value = resp
        MockClient.return_value = mock_client

        llm_mod._OMLX_CLIENT = None
        barrier = threading.Barrier(5)
        results = []

        def worker():
            barrier.wait()
            client = llm_mod._ensure_omlx()
            results.append(client)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All threads should get the same client instance
            assert all(r is results[0] for r in results)
            # httpx.Client() should only be called once
            assert MockClient.call_count == 1
        finally:
            llm_mod._OMLX_CLIENT = None


class TestOmlxPayload:
    """Tests for structured output and sampling params in oMLX payload."""

    @patch("athena.agents.llm._OMLX_CLIENT")
    def test_json_schema_in_omlx_payload(self, _):
        mock_client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _make_omlx_response()
        resp.raise_for_status = MagicMock()
        mock_client.post.return_value = resp

        llm_mod._OMLX_CLIENT = mock_client
        old_calls = llm_mod._stats["calls"]
        try:
            schema = {"type": "object", "properties": {"x": {"type": "number"}}}
            llm_mod._call_model_omlx("sys", "usr", 0.5, json_schema=schema)
            payload = mock_client.post.call_args[1]["json"]
            assert payload["response_format"] == {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            }
        finally:
            llm_mod._OMLX_CLIENT = None
            llm_mod._stats["calls"] = old_calls

    @patch("athena.agents.llm._OMLX_CLIENT")
    def test_json_schema_none_omits_response_format(self, _):
        mock_client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _make_omlx_response()
        resp.raise_for_status = MagicMock()
        mock_client.post.return_value = resp

        llm_mod._OMLX_CLIENT = mock_client
        old_calls = llm_mod._stats["calls"]
        try:
            llm_mod._call_model_omlx("sys", "usr", 0.5)
            payload = mock_client.post.call_args[1]["json"]
            assert "response_format" not in payload
        finally:
            llm_mod._OMLX_CLIENT = None
            llm_mod._stats["calls"] = old_calls

    @patch("athena.agents.llm._OMLX_CLIENT")
    def test_sampling_params_in_payload(self, _):
        mock_client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _make_omlx_response()
        resp.raise_for_status = MagicMock()
        mock_client.post.return_value = resp

        llm_mod._OMLX_CLIENT = mock_client
        old_calls = llm_mod._stats["calls"]
        try:
            llm_mod._call_model_omlx("sys", "usr", 0.5)
            payload = mock_client.post.call_args[1]["json"]
            assert payload["repetition_penalty"] == 1.3
            assert payload["top_p"] == 0.8
            assert payload["top_k"] == 20
            assert payload["repetition_context_size"] == 256
        finally:
            llm_mod._OMLX_CLIENT = None
            llm_mod._stats["calls"] = old_calls


class TestResetStats:
    """Test reset_stats function."""

    def test_reset_clears_all(self):
        llm_mod._stats["calls"] = 42
        llm_mod._stats["total_tokens"] = 1000
        llm_mod._stats["total_time"] = 5.0
        llm_mod._stats["repair_types"] = {"foo": 3}
        reset_stats()
        assert llm_mod._stats["calls"] == 0
        assert llm_mod._stats["total_tokens"] == 0
        assert llm_mod._stats["total_time"] == 0.0
        assert llm_mod._stats["repair_types"] == {}
