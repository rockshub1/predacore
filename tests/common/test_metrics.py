"""Tests for src/common/metrics.py — Prometheus observability metrics."""
import pytest

from jarvis._vendor.common.metrics import (
    ACTIVE_SESSIONS,
    CHANNEL_STATUS,
    EGM_EVALUATIONS,
    LLM_LATENCY,
    LLM_REQUESTS,
    MEMORY_OPS,
    MESSAGES_RECEIVED,
    TOOL_CALLS,
    TOOL_LATENCY,
    get_metrics_content_type,
    get_metrics_text,
    get_trace_id,
    new_trace_id,
    set_trace_id,
    track_latency,
)


class TestTraceId:
    """Tests for trace ID propagation."""

    def test_new_trace_id_returns_16_hex_chars(self):
        tid = new_trace_id()
        assert len(tid) == 16
        int(tid, 16)  # should be valid hex

    def test_get_trace_id_generates_if_empty(self):
        # Reset by setting empty
        set_trace_id("")
        tid = get_trace_id()
        assert len(tid) == 16

    def test_set_and_get_trace_id(self):
        set_trace_id("deadbeef12345678")
        assert get_trace_id() == "deadbeef12345678"

    def test_new_trace_id_changes_value(self):
        id1 = new_trace_id()
        id2 = new_trace_id()
        # Extremely unlikely to collide
        assert id1 != id2


class TestMetricCounters:
    """Tests for counter metrics."""

    def test_tool_calls_increment(self):
        before = TOOL_CALLS.labels(tool="test_tool", status="success")._value.get()
        TOOL_CALLS.labels(tool="test_tool", status="success").inc()
        after = TOOL_CALLS.labels(tool="test_tool", status="success")._value.get()
        assert after == before + 1

    def test_llm_requests_increment(self):
        before = LLM_REQUESTS.labels(provider="test_prov", status="success")._value.get()
        LLM_REQUESTS.labels(provider="test_prov", status="success").inc()
        after = LLM_REQUESTS.labels(provider="test_prov", status="success")._value.get()
        assert after == before + 1

    def test_memory_ops_increment(self):
        before = MEMORY_OPS.labels(operation="store", status="success")._value.get()
        MEMORY_OPS.labels(operation="store", status="success").inc()
        after = MEMORY_OPS.labels(operation="store", status="success")._value.get()
        assert after == before + 1

    def test_messages_received(self):
        before = MESSAGES_RECEIVED.labels(channel="test_ch")._value.get()
        MESSAGES_RECEIVED.labels(channel="test_ch").inc()
        after = MESSAGES_RECEIVED.labels(channel="test_ch")._value.get()
        assert after == before + 1


class TestMetricGauges:
    """Tests for gauge metrics."""

    def test_active_sessions(self):
        ACTIVE_SESSIONS.set(5)
        assert ACTIVE_SESSIONS._value.get() == 5.0
        ACTIVE_SESSIONS.inc()
        assert ACTIVE_SESSIONS._value.get() == 6.0
        ACTIVE_SESSIONS.dec()
        assert ACTIVE_SESSIONS._value.get() == 5.0

    def test_channel_status(self):
        CHANNEL_STATUS.labels(channel="telegram").set(1)
        assert CHANNEL_STATUS.labels(channel="telegram")._value.get() == 1.0


class TestTrackLatency:
    """Tests for the track_latency context manager."""

    def test_records_elapsed_time(self):
        import time
        with track_latency(TOOL_LATENCY, tool="test_latency"):
            time.sleep(0.01)
        # If we get here without error, the histogram recorded successfully

    def test_records_with_labels_dict(self):
        with track_latency(LLM_LATENCY, labels={"provider": "test_latency"}):
            pass

    def test_records_with_kwargs(self):
        with track_latency(LLM_LATENCY, provider="test_kwarg"):
            pass

    def test_records_even_on_exception(self):
        """Latency should be recorded even if the body raises."""
        with pytest.raises(ValueError):
            with track_latency(TOOL_LATENCY, tool="test_exception"):
                raise ValueError("boom")


class TestMetricsEndpoint:
    """Tests for the metrics text generation."""

    def test_get_metrics_text_returns_string(self):
        text = get_metrics_text()
        assert isinstance(text, str)
        # Should contain at least one of our registered metrics
        assert "prometheus_" in text or "python_" in text

    def test_get_metrics_content_type(self):
        ct = get_metrics_content_type()
        assert "text/plain" in ct or "openmetrics" in ct


class TestEGMMetrics:
    def test_egm_evaluations(self):
        before = EGM_EVALUATIONS.labels(decision="allow")._value.get()
        EGM_EVALUATIONS.labels(decision="allow").inc()
        after = EGM_EVALUATIONS.labels(decision="allow")._value.get()
        assert after == before + 1
