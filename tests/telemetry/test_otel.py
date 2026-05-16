"""
Tests for OpenTelemetry integration module.
"""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestOtelConfiguration:
    """Tests for OTEL configuration functions."""

    def test_otel_disabled_by_default(self):
        """OTEL should be disabled when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SWARMS_OTEL_ENABLED", None)
            from swarms.telemetry import otel

            otel._tracer = None
            otel._meter = None
            assert not otel._is_otel_enabled()

    def test_otel_enabled_with_true(self):
        """OTEL should be enabled when set to 'true'."""
        if not otel_available():
            pytest.skip("OTEL packages not installed")

        with patch.dict(os.environ, {"SWARMS_OTEL_ENABLED": "true"}):
            from swarms.telemetry import otel

            otel._tracer = None
            assert otel._is_otel_enabled()

    def test_otel_enabled_with_1(self):
        """OTEL should be enabled when set to '1'."""
        if not otel_available():
            pytest.skip("OTEL packages not installed")

        with patch.dict(os.environ, {"SWARMS_OTEL_ENABLED": "1"}):
            from swarms.telemetry import otel

            otel._tracer = None
            assert otel._is_otel_enabled()

    def test_service_name_default(self):
        """Service name should default to 'swarms'."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OTEL_SERVICE_NAME", None)
            from swarms.telemetry.otel import _get_service_name

            assert _get_service_name() == "swarms"

    def test_service_name_custom(self):
        """Service name should use env var when set."""
        with patch.dict(
            os.environ, {"OTEL_SERVICE_NAME": "my-service"}
        ):
            from swarms.telemetry.otel import _get_service_name

            assert _get_service_name() == "my-service"


class TestTraceContext:
    """Tests for trace_context context manager."""

    def test_trace_context_disabled(self):
        """trace_context should yield None when OTEL disabled."""
        with patch.dict(os.environ, {"SWARMS_OTEL_ENABLED": "false"}):
            from swarms.telemetry.otel import trace_context

            with trace_context("test.span") as span:
                assert span is None

    def test_trace_context_no_error_when_disabled(self):
        """trace_context should not raise errors when disabled."""
        with patch.dict(os.environ, {"SWARMS_OTEL_ENABLED": "false"}):
            from swarms.telemetry.otel import trace_context

            try:
                with trace_context(
                    "test.span", {"key": "value"}
                ) as span:
                    pass
            except Exception as e:
                pytest.fail(f"trace_context raised: {e}")


class TestOtelMetrics:
    """Tests for OTelMetrics singleton."""

    def test_metrics_singleton(self):
        """OTelMetrics should be a singleton."""
        from swarms.telemetry.otel import OTelMetrics

        m1 = OTelMetrics()
        m2 = OTelMetrics()
        assert m1 is m2

    def test_metrics_none_when_disabled(self):
        """Metrics instruments should be None when OTEL disabled."""
        with patch.dict(os.environ, {"SWARMS_OTEL_ENABLED": "false"}):
            from swarms.telemetry.otel import OTelMetrics

            OTelMetrics._instance = None
            metrics = OTelMetrics()
            assert metrics.agent_runs is None
            assert metrics.agent_duration is None


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_otel_enabled_export(self):
        """is_otel_enabled should be exported."""
        from swarms.telemetry import is_otel_enabled

        assert callable(is_otel_enabled)

    def test_otel_available_export(self):
        """otel_available should be exported."""
        from swarms.telemetry import otel_available

        assert callable(otel_available)

    def test_get_tracer_export(self):
        """get_tracer should be exported."""
        from swarms.telemetry import get_tracer

        assert callable(get_tracer)


def otel_available():
    """Check if OTEL packages are installed."""
    try:
        from opentelemetry import trace

        return True
    except ImportError:
        return False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
