"""
Unit tests for Dynamic Agent Fabric Self-Optimization module
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from predacore.agents.daf.self_optimization import (
    AdaptiveConfiguration,
    OptimizationStrategy,
    PerformanceMetrics,
    PerformanceMonitor,
    ResourceMonitor,
    SelfOptimizer,
)

# Mark all async tests in this module
pytestmark = pytest.mark.asyncio

@pytest.fixture
def resource_monitor():
    return ResourceMonitor(
        cpu_threshold=80,
        memory_threshold=85,
        network_threshold=90
    )

@pytest.fixture
def optimization_strategy():
    return OptimizationStrategy(
        optimization_interval=60,
        resource_weights={
            "cpu": 0.4,
            "memory": 0.3,
            "network": 0.3
        }
    )

@pytest.fixture
def self_optimizer():
    return SelfOptimizer(
        optimization_interval=60,
        cpu_threshold=80,
        memory_threshold=85,
        network_threshold=90
    )

@patch("predacore.agents.daf.self_optimization.psutil")
async def test_resource_monitor_cpu_usage(mock_psutil):
    mock_psutil.cpu_percent.return_value = 75.5

    monitor = ResourceMonitor()
    cpu_usage = await monitor.get_cpu_usage()

    assert cpu_usage == 75.5
    mock_psutil.cpu_percent.assert_called_once_with(interval=0.1)

@patch("predacore.agents.daf.self_optimization.psutil")
async def test_resource_monitor_memory_usage(mock_psutil):
    mock_psutil.virtual_memory.return_value = MagicMock(percent=65.2)

    monitor = ResourceMonitor()
    memory_usage = await monitor.get_memory_usage()

    assert memory_usage == 65.2
    mock_psutil.virtual_memory.assert_called_once()

@patch("predacore.agents.daf.self_optimization.psutil")
async def test_resource_monitor_network_usage(mock_psutil):
    # Return increasing byte counts on successive calls so rate > 0
    mock_psutil.net_io_counters.side_effect = [
        MagicMock(bytes_sent=1000000, bytes_recv=2000000),   # First call: baseline
        MagicMock(bytes_sent=2000000, bytes_recv=3000000),   # Second call: increased
    ]

    monitor = ResourceMonitor()
    # First call establishes baseline
    await monitor.get_network_usage()

    # Second call calculates rate
    network_usage = await monitor.get_network_usage()

    # Rate should be positive since total bytes increased
    assert network_usage > 0  # Should show some usage rate
    mock_psutil.net_io_counters.assert_called()

async def test_optimization_strategy_calculation():
    strategy = OptimizationStrategy(
        resource_weights={
            "cpu": 0.4,
            "memory": 0.3,
            "network": 0.3
        }
    )

    # Test with normal resource usage
    resource_usage = {
        "cpu": 70,
        "memory": 60,
        "network": 50
    }

    score = await strategy.calculate_optimization_score(resource_usage)
    assert 0 <= score <= 1  # Score should be normalized between 0 and 1

    # Test with high CPU usage
    resource_usage["cpu"] = 95
    high_cpu_score = await strategy.calculate_optimization_score(resource_usage)
    assert high_cpu_score > score  # Score should be higher due to increased CPU usage

@patch("predacore.agents.daf.self_optimization.ResourceMonitor")
@patch("predacore.agents.daf.self_optimization.OptimizationStrategy")
async def test_self_optimizer_initialization(mock_strategy_class, mock_monitor_class):
    mock_monitor = AsyncMock()
    mock_strategy = AsyncMock()

    mock_monitor_class.return_value = mock_monitor
    mock_strategy_class.return_value = mock_strategy

    optimizer = SelfOptimizer(
        optimization_interval=30,
        cpu_threshold=75,
        memory_threshold=80,
        network_threshold=85
    )

    assert optimizer.optimization_interval == 30
    assert optimizer.resource_monitor == mock_monitor
    assert optimizer.optimization_strategy == mock_strategy

async def test_self_optimizer_optimization_cycle():
    optimizer = SelfOptimizer(
        optimization_interval=1,
        cpu_threshold=80,
        memory_threshold=85,
        network_threshold=90
    )

    # Inject mocks after construction
    mock_monitor = AsyncMock()
    mock_monitor.get_resource_usage = AsyncMock(return_value={"cpu": 92, "memory": 88, "network": 70})
    optimizer.resource_monitor = mock_monitor

    mock_strategy = AsyncMock()
    mock_strategy.calculate_optimization_score = AsyncMock(return_value=0.92)
    mock_strategy.get_optimization_actions = AsyncMock(return_value=["scale_agents", "optimize_network"])
    optimizer.optimization_strategy = mock_strategy

    mock_orchestrator = AsyncMock()
    optimizer.orchestrator = mock_orchestrator

    # Run optimization cycle
    await optimizer.run_optimization_cycle()

    # Verify resource monitoring
    mock_monitor.get_resource_usage.assert_called()

    # Verify optimization score calculation
    mock_strategy.calculate_optimization_score.assert_called()

    # Verify optimization actions
    mock_strategy.get_optimization_actions.assert_called_with(0.92)

    # Verify orchestrator actions
    mock_orchestrator.scale_agents.assert_called()
    mock_orchestrator.optimize_network.assert_called()

async def test_self_optimizer_threshold_alerts():
    optimizer = SelfOptimizer(
        optimization_interval=1,
        cpu_threshold=80,
        memory_threshold=85,
        network_threshold=90
    )

    # Inject mocks after construction
    mock_monitor = AsyncMock()
    mock_monitor.get_resource_usage = AsyncMock(return_value={
        "cpu": 85,
        "memory": 90,
        "network": 95
    })
    optimizer.resource_monitor = mock_monitor

    mock_strategy = AsyncMock()
    mock_strategy.calculate_optimization_score = AsyncMock(return_value=0.95)
    mock_strategy.get_optimization_actions = AsyncMock(return_value=["scale_agents", "throttle_noncritical"])
    optimizer.optimization_strategy = mock_strategy

    optimizer.alert_system = AsyncMock()

    # Run optimization cycle
    await optimizer.run_optimization_cycle()

    # Verify alerts were triggered
    optimizer.alert_system.send_alert.assert_called_with(
        "high_resource_usage",
        "Resource usage above thresholds: cpu=85% (threshold:80%), memory=90% (threshold:85%), network=95% (threshold:90%)"
    )

@patch("predacore.agents.daf.self_optimization.ResourceMonitor")
@patch("predacore.agents.daf.self_optimization.OptimizationStrategy")
async def test_self_optimizer_background_task(mock_strategy, mock_monitor):
    mock_monitor.get_resource_usage = AsyncMock(return_value={
        "cpu": 70,
        "memory": 65,
        "network": 50
    })

    optimizer = SelfOptimizer(optimization_interval=0.1)
    optimizer.run_optimization_cycle = AsyncMock()

    # Start background task
    task = asyncio.create_task(optimizer.start_monitoring())

    # Let it run a few cycles
    await asyncio.sleep(0.3)

    # Stop the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Verify it ran multiple times
    assert optimizer.run_optimization_cycle.call_count >= 2


async def test_adaptive_configuration_policy_actions():
    adaptive = AdaptiveConfiguration()
    metrics = PerformanceMetrics(
        cpu_usage=88.0,
        memory_usage=81.0,
        network_latency=20.0,
        request_rate=220.0,
        error_rate=0.12,
    )
    decision = adaptive.evaluate(metrics, score=0.84)
    assert "scale_agents" in decision["actions"]
    assert "throttle_noncritical" in decision["actions"]
    assert decision["rollout_fraction"] > 0


async def test_performance_monitor_anomaly_detection(resource_monitor):
    monitor = PerformanceMonitor(resource_monitor)
    metrics = PerformanceMetrics(
        cpu_usage=95.0,
        memory_usage=93.0,
        network_latency=95.0,
        request_rate=10.0,
        error_rate=0.18,
    )
    anomalies = monitor.detect_anomalies(
        metrics,
        cpu_threshold=80,
        memory_threshold=85,
        network_threshold=90,
    )
    assert any("cpu=" in item for item in anomalies)
    assert any("memory=" in item for item in anomalies)
    assert any("network=" in item for item in anomalies)
    assert any("error_rate=" in item for item in anomalies)
