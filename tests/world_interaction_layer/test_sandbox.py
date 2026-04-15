"""
Unit tests for the DockerSandboxManager in WIL.
"""
import pytest

try:
    from jarvis._vendor.world_interaction_layer.sandbox import DockerSandboxManager
except ImportError:
    pytest.skip("world_interaction_layer not available in _vendor", allow_module_level=True)

from unittest.mock import MagicMock, patch


@pytest.fixture
def docker_manager():
    # Patch docker.from_env at the correct import location
    with patch("jarvis._vendor.world_interaction_layer.sandbox.docker.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        manager = DockerSandboxManager(logger=None)
        # Ensure the docker_client is our mock (in case try-except swallowed it)
        manager.docker_client = mock_client
        yield manager

def test_docker_manager_init(docker_manager):
    assert docker_manager.image.startswith("python")
    assert docker_manager.cpu_limit > 0
    assert docker_manager.mem_limit.endswith("m")

@pytest.mark.asyncio
async def test_run_success(docker_manager):
    # Patch container run and logs
    mock_container = MagicMock()
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.side_effect = [b"output", b""] # stdout, stderr
    docker_manager.docker_client.containers.run.return_value = mock_container

    result = await docker_manager.run("print('hi')", None, timeout=5)
    assert result["status"] == "SUCCESS"
    assert result["stdout"] == "output"
    assert result["stderr"] == ""
    assert result["error_message"] == ""

@pytest.mark.asyncio
async def test_run_failure(docker_manager):
    mock_container = MagicMock()
    mock_container.wait.return_value = {"StatusCode": 1}
    mock_container.logs.side_effect = [b"", b"error"] # stdout, stderr
    docker_manager.docker_client.containers.run.return_value = mock_container

    result = await docker_manager.run("raise Exception()", None, timeout=5)
    assert result["status"] == "FAILED"
    assert result["stderr"] == "error"
    assert "Execution failed with return code" in result["error_message"]

@pytest.mark.asyncio
async def test_run_timeout(docker_manager):
    mock_container = MagicMock()
    # Simulate timeout by raising Exception in wait
    mock_container.wait.side_effect = Exception("Timeout")
    docker_manager.docker_client.containers.run.return_value = mock_container

    result = await docker_manager.run("while True: pass", None, timeout=1)
    assert result["status"] == "TIMEOUT" or result["status"] == "ERROR"
    assert "timed out" in result["error_message"] or "Timeout" in result["error_message"]

@pytest.mark.asyncio
async def test_run_docker_error(docker_manager):
    # Simulate Docker error
    docker_manager.docker_client.containers.run.side_effect = Exception("Docker error")
    result = await docker_manager.run("print('fail')", None, timeout=5)
    assert result["status"] == "ERROR"
    assert "Docker error" in result["error_message"]
