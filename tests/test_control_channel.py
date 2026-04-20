# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for control channel."""

import time

import pytest

from vexact.ipc.control_channel import ControlChannelClient, ControlChannelServer


class MockTarget:
    """Mock target object for testing control channel."""

    def __init__(self):
        self.sleep_count = 0
        self.wake_up_count = 0

    def sleep(self, tag: str = None):
        """Mock sleep method."""
        self.sleep_count += 1
        return f"slept with tag={tag}"

    def wake_up(self, tag: str = None):
        """Mock wake_up method."""
        self.wake_up_count += 1
        return f"woke up with tag={tag}"

    def forbidden_method(self):
        """Method not in allowed list."""
        return "should not be called"

    def raise_error(self):
        """Method that raises an error."""
        raise ValueError("Test error")


class TestControlChannel:
    """Test control channel server and client."""

    def test_single_server_client(self):
        """Test basic server-client communication."""
        # Create mock target
        target = MockTarget()

        # Create server
        address = "ipc:///tmp/vexact_test_ctrl_single.sock"
        server = ControlChannelServer(
            address=address,
            target=target,
            allowed_methods=["sleep", "wake_up"],
        )
        server.start()
        time.sleep(0.1)  # Let server start

        try:
            # Create client
            client = ControlChannelClient([address])

            # Test sleep
            results = client.collective_rpc("sleep", tag="test_tag")
            assert len(results) == 1
            assert results[0] == "slept with tag=test_tag"
            assert target.sleep_count == 1

            # Test wake_up
            results = client.collective_rpc("wake_up", tag="test_tag")
            assert len(results) == 1
            assert results[0] == "woke up with tag=test_tag"
            assert target.wake_up_count == 1

            client.close()

        finally:
            server.stop()

    def test_multiple_servers(self):
        """Test client communicating with multiple servers."""
        # Create multiple targets
        targets = [MockTarget() for _ in range(3)]

        # Create multiple servers
        addresses = [
            "ipc:///tmp/vexact_test_ctrl_0.sock",
            "ipc:///tmp/vexact_test_ctrl_1.sock",
            "ipc:///tmp/vexact_test_ctrl_2.sock",
        ]
        servers = []
        for i, address in enumerate(addresses):
            server = ControlChannelServer(
                address=address,
                target=targets[i],
                allowed_methods=["sleep", "wake_up"],
            )
            server.start()
            servers.append(server)

        time.sleep(0.1)  # Let servers start

        try:
            # Create client connected to all servers
            client = ControlChannelClient(addresses)

            # Test collective sleep
            results = client.collective_rpc("sleep", tag="collective_test")
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result == "slept with tag=collective_test"
                assert targets[i].sleep_count == 1

            # Test collective wake_up
            results = client.collective_rpc("wake_up")
            assert len(results) == 3
            for target in targets:
                assert target.wake_up_count == 1

            client.close()

        finally:
            for server in servers:
                server.stop()

    def test_forbidden_method(self):
        """Test that calling a forbidden method raises an error."""
        target = MockTarget()

        address = "ipc:///tmp/vexact_test_ctrl_forbidden.sock"
        server = ControlChannelServer(
            address=address,
            target=target,
            allowed_methods=["sleep", "wake_up"],
        )
        server.start()
        time.sleep(0.1)

        try:
            client = ControlChannelClient([address])

            # Try to call forbidden method
            with pytest.raises(RuntimeError, match="Method not allowed"):
                client.collective_rpc("forbidden_method")

            client.close()

        finally:
            server.stop()

    def test_method_raises_error(self):
        """Test that errors from target methods are properly propagated."""
        target = MockTarget()

        address = "ipc:///tmp/vexact_test_ctrl_error.sock"
        server = ControlChannelServer(
            address=address,
            target=target,
            allowed_methods=["raise_error"],
        )
        server.start()
        time.sleep(0.1)

        try:
            client = ControlChannelClient([address])

            # Try to call method that raises error
            with pytest.raises(RuntimeError, match="Test error"):
                client.collective_rpc("raise_error")

            client.close()

        finally:
            server.stop()

    def test_nonexistent_method(self):
        """Test that calling a non-existent method raises an error."""
        target = MockTarget()

        address = "ipc:///tmp/vexact_test_ctrl_nonexistent.sock"
        server = ControlChannelServer(
            address=address,
            target=target,
            allowed_methods=["nonexistent_method"],
        )
        server.start()
        time.sleep(0.1)

        try:
            client = ControlChannelClient([address])

            # Try to call non-existent method
            with pytest.raises(RuntimeError, match="Method not found"):
                client.collective_rpc("nonexistent_method")

            client.close()

        finally:
            server.stop()

    def test_partial_failure(self):
        """Test that if one server fails, the error is raised."""
        # Create targets where one will fail
        targets = [MockTarget(), MockTarget()]

        addresses = [
            "ipc:///tmp/vexact_test_ctrl_partial_0.sock",
            "ipc:///tmp/vexact_test_ctrl_partial_1.sock",
        ]

        servers = []
        for i, address in enumerate(addresses):
            allowed = ["sleep", "wake_up"]
            if i == 1:
                allowed.append("raise_error")  # Only second server allows this

            server = ControlChannelServer(
                address=address,
                target=targets[i],
                allowed_methods=allowed,
            )
            server.start()
            servers.append(server)

        time.sleep(0.1)

        try:
            client = ControlChannelClient(addresses)

            # First server will reject, second server will allow but raise error
            with pytest.raises(RuntimeError):
                client.collective_rpc("raise_error")

            client.close()

        finally:
            for server in servers:
                server.stop()
