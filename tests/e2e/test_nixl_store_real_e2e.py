# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
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

import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import pytest
import ray
import torch
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tensordict import TensorDict

import transfer_queue as tq
from tests.e2e.nixl_store_ray_helper import read_nixl_store_key


pytestmark = pytest.mark.skipif(
    os.environ.get("TQ_NIXL_STORE_E2E") != "1",
    reason="set TQ_NIXL_STORE_E2E=1 to run the real NIXL Store e2e test",
)


def _required_binary(env_key: str, default: str) -> str:
    binary = os.environ.get(env_key, default)
    if shutil.which(binary) or Path(binary).exists():
        return binary
    pytest.skip(f"{binary!r} was not found; set {env_key} or install NIXL Store binaries")


def _reserve_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_log(path: Path) -> str:
    try:
        return path.read_text()
    except Exception as exc:
        return f"failed to read {path}: {exc}"


def _wait_for_tcp(address: str, process: subprocess.Popen, log_path: Path, timeout_seconds: float = 30) -> None:
    host, port_text = address.rsplit(":", 1)
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited before TCP readiness. Log {log_path}:\n{_read_log(log_path)}")
        try:
            with socket.create_connection((host, int(port_text)), timeout=0.5):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"TCP address {address} did not become ready: {last_error}. Log {log_path}:\n{_read_log(log_path)}")


def _wait_for_unix_socket(path: Path, process: subprocess.Popen, log_path: Path, timeout_seconds: float = 30) -> None:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited before Unix socket readiness. Log {log_path}:\n{_read_log(log_path)}")
        if path.exists():
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                try:
                    sock.connect(str(path))
                    return
                finally:
                    sock.close()
            except OSError as exc:
                last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"Unix socket {path} did not become ready: {last_error}. Log {log_path}:\n{_read_log(log_path)}")


def _terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _short_namespace() -> str:
    return f"nixl{uuid4().hex[:8]}"


def _short_socket_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix=f"tqnixl{uuid4().hex[:6]}-", dir="/tmp"))


def _sample_fields() -> TensorDict:
    return TensorDict(
        {
            "x": torch.arange(8, dtype=torch.int64).reshape(1, 8),
            "y": torch.ones(1, 4, dtype=torch.float32),
        },
        batch_size=1,
    )


def _nixl_config(
    *,
    namespace: str,
    manager_binary: str,
    nsclient_binary: str,
    socket_dir: Path,
    auto_init: bool,
    manager_address: str = "127.0.0.1:0",
    socket_path: str | None = None,
) -> OmegaConf:
    nixl = {
        "auto_init": auto_init,
        "manager_address": manager_address,
        "manager_binary": manager_binary,
        "nsclient_binary": nsclient_binary,
        "socket_dir": str(socket_dir),
        "namespace": namespace,
        "startup_timeout_seconds": 60,
        "shutdown_timeout_seconds": 10,
        "nsclient_config": os.environ.get("TQ_NIXL_STORE_NSCLIENT_CONFIG"),
    }
    if socket_path is not None:
        nixl["socket_path"] = socket_path
    return OmegaConf.create({"backend": {"storage_backend": "NixlStore", "NixlStore": nixl}})


def _read_from_attached_process(namespace: str, partition_id: str, attached_node_id: str | None = None) -> str:
    options = {}
    if attached_node_id is not None:
        options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(node_id=attached_node_id, soft=False)
    return ray.get(read_nixl_store_key.options(**options).remote(namespace, partition_id), timeout=60)


def _exercise_public_kv(partition_id: str, namespace: str | None = None, attached_node_id: str | None = None) -> str | None:
    keys = ["sample"]
    fields = _sample_fields()

    tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields)
    got = tq.kv_batch_get(keys=keys, partition_id=partition_id)
    assert torch.equal(got["x"][0], fields["x"][0])
    assert torch.equal(got["y"][0], fields["y"][0])

    attached_ip = None
    if namespace is not None:
        attached_ip = _read_from_attached_process(namespace, partition_id, attached_node_id)

    got_again = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields="x")
    assert torch.equal(got_again["x"][0], fields["x"][0])

    tq.kv_clear(keys=keys, partition_id=partition_id, fields="y")
    remaining = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields="x")
    assert torch.equal(remaining["x"][0], fields["x"][0])

    tq.kv_clear(keys=keys, partition_id=partition_id)
    with pytest.raises(ValueError, match="not found"):
        tq.kv_batch_get(keys=keys, partition_id=partition_id)
    return attached_ip


def _ray_init_local(namespace: str) -> bool:
    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(namespace=namespace)
    return started_ray


def test_nixl_store_auto_init_real_service_e2e(monkeypatch):
    pytest.importorskip("nixl_store")

    manager_binary = _required_binary("TQ_NIXL_STORE_MANAGER_BINARY", "nixl-store-manager")
    nsclient_binary = _required_binary("TQ_NIXL_STORE_NSCLIENT_BINARY", "nixl-store-nsclient")
    namespace = _short_namespace()
    monkeypatch.setenv("TQ_RAY_NAMESPACE", namespace)

    started_ray = _ray_init_local(namespace)
    socket_dir = _short_socket_dir()
    try:
        tq.init(
            _nixl_config(
                namespace=namespace,
                manager_binary=manager_binary,
                nsclient_binary=nsclient_binary,
                socket_dir=socket_dir,
                auto_init=True,
                manager_address=os.environ.get("TQ_NIXL_STORE_MANAGER_ADDRESS", "127.0.0.1:0"),
            )
        )
        _exercise_public_kv("nixl-store-auto-init-e2e", namespace=namespace)
    finally:
        tq.close()
        if started_ray:
            ray.shutdown()
        shutil.rmtree(socket_dir, ignore_errors=True)


def test_nixl_store_external_service_real_e2e(monkeypatch):
    pytest.importorskip("nixl_store")

    manager_binary = _required_binary("TQ_NIXL_STORE_MANAGER_BINARY", "nixl-store-manager")
    nsclient_binary = _required_binary("TQ_NIXL_STORE_NSCLIENT_BINARY", "nixl-store-nsclient")
    namespace = _short_namespace()
    monkeypatch.setenv("TQ_RAY_NAMESPACE", namespace)

    started_ray = _ray_init_local(namespace)
    socket_dir = _short_socket_dir()
    socket_path = Path(f"/tmp/tqnixl{uuid4().hex[:8]}.sock")
    manager_address = f"127.0.0.1:{_reserve_tcp_port()}"
    manager_log = socket_dir / "manager.log"
    nsclient_log = socket_dir / "nsclient.log"
    manager = None
    nsclient = None
    try:
        with manager_log.open("w") as output:
            manager = subprocess.Popen(
                [manager_binary, "--listen", manager_address],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        _wait_for_tcp(manager_address, manager, manager_log)

        cmd = [
            nsclient_binary,
            "--manager",
            manager_address,
            "--socket",
            str(socket_path),
            "--shutdown-grace",
            "10",
        ]
        nsclient_config = os.environ.get("TQ_NIXL_STORE_NSCLIENT_CONFIG")
        if nsclient_config:
            cmd.extend(["--config", nsclient_config])
        with nsclient_log.open("w") as output:
            nsclient = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        _wait_for_unix_socket(socket_path, nsclient, nsclient_log)

        tq.init(
            _nixl_config(
                namespace=namespace,
                manager_binary=manager_binary,
                nsclient_binary=nsclient_binary,
                socket_dir=socket_dir,
                auto_init=False,
                manager_address=manager_address,
                socket_path=str(socket_path),
            )
        )
        _exercise_public_kv("nixl-store-external-e2e")
        assert manager.poll() is None
        assert nsclient.poll() is None
    finally:
        tq.close()
        _terminate_process(nsclient)
        _terminate_process(manager)
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
        if started_ray:
            ray.shutdown()
        shutil.rmtree(socket_dir, ignore_errors=True)


@pytest.mark.skipif(
    os.environ.get("TQ_NIXL_STORE_TWO_NODE_E2E") != "1",
    reason="set TQ_NIXL_STORE_TWO_NODE_E2E=1 to run the two-node NIXL Store e2e test",
)
def test_nixl_store_auto_init_two_node_real_e2e(monkeypatch):
    pytest.importorskip("nixl_store")

    manager_binary = _required_binary("TQ_NIXL_STORE_MANAGER_BINARY", "nixl-store-manager")
    nsclient_binary = _required_binary("TQ_NIXL_STORE_NSCLIENT_BINARY", "nixl-store-nsclient")
    namespace = _short_namespace()
    monkeypatch.setenv("TQ_RAY_NAMESPACE", namespace)

    started_ray = not ray.is_initialized()
    if started_ray:
        ray_address = os.environ.get("RAY_ADDRESS")
        if not ray_address:
            pytest.skip("two-node NixlStore e2e requires RAY_ADDRESS for an existing Ray cluster")
        ray.init(address=ray_address, namespace=namespace)

    socket_dir = _short_socket_dir()
    try:
        nodes = [node for node in ray.nodes() if node.get("Alive", False)]
        if len(nodes) < 2:
            pytest.skip(f"two-node NixlStore e2e requires at least two alive Ray nodes, got {len(nodes)}")
        nodes = sorted(nodes, key=lambda node: node["NodeManagerAddress"])
        local_ip = ray.util.get_node_ip_address()
        consumer_node = next((node for node in nodes if node["NodeManagerAddress"] != local_ip), None)
        if consumer_node is None:
            pytest.skip("could not find a live Ray node different from the driver node")

        tq.init(
            _nixl_config(
                namespace=namespace,
                manager_binary=manager_binary,
                nsclient_binary=nsclient_binary,
                socket_dir=socket_dir,
                auto_init=True,
                manager_address=os.environ.get("TQ_NIXL_STORE_MANAGER_ADDRESS", "127.0.0.1:0"),
            )
        )
        attached_ip = _exercise_public_kv(
            "nixl-store-two-node-e2e",
            namespace=namespace,
            attached_node_id=consumer_node["NodeID"],
        )
        assert attached_ip == consumer_node["NodeManagerAddress"]
    finally:
        tq.close()
        if started_ray:
            ray.shutdown()
        shutil.rmtree(socket_dir, ignore_errors=True)
