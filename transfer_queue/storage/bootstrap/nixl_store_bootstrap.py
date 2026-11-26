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
import time
from pathlib import Path
from typing import Any

import ray
from omegaconf import DictConfig, open_dict
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from transfer_queue.utils.logging_utils import get_logger

from .provider import StorageBootstrapProvider

logger = get_logger(__name__)


def _split_host_port(address: str) -> tuple[str, int]:
    host, separator, port_text = str(address).rpartition(":")
    if not separator or not host or not port_text:
        raise ValueError(f"NixlStore manager_address must be 'host:port', got {address!r}")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"NixlStore manager_address port must be an integer, got {address!r}") from exc
    if port < 0 or port > 65535:
        raise ValueError(f"NixlStore manager_address port must be between 0 and 65535, got {port!r}")
    return host, port


def _reserve_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _manager_addresses(config: DictConfig, node_count: int) -> tuple[str, str]:
    host, port = _split_host_port(config.get("manager_address", "127.0.0.1:0"))
    if port == 0:
        port = _reserve_tcp_port()

    driver_ip = ray.util.get_node_ip_address()
    if host in {"127.0.0.1", "localhost", "0.0.0.0"}:
        listen_host = "0.0.0.0" if node_count > 1 or host == "0.0.0.0" else "127.0.0.1"
        connect_host = driver_ip if listen_host == "0.0.0.0" else host
    else:
        listen_host = connect_host = host
    return f"{listen_host}:{port}", f"{connect_host}:{port}"


def _wait_for_tcp(address: str, process: subprocess.Popen, timeout_seconds: float, log_path: Path) -> None:
    host, port = _split_host_port(address)
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = _read_log(log_path)
            raise RuntimeError(f"NixlStore manager exited before readiness. Log {log_path}:\n{output}")
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.1)
    output = _read_log(log_path)
    raise RuntimeError(f"NixlStore manager at {address} did not become ready: {last_error}. Log {log_path}:\n{output}")


def _wait_for_unix_socket(path: str, process: subprocess.Popen, timeout_seconds: float, log_path: Path) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = _read_log(log_path)
            raise RuntimeError(f"NixlStore NSClient exited before readiness. Log {log_path}:\n{output}")
        if os.path.exists(path):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.5)
                    sock.connect(path)
                    return
            except OSError as exc:
                last_error = exc
        time.sleep(0.1)
    output = _read_log(log_path)
    raise RuntimeError(f"NixlStore NSClient socket {path!r} did not become ready: {last_error}. Log {log_path}:\n{output}")


def _read_log(path: Path) -> str:
    try:
        return path.read_text()
    except Exception as exc:
        return f"failed to read log: {exc}"


def _process_env_overrides(config: DictConfig) -> dict[str, str]:
    raw = config.get("process_env", {})
    if not raw:
        return {}
    if not hasattr(raw, "items"):
        raise ValueError("NixlStore process_env must be a mapping of environment variable names to values")
    return {str(key): str(value) for key, value in raw.items()}


def _subprocess_env(overrides: dict[str, str]) -> dict[str, str] | None:
    if not overrides:
        return None
    env = os.environ.copy()
    env.update(overrides)
    return env


def _terminate_process(process: subprocess.Popen | None, timeout_seconds: float) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_seconds)


def _node_socket_path(socket_dir: str, namespace: str, node_ip: str) -> str:
    safe_namespace = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in namespace)
    safe_node = node_ip.replace(":", "_").replace(".", "_")
    return str(Path(socket_dir) / f"nixl-store-{safe_namespace}-{safe_node}.sock")


def _alive_nodes() -> list[dict[str, Any]]:
    nodes = [node for node in ray.nodes() if node.get("Alive", False)]
    if not nodes:
        raise RuntimeError("NixlStore auto_init requires at least one alive Ray node")
    return sorted(nodes, key=lambda node: node["NodeManagerAddress"])


@ray.remote(num_cpus=0.01)
class NixlStoreNSClientActor:
    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._socket_path: str | None = None

    def start(
        self,
        *,
        nsclient_binary: str,
        manager_address: str,
        socket_path: str,
        startup_timeout_seconds: float,
        shutdown_timeout_seconds: float,
        nsclient_config: str | None,
        process_env_overrides: dict[str, str],
        log_path: str,
    ) -> dict[str, Any]:
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("NixlStore NSClient actor already owns a running process")

        socket_file = Path(socket_path)
        socket_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            socket_file.unlink()
        except FileNotFoundError:
            pass

        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            nsclient_binary,
            "--manager",
            manager_address,
            "--socket",
            socket_path,
            "--shutdown-grace",
            str(int(shutdown_timeout_seconds)),
        ]
        if nsclient_config:
            cmd.extend(["--config", nsclient_config])

        with log_file.open("w") as output:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=_subprocess_env(process_env_overrides),
            )
        self._socket_path = socket_path
        try:
            _wait_for_unix_socket(socket_path, self._process, startup_timeout_seconds, log_file)
        except Exception:
            _terminate_process(self._process, shutdown_timeout_seconds)
            self._process = None
            raise
        return {"pid": self._process.pid, "socket_path": socket_path, "log_path": str(log_file)}

    def stop(self, shutdown_timeout_seconds: float) -> None:
        _terminate_process(self._process, shutdown_timeout_seconds)
        self._process = None
        if self._socket_path:
            try:
                Path(self._socket_path).unlink()
            except FileNotFoundError:
                pass
            self._socket_path = None


def _start_manager(config: DictConfig, node_count: int, socket_dir: str) -> tuple[subprocess.Popen, str, Path]:
    startup_timeout = float(config.get("startup_timeout_seconds", 60))
    shutdown_timeout = float(config.get("shutdown_timeout_seconds", 10))
    process_env_overrides = _process_env_overrides(config)
    manager_binary = str(config.get("manager_binary", "nixl-store-manager"))
    if shutil.which(manager_binary) is None and not Path(manager_binary).exists():
        raise RuntimeError(f"NixlStore manager_binary {manager_binary!r} was not found")

    listen_address, connect_address = _manager_addresses(config, node_count)
    log_path = Path(socket_dir) / "nixl-store-manager.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as output:
        process = subprocess.Popen(
            [manager_binary, "--listen", listen_address],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=_subprocess_env(process_env_overrides),
        )
    try:
        _wait_for_tcp(connect_address, process, startup_timeout, log_path)
    except Exception:
        _terminate_process(process, shutdown_timeout)
        raise
    return process, connect_address, log_path


def _start_nsclients(
    config: DictConfig,
    nodes: list[dict[str, Any]],
    manager_address: str,
    socket_dir: str,
    namespace: str,
) -> tuple[list[Any], dict[str, str], list[dict[str, Any]]]:
    nsclient_binary = str(config.get("nsclient_binary", "nixl-store-nsclient"))
    if shutil.which(nsclient_binary) is None and not Path(nsclient_binary).exists():
        raise RuntimeError(f"NixlStore nsclient_binary {nsclient_binary!r} was not found")

    startup_timeout = float(config.get("startup_timeout_seconds", 60))
    shutdown_timeout = float(config.get("shutdown_timeout_seconds", 10))
    nsclient_config = config.get("nsclient_config")
    nsclient_config_path = str(nsclient_config) if nsclient_config else None
    process_env_overrides = _process_env_overrides(config)

    actors = []
    socket_paths: dict[str, str] = {}
    starts: list[dict[str, Any]] = []
    for node in nodes:
        node_id = node["NodeID"]
        node_ip = node["NodeManagerAddress"]
        socket_path = _node_socket_path(socket_dir, namespace, node_ip)
        log_path = str(Path(socket_dir) / f"nixl-store-nsclient-{node_ip.replace('.', '_')}.log")
        actor = NixlStoreNSClientActor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        ).remote()
        actors.append(actor)
        try:
            start_info = ray.get(
                actor.start.remote(
                    nsclient_binary=nsclient_binary,
                    manager_address=manager_address,
                    socket_path=socket_path,
                    startup_timeout_seconds=startup_timeout,
                    shutdown_timeout_seconds=shutdown_timeout,
                    nsclient_config=nsclient_config_path,
                    process_env_overrides=process_env_overrides,
                    log_path=log_path,
                ),
                timeout=startup_timeout + 5,
            )
        except Exception:
            cleanup_nixl_store_resources(
                {
                    "managed": True,
                    "sidecar_actors": actors,
                    "manager_process": None,
                    "shutdown_timeout_seconds": shutdown_timeout,
                }
            )
            raise
        socket_paths[node_ip] = socket_path
        starts.append({"node_ip": node_ip, **start_info})
    return actors, socket_paths, starts


def cleanup_nixl_store_resources(resources: Any) -> None:
    if not isinstance(resources, dict) or not resources.get("managed", False):
        return

    shutdown_timeout = float(resources.get("shutdown_timeout_seconds", 10))
    sidecar_actors = list(resources.get("sidecar_actors") or [])
    for actor in reversed(sidecar_actors):
        try:
            ray.get(actor.stop.remote(shutdown_timeout), timeout=shutdown_timeout + 5)
        except Exception as exc:
            logger.warning(f"Failed to stop NixlStore NSClient sidecar: {exc}")
        try:
            ray.kill(actor)
        except Exception:
            pass

    _terminate_process(resources.get("manager_process"), shutdown_timeout)


@StorageBootstrapProvider.register_provider("NixlStore")
def initialize_nixl_store(conf: DictConfig) -> dict[str, Any]:
    """Initialize externally managed or TQ-owned NIXL Store services."""
    config = conf.backend.NixlStore
    if not bool(config.get("auto_init", False)):
        return {"managed": False, "external": True}

    nodes = _alive_nodes()
    socket_dir = str(config.get("socket_dir", "/tmp/nixl-store"))
    namespace = str(config.get("namespace", "transfer-queue"))
    shutdown_timeout = float(config.get("shutdown_timeout_seconds", 10))
    manager_process = None
    sidecar_actors: list[Any] = []

    try:
        manager_process, manager_address, manager_log = _start_manager(config, len(nodes), socket_dir)
        sidecar_actors, socket_paths, starts = _start_nsclients(config, nodes, manager_address, socket_dir, namespace)

        local_ip = ray.util.get_node_ip_address()
        with open_dict(config):
            config.manager_address = manager_address
            config.runtime_socket_paths = socket_paths
            config.socket_path = socket_paths.get(local_ip, next(iter(socket_paths.values())))

        return {
            "managed": True,
            "manager_process": manager_process,
            "manager_address": manager_address,
            "manager_log": str(manager_log),
            "sidecar_actors": sidecar_actors,
            "sidecars": starts,
            "runtime_socket_paths": socket_paths,
            "shutdown_timeout_seconds": shutdown_timeout,
        }
    except Exception:
        cleanup_nixl_store_resources(
            {
                "managed": True,
                "manager_process": manager_process,
                "sidecar_actors": sidecar_actors,
                "shutdown_timeout_seconds": shutdown_timeout,
            }
        )
        raise
