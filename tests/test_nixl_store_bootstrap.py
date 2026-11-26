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

from omegaconf import OmegaConf

import transfer_queue.storage.bootstrap.nixl_store_bootstrap as nixl_bootstrap


class FakeProcess:
    next_pid = 1000

    def __init__(self, events):
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode = None
        self.events = events

    def poll(self):
        return self.returncode

    def terminate(self):
        self.events.append("manager-terminate")
        self.returncode = 0

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.events.append("manager-kill")
        self.returncode = -9


class RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeActor:
    def __init__(self, index, events, fail_start=False):
        self.index = index
        self.events = events
        self.fail_start = fail_start
        self.start_kwargs = None
        self.start = RemoteMethod(self._start)
        self.stop = RemoteMethod(self._stop)

    def _start(self, **kwargs):
        self.start_kwargs = kwargs
        self.events.append(f"start-{self.index}")
        if self.fail_start:
            raise RuntimeError(f"sidecar {self.index} failed")
        return {
            "pid": 2000 + self.index,
            "socket_path": kwargs["socket_path"],
            "log_path": kwargs["log_path"],
        }

    def _stop(self, shutdown_timeout_seconds):
        self.events.append(f"stop-{self.index}")


class FakeActorClass:
    def __init__(self, events, fail_indexes=()):
        self.events = events
        self.fail_indexes = set(fail_indexes)
        self.actors = []
        self.manager_popen_kwargs = None
        self.options_kwargs = []

    def options(self, **kwargs):
        self.options_kwargs.append(kwargs)
        index = len(self.actors)
        actor = FakeActor(index, self.events, fail_start=index in self.fail_indexes)
        self.actors.append(actor)

        class Builder:
            def remote(self_nonlocal):
                return actor

        return Builder()


def _conf(**nixl_overrides):
    nixl = {
        "auto_init": False,
        "socket_path": "/external/nsclient.sock",
        "namespace": "phase10",
        "max_batch_items": 4096,
        "manager_address": "127.0.0.1:0",
        "manager_binary": "nixl-store-manager",
        "nsclient_binary": "nixl-store-nsclient",
        "socket_dir": "/tmp/nixl-store-test",
        "startup_timeout_seconds": 60,
        "shutdown_timeout_seconds": 10,
        "nsclient_config": None,
        "runtime_socket_paths": {},
    }
    nixl.update(nixl_overrides)
    return OmegaConf.create({"backend": {"storage_backend": "NixlStore", "NixlStore": nixl}})


def _nodes(*ips):
    return [
        {
            "Alive": True,
            "NodeID": f"node-{index}",
            "NodeManagerAddress": ip,
        }
        for index, ip in enumerate(ips)
    ]


def _patch_common(monkeypatch, events, *, nodes=None, local_ip="10.0.0.1", fail_sidecars=()):
    nodes = nodes or _nodes(local_ip)
    fake_actor_class = FakeActorClass(events, fail_indexes=fail_sidecars)
    monkeypatch.setattr(nixl_bootstrap.NixlStoreNSClientActor, "options", fake_actor_class.options)
    monkeypatch.setattr(nixl_bootstrap.ray, "nodes", lambda: nodes)
    monkeypatch.setattr(nixl_bootstrap.ray, "get", lambda value, timeout=None: value)
    monkeypatch.setattr(nixl_bootstrap.ray, "kill", lambda actor: events.append(f"kill-{actor.index}"))
    monkeypatch.setattr(nixl_bootstrap.ray.util, "get_node_ip_address", lambda: local_ip)
    monkeypatch.setattr(
        nixl_bootstrap,
        "NodeAffinitySchedulingStrategy",
        lambda node_id, soft: {"node_id": node_id, "soft": soft},
    )
    monkeypatch.setattr(nixl_bootstrap.shutil, "which", lambda binary: binary)
    monkeypatch.setattr(nixl_bootstrap, "_reserve_tcp_port", lambda: 53123)
    monkeypatch.setattr(nixl_bootstrap, "_wait_for_tcp", lambda *args, **kwargs: None)

    def popen(*args, **kwargs):
        fake_actor_class.manager_popen_kwargs = kwargs
        events.append("manager-start")
        return FakeProcess(events)

    monkeypatch.setattr(nixl_bootstrap.subprocess, "Popen", popen)
    return fake_actor_class


def test_nixl_store_auto_init_starts_sidecars_and_publishes_runtime_config(monkeypatch):
    events = []
    actors = _patch_common(monkeypatch, events, nodes=_nodes("10.0.0.1", "10.0.0.2"))
    conf = _conf(auto_init=True)

    resources = nixl_bootstrap.initialize_nixl_store(conf)

    assert resources["managed"] is True
    assert resources["manager_address"] == "10.0.0.1:53123"
    assert conf.backend.NixlStore.manager_address == "10.0.0.1:53123"
    assert conf.backend.NixlStore.socket_path == "/tmp/nixl-store-test/nixl-store-phase10-10_0_0_1.sock"
    assert dict(conf.backend.NixlStore.runtime_socket_paths) == {
        "10.0.0.1": "/tmp/nixl-store-test/nixl-store-phase10-10_0_0_1.sock",
        "10.0.0.2": "/tmp/nixl-store-test/nixl-store-phase10-10_0_0_2.sock",
    }
    assert actors.options_kwargs == [
        {"scheduling_strategy": {"node_id": "node-0", "soft": False}},
        {"scheduling_strategy": {"node_id": "node-1", "soft": False}},
    ]
    assert events == ["manager-start", "start-0", "start-1"]

    nixl_bootstrap.cleanup_nixl_store_resources(resources)
    assert events[-5:] == ["stop-1", "kill-1", "stop-0", "kill-0", "manager-terminate"]


def test_nixl_store_auto_init_passes_process_env_to_owned_services(monkeypatch):
    events = []
    actors = _patch_common(monkeypatch, events)
    conf = _conf(auto_init=True, process_env={"LD_LIBRARY_PATH": "/nixl/lib", "NIXL_PLUGIN_DIR": "/nixl/plugins"})

    resources = nixl_bootstrap.initialize_nixl_store(conf)

    manager_env = actors.manager_popen_kwargs["env"]
    assert manager_env["LD_LIBRARY_PATH"] == "/nixl/lib"
    assert manager_env["NIXL_PLUGIN_DIR"] == "/nixl/plugins"
    assert actors.actors[0].start_kwargs["process_env_overrides"] == {
        "LD_LIBRARY_PATH": "/nixl/lib",
        "NIXL_PLUGIN_DIR": "/nixl/plugins",
    }
    nixl_bootstrap.cleanup_nixl_store_resources(resources)


def test_nixl_store_auto_init_cleans_up_partial_sidecar_start(monkeypatch):
    events = []
    _patch_common(monkeypatch, events, nodes=_nodes("10.0.0.1", "10.0.0.2"), fail_sidecars={1})

    try:
        nixl_bootstrap.initialize_nixl_store(_conf(auto_init=True))
    except RuntimeError as exc:
        assert "sidecar 1 failed" in str(exc)
    else:
        raise AssertionError("expected sidecar startup failure")

    assert events == [
        "manager-start",
        "start-0",
        "start-1",
        "stop-1",
        "kill-1",
        "stop-0",
        "kill-0",
        "manager-terminate",
    ]
