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

import ctypes
from types import SimpleNamespace

import pytest
import torch

from transfer_queue.storage.clients.nixl_store_client import NixlStoreClient


class FakeStoreError(Exception):
    pass


class FakeStatus:
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    MISSING_KEY = "MISSING_KEY"
    BUFFER_TOO_SMALL = "BUFFER_TOO_SMALL"


class FakePutBufferItem:
    constructed = []

    def __init__(self, key, buffer, size):
        self.key = key
        self.buffer = buffer
        self.size = size
        self.constructed.append(self)


class FakeGetItem:
    constructed = []

    def __init__(self, key, buffer, size):
        self.key = key
        self.buffer = buffer
        self.size = size
        self.constructed.append(self)


class FakeBufferRef:
    constructed = []
    addresses = []

    @staticmethod
    def from_buffer(buffer):
        FakeBufferRef.constructed.append(buffer)
        return buffer

    @staticmethod
    def from_address(ptr, device="CPU_DRAM"):
        ref = FakeAddressRef(ptr, device)
        FakeBufferRef.addresses.append(ref)
        return ref


class FakeAddressRef:
    def __init__(self, ptr, device):
        self.ptr = ptr
        self.device = device

    def __setitem__(self, key, value):
        if not isinstance(key, slice):
            raise TypeError("FakeAddressRef only supports slice assignment")
        start = 0 if key.start is None else key.start
        stop = key.stop
        if start != 0 or stop is None:
            raise TypeError("FakeAddressRef tests only assign from offset 0 with a stop")
        data = bytes(value)
        if len(data) != stop:
            raise ValueError("FakeAddressRef assignment length mismatch")
        ctypes.memmove(self.ptr, data, len(data))


class FakeFuture:
    def __init__(self, results, error=None):
        self.results = results
        self.error = error
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1
        if self.error is not None:
            raise self.error
        return self.results


class FakeConnector:
    instances = []
    connect_error = None

    def __init__(self, socket_path, namespace):
        self.socket_path = socket_path
        self.namespace = namespace
        self.close_calls = 0
        self.upsert_buffer_calls = []
        self.upsert_results = []
        self.upsert_error = None
        self.get_calls = []
        self.get_results = []
        self.get_error = None
        self.clear_calls = []
        self.clear_results = []
        self.clear_error = None
        self.clear_wait_error = None
        self.clear_futures = []
        self.payloads = {}
        self.copy_fn = None

    @classmethod
    def connect(cls, socket_path, namespace):
        if cls.connect_error is not None:
            raise cls.connect_error
        connector = cls(socket_path, namespace)
        cls.instances.append(connector)
        return connector

    def upsert_buffer_many(self, items):
        self.upsert_buffer_calls.append(items)
        if self.upsert_error is not None:
            raise self.upsert_error
        if self.upsert_results:
            return self.upsert_results.pop(0)
        return [SimpleNamespace(status=FakeStatus.COMPLETED) for _ in items]

    def get_many(self, items):
        self.get_calls.append(items)
        if self.get_error is not None:
            raise self.get_error
        if self.get_results:
            return self.get_results.pop(0)

        results = []
        for item in items:
            payload = self.payloads.get(item.key)
            if payload is None:
                results.append(SimpleNamespace(status=FakeStatus.MISSING_KEY, bytes_copied=None))
            elif item.size < len(payload):
                results.append(SimpleNamespace(status=FakeStatus.BUFFER_TOO_SMALL, bytes_copied=None))
            else:
                if isinstance(item.buffer, FakeAddressRef):
                    item.buffer[: len(payload)] = payload
                else:
                    memoryview(item.buffer)[: len(payload)] = payload
                results.append(SimpleNamespace(status=FakeStatus.COMPLETED, bytes_copied=len(payload)))
        return results

    def clear_many(self, keys):
        self.clear_calls.append(keys)
        if self.clear_error is not None:
            raise self.clear_error
        results = (
            self.clear_results.pop(0)
            if self.clear_results
            else [SimpleNamespace(status=FakeStatus.COMPLETED) for _ in keys]
        )
        future = FakeFuture(results, self.clear_wait_error)
        self.clear_futures.append(future)
        return future

    def register_copy_fn(self, src_device, dst_device, fn):
        self.copy_fn = fn

    def close(self):
        self.close_calls += 1


def _install_fake_nixl_store(monkeypatch):
    FakeConnector.instances = []
    FakeConnector.connect_error = None
    FakePutBufferItem.constructed = []
    FakeGetItem.constructed = []
    FakeBufferRef.constructed = []
    FakeBufferRef.addresses = []
    module_kwargs = dict(
        Connector=FakeConnector,
        PutBufferItem=FakePutBufferItem,
        BufferRef=FakeBufferRef,
        GetItem=FakeGetItem,
        Status=FakeStatus,
        StoreError=FakeStoreError,
        MemoryType=SimpleNamespace(CPU_DRAM="CPU_DRAM"),
    )
    module = SimpleNamespace(**module_kwargs)

    def import_module(name, package=None):
        if name == "nixl_store":
            return module
        return __import__(name)

    monkeypatch.setattr("importlib.import_module", import_module)
    return module


def _config(**overrides):
    config = {"socket_path": "/tmp/nixl-store.sock", "namespace": "transfer-queue", "max_batch_items": 2}
    config.update(overrides)
    return config


def _make_client(monkeypatch, **overrides):
    _install_fake_nixl_store(monkeypatch)
    return NixlStoreClient(_config(**overrides))


def _encoded(client, value):
    return client._encode_values([value])[0]


def _assert_values_equal(actual, expected):
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        if torch.is_tensor(expected_value):
            assert torch.equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def test_config_parsing_connector_creation_and_close(monkeypatch):
    client = _make_client(monkeypatch, socket_path="/tmp/custom.sock", namespace="ns", max_batch_items=3)
    connector = FakeConnector.instances[0]

    assert client.socket_path == "/tmp/custom.sock"
    assert client.namespace == b"ns"
    assert client.max_batch_items == 3
    assert connector.socket_path == "/tmp/custom.sock"
    assert connector.namespace == b"ns"

    client.close()
    client.close()

    assert connector.close_calls == 1
    assert client._connector is None


def test_runtime_socket_paths_select_local_node_socket(monkeypatch):
    import ray

    _install_fake_nixl_store(monkeypatch)
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "10.1.2.3")

    client = NixlStoreClient(
        _config(
            socket_path="/tmp/fallback.sock",
            runtime_socket_paths={
                "10.1.2.3": "/tmp/local-node.sock",
                "10.1.2.4": "/tmp/other-node.sock",
            },
        )
    )

    assert client.socket_path == "/tmp/local-node.sock"
    assert FakeConnector.instances[0].socket_path == "/tmp/local-node.sock"


def test_connector_construction_failure_has_context_and_partial_close_is_safe(monkeypatch):
    _install_fake_nixl_store(monkeypatch)
    FakeConnector.connect_error = FakeStoreError("unavailable")

    with pytest.raises(RuntimeError) as exc_info:
        NixlStoreClient(_config(socket_path="/tmp/bad.sock", namespace="ns"))

    message = str(exc_info.value)
    assert exc_info.value.__cause__ is FakeConnector.connect_error
    assert "failed to connect" in message
    assert "/tmp/bad.sock" in message
    assert "namespace" in message
    assert FakeConnector.instances == []

    partial_client = NixlStoreClient.__new__(NixlStoreClient)
    partial_client.close()
    partial_client._connector = None
    partial_client.close()


def test_invalid_config_errors_before_optional_import(monkeypatch):
    monkeypatch.setattr(
        "importlib.import_module",
        lambda name, package=None: pytest.fail(f"unexpected optional dependency import: {name}"),
    )

    with pytest.raises(ValueError, match="socket_path"):
        NixlStoreClient({"namespace": "ns"})
    with pytest.raises(ValueError, match="max_batch_items"):
        NixlStoreClient({"socket_path": "/tmp/ns.sock", "namespace": "ns", "max_batch_items": 0})


def test_put_writes_buffer_items_and_returns_metadata(monkeypatch):
    client = _make_client(monkeypatch, max_batch_items=1)
    connector = FakeConnector.instances[0]
    monkeypatch.setattr(
        NixlStoreClient,
        "_encode_values",
        staticmethod(lambda values: [bytearray(b"encoded-a"), bytearray(b"encoded-b")]),
    )

    metadata = client.put(["a", "b"], ["raw-a", "raw-b"])

    assert [[item.key for item in call] for call in connector.upsert_buffer_calls] == [
        [b"a"],
        [b"b"],
    ]
    assert [item.buffer for item in FakePutBufferItem.constructed] == [bytearray(b"encoded-a"), bytearray(b"encoded-b")]
    assert [item.size for item in FakePutBufferItem.constructed] == [len(b"encoded-a"), len(b"encoded-b")]
    assert [type(buffer) for buffer in FakeBufferRef.constructed] == [bytearray, bytearray]
    assert metadata == [{"encoded_size": len(b"encoded-a")}, {"encoded_size": len(b"encoded-b")}]


def test_put_uses_raw_cpu_contiguous_tensor_buffer(monkeypatch):
    client = _make_client(monkeypatch)
    connector = FakeConnector.instances[0]
    tensor = torch.arange(8, dtype=torch.float32)

    metadata = client.put(["tensor"], [tensor])

    assert [[item.key for item in call] for call in connector.upsert_buffer_calls] == [[b"tensor"]]
    assert FakeBufferRef.constructed == []
    assert len(FakeBufferRef.addresses) == 1
    assert FakeBufferRef.addresses[0].ptr == tensor.data_ptr()
    assert FakePutBufferItem.constructed[0].buffer is FakeBufferRef.addresses[0]
    assert FakePutBufferItem.constructed[0].size == tensor.nbytes
    assert metadata == [{"nixl_store_format": "raw_tensor_v1", "encoded_size": tensor.nbytes}]


def test_put_packs_non_contiguous_tensor(monkeypatch):
    client = _make_client(monkeypatch)
    connector = FakeConnector.instances[0]
    monkeypatch.setattr(NixlStoreClient, "_encode_values", staticmethod(lambda values: [bytearray(b"packed")]))
    tensor = torch.arange(8, dtype=torch.float32)[::2]

    metadata = client.put(["tensor"], [tensor])

    assert [[item.key for item in call] for call in connector.upsert_buffer_calls] == [[b"tensor"]]
    assert FakeBufferRef.addresses == []
    assert [item.buffer for item in FakePutBufferItem.constructed] == [bytearray(b"packed")]
    assert metadata == [{"encoded_size": len(b"packed")}]


def test_put_per_item_non_completed_status_raises_with_key_and_status(monkeypatch):
    client = _make_client(monkeypatch)
    connector = FakeConnector.instances[0]
    connector.upsert_results = [
        [SimpleNamespace(status=FakeStatus.COMPLETED), SimpleNamespace(status=FakeStatus.FAILED)],
    ]

    with pytest.raises(RuntimeError) as exc_info:
        client.put(["ok", "bad"], [1, 2])

    message = str(exc_info.value)
    assert "bad" in message
    assert FakeStatus.FAILED in message


def test_put_whole_call_store_error_has_context_and_is_not_replayed(monkeypatch):
    client = _make_client(monkeypatch, max_batch_items=2)
    connector = FakeConnector.instances[0]
    connector.upsert_error = FakeStoreError("native Store failure: UNAVAILABLE")

    with pytest.raises(RuntimeError) as exc_info:
        client.put(["a", "b", "c"], [1, 2, 3])

    assert exc_info.value.__cause__ is connector.upsert_error
    assert "upsert_buffer_many failed" in str(exc_info.value)
    assert len(connector.upsert_buffer_calls) == 1
    assert [item.key for item in connector.upsert_buffer_calls[0]] == [b"a", b"b"]


def test_get_uses_encoded_size_metadata(monkeypatch):
    client = _make_client(monkeypatch)
    connector = FakeConnector.instances[0]
    values = [torch.tensor([1, 2, 3]), {"kind": "object"}]
    payloads = [_encoded(client, value) for value in values]
    connector.payloads = {b"tensor": payloads[0], b"object": payloads[1]}

    result = client.get(
        ["tensor", "object"],
        custom_backend_meta=[{"encoded_size": len(payload)} for payload in payloads],
    )

    _assert_values_equal(result, values)
    assert [[item.key for item in call] for call in connector.get_calls] == [[b"tensor", b"object"]]


def test_get_reads_raw_tensor_into_allocated_tensor(monkeypatch):
    client = _make_client(monkeypatch)
    connector = FakeConnector.instances[0]
    expected = torch.arange(8, dtype=torch.float32)
    connector.payloads = {b"tensor": expected.numpy().tobytes()}

    result = client.get(
        ["tensor"],
        shapes=[expected.shape],
        dtypes=[expected.dtype],
        custom_backend_meta=[{"nixl_store_format": "raw_tensor_v1", "encoded_size": expected.nbytes}],
    )

    assert len(result) == 1
    assert torch.equal(result[0], expected)
    assert len(connector.get_calls) == 1
    assert isinstance(connector.get_calls[0][0].buffer, FakeAddressRef)
    assert FakeGetItem.constructed[0].size == expected.nbytes


def test_get_mixed_raw_tensor_and_packed_object_preserves_order(monkeypatch):
    client = _make_client(monkeypatch)
    connector = FakeConnector.instances[0]
    tensor = torch.arange(4, dtype=torch.int64)
    obj = {"kind": "packed"}
    packed = _encoded(client, obj)
    connector.payloads = {
        b"object": packed,
        b"tensor": tensor.numpy().tobytes(),
    }

    result = client.get(
        ["object", "tensor"],
        shapes=[None, tensor.shape],
        dtypes=[None, tensor.dtype],
        custom_backend_meta=[
            {"encoded_size": len(packed)},
            {"nixl_store_format": "raw_tensor_v1", "encoded_size": tensor.nbytes},
        ],
    )

    assert result[0] == obj
    assert torch.equal(result[1], tensor)


def test_get_raw_tensor_metadata_mismatch_has_key_context(monkeypatch):
    client = _make_client(monkeypatch)

    with pytest.raises(RuntimeError) as exc_info:
        client.get(
            ["tensor"],
            shapes=[(8,)],
            dtypes=[torch.float32],
            custom_backend_meta=[{"nixl_store_format": "raw_tensor_v1", "encoded_size": 1}],
        )

    message = str(exc_info.value)
    assert "tensor" in message
    assert "expected 1 bytes" in message


def test_get_requires_encoded_size_metadata_for_packed_values(monkeypatch):
    client = _make_client(monkeypatch)
    connector = FakeConnector.instances[0]

    with pytest.raises(ValueError) as exc_info:
        client.get(["key"], custom_backend_meta=[{}])

    assert "encoded_size" in str(exc_info.value)
    assert "key" in str(exc_info.value)
    assert connector.get_calls == []


def test_get_uses_bytes_copied_and_ignores_destination_tail(monkeypatch):
    client = _make_client(monkeypatch)
    connector = FakeConnector.instances[0]
    connector.payloads = {b"key": b"abc"}
    decoded_payloads = []
    decoded_payload_types = []

    def decode(payloads):
        decoded_payload_types.extend(type(payload) for payload in payloads)
        decoded_payloads.extend(bytes(payload) for payload in payloads)
        return ["decoded"]

    monkeypatch.setattr(NixlStoreClient, "_decode_payloads", staticmethod(decode))

    assert client.get(["key"], custom_backend_meta=[{"encoded_size": 8}]) == ["decoded"]
    assert decoded_payload_types == [memoryview]
    assert decoded_payloads == [b"abc"]


def test_registered_copy_function_copies_between_raw_buffers(monkeypatch):
    _make_client(monkeypatch)
    copy_fn = FakeConnector.instances[0].copy_fn
    src = (ctypes.c_char * 4).from_buffer_copy(b"abcd")
    dst = (ctypes.c_char * 4)()

    status = copy_fn(SimpleNamespace(ptr=ctypes.addressof(dst)), SimpleNamespace(ptr=ctypes.addressof(src)), 4)

    assert status == FakeStatus.COMPLETED
    assert dst.raw == b"abcd"
    assert copy_fn(None, None, 0) == FakeStatus.COMPLETED  # an empty copy must not reach torch.frombuffer


def test_get_stale_too_small_size_hint_raises_with_key_context(monkeypatch):
    client = _make_client(monkeypatch)
    connector = FakeConnector.instances[0]
    value = {"fresh": "value"}
    payload = _encoded(client, value)
    connector.payloads = {b"key": payload}

    with pytest.raises(RuntimeError) as exc_info:
        client.get(["key"], custom_backend_meta=[{"encoded_size": len(payload) - 1}])

    message = str(exc_info.value)
    assert "encoded_size" in message
    assert "key" in message
    assert len(connector.get_calls) == 1
    assert connector.get_calls[0][0].size == len(payload) - 1


def test_get_missing_key_raises_useful_error(monkeypatch):
    client = _make_client(monkeypatch)

    with pytest.raises(RuntimeError) as exc_info:
        client.get(["missing"], custom_backend_meta=[{"encoded_size": 1}])

    message = str(exc_info.value)
    assert "missing" in message
    assert FakeStatus.MISSING_KEY in message


def test_get_whole_call_store_error_has_context(monkeypatch):
    client = _make_client(monkeypatch)
    connector = FakeConnector.instances[0]
    error = FakeStoreError("native Store failure")
    connector.get_error = error

    with pytest.raises(RuntimeError) as exc_info:
        client.get(["a"], custom_backend_meta=[{"encoded_size": 1}])

    assert exc_info.value.__cause__ is error
    assert "get_many" in str(exc_info.value)
    assert "a" in str(exc_info.value)


def test_get_chunks_get_many(monkeypatch):
    client = _make_client(monkeypatch, max_batch_items=2)
    connector = FakeConnector.instances[0]
    connector.payloads = {key: key.upper() for key in (b"a", b"b", b"c")}
    monkeypatch.setattr(NixlStoreClient, "_decode_payloads", staticmethod(lambda payloads: list(payloads)))

    assert client.get(
        ["a", "b", "c"],
        custom_backend_meta=[{"encoded_size": 1}, {"encoded_size": 1}, {"encoded_size": 1}],
    ) == [b"A", b"B", b"C"]
    assert [[item.key for item in call] for call in connector.get_calls] == [[b"a", b"b"], [b"c"]]


def test_clear_chunks_converts_keys_waits_and_accepts_missing(monkeypatch):
    client = _make_client(monkeypatch, max_batch_items=2)
    connector = FakeConnector.instances[0]
    connector.clear_results = [
        [SimpleNamespace(status=FakeStatus.COMPLETED), SimpleNamespace(status=FakeStatus.MISSING_KEY)],
        [FakeStatus.COMPLETED],
    ]

    client.clear(["a", "b", "c"], custom_backend_meta=[{"ignored": True}, None, {}])

    assert connector.clear_calls == [[b"a", b"b"], [b"c"]]
    assert [future.wait_calls for future in connector.clear_futures] == [1, 1]


def test_clear_non_tolerated_status_raises_with_key_and_status(monkeypatch):
    client = _make_client(monkeypatch)
    connector = FakeConnector.instances[0]
    connector.clear_results = [[SimpleNamespace(status=FakeStatus.FAILED)]]

    with pytest.raises(RuntimeError) as exc_info:
        client.clear(["bad"])

    message = str(exc_info.value)
    assert "bad" in message
    assert FakeStatus.FAILED in message


@pytest.mark.parametrize("failing_call", ["clear_many", "wait"])
def test_clear_whole_call_and_wait_errors_have_context(monkeypatch, failing_call):
    client = _make_client(monkeypatch)
    connector = FakeConnector.instances[0]
    error = FakeStoreError("native Store failure")
    if failing_call == "clear_many":
        connector.clear_error = error
    else:
        connector.clear_wait_error = error

    with pytest.raises(RuntimeError) as exc_info:
        client.clear(["a"])

    assert exc_info.value.__cause__ is error
    assert failing_call in str(exc_info.value)
    assert "a" in str(exc_info.value)
