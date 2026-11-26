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
import importlib
import socket
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch

from transfer_queue.storage.clients.base import StorageClientFactory, StorageKVClient
from transfer_queue.utils.serial_utils import batch_decode_from, batch_encode_into


NIXL_STORE_MAX_BATCH_ITEMS = 4096
NIXL_STORE_FORMAT_KEY = "nixl_store_format"
NIXL_STORE_FORMAT_RAW_TENSOR = "raw_tensor_v1"


@StorageClientFactory.register("NixlStoreClient")
class NixlStoreClient(StorageKVClient):
    """NIXL Store KV client skeleton with local connector lifecycle and encoding helpers."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.socket_path = self._parse_socket_path(config)
        self.namespace = self._parse_namespace(config)
        self.max_batch_items = self._parse_max_batch_items(config)
        self._connector = None
        self._nixl_store = self._import_nixl_store()
        self._connector = self._connect()
        if not self._supports_buffer_put():
            raise RuntimeError(
                "NixlStoreClient requires nixl_store BufferRef PUT support "
                "(PutBufferItem, BufferRef, and connector.upsert_buffer_many)."
            )
        cpu = self._nixl_store.MemoryType.CPU_DRAM
        self._connector.register_copy_fn(cpu, cpu, self._copy_with_torch_threads)

    def _copy_with_torch_threads(self, dst, src, nbytes: int):
        """Store copy function: the Store's built-in CPU copy is single-threaded, torch's copy_ is not.

        It runs inside TQ's get/put, so torch uses the thread count TQ sets around those calls.
        """
        if nbytes:  # torch.frombuffer rejects empty buffers
            dst_view = torch.frombuffer((ctypes.c_char * nbytes).from_address(dst.ptr), dtype=torch.uint8)
            src_view = torch.frombuffer((ctypes.c_char * nbytes).from_address(src.ptr), dtype=torch.uint8)
            dst_view.copy_(src_view)
        return self._nixl_store.Status.COMPLETED

    def _parse_socket_path(self, config: dict[str, Any]) -> str:
        runtime_socket_paths = config.get("runtime_socket_paths")
        if runtime_socket_paths is not None and hasattr(runtime_socket_paths, "items"):
            runtime_socket_paths = dict(runtime_socket_paths.items())
        if isinstance(runtime_socket_paths, dict) and runtime_socket_paths:
            local_keys = self._local_socket_keys()
            for key in local_keys:
                socket_path = runtime_socket_paths.get(key)
                if isinstance(socket_path, str) and socket_path:
                    return socket_path
            known_nodes = sorted(str(key) for key in runtime_socket_paths.keys())
            raise ValueError(
                "NixlStoreClient could not find a runtime NSClient socket for this node. "
                f"Tried {local_keys!r}; known nodes are {known_nodes!r}."
            )

        socket_path = config.get("socket_path")
        if not isinstance(socket_path, str) or not socket_path:
            raise ValueError("NixlStoreClient requires a non-empty string 'socket_path' config value")
        return socket_path

    def _local_socket_keys(self) -> list[str]:
        keys = []
        try:
            import ray

            keys.append(ray.util.get_node_ip_address())
        except Exception:
            pass
        try:
            hostname = socket.gethostname()
            keys.append(hostname)
            keys.append(socket.gethostbyname(hostname))
        except Exception:
            pass
        return list(dict.fromkeys(key for key in keys if key))

    def _parse_namespace(self, config: dict[str, Any]) -> bytes:
        namespace = config.get("namespace")
        if isinstance(namespace, str) and namespace:
            return namespace.encode("utf-8")
        if isinstance(namespace, bytes | bytearray) and namespace:
            return bytes(namespace)
        raise ValueError("NixlStoreClient requires a non-empty 'namespace' config value as str or bytes")

    def _parse_max_batch_items(self, config: dict[str, Any]) -> int:
        max_batch_items = config.get("max_batch_items", NIXL_STORE_MAX_BATCH_ITEMS)
        if (
            not isinstance(max_batch_items, int)
            or isinstance(max_batch_items, bool)
            or max_batch_items < 1
            or max_batch_items > NIXL_STORE_MAX_BATCH_ITEMS
        ):
            raise ValueError(
                f"NixlStoreClient 'max_batch_items' must be an integer between 1 and "
                f"{NIXL_STORE_MAX_BATCH_ITEMS}, got {max_batch_items!r}"
            )
        return max_batch_items

    def _import_nixl_store(self):
        try:
            return importlib.import_module("nixl_store")
        except ImportError as exc:
            raise ImportError(
                "NixlStore backend requires the optional `nixl_store` Python package. "
                "Install or build the NIXL Store Python bindings, or choose another "
                "TransferQueue backend."
            ) from exc

    def _connect(self):
        try:
            connector_cls = self._nixl_store.Connector
        except AttributeError as exc:
            raise RuntimeError("nixl_store module does not expose Connector") from exc
        try:
            return connector_cls.connect(self.socket_path, self.namespace)
        except AttributeError as exc:
            raise RuntimeError("nixl_store.Connector does not expose connect(socket, namespace)") from exc
        except Exception as exc:
            raise RuntimeError(
                f"NixlStoreClient failed to connect to NIXL Store at socket {self.socket_path!r} "
                f"with namespace {self.namespace!r}"
            ) from exc

    def _iter_chunks(self, items: Sequence[Any]) -> Iterator[Sequence[Any]]:
        for start in range(0, len(items), self.max_batch_items):
            yield items[start : start + self.max_batch_items]

    def _encode_values(self, values: list[Any]) -> list[bytearray]:
        buffers, _ = batch_encode_into(values, lambda sizes: [bytearray(size) for size in sizes])
        return buffers

    def _decode_payloads(self, payloads: Sequence[bytes | bytearray | memoryview]) -> list[Any]:
        return batch_decode_from([memoryview(payload) for payload in payloads])

    def _encode_keys(self, keys: list[str]) -> list[bytes]:
        return [key.encode("utf-8") for key in keys]

    def _validate_optional_lengths(
        self,
        keys: list[str],
        shapes: Sequence[Any] | None,
        dtypes: Sequence[Any] | None,
        custom_backend_meta: Sequence[Any] | None,
    ) -> None:
        for name, values in (
            ("shapes", shapes),
            ("dtypes", dtypes),
            ("custom_backend_meta", custom_backend_meta),
        ):
            if values is not None and len(values) != len(keys):
                raise ValueError(
                    f"NixlStoreClient.get requires {name} length to match keys length: "
                    f"{len(values)} != {len(keys)}"
                )

    def _encoded_size_from_meta(self, meta: Any) -> int | None:
        if not isinstance(meta, Mapping):
            return None
        encoded_size = meta.get("encoded_size")
        if isinstance(encoded_size, int) and not isinstance(encoded_size, bool) and encoded_size >= 0:
            return encoded_size
        return None

    def _validate_capacity(self, key: str, size: Any) -> int:
        if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
            return size
        raise RuntimeError(f"NixlStoreClient.get received invalid encoded size {size!r} for key {key!r}")

    def _encoded_size_required_from_meta(self, key: str, meta: Any) -> int:
        encoded_size = self._encoded_size_from_meta(meta)
        if encoded_size is None:
            raise ValueError(f"NixlStoreClient.get requires encoded_size metadata for packed key {key!r}")
        return encoded_size

    def _supports_buffer_put(self) -> bool:
        return (
            hasattr(self._connector, "upsert_buffer_many")
            and hasattr(self._nixl_store, "PutBufferItem")
            and hasattr(self._nixl_store, "BufferRef")
            and hasattr(self._nixl_store.BufferRef, "from_buffer")
            and hasattr(self._nixl_store.BufferRef, "from_address")
        )

    def _buffer_ref_from_address(self, ptr: int):
        memory_type = getattr(getattr(self._nixl_store, "MemoryType", None), "CPU_DRAM", None)
        if memory_type is None:
            return self._nixl_store.BufferRef.from_address(ptr)
        return self._nixl_store.BufferRef.from_address(ptr, memory_type)

    def _can_put_raw_tensor(self, value: Any) -> bool:
        return (
            torch.is_tensor(value)
            and value.device.type == "cpu"
            and value.is_contiguous()
            and value.nbytes > 0
        )

    def _is_raw_tensor_meta(self, meta: Any) -> bool:
        return isinstance(meta, Mapping) and meta.get(NIXL_STORE_FORMAT_KEY) == NIXL_STORE_FORMAT_RAW_TENSOR

    def _get_packed_payloads(
        self,
        keys: list[str],
        key_bytes: list[bytes],
        capacities: list[int],
    ) -> list[memoryview]:
        completed_status = self._nixl_store.Status.COMPLETED
        missing_status = self._nixl_store.Status.MISSING_KEY
        buffer_too_small_status = self._nixl_store.Status.BUFFER_TOO_SMALL
        payloads: list[memoryview] = []

        for chunk_indexes in self._iter_chunks(range(len(keys))):
            chunk_keys = [keys[index] for index in chunk_indexes]
            buffers = []
            items = []
            for index in chunk_indexes:
                capacity = self._validate_capacity(keys[index], capacities[index])
                # Uninitialized, unlike bytearray: no zero-fill pass, and the copy-out pays the page faults on several threads.
                buffer = torch.empty(capacity, dtype=torch.uint8, device="cpu").numpy()
                buffers.append(buffer)
                items.append(
                    self._nixl_store.GetItem(
                        key_bytes[index],
                        self._nixl_store.BufferRef.from_buffer(buffer),
                        capacity,
                    )
                )

            try:
                results = self._connector.get_many(items)
            except self._nixl_store.StoreError as exc:
                raise RuntimeError(f"NixlStoreClient.get get_many failed for keys {chunk_keys!r}") from exc

            if len(results) != len(chunk_indexes):
                raise RuntimeError(
                    f"NixlStoreClient.get expected {len(chunk_indexes)} get_many results for keys "
                    f"{chunk_keys!r}, got {len(results)}"
                )

            failures = []
            for index, result, buffer in zip(chunk_indexes, results, buffers, strict=True):
                if result.status == completed_status:
                    bytes_copied = result.bytes_copied
                    if (
                        not isinstance(bytes_copied, int)
                        or isinstance(bytes_copied, bool)
                        or bytes_copied < 0
                        or bytes_copied > len(buffer)
                    ):
                        raise RuntimeError(
                            f"NixlStoreClient.get received invalid bytes_copied {bytes_copied!r} "
                            f"for key {keys[index]!r}"
                        )
                    payloads.append(memoryview(buffer)[:bytes_copied])
                elif result.status == buffer_too_small_status:
                    raise RuntimeError(
                        f"NixlStoreClient.get packed key {keys[index]!r} encoded_size metadata "
                        f"{capacities[index]} was too small (status={result.status!r})"
                    )
                elif result.status == missing_status:
                    raise RuntimeError(
                        f"NixlStoreClient.get missing Store key {keys[index]!r} during get_many "
                        f"(status={result.status!r})"
                    )
                else:
                    failures.append({"key": keys[index], "status": result.status})
            if failures:
                raise RuntimeError(f"NixlStoreClient.get failed for Store items: {failures!r}")
        return payloads

    def close(self) -> None:
        connector = getattr(self, "_connector", None)
        if connector is None:
            return
        connector.close()
        self._connector = None

    def put(self, keys: list[str], values: list[Any]) -> list[dict]:
        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError("NixlStoreClient.put requires keys and values to be lists")
        if len(keys) != len(values):
            raise ValueError("NixlStoreClient.put requires the same number of keys and values")
        if not keys:
            return []

        key_bytes = self._encode_keys(keys)
        completed_status = self._nixl_store.Status.COMPLETED
        metadata: list[dict] = [{} for _ in values]
        entries: list[tuple[str, bytes, str, Any, int]] = []
        raw_flags = [self._can_put_raw_tensor(value) for value in values]
        packed_payloads = iter(
            self._encode_values([value for value, is_raw in zip(values, raw_flags, strict=True) if not is_raw])
        )

        for index, (key, encoded_key, value, is_raw) in enumerate(zip(keys, key_bytes, values, raw_flags, strict=True)):
            if is_raw:
                assert torch.is_tensor(value)
                metadata[index] = {NIXL_STORE_FORMAT_KEY: NIXL_STORE_FORMAT_RAW_TENSOR, "encoded_size": value.nbytes}
                entries.append((key, encoded_key, NIXL_STORE_FORMAT_RAW_TENSOR, value, value.nbytes))
            else:
                payload = next(packed_payloads)
                metadata[index] = {"encoded_size": len(payload)}
                entries.append((key, encoded_key, "packed", payload, len(payload)))

        for chunk in self._iter_chunks(entries):
            items = []
            for _, encoded_key, value_format, payload, size in chunk:
                buffer_ref = (
                    self._buffer_ref_from_address(payload.data_ptr())
                    if value_format == NIXL_STORE_FORMAT_RAW_TENSOR
                    else self._nixl_store.BufferRef.from_buffer(payload)
                )
                items.append(self._nixl_store.PutBufferItem(encoded_key, buffer_ref, size))
            try:
                results = self._connector.upsert_buffer_many(items)
            except self._nixl_store.StoreError as exc:
                chunk_keys = [key for key, _, _, _, _ in chunk]
                raise RuntimeError(f"NixlStoreClient.put upsert_buffer_many failed for keys {chunk_keys!r}") from exc
            self._validate_put_results(chunk, results, completed_status, "upsert_buffer_many")

        return metadata

    def _validate_put_results(
        self,
        chunk: Sequence[tuple[str, bytes, str, Any, int]],
        results: Sequence[Any],
        completed_status: Any,
        operation: str,
    ) -> None:
        if len(results) != len(chunk):
            chunk_keys = [key for key, _, _, _, _ in chunk]
            raise RuntimeError(
                f"NixlStoreClient.put expected {len(chunk)} {operation} results for keys {chunk_keys!r}, "
                f"got {len(results)}"
            )

        failures = [
            {"key": key, "status": result.status}
            for (key, _, _, _, _), result in zip(chunk, results, strict=True)
            if result.status != completed_status
        ]
        if failures:
            raise RuntimeError(f"NixlStoreClient.put failed for Store items: {failures!r}")

    def _get_packed_values(
        self,
        keys: list[str],
        key_bytes: list[bytes],
        custom_backend_meta: list[Any] | None,
    ) -> list[Any]:
        meta_list = custom_backend_meta if custom_backend_meta is not None else [None] * len(keys)
        capacities = [
            self._encoded_size_required_from_meta(key, meta)
            for key, meta in zip(keys, meta_list, strict=True)
        ]
        return self._decode_payloads(self._get_packed_payloads(keys, key_bytes, capacities))

    def _get_raw_tensors(
        self,
        keys: list[str],
        key_bytes: list[bytes],
        shapes: Sequence[Any],
        dtypes: Sequence[Any],
        custom_backend_meta: Sequence[Any],
        indexes: Sequence[int],
        output: list[Any],
    ) -> None:
        completed_status = self._nixl_store.Status.COMPLETED
        missing_status = self._nixl_store.Status.MISSING_KEY

        for chunk_indexes in self._iter_chunks(indexes):
            chunk_keys = [keys[index] for index in chunk_indexes]
            tensors: dict[int, torch.Tensor] = {}
            item_indexes: list[int] = []
            items = []

            for index in chunk_indexes:
                dtype = dtypes[index]
                if dtype is None:
                    raise RuntimeError(f"NixlStoreClient.get raw tensor key {keys[index]!r} is missing dtype metadata")
                tensor = torch.empty(tuple(shapes[index]), dtype=dtype, device="cpu")
                expected_size = self._encoded_size_from_meta(custom_backend_meta[index])
                if expected_size != tensor.nbytes:
                    raise RuntimeError(
                        f"NixlStoreClient.get raw tensor key {keys[index]!r} expected {expected_size} bytes, "
                        f"but shape/dtype requires {tensor.nbytes} bytes"
                    )
                tensors[index] = tensor
                if tensor.nbytes == 0:
                    output[index] = tensor
                    continue
                items.append(
                    self._nixl_store.GetItem(
                        key_bytes[index],
                        self._buffer_ref_from_address(tensor.data_ptr()),
                        tensor.nbytes,
                    )
                )
                item_indexes.append(index)

            if not items:
                continue

            try:
                results = self._connector.get_many(items)
            except self._nixl_store.StoreError as exc:
                raise RuntimeError(f"NixlStoreClient.get raw tensor get_many failed for keys {chunk_keys!r}") from exc

            if len(results) != len(items):
                raise RuntimeError(
                    f"NixlStoreClient.get expected {len(items)} raw tensor get_many results for keys "
                    f"{chunk_keys!r}, got {len(results)}"
                )

            failures = []
            for index, result in zip(item_indexes, results, strict=True):
                if result.status == completed_status:
                    if result.bytes_copied != tensors[index].nbytes:
                        raise RuntimeError(
                            f"NixlStoreClient.get raw tensor key {keys[index]!r} copied "
                            f"{result.bytes_copied!r} bytes, expected {tensors[index].nbytes}"
                        )
                    output[index] = tensors[index]
                elif result.status == missing_status:
                    raise RuntimeError(
                        f"NixlStoreClient.get missing Store key {keys[index]!r} during raw tensor get_many "
                        f"(status={result.status!r})"
                    )
                else:
                    failures.append({"key": keys[index], "status": result.status})
            if failures:
                raise RuntimeError(f"NixlStoreClient.get raw tensor failed for Store items: {failures!r}")

    def get(self, keys: list[str], shapes=None, dtypes=None, custom_backend_meta=None) -> list[Any]:
        if not isinstance(keys, list):
            raise ValueError("NixlStoreClient.get requires keys to be a list")
        self._validate_optional_lengths(keys, shapes, dtypes, custom_backend_meta)
        if not keys:
            return []

        key_bytes = self._encode_keys(keys)
        meta_list = list(custom_backend_meta) if custom_backend_meta is not None else [None] * len(keys)
        raw_indexes = []
        packed_indexes = []
        for index, meta in enumerate(meta_list):
            if self._is_raw_tensor_meta(meta):
                raw_indexes.append(index)
            else:
                packed_indexes.append(index)
        output: list[Any] = [None] * len(keys)

        if raw_indexes:
            if shapes is None or dtypes is None:
                raise RuntimeError("NixlStoreClient.get raw tensor values require shapes and dtypes metadata")
            self._get_raw_tensors(keys, key_bytes, shapes, dtypes, meta_list, raw_indexes, output)

        if packed_indexes:
            packed_values = self._get_packed_values(
                [keys[index] for index in packed_indexes],
                [key_bytes[index] for index in packed_indexes],
                [meta_list[index] for index in packed_indexes],
            )
            for index, value in zip(packed_indexes, packed_values, strict=True):
                output[index] = value

        return output

    def clear(self, keys: list[str], custom_backend_meta=None) -> None:
        if not isinstance(keys, list):
            raise ValueError("NixlStoreClient.clear requires keys to be a list")
        if not keys:
            return

        completed_status = self._nixl_store.Status.COMPLETED
        missing_status = self._nixl_store.Status.MISSING_KEY
        entries = list(zip(keys, self._encode_keys(keys), strict=True))

        for chunk in self._iter_chunks(entries):
            chunk_keys = [key for key, _ in chunk]
            try:
                future = self._connector.clear_many([key_bytes for _, key_bytes in chunk])
            except self._nixl_store.StoreError as exc:
                raise RuntimeError(f"NixlStoreClient.clear clear_many failed for keys {chunk_keys!r}") from exc

            try:
                results = future.wait()
            except Exception as exc:
                raise RuntimeError(f"NixlStoreClient.clear wait failed for keys {chunk_keys!r}") from exc

            if len(results) != len(chunk):
                raise RuntimeError(
                    f"NixlStoreClient.clear expected {len(chunk)} clear_many results for keys {chunk_keys!r}, "
                    f"got {len(results)}"
                )

            failures = []
            for (key, _), result in zip(chunk, results, strict=True):
                status = getattr(result, "status", result)
                if status not in (completed_status, missing_status):
                    failures.append({"key": key, "status": status})
            if failures:
                raise RuntimeError(f"NixlStoreClient.clear failed for Store items: {failures!r}")
