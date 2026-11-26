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

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from transfer_queue.metadata import BatchMeta, extract_field_schema
from transfer_queue.storage.managers.base import KVStorageManager
from transfer_queue.storage.managers.nixl_store_manager import NixlStoreManager


STORAGE_CLIENT_FACTORY_PATH = "transfer_queue.storage.managers.base.StorageClientFactory"


class RecordingStorageClient:
    def __init__(self):
        self.put_calls = []
        self.get_calls = []
        self.clear_calls = []
        self.close_calls = 0
        self.values_by_key = {}

    def put(self, keys, values):
        keys = list(keys)
        values = list(values)
        self.put_calls.append((keys, values))
        self.values_by_key.update(zip(keys, values, strict=True))
        return [{"encoded_size": 100 + index} for index in range(len(keys))]

    def get(self, keys, shapes=None, dtypes=None, custom_backend_meta=None):
        keys = list(keys)
        self.get_calls.append(
            {
                "keys": keys,
                "shapes": list(shapes),
                "dtypes": list(dtypes),
                "custom_backend_meta": list(custom_backend_meta),
            }
        )
        return [self.values_by_key[key] for key in keys]

    def clear(self, keys, custom_backend_meta=None):
        self.clear_calls.append(
            {
                "keys": list(keys),
                "custom_backend_meta": list(custom_backend_meta),
            }
        )

    def close(self):
        self.close_calls += 1


def _sample_data() -> TensorDict:
    return TensorDict(
        {
            "tensor": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
            "object": ["first", {"row": 1}],
        },
        batch_size=2,
    )


def _metadata_for(data: TensorDict) -> BatchMeta:
    return BatchMeta(
        global_indexes=[11, 12],
        partition_ids=["partition-a", "partition-a"],
        field_schema=extract_field_schema(data),
        production_status=np.ones(data.batch_size[0], dtype=np.int8),
    )


def _assert_tensor_rows_equal(actual, expected):
    actual_rows = actual.unbind(0)
    expected_rows = expected.unbind(0)
    assert len(actual_rows) == len(expected_rows)
    for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True):
        assert torch.equal(actual_row, expected_row)


@patch.object(KVStorageManager, "_connect_to_controller", lambda self: None)
def test_nixl_store_manager_sets_client_name_and_passes_config():
    storage_client = RecordingStorageClient()
    config = {
        "socket_path": "/tmp/nixl-store.sock",
        "namespace": "phase5",
        "max_batch_items": 7,
    }

    with patch(f"{STORAGE_CLIENT_FACTORY_PATH}.create", return_value=storage_client) as create:
        manager = NixlStoreManager(controller_info=MagicMock(), config=config)

    assert isinstance(manager, KVStorageManager)
    create.assert_called_once()
    client_name, client_config = create.call_args.args
    assert client_name == "NixlStoreClient"
    assert client_config["client_name"] == "NixlStoreClient"
    assert client_config["socket_path"] == "/tmp/nixl-store.sock"
    assert client_config["namespace"] == "phase5"
    assert client_config["max_batch_items"] == 7
    assert "client_name" not in config


@patch.object(KVStorageManager, "_connect_to_controller", lambda self: None)
def test_nixl_store_manager_rejects_data_parser_before_store_write():
    storage_client = RecordingStorageClient()
    data = _sample_data()
    metadata = _metadata_for(data)

    with patch(f"{STORAGE_CLIENT_FACTORY_PATH}.create", return_value=storage_client):
        manager = NixlStoreManager(controller_info=MagicMock(), config={})

    with pytest.raises(NotImplementedError, match="NixlStore"):
        asyncio.run(manager.put_data(data, metadata, data_parser=lambda value: value))

    assert storage_client.put_calls == []


@patch.object(KVStorageManager, "_connect_to_controller", lambda self: None)
@patch.object(KVStorageManager, "notify_data_update", new_callable=AsyncMock)
def test_nixl_store_manager_preserves_order_and_backend_metadata(mock_notify):
    storage_client = RecordingStorageClient()
    data = _sample_data()
    metadata = _metadata_for(data)

    with patch(f"{STORAGE_CLIENT_FACTORY_PATH}.create", return_value=storage_client):
        manager = NixlStoreManager(controller_info=MagicMock(), config={})

    asyncio.run(manager.put_data(data, metadata))

    expected_keys = ["11@object", "12@object", "11@tensor", "12@tensor"]
    assert storage_client.put_calls[0][0] == expected_keys
    assert storage_client.put_calls[0][1][0] == "first"
    assert storage_client.put_calls[0][1][1] == {"row": 1}
    assert torch.equal(storage_client.put_calls[0][1][2], data["tensor"][0])
    assert torch.equal(storage_client.put_calls[0][1][3], data["tensor"][1])

    per_field_meta = mock_notify.call_args.args[3]
    assert per_field_meta == {
        11: {"object": {"encoded_size": 100}, "tensor": {"encoded_size": 102}},
        12: {"object": {"encoded_size": 101}, "tensor": {"encoded_size": 103}},
    }
    assert metadata._custom_backend_meta == [
        {"object": {"encoded_size": 100}, "tensor": {"encoded_size": 102}},
        {"object": {"encoded_size": 101}, "tensor": {"encoded_size": 103}},
    ]

    result = asyncio.run(manager.get_data(metadata))

    assert storage_client.get_calls == [
        {
            "keys": expected_keys,
            "shapes": [None, None, torch.Size([2]), torch.Size([2])],
            "dtypes": [None, None, torch.int64, torch.int64],
            "custom_backend_meta": [
                {"encoded_size": 100},
                {"encoded_size": 101},
                {"encoded_size": 102},
                {"encoded_size": 103},
            ],
        }
    ]
    assert list(result["object"]) == ["first", {"row": 1}]
    _assert_tensor_rows_equal(result["tensor"], data["tensor"])


@patch.object(KVStorageManager, "_connect_to_controller", lambda self: None)
def test_nixl_store_manager_selected_field_clear_forwards_keys_and_metadata():
    storage_client = RecordingStorageClient()
    metadata = _metadata_for(_sample_data())
    metadata._custom_backend_meta = [
        {"object": {"encoded_size": 11}, "tensor": {"encoded_size": 21}},
        {"object": {"encoded_size": 12}, "tensor": {"encoded_size": 22}},
    ]

    with patch(f"{STORAGE_CLIENT_FACTORY_PATH}.create", return_value=storage_client):
        manager = NixlStoreManager(controller_info=MagicMock(), config={})

    asyncio.run(manager.clear_data(metadata, fields=["tensor"]))

    assert storage_client.clear_calls == [
        {
            "keys": ["11@tensor", "12@tensor"],
            "custom_backend_meta": [{"encoded_size": 21}, {"encoded_size": 22}],
        }
    ]


@patch.object(KVStorageManager, "_connect_to_controller", lambda self: None)
def test_nixl_store_manager_close_closes_storage_client_once():
    storage_client = RecordingStorageClient()

    with patch(f"{STORAGE_CLIENT_FACTORY_PATH}.create", return_value=storage_client):
        manager = NixlStoreManager(controller_info=MagicMock(), config={})

    manager.close()
    manager.close()

    assert storage_client.close_calls == 1
    assert manager.storage_client is None
