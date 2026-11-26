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

import importlib
import sys

import pytest
from omegaconf import OmegaConf

import transfer_queue.storage
import transfer_queue.storage.bootstrap
import transfer_queue.storage.clients
from transfer_queue.storage.clients.base import StorageClientFactory
from transfer_queue.storage.bootstrap import StorageBootstrapProvider
from transfer_queue.storage.clients.nixl_store_client import NixlStoreClient
from transfer_queue.storage.managers import NixlStoreManager
from transfer_queue.storage.managers.base import StorageManagerFactory


def _block_nixl_store_import(monkeypatch):
    original_import_module = importlib.import_module

    def guarded_import_module(name, package=None):
        if name == "nixl_store" or name.startswith("nixl_store."):
            raise ImportError("blocked optional nixl_store import", name="nixl_store")
        return original_import_module(name, package)

    monkeypatch.delitem(sys.modules, "nixl_store", raising=False)
    monkeypatch.setattr(importlib, "import_module", guarded_import_module)


def test_nixl_store_registrations_exist():
    assert StorageClientFactory._registry["NixlStoreClient"] is NixlStoreClient
    assert StorageManagerFactory._registry["NixlStore"] is NixlStoreManager
    provider = StorageBootstrapProvider.get_provider("NixlStore")
    assert provider is not None
    conf = OmegaConf.create({"backend": {"NixlStore": {"auto_init": False}}})
    assert provider(conf) == {"managed": False, "external": True}


def test_storage_import_does_not_require_nixl_store(monkeypatch):
    _block_nixl_store_import(monkeypatch)

    importlib.reload(transfer_queue.storage.clients)
    importlib.reload(transfer_queue.storage)

    assert StorageClientFactory._registry["NixlStoreClient"] is NixlStoreClient
    assert "nixl_store" not in sys.modules


def test_nixl_store_selected_without_dependency_is_actionable(monkeypatch):
    _block_nixl_store_import(monkeypatch)

    with pytest.raises(ImportError) as exc_info:
        StorageClientFactory.create(
            "NixlStoreClient",
            {"socket_path": "/tmp/nixl-store.sock", "namespace": "test", "max_batch_items": 1},
        )

    message = str(exc_info.value)
    assert "NixlStore backend" in message
    assert "nixl_store" in message
    assert "Install or build" in message
    assert "choose another TransferQueue backend" in message
