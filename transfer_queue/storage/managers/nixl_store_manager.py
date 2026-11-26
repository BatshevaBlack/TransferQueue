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

from typing import Any

import zmq

from transfer_queue.storage.managers.base import KVStorageManager, StorageManagerFactory
from transfer_queue.utils.zmq_utils import ZMQServerInfo


@StorageManagerFactory.register("NixlStore")
class NixlStoreManager(KVStorageManager):
    """Storage manager for the NIXL Store backend."""

    def __init__(
        self,
        controller_info: ZMQServerInfo,
        config: dict[str, Any],
        zmq_context: zmq.asyncio.Context | None = None,
    ):
        config = (config or {}).copy()
        if config.get("client_name") not in (None, "NixlStoreClient"):
            raise ValueError(f"NixlStoreManager only supports 'NixlStoreClient', got: {config.get('client_name')}")
        super().__init__(controller_info, {**config, "client_name": "NixlStoreClient"}, zmq_context=zmq_context)

    def close(self) -> None:
        try:
            storage_client = getattr(self, "storage_client", None)
            if storage_client is not None:
                storage_client.close()
                self.storage_client = None
        finally:
            super().close()
