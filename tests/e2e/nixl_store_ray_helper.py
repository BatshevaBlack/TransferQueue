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

import ray
import torch


@ray.remote(num_cpus=0.01)
def read_nixl_store_key(namespace: str, partition_id: str) -> str:
    import transfer_queue as tq

    os.environ["TQ_RAY_NAMESPACE"] = namespace
    tq.init()
    try:
        got = tq.kv_batch_get(keys=["sample"], partition_id=partition_id, select_fields="x")
        assert torch.equal(got["x"][0], torch.arange(8, dtype=torch.int64))
        return ray.util.get_node_ip_address()
    finally:
        tq.close()
