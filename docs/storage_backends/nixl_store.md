# NixlStore Backend

## Overview

`NixlStore` is an MVP KV storage backend for TransferQueue that stores
TransferQueue payloads in a NIXL Store deployment.

The MVP stores values through NIXL Store buffers. General Python values and
tensors that cannot use the raw path are packed with TransferQueue's existing
serialization helpers. Contiguous non-empty CPU tensors use a raw tensor fast
path that lets NIXL Store read and write tensor memory directly. Direct GPU
transfers are not part of this MVP.

## Support Level

`NixlStore` is an experimental MVP backend. The implementation supports the KV
put/get/clear path and has opt-in real-service validation, but it is not yet as
turnkey as the default backend or the packaged optional backends.

Before treating this backend as release-ready in a new environment, run the
manual validation checklist in this document. In particular, verify that the
target environment can install the `nixl_store` Python module, find the Store
Manager and NSClient binaries, and run the opt-in real-service e2e tests.

## Service Lifecycle

By default, TransferQueue does not start or stop NIXL Store services for this
backend. Start the Store services before calling `tq.init()`:

- one `nixl-store-manager` for the Store deployment,
- one local `nixl-store-nsclient` sidecar on each node that runs TQ clients,
- a Unix socket path that each local TQ process can use to reach its local
  NSClient.

Each `NixlStoreClient` creates its own connector to the configured local
NSClient socket and closes only that connector. Store services remain externally
owned when `auto_init` is `false`.

When `auto_init` is `true`, the process that creates the TQ controller starts
one Store Manager and one NSClient sidecar per alive Ray node. Attached TQ
processes reuse the stored runtime socket map and do not start duplicate Store
services. `tq.close()` stops only services that TQ started, with NSClients
stopped before the Manager.

## Installation Notes

There is not yet a `TransferQueue[nixl]` extra that installs NIXL Store for you.
Install NIXL Store separately before selecting this backend.

The current NIXL Store package metadata and runtime shape do not yet provide a
complete TransferQueue extra:

- the local NIXL Store package is named `nixl-store`, but its current
  `requires-python` range is narrower than the Python versions TransferQueue
  validates,
- TransferQueue needs the `nixl_store` Python module and the
  `nixl-store-manager` and `nixl-store-nsclient` binaries,
- NSClient also needs the native NIXL runtime libraries and plugins available
  in its process environment.

Add a `TransferQueue[nixl]` extra only after the NIXL Store package provides
that full runtime contract for the Python versions TransferQueue supports.

The Python environment that runs TransferQueue must be able to import
`nixl_store`. The Store Manager and NSClient binaries must be built or installed
from the same NIXL Store release or checkout, and the NSClient process must see
the native NIXL runtime libraries and plugins through its environment.

A source install usually follows this shape:

```bash
cd Nixl-Store
python -m pip install -r python/requirements-dev.txt
cargo build --release --locked -p nixl-store-manager -p nixl-store-nsclient
python -m maturin build --release --locked --compatibility linux --out target/python-wheels
python -m pip install target/python-wheels/nixl_store-*.whl
export PATH="$PWD/target/release:$PATH"
```

If the wheel stores native libraries outside a standard loader path, also export
the wheel library directory on `LD_LIBRARY_PATH` and its `plugins` directory on
`NIXL_PLUGIN_DIR`. Prefer a packaged NIXL Store release when one is available in
your environment; the commands above are a reproducible source-build fallback.

## Quickstart: Auto-Start Single Node

After `nixl_store`, `nixl-store-manager`, and `nixl-store-nsclient` are
available in your environment, the shortest TQ-owned path is `auto_init: true`.
This starts one Store Manager and one local NSClient sidecar for the Ray node,
runs normal KV operations, then stops the TQ-owned Store services on `tq.close()`.

```python
from omegaconf import OmegaConf
import ray
import torch
from tensordict import TensorDict

import transfer_queue as tq

ray.init()

conf = OmegaConf.create(
    {
        "backend": {
            "storage_backend": "NixlStore",
            "NixlStore": {
                "auto_init": True,
                "namespace": "transfer-queue-demo",
                "manager_address": "127.0.0.1:0",
                "manager_binary": "nixl-store-manager",
                "nsclient_binary": "nixl-store-nsclient",
                "socket_dir": "/tmp/nixl-store",
            },
        },
    }
)

tq.init(conf)
fields = TensorDict({"x": torch.arange(8).reshape(1, 8)}, batch_size=1)
tq.kv_batch_put(keys=["sample"], partition_id="demo", fields=fields)
print(tq.kv_batch_get(keys=["sample"], partition_id="demo")["x"])
tq.close()
ray.shutdown()
```

Validate the same real-service path with the opt-in e2e test:

```bash
TQ_NIXL_STORE_E2E=1 python -m pytest -q tests/e2e/test_nixl_store_real_e2e.py
```

## Configuration

```python
import transfer_queue as tq
from omegaconf import OmegaConf

conf = OmegaConf.create(
    {
        "backend": {
            "storage_backend": "NixlStore",
            "NixlStore": {
                "auto_init": False,
                "socket_path": "/run/nixl-store/nsclient.sock",
                "namespace": "transfer-queue",
                "max_batch_items": 4096,
            },
        },
    }
)

tq.init(conf)
```

Configuration fields:

| Field | Description |
|---|---|
| `auto_init` | `false` uses externally managed Store services. `true` lets the TQ owner start and stop the Manager and NSClient sidecars. |
| `socket_path` | Local Unix socket of the node-local `nixl-store-nsclient`. |
| `namespace` | Stable NIXL Store namespace for this TQ deployment. Use the same namespace for producer and consumer processes that share data. |
| `max_batch_items` | Client chunk size for Store batch calls. The current binding limit is 4096. |

For multi-node runs, each node normally uses the same socket path, but the
socket is local to that node's NSClient sidecar.

Auto-start fields:

| Field | Description |
|---|---|
| `manager_address` | Manager address. Port `0` asks TQ to choose a free port. For multi-node auto-start, loopback hosts are converted to a driver-node address for NSClient connections. |
| `manager_binary` | `nixl-store-manager` executable or absolute path. |
| `nsclient_binary` | `nixl-store-nsclient` executable or absolute path. |
| `socket_dir` | Node-local directory for generated NSClient sockets and logs. |
| `startup_timeout_seconds` | Readiness timeout for Manager TCP and NSClient Unix socket startup. |
| `shutdown_timeout_seconds` | Grace period before owned Store processes are killed. |
| `nsclient_config` | Optional path to a `nixl-store-nsclient` TOML file for staging/global-segment sizing. `null` uses NSClient built-in defaults. |
| `process_env` | Environment variable overrides for TQ-owned Store Manager and NSClient processes, for example `LD_LIBRARY_PATH` or `NIXL_PLUGIN_DIR` while native runtime discovery is not packaged. Empty by default, so owned processes inherit the parent environment unchanged. |
| `runtime_socket_paths` | Filled by auto-start; users normally leave it empty. |

## Large Payload Sizing

Large values are controlled by the NSClient TOML, not by TransferQueue. The
NSClient built-in defaults are small smoke-test values: staging has 16 slots of
64 KiB and the default Global segment is 2 MiB. Configure larger staging slots
and Global segment capacity before using GiB-scale TQ values.

The Store code admits object sizes as 64-bit lengths, but each PUT and GET still
must fit in an available staging slot, and region lengths must fit the process
address space and be allocatable on the node. If a value is larger than the
configured staging maximum, NSClient reports it as too large. If memory cannot
be allocated or registered, NSClient startup fails and TQ reports the sidecar log
path in the startup error.

For large-payload runs, record the exact `nsclient_config` used next to any
performance or validation numbers. A larger config is an expected operating
mode, not a TransferQueue code change.

## Supported APIs

The MVP supports the public KV APIs that map to create/read/delete operations:

- `kv_batch_put` and `async_kv_batch_put`
- `kv_batch_get` and `async_kv_batch_get`
- `kv_clear` and `async_kv_clear`
- repeated `kv_batch_get` after a put
- selected-field clear, for example clearing only one field of a key
- full key clear

The backend uses the same KV key layout as other `KVStorageManager` backends:

```text
{global_index}@{field_name}
```

## Unsupported MVP Items

These features are not implemented for the NixlStore MVP:

- direct GPU/application-buffer transfers,
- storage-side parser support,
- `kv_update`,
- `kv_link`,
- checkpointing Store contents,
- selective dump/load,
- recovery after Store peer failure.

The backend also rejects `data_parser`, matching current KV-backend limitations
for storage-side parsing.

Automatic Store service startup is supported when `auto_init` is `true`; the
required NIXL Store Python module, native runtime, and binaries must still be
installed first.

## Data Path

The MVP data path is:

1. Contiguous non-empty CPU tensors are written as raw tensor buffers with
   NIXL Store `PutBufferItem` and `upsert_buffer_many`.
2. Other values are packed into byte buffers with `transfer_queue.utils.serial_utils`
   and written with `PutBufferItem` and `upsert_buffer_many`.
3. Raw tensor reads allocate the destination tensor and call `get_many` with a
   buffer reference to the tensor storage.
4. Packed reads require `encoded_size` metadata, then call `get_many` to fetch
   payload bytes for decoding.
5. Clear calls `clear_many(...).wait()`.

The raw tensor path is CPU-only until the Store binding exposes supported GPU
memory types and device identity for application-owned buffers.

Put uses upsert semantics because TQ public KV put can add or replace fields.
The client does not blindly replay failed or uncertain writes.

## Clear Semantics

`kv_clear` maps each requested key and field to Store keys and waits for
`clear_many(...).wait()` to complete. Missing Store keys are tolerated during
clear so repeated cleanup and partial-field cleanup remain deterministic. Other
Store statuses are treated as errors and include the affected key/status.

## Validation Topology

The MVP has been validated in these real Store topologies:

- one-node real Store validation: one Manager, one NSClient, one TQ process,
- one-node multiprocess validation: one Manager, one NSClient, multiple TQ
  processes attaching to the same externally started Store services,
- two-node producer/consumer validation: one Manager, one NSClient per node,
  producer on one node, consumer on another node, both using their node-local
  NSClient sockets.

The two-node validation proves the intended deployment shape: payloads move
through NIXL Store's data plane, while the TQ controller tracks metadata and
readiness.

## Optional Real-Service Tests

The committed NixlStore unit tests use fakes so normal CI does not require NIXL
Store. A real-service e2e hook is available and skipped by default:

```bash
TQ_NIXL_STORE_E2E=1 \
python -m pytest -q tests/e2e/test_nixl_store_real_e2e.py
```

That test expects `nixl_store` to be importable and both `nixl-store-manager`
and `nixl-store-nsclient` to be on `PATH`. Override binary paths with
`TQ_NIXL_STORE_MANAGER_BINARY` and `TQ_NIXL_STORE_NSCLIENT_BINARY`; override the
NSClient TOML path with `TQ_NIXL_STORE_NSCLIENT_CONFIG`.

The default opt-in run covers:

- one-node `auto_init: true`,
- one-node attached Ray worker process,
- one-node externally managed Manager/NSClient with `auto_init: false`.

Two-node validation is a separate manual check because generic GitHub-hosted CI
does not provide a multi-node NIXL/Ray environment:

```bash
TQ_NIXL_STORE_E2E=1 \
TQ_NIXL_STORE_TWO_NODE_E2E=1 \
RAY_ADDRESS=<existing-ray-head-address> \
python -m pytest -q tests/e2e/test_nixl_store_real_e2e.py -k two_node
```

The two-node check requires an existing Ray cluster with at least two alive
nodes and the same NIXL Store package/binaries available on every node.

Manual release validation should include:

1. local compile and lint checks,
2. the default opt-in real-service e2e test on one NIXL-capable node,
3. the two-node opt-in e2e test on a Ray cluster with producer and consumer on
   different nodes,
4. confirmation that the install instructions for the target environment provide
   the Python module, binaries, native libraries, and plugins.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `nixl_store` import error when selecting `NixlStore` | The Python binding is not installed in the selected environment. Install or build `nixl_store`, or use another backend. |
| Connector construction fails with a socket error | The local NSClient is not running, the socket path is wrong, or the process cannot access the socket. |
| `NixlUnavailable` from NSClient startup | The NIXL native runtime libraries were not available to NSClient. Ensure `LD_LIBRARY_PATH` points at the NIXL wheel libraries and `NIXL_PLUGIN_DIR` points at their `plugins` directory. |
| Cross-node Ray actor imports differ | Ensure the local TransferQueue checkout is first on `PYTHONPATH` for Ray head, Ray workers, and TQ driver processes. |
| Missing-key get error | The key was not written, was cleared, or the producer and consumer used different namespaces or partitions. |

## Baseline Performance

Phase 9 records implementation-only baseline probes outside the repository. The
MVP numbers are environment-specific and should be read as a pre-optimization
baseline from before the raw CPU tensor fast path and packed BufferRef PUT path.

One Phase 9 baseline run used one Slurm node, one external Store Manager, one
external NSClient, and one TQ benchmark driver. It ran in the investigation
Ray image on `eos0418.eos.clusters.nvidia.com` with TransferQueue commit
`6213e61`. The implementation-only harness built release `nixl-store-manager`
and `nixl-store-nsclient` binaries and a `nixl_store` Python wheel from the
local NIXL Store checkout. It used three repeats per case:

| Case | Payload bytes | Put median ms | Put MiB/s | Get median ms | Get MiB/s | Clear median ms |
|---|---:|---:|---:|---:|---:|---:|
| small tensor | 4,096 | 5.478 | 0.71 | 4.492 | 0.87 | 1.291 |
| large tensor | 4,194,304 | 6.617 | 604.49 | 6.060 | 660.11 | 1.283 |
| structured object | 1,224 | 4.653 | 0.25 | 3.939 | 0.30 | 1.121 |
| multi-field batch | 131,288 | 5.516 | 22.70 | 5.380 | 23.27 | 1.419 |

The raw implementation-only probe output is intentionally kept outside the TQ
repo. Do not treat these numbers as product-level performance claims; they are
a reproducibility baseline for this MVP data path and environment.
