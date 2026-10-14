# TransferQueue Backend Comparison

This script compares TransferQueue backend/transport cases across increasing payload sizes. It measures
individual PUT/GET latency and separate PUT/GET phase throughput, and saves the results as JSON.

## Prerequisites

1. A running Ray cluster. Node allocation and `ray start` (Slurm or otherwise) happen outside this script,
   which only connects to it (`benchmark.ray_address`).

2. The same TransferQueue version and the optional backends you want to compare (Mooncake, NIXL)
   installed on every node. Cases whose dependencies, hardware, or memory are missing are recorded as
   `skipped` with a reason; there is no fallback to another mode.

3. For Mooncake, the master starts on the first writer node and must be reachable from every client node.

## Usage

```bash
python backend_comparison.py --config=backend_comparison.yaml
```

## Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--config` | Path to the benchmark YAML file | `backend_comparison.yaml` | No |
| `--output` | Path to the output JSON file | `results/backend_comparison_<timestamp>.json` | No |

## Configuration

All settings are in the `benchmark:` section of `backend_comparison.yaml`.

| Key | Description |
|-----|-------------|
| `ray_address` | Ray cluster address (`auto`) |
| `clients` | Number of writers; the same number of readers is created |
| `reader_mapping` | `paired` (reader *i* reads writer *i*) or `shuffled` (seeded one-to-one permutation) |
| `shuffle_seed` | Makes the shuffled assignment repeatable; ignored for `paired` |
| `sizes` | `start_kib` and `count`; sizes double each step (128 KiB through 1 GiB by default) |
| `ops_per_iteration`, `max_iteration_gib` | Upper bounds on operations per writer (see Test Flow) |
| `warmup_iterations`, `measured_iterations` | Iterations per size; warmup results are discarded |
| `placement` | `writer_nodes`, `reader_nodes`, `max_clients_per_node` |
| `backends` | List of cases: `label`, `device` (`cpu` or `cuda`) and a normal TQ `tq_config` |

Each `tq_config` is merged over the library defaults. The effective config is recorded in the JSON and
checked against the running controller.

### Cases

| Label | Backend | Transport | Device |
|-------|---------|-----------|--------|
| `simple_tcp` | SimpleStorage | ZMQ/TCP | CPU |
| `mooncake_tcp` | MooncakeStore | `protocol: tcp` | CPU |
| `mooncake_rdma` | MooncakeStore | `protocol: rdma` | CPU |
| `ray_nixl` | RayStore | NIXL tensor transport, no pool | CPU |

TODO: add `mooncake_gdr` (CUDA) once it has run successfully; its config is kept commented out in the YAML.
Yuanrong is not implemented yet.

**Temporary note, remove or update once the pool / no-pool setup is settled:** `ray_nixl` needs a Ray with
NIXL tensor transport; the results so far used a patched Ray RDT image. A NIXL memory pool
(`backend.RayStore.rdt_pool_bytes` and `rdt_pool_device`) is read only by an overlay `RayStorageClient`,
not by this repo's client, so `ray_nixl` here is the no-pool variant. On the patched Ray, the no-pool run
only worked with an overlay client that keeps the writer's ObjectRefs alive. Both overlays are described in
`RUN_AND_SUBMIT.md`.

### Placement

Writers and readers are placed on `writer_nodes` / `reader_nodes` round-robin, at most
`max_clients_per_node` each, with hard placement. If both lists are null, the alive Ray nodes are split in
half: the first half hosts writers, the second half readers. Writer 0 starts TransferQueue and the other
clients attach. The driver creates no TQ client. Default SimpleStorage is restricted to the first writer
node. The JSON records the actual placement of clients and services.

## Test Flow

Per size, each writer performs `ops = max(1, min(ops_per_iteration, floor(max_iteration_gib / size)))`
PUTs, each with its own key (`writer_<w>_op_<j>`). Each reader performs the same number of GETs on its
assigned writer's keys. Each iteration:

1. **Prepare**: each writer builds its payloads (one contiguous `uint8` field of exactly the requested
   bytes, derived from the key and iteration marker).
2. **PUT** (`kv_put`): all writers put their keys sequentially.
3. **GET** (`kv_batch_get`): after all writers finish, all readers get their keys sequentially.
4. **DELETE** (`kv_clear`): writers clear their keys.

The phases do not overlap. Each size runs `warmup_iterations` and then `measured_iterations`. Keys are
reused by every iteration; cases and sizes are isolated by partition. The reader assignment is generated
once and reused for all sizes and cases.

## Validation

Each reader validates only the first GET of every iteration (warmup included), after its timer stops:
field, dtype, logical sample shape, and the first, middle and last byte. There is no full checksum. A
validation failure marks the case failed.

## Timing

- **Latency**: each call is timed inside its actor with a monotonic clock, excluding preparation,
  validation and histogram updates; CUDA is synchronized first. Reported in ms: count (`n`, the timed
  calls at that size over all clients and measured iterations), min, mean, max, p50 and p90. Percentiles
  are estimated from log-spaced buckets about 5% apart (1 us to 100 s, plus underflow and overflow),
  which are also saved. With few samples (for example 20 at 1 GiB) they are coarse.
- **Throughput** (GiB/s): bytes of completed calls divided by the phase time on the driver's clock, from
  before dispatch until the last actor finishes. The GET phase includes the first-GET validation. Per size
  it is the total measured bytes over the total measured phase time.

## Capacity

Live data is about `clients × ops(size) × size`. The preflight requires
`1.5 × max(max_iteration_bytes, largest_payload) × clients` bytes of backend capacity, and checks writer
input buffers, reader outputs (one result plus one copy) and per-client Mooncake segments and local
buffers against node memory (GPU memory for `cuda`). SimpleStorage `total_storage_size` is a sample count,
calculated as `1.25 × clients × largest ops`.

## Failures and Cleanup

On a failure, the case, size, phase, actor and reason are recorded, the partial result is saved, and larger
sizes are skipped. Clients are then closed (attached clients first, owner last), benchmark actors are
killed, and the old TQ actors are verified gone before the next case. `cleanup_verified` records whether
`kv_list` was empty after each size. A case is refused if a TransferQueue controller already exists.

## Output

The JSON is rewritten atomically after every size and failure. It contains:

- Settings, effective TQ configs, software versions, requested and actual placement
- Mapping policy, shuffle seed and reader assignment
- Per size: payload bytes, operation counts, transferred bytes, per-iteration phase times, throughput,
  latency summaries and raw histogram buckets
- The validation policy, errors, skips and cleanup status

## Examples

### Default run
```bash
python backend_comparison.py
```

### Custom config and output
```bash
python backend_comparison.py --config=my_comparison.yaml --output=results/my_run.json
```
