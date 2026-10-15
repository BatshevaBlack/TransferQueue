# Running and Submitting the Backend Comparison on EOS

**Temporary file: remove before pushing upstream.** It records how the benchmark was submitted during
development. The submission scripts and overlays live outside this repo, in the investigation notes
(`notes_experiences_path/investigation/scripts/`).

## How a job runs

`sbatch` starts a Ray cluster through the data-plane-bench `ray.sub` inside a container image, then runs
`backend_comparison.py` on the head node (`ray_address: auto`). Two nodes give one writer node and one
reader node (the default placement splits the alive nodes in half).

The container's pinned TQ is too old (it has no namespaced actors), so the jobs put this checkout on
`PYTHONPATH`; it is visible in the container through the investigation mount.

## Submit scripts

| Script | Image | Extra mounts |
|--------|-------|--------------|
| `24_backend_comparison.sh` | Stock DPB image (Ray 2.54.0) | None |
| `25_backend_comparison_fix3ideal.sh` | Ray `ca01ff1` UCX image | Ray RDT patches (`rdt_manager.py`, `util.py`, `nixl_tensor_transport.py`) and a `RayStorageClient` overlay |

Both take optional environment variables: `CONFIG` (benchmark YAML), `TAG` (run directory name), `WALL`
(time limit, default `01:30:00`), `DRY=1` (print the `sbatch` command only). Script 25 also takes
`TQ_CLIENT_OVERLAY` (the file mounted over `ray_storage_client.py`).

```bash
cd notes_experiences_path/investigation/scripts

# TCP / Mooncake cases on the stock image
bash 24_backend_comparison.sh

# Same, with another config and run directory name
CONFIG=$PWD/24_backend_comparison_large.yaml TAG=large bash 24_backend_comparison.sh

# Ray NIXL cases (with and without the pool), 512 MiB and 1 GiB
WALL=01:00:00 CONFIG=$PWD/27_backend_comparison_2sizes.yaml \
  TQ_CLIENT_OVERLAY=$PWD/26_ray_storage_client_nixl_pool.py TAG=2sizes \
  bash 25_backend_comparison_fix3ideal.sh
```

Results go to `runs/backend-comparison[-<TAG>]-<timestamp>/backend_comparison.json`, next to the `sbatch`
log and the Ray driver log (`<jobid>-logs/ray-driver.log`).

## Configs used

| Config | Cases | Sizes |
|--------|-------|-------|
| `24_backend_comparison.yaml` | `simple_tcp`, `mooncake_tcp`, `mooncake_rdma` | 128 KiB to 1 GiB |
| `24_backend_comparison_large.yaml` | same | 512 MiB, 1 GiB |
| `26_backend_comparison_nixl_pool.yaml` | `ray_nixl` with pool | 1 GiB |
| `27_backend_comparison_2sizes.yaml` | the three above plus `ray_nixl_pool` and `ray_nixl_nopool` | 512 MiB, 1 GiB |

Each is a copy of `backend_comparison.yaml` with some cases commented out or added.

## Ray NIXL cases

- The Ray image is `ca01ff1`; the three RDT files come from `data-plane-bench/patches/ray/python/ray/experimental/rdt`.
- The `RayStorageClient` overlay (`26_ray_storage_client_nixl_pool.py`) is this checkout's client plus
  Ray NIXL memory pool registration. The pool is enabled by `backend.RayStore.rdt_pool_bytes` and
  `rdt_pool_device` in the case's `tq_config`; leave them out for the no-pool case.
- The benchmark creates its client actors with `enable_tensor_transport=True` for RayStore.
- With NIXL, PUT only publishes the tensor; the data moves on GET. Compare PUT + GET time (`put_get_gib_s`), not PUT alone.
- The benchmark sets `TQ_RDT_RECV_DEVICE` (the reader device) in every client. Only a patched receive path reads it:
  `30_nixl_tensor_transport_recv_device.py` is the mounted `nixl_tensor_transport.py` with that change; pass it to
  `25_backend_comparison_fix3ideal.sh` as `NIXL_TRANSPORT_OVERLAY`. Without it a CPU reader that sees CUDA gets
  GPU results and `get_device_warning` is set.
- CPU writer with a GPU reader also needs `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` in the submit environment.

## Notes

- `tq_simple` in the data-plane-bench notes is the `simple_tcp` case here (SimpleStorage over ZMQ/TCP).
