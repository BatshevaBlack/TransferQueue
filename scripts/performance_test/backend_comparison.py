#!/usr/bin/env python3
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

"""Compare TransferQueue backend/transport cases across payload sizes.

Configuration, usage and definitions: README_BACKEND_COMPARISON.md.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import random
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf
from ray.util.state import list_actors

import transfer_queue as tq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GIB = 1024**3
NAMESPACE = "transfer_queue"
CONTROLLER = "TransferQueueController"
# Named actors that outlive tq.close() and would leak into the next case.
TQ_ACTORS = (CONTROLLER, "RayObjectRefStorage")
PACKAGES = {"MooncakeStore": "mooncake", "RayStore": "nixl"}
# Settings that select the transport; the running controller must report the requested values.
TRANSPORT_KEYS = ("protocol", "use_gdr")

# Log-spaced latency buckets, ~5% apart, plus underflow (index 0) and overflow (last) buckets.
LOW_S, HIGH_S, STEP = 1e-6, 100.0, 1.05
NUM_BUCKETS = math.ceil(math.log(HIGH_S / LOW_S, STEP))


class Histogram:
    def __init__(self):
        self.counts = [0] * (NUM_BUCKETS + 2)
        self.n, self.total, self.min, self.max = 0, 0.0, math.inf, 0.0

    def add(self, seconds: float) -> None:
        if seconds < LOW_S:
            i = 0
        elif seconds >= HIGH_S:
            i = NUM_BUCKETS + 1
        else:
            i = min(1 + int(math.log(seconds / LOW_S, STEP)), NUM_BUCKETS)
        self.counts[i] += 1
        self.n += 1
        self.total += seconds
        self.min, self.max = min(self.min, seconds), max(self.max, seconds)

    def percentile(self, q: float) -> float:
        """Upper edge of the bucket holding the q-quantile, clipped to the exact min/max."""
        target, seen = math.ceil(q * self.n), 0
        for i, count in enumerate(self.counts):
            seen += count
            if seen >= target:
                return min(max(LOW_S * STEP**i, self.min), self.max)

    def summary(self) -> dict:
        return {
            "count": self.n,
            "min_ms": self.min * 1e3,
            "mean_ms": self.total / self.n * 1e3,
            "max_ms": self.max * 1e3,
            "p50_ms": self.percentile(0.5) * 1e3,
            "p90_ms": self.percentile(0.9) * 1e3,
        }


def _payload(key: str, marker: str, size: int, device: str) -> torch.Tensor:
    """Contiguous uint8 payload whose fill and first/middle/last bytes depend on key and marker."""
    first, middle, last, fill = _probe(key, marker)
    t = torch.full((size,), fill, dtype=torch.uint8, device=device)
    t[0], t[size // 2], t[-1] = first, middle, last
    return t


def _probe(key: str, marker: str) -> tuple[int, int, int, int]:
    h = zlib.crc32(f"{key}:{marker}".encode())
    return (h >> 8) % 256, (h >> 16) % 256, (h >> 24) % 256, h % 251


class Client:
    """One TQ client process; payload generation, storage calls and validation stay here."""

    def __init__(self, device: str):
        self.device = device
        self.payloads = {}
        if device == "cuda":
            torch.zeros(1, device="cuda")  # CUDA must exist before the client decides on the GDR path

    def node_ip(self) -> str:
        return ray.util.get_node_ip_address()

    def init_tq(self, conf: dict | None = None) -> None:
        tq.init(OmegaConf.create(conf) if conf else None)  # no conf: attach to the owner's controller

    def gdr_active(self) -> bool:
        return tq.get_client().storage_manager.storage_client._gdr_staging is not None

    def prepare(self, keys: list[str], size: int, marker: str) -> None:
        self.payloads = {k: _payload(k, marker, size, self.device) for k in keys}

    def _sync(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()

    def put(self, partition: str) -> list[float]:
        times = []
        for key, tensor in self.payloads.items():
            start = time.perf_counter()
            tq.kv_put(key, partition, {"data": tensor})
            self._sync()
            times.append(time.perf_counter() - start)
        return times

    def get(self, partition: str, keys: list[str], marker: str, size: int) -> list[float]:
        times = []
        for i, key in enumerate(keys):
            start = time.perf_counter()
            out = tq.kv_batch_get(key, partition, select_fields="data")
            self._sync()
            times.append(time.perf_counter() - start)
            if i == 0:  # only the first GET of an iteration is validated
                self._validate(out, key, marker, size)
            del out
        return times

    def _validate(self, out, key: str, marker: str, size: int) -> None:
        sample = out["data"][0]  # dense and nested returns both index to the single logical sample
        if sample.dtype != torch.uint8 or tuple(sample.shape) != (size,):
            raise ValueError(f"{key}: got {sample.dtype} {tuple(sample.shape)}, expected uint8 ({size},)")
        got = (sample[0].item(), sample[size // 2].item(), sample[-1].item())
        if got != _probe(key, marker)[:3]:
            raise ValueError(f"{key}: stale or corrupt data, first/middle/last bytes {got}")

    def clear(self, partition: str) -> None:
        tq.kv_clear(list(self.payloads), partition)
        self.payloads = {}

    def live_keys(self, partition: str) -> int:
        return len(tq.kv_list(partition).get(partition, {}))

    def close(self) -> None:
        self.payloads = {}
        tq.close()


@ray.remote
def node_info(device: str) -> dict:
    with open("/proc/meminfo") as f:
        mem = next(int(line.split()[1]) * 1024 for line in f if line.startswith("MemAvailable"))
    ib = "/sys/class/infiniband"

    def version(pkg):
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "mem_available": mem,
        "rdma": os.listdir(ib) if os.path.isdir(ib) else [],
        "gpu_free": torch.cuda.mem_get_info()[0] if device == "cuda" else None,
        "packages": {p: importlib.util.find_spec(p) is not None for p in PACKAGES.values()},
        "versions": {
            "transfer_queue": tq.__version__,
            **{p: version(p) for p in ("torch", "ray", "mooncake-transfer-engine", "nixl")},
        },
    }


@dataclass
class Plan:
    writer_ips: list[str]
    reader_ips: list[str]
    assignment: list[int]  # assignment[r] is the writer read by reader r
    sizes: list[int]
    ops: dict[int, int]  # payload bytes -> operations per writer
    max_iter_bytes: int
    warmup: int
    measured: int
    nodes: dict  # node ip -> Ray resources


def make_plan(b) -> Plan:
    nodes = {n["NodeManagerAddress"]: n["Resources"] for n in ray.nodes() if n["Alive"]}
    p = b.placement
    if p.writer_nodes is None and p.reader_nodes is None:
        ips = sorted(nodes)
        if len(ips) < 2 or len(ips) % 2:
            raise ValueError(f"Default placement needs an even number (>= 2) of alive nodes, found {len(ips)}.")
        p.writer_nodes, p.reader_nodes = ips[: len(ips) // 2], ips[len(ips) // 2 :]
    elif p.writer_nodes is None or p.reader_nodes is None:
        raise ValueError("Set both writer_nodes and reader_nodes, or neither.")
    for side in (p.writer_nodes, p.reader_nodes):
        if set(side) - set(nodes):
            raise ValueError(f"Nodes not alive in the Ray cluster: {sorted(set(side) - set(nodes))}")
        if len(side) * p.max_clients_per_node < b.clients:
            raise ValueError(f"{b.clients} clients exceed max_clients_per_node={p.max_clients_per_node} on {side}.")

    writer_ips = [p.writer_nodes[i % len(p.writer_nodes)] for i in range(b.clients)]
    reader_ips = [p.reader_nodes[i % len(p.reader_nodes)] for i in range(b.clients)]
    for ip, count in Counter(writer_ips + reader_ips).items():
        if nodes[ip].get("CPU", 0) < count:
            raise ValueError(f"Node {ip} has {nodes[ip].get('CPU', 0)} CPUs for {count} clients.")

    if b.reader_mapping == "paired":
        assignment = list(range(b.clients))
    elif b.reader_mapping == "shuffled":
        assignment = random.Random(b.shuffle_seed).sample(range(b.clients), b.clients)
    else:
        raise ValueError(f"reader_mapping must be paired or shuffled, got {b.reader_mapping}")

    sizes = [b.sizes.start_kib * 1024 * 2**i for i in range(b.sizes.count)]
    max_iter_bytes = int(b.max_iteration_gib * GIB)
    ops = {s: max(1, min(b.ops_per_iteration, max_iter_bytes // s)) for s in sizes}
    return Plan(
        writer_ips,
        reader_ips,
        assignment,
        sizes,
        ops,
        max_iter_bytes,
        b.warmup_iterations,
        b.measured_iterations,
        nodes,
    )


def preflight(conf, device: str, plan: Plan, result: dict) -> str | None:
    """Return the reason this case cannot run, or None. Fills result["capacity"] and the versions."""
    backend = conf.backend.storage_backend
    bconf = conf.backend.get(backend) or {}
    storage_node = (bconf.get("required_node_resource") or "")[len("node:") :]
    ips = sorted({*plan.writer_ips, *plan.reader_ips, *([storage_node] if storage_node else [])})
    refs = [
        node_info.options(num_gpus=int(device == "cuda"), resources={f"node:{ip}": 0.001}).remote(device)
        for ip in ips
    ]
    info = dict(zip(ips, ray.get(refs), strict=True))
    result["software_versions"] = info[ips[0]]["versions"]
    load = Counter(plan.writer_ips + plan.reader_ips)

    package = PACKAGES.get(backend)
    if package and not all(info[ip]["packages"][package] for ip in load):
        return f"Python package '{package}' is missing on a client node"
    if device == "cuda" and any(plan.nodes[ip].get("GPU", 0) < n for ip, n in load.items()):
        return "a client node has fewer GPUs than clients"
    needs_rdma = backend == "MooncakeStore" and bconf.protocol == "rdma"
    if needs_rdma and not all(info[ip]["rdma"] for ip in load):
        return "no RDMA device on a client node"

    # Capacity: 1.5 x the largest live payload per writer, summed over writers (bytes).
    largest = plan.sizes[-1]
    live = max(plan.max_iter_bytes, largest)
    data = 1.5 * live * len(plan.writer_ips)
    result["capacity"] = {"data_bytes_required": data}

    # Writers hold their input buffers; readers hold one result plus one temporary copy.
    buffers = [(ip, live) for ip in plan.writer_ips] + [(ip, 2 * largest) for ip in plan.reader_ips]
    need = Counter()  # host memory per node
    for ip, size in buffers:
        if device == "cuda":
            if size > info[ip]["gpu_free"]:
                return f"GPU memory on {ip}: need {size / GIB:.1f} GiB, free {info[ip]['gpu_free'] / GIB:.1f} GiB"
        else:
            need[ip] += size

    if backend == "SimpleStorage":
        total = sum(info[ip]["mem_available"] for ip in ([storage_node] if storage_node else ips))
        if data > total:
            return f"SimpleStorage nodes: need {data / GIB:.1f} GiB, available {total / GIB:.1f} GiB"
    elif backend == "MooncakeStore":
        segment, local = bconf.global_segment_size, bconf.local_buffer_size
        if local < largest:
            return f"local_buffer_size {local} is smaller than the largest payload {largest}"
        if segment * len(plan.writer_ips + plan.reader_ips) < data:
            return f"Mooncake segments hold {segment * 2 * len(plan.writer_ips) / GIB:.1f} GiB, need {data / GIB:.1f}"
        for ip, n in load.items():
            need[ip] += (segment + local) * n

    for ip, size in need.items():
        if size > info[ip]["mem_available"]:
            return f"host memory on {ip}: need {size / GIB:.1f} GiB, available {info[ip]['mem_available'] / GIB:.1f}"
    return None


class PhaseError(Exception):
    def __init__(self, phase: str, actor: str, reason: str, size: int | None = None):
        super().__init__(reason)
        self.phase, self.actor, self.reason, self.size = phase, actor, reason, size


def run_phase(phase: str, jobs: list) -> tuple[list, float]:
    """Run (actor name, dispatch function) jobs concurrently; the timer covers dispatch to completion."""
    start = time.perf_counter()
    refs = [(name, dispatch()) for name, dispatch in jobs]
    out = []
    for name, ref in refs:
        try:
            out.append(ray.get(ref))
        except Exception as e:
            raise PhaseError(phase, name, repr(e)) from e
    return out, time.perf_counter() - start


def run_iteration(writers, readers, plan: Plan, partition: str, size: int, marker: str):
    keys = [[f"writer_{w}_op_{j}" for j in range(plan.ops[size])] for w in range(len(writers))]
    prepare_jobs = [
        (f"writer_{w}", lambda a=a, k=keys[w]: a.prepare.remote(k, size, marker)) for w, a in enumerate(writers)
    ]
    put_jobs = [(f"writer_{w}", lambda a=a: a.put.remote(partition)) for w, a in enumerate(writers)]
    run_phase("prepare", prepare_jobs)
    put_times, put_s = run_phase("put", put_jobs)
    get_jobs = [
        (f"reader_{r}", lambda a=a, k=keys[plan.assignment[r]]: a.get.remote(partition, k, marker, size))
        for r, a in enumerate(readers)
    ]
    get_times, get_s = run_phase("get", get_jobs)
    run_phase("clear", [(f"writer_{w}", lambda a=a: a.clear.remote(partition)) for w, a in enumerate(writers)])
    return put_times, put_s, get_times, get_s


def run_size(label: str, writers, readers, plan: Plan, size: int) -> dict:
    partition = f"{label}_{size}"  # isolates cases and sizes while keys stay identical
    try:
        for i in range(plan.warmup):
            run_iteration(writers, readers, plan, partition, size, f"warmup_{i}")
        iterations = [
            run_iteration(writers, readers, plan, partition, size, f"measured_{i}") for i in range(plan.measured)
        ]
        live, _ = run_phase("verify_cleanup", [("writer_0", lambda: writers[0].live_keys.remote(partition))])
    except PhaseError as e:
        e.size = size
        raise

    out = {"payload_bytes": size, "ops_per_writer": plan.ops[size], "cleanup_verified": live[0] == 0}
    out["iterations"] = [
        {"put_s": put_s, "get_s": get_s, "put_calls": [len(t) for t in put_t], "get_calls": [len(t) for t in get_t]}
        for put_t, put_s, get_t, get_s in iterations
    ]
    for op, times_idx, secs_idx in (("put", 0, 1), ("get", 2, 3)):
        hist = Histogram()
        for it in iterations:
            for actor_times in it[times_idx]:
                for t in actor_times:
                    hist.add(t)
        total_bytes = hist.n * size
        total_s = sum(it[secs_idx] for it in iterations)
        out[op] = {
            "calls": hist.n,
            "bytes": total_bytes,
            "throughput_gib_s": total_bytes / total_s / GIB,
            "latency": hist.summary(),
            "latency_buckets": hist.counts,
        }
    return out


def service_placement() -> dict:
    ip_of = {n["NodeID"]: n["NodeManagerAddress"] for n in ray.nodes()}
    services = {}
    for a in list_actors(filters=[("state", "=", "ALIVE")], limit=1000):
        if a.class_name != "Client":
            services.setdefault(a.class_name, []).append(ip_of.get(a.node_id))
    return services


def start_tq(conf: dict, device: str, plan: Plan, writers, readers, result: dict) -> None:
    run_phase("init", [("writer_0", lambda: writers[0].init_tq.remote(conf))])  # the owner creates the services
    attached = writers[1:] + readers
    run_phase("attach", [(f"client_{i}", lambda a=a: a.init_tq.remote()) for i, a in enumerate(attached)])

    actual = OmegaConf.to_container(ray.get(ray.get_actor(CONTROLLER, namespace=NAMESPACE).get_config.remote()))
    backend = conf["backend"]["storage_backend"]
    wanted = {"storage_backend": backend}
    wanted.update({k: v for k, v in (conf["backend"].get(backend) or {}).items() if k in TRANSPORT_KEYS})
    got = {"storage_backend": actual["backend"]["storage_backend"]}
    got.update({k: (actual["backend"].get(backend) or {}).get(k) for k in wanted if k != "storage_backend"})
    if got != wanted:
        raise RuntimeError(f"Running config differs from the requested one: requested {wanted}, running {got}")
    if wanted.get("use_gdr"):
        checks = [(f"client_{i}", lambda a=a: a.gdr_active.remote()) for i, a in enumerate(writers + readers)]
        active, _ = run_phase("gdr_check", checks)
        if not all(active):
            raise RuntimeError("use_gdr is set but the GDR path is not active on every client")

    owner_ip = ray.get(writers[0].node_ip.remote())
    result["actual_placement"] = {
        "writer_nodes": ray.get([w.node_ip.remote() for w in writers]),
        "reader_nodes": ray.get([r.node_ip.remote() for r in readers]),
        "ray_actors": service_placement(),
        "tq_init_node": owner_ip,  # TQ-started processes (e.g. the Mooncake master) run here
    }


def teardown(writers, readers) -> dict:
    errors = []
    for group in (writers[1:] + readers, writers[:1]):  # attached clients close before the owner
        try:
            ray.get([a.close.remote() for a in group])
        except Exception as e:
            errors.append(repr(e))
    for actor in writers + readers:
        ray.kill(actor)
    for name in TQ_ACTORS:
        try:
            ray.kill(ray.get_actor(name, namespace=NAMESPACE))
        except ValueError:
            pass
    for _ in range(20):  # ray.kill is asynchronous
        if not _tq_actor_names():
            break
        time.sleep(0.5)
    else:
        errors.append(f"TQ actors still alive: {_tq_actor_names()}")
    return {"verified": not errors, "errors": errors}


def _tq_actor_names() -> list[str]:
    alive = []
    for name in TQ_ACTORS:
        try:
            ray.get_actor(name, namespace=NAMESPACE)
            alive.append(name)
        except ValueError:
            pass
    return alive


def run_case(case, plan: Plan, result: dict, save) -> None:
    device = case.device
    conf = OmegaConf.merge(
        OmegaConf.load(resources.files("transfer_queue") / "config.yaml"),
        OmegaConf.create(OmegaConf.to_container(case.tq_config, resolve=True)),
    )
    if conf.backend.storage_backend == "SimpleStorage":
        # total_storage_size counts samples: every live key of every writer plus 25% margin.
        conf.backend.SimpleStorage.total_storage_size = math.ceil(1.25 * len(plan.writer_ips) * max(plan.ops.values()))
    conf = OmegaConf.to_container(conf, resolve=True)
    result["effective_tq_config"] = conf

    reason = preflight(OmegaConf.create(conf), device, plan, result)
    if reason:
        result.update(status="skipped", reason=reason)
        return
    if _tq_actor_names():
        raise RuntimeError(f"Unrelated TQ actors exist ({_tq_actor_names()}); refusing to attach to them.")

    # RayStore publishes tensors through Ray's tensor transport, which the actor must opt into.
    if conf["backend"]["storage_backend"] == "RayStore":
        client_cls = ray.remote(enable_tensor_transport=True)(Client)
    else:
        client_cls = ray.remote(Client)

    def spawn(ip):
        opts = {"num_cpus": 1, "num_gpus": int(device == "cuda"), "resources": {f"node:{ip}": 0.001}}
        return client_cls.options(**opts).remote(device)

    writers = [spawn(ip) for ip in plan.writer_ips]
    readers = [spawn(ip) for ip in plan.reader_ips]
    result["sizes"] = []
    try:
        start_tq(conf, device, plan, writers, readers, result)
        for size in plan.sizes:
            result["sizes"].append(run_size(case.label, writers, readers, plan, size))
            save()
            if not result["sizes"][-1]["cleanup_verified"]:
                raise PhaseError("verify_cleanup", "writer_0", "keys remain after clear", size)
        result["status"] = "ok"
    except Exception as e:
        err = e if isinstance(e, PhaseError) else PhaseError("setup", "driver", repr(e))
        result["status"] = "failed"
        result["error"] = {
            "case": case.label,
            "size": err.size,
            "phase": err.phase,
            "actor": err.actor,
            "reason": err.reason,
        }
        logger.error(f"{case.label} failed: {result['error']}")
        save()
    finally:
        result["cleanup"] = teardown(writers, readers)


def write_json(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str))
    os.replace(tmp, path)


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(here / "backend_comparison.yaml"))
    parser.add_argument("--output", default=None, help="Default: results/backend_comparison_<timestamp>.json")
    args = parser.parse_args()

    b = OmegaConf.load(args.config).benchmark
    ray.init(address=b.ray_address)
    plan = make_plan(b)
    output = Path(args.output or here / "results" / f"backend_comparison_{datetime.now():%Y%m%d_%H%M%S}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    doc = {
        "settings": OmegaConf.to_container(b, resolve=True),
        "reader_assignment": {
            "policy": b.reader_mapping,
            "shuffle_seed": b.shuffle_seed if b.reader_mapping == "shuffled" else None,
            "reader_to_writer": plan.assignment,
        },
        "requested_placement": {"writer_nodes": plan.writer_ips, "reader_nodes": plan.reader_ips},
        "units": {
            "latency": "ms; count = timed calls at that size, all clients and measured iterations (no warmup)",
            "throughput_gib_s": "GiB/s; bytes of completed calls / total measured phase time",
            "iteration put_s/get_s": "seconds",
        },
        "histogram": {
            "low_s": LOW_S,
            "high_s": HIGH_S,
            "step": STEP,
            "latency_buckets": "call counts: [0] is < low_s, [i] covers [low_s*step^(i-1), low_s*step^i), "
            "the last is >= high_s; percentiles are bucket upper edges clipped to the exact min/max",
        },
        "validation_policy": "first GET of every iteration (warmup and measured) at every size: field, dtype, "
        "shape, and first/middle/last byte; no full checksum",
        "cases": [],
    }
    save = lambda: write_json(output, doc)  # noqa: E731
    for case in b.backends:
        result = {"label": case.label, "device": case.device, "status": "running"}
        doc["cases"].append(result)
        try:
            run_case(case, plan, result, save)
        except Exception as e:  # preflight or pre-start failure
            result["status"] = "failed"
            result["error"] = {"case": case.label, "size": None, "phase": "setup", "actor": "driver", "reason": repr(e)}
        save()
        logger.info(f"{case.label}: {result['status']}")
    logger.info(f"Results written to {output}")


if __name__ == "__main__":
    main()
