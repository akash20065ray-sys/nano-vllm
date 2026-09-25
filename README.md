# nano-vllm ⚡

> **A Minimal, Educational Systems Implementation of Paged KV-Cache Memory Management, Copy-On-Write Prefix Sharing, and Continuous Batching for LLM Serving.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests: 12 Passed](https://img.shields.io/badge/Tests-12%20Passed-brightgreen.svg)]()

---

## Overview

Serving Large Language Models (LLMs) autoregressively is fundamentally **memory-bandwidth and capacity bound**, not compute bound. During generation, models save past Key and Value projection vectors in GPU VRAM (the **KV-Cache**) to avoid redundant $O(N^2)$ recomputation.

However, standard serving implementations allocate memory **contiguously** for worst-case context lengths (e.g. 2048 tokens). This causes up to **80% internal and external memory fragmentation**, stranding GPU Tensor Cores and triggering premature `CUDA: Out-of-Memory` crashes.

**`nano-vllm`** bridges classical **Operating Systems Virtual Memory Paging** with **Deep Learning Inference Runtimes**:
* **Fixed Physical Block Allocation:** Partitions VRAM into uniform 16-token physical blocks with $O(1)$ allocation/reclamation.
* **Virtual Page Tables:** Maps logical token sequence indices to arbitrary non-contiguous physical blocks in VRAM.
* **Copy-On-Write (CoW) Prefix Sharing:** Shares immutable system prompt blocks across concurrent sessions with atomic reference counts (`ref_count`), duplicating blocks only upon mutation.
* **Iteration-Level Continuous Batching:** Retires completed sequences and admits queued requests on every individual token step.
* **Preemption Controller:** Gracefully evicts cold prefixes via LRU and pauses lower-priority sequences under 100% VRAM saturation instead of crashing with CUDA OOM.
* **Real-Time Observability Matrix:** Interactive 2D physical block visualizer tracking memory states and per-request latency diagnostics.

---

## Architecture

```
                  [ Client / Web Browser / Benchmark Harness ]
                                      │
                                      ▼
                           [ FastAPI SSE Server ]
                                      │
                                      ▼
                        [ Continuous Batch Scheduler ]
                        - Dynamic Request Admission
                        - Iteration-Level Priority Queue
                        - Preemption Controller
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        [ Prefix Cache (LRU) ]                 [ Page Table Manager ]
        - Hash-based prefix tree               - Logical -> Physical mapping
        - Atomic ref_count management          - Copy-on-Write (CoW) triggers
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                           [ Physical Block Manager ]
                           - BlockAllocator
                           - FreeBlockPool (O(1) pop/push)
                                      │
                                      ▼
                         [ Paged KV-Cache Storage ]
                         - GPU VRAM / CPU RAM Tensor Pool
                         - Vectorized Paged Gather Kernel
```

---

## Project Structure

```
nano-vllm/
├── core/
│   ├── block_manager.py     # PhysicalBlock & O(1) lock-free BlockAllocator
│   ├── page_table.py        # Logical-to-physical address translation & CoW logic
│   ├── prefix_cache.py      # Chained SHA-256 rolling hash tree & LRU eviction
│   ├── kv_cache.py          # Unified GPU VRAM / CPU RAM 5D tensor storage pool
│   └── request.py           # Sequence state machine & latency instrumentation
├── kernels/
│   └── paged_gather.py      # Vectorized non-contiguous block indexing benchmark
├── dashboard/               # Systems Observability UI (VRAM Matrix & Diagnostics)
│   ├── index.html           # 2D physical block grid & prompt terminal
│   ├── styles.css           # NVIDIA-green cyberpunk dark mode theme
│   └── app.js               # Real-time state controller & failure injection
├── server/
│   └── app.py               # FastAPI streaming backend & telemetry API
├── prd/                     # Technical specifications, math formulations & architecture
│   ├── brain.md             # The memory-wall problem & KV-cache formulation
│   ├── system_architecture.md # Data flows & Alice/Bob state transition table
│   ├── tech_stack.md        # Hardware compatibility (RTX 3050 to H100)
│   ├── benchmark_plan.md    # The 4 baselines & 5-stage ablation matrix
│   └── roadmap.md           # 3-week implementation milestones
├── tests/                   # Automated pytest verification suite
│   ├── test_allocator.py    # O(1) allocation, out-of-blocks, double-free tests
│   ├── test_page_table.py   # Address translation & Copy-On-Write tests
│   ├── test_prefix_cache.py # Prefix matching & LRU eviction order tests
│   └── test_kv_cache.py     # Tensor read/write & paged gather tests
├── run_dashboard.py         # One-click launcher (starts server + opens browser)
└── requirements.txt         # Dependencies
```

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/akash20065ray-sys/nano-vllm.git
cd nano-vllm
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m pytest tests/ -v
```

### 3. Launch Observability Dashboard
```bash
python run_dashboard.py
```
Open **`http://localhost:8000`** in your browser to interact with the real-time VRAM block matrix and test prefix caching and preemption live!

---

## Benchmark Results (Local RTX GPU / CPU)

```
============================================================
  PAGED GATHER KERNEL BENCHMARK RESULTS
============================================================
  Sequence Length               : 512 Tokens
  Physical Blocks Spanned       : 32 Blocks
  Effective Memory Bandwidth    : 11.08 GB/s
  Paged Gather Latency          : 176.2 us
============================================================
```

---

## License
MIT License. Created by Akash (VIT Pune).
