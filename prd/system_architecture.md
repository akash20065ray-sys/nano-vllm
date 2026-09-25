# MicroServe-LLM: System Architecture & Data Flow

## 1. High-Level Systems Architecture

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
                                      │
                                      ▼
                          [ Dual Execution Engine ]
                          - Mode A: Real LLM (Qwen2.5-0.5B)
                          - Mode B: High-Throughput Simulator
                                      │
                                      ▼
                          [ Telemetry Profiler ]
                          - TTFT, ITL, TPS, VRAM Fragmentation %
```

---

## 2. Core Modules & Responsibilities

| Module Path | Primary Responsibility | Key Data Structures & Algorithms |
| :--- | :--- | :--- |
| `core/request.py` | Tracks individual sequence lifecycle, tokens, and latency timestamps. | `Sequence`, `SequenceStatus` (WAITING, RUNNING, PREEMPTED, COMPLETED). |
| `core/block_manager.py` | Allocates and reclaims fixed-size physical blocks in $O(1)$ time. | `PhysicalBlock`, `BlockAllocator`, `free_blocks` deque, atomic `ref_count`. |
| `core/page_table.py` | Maps logical token slots to physical blocks; isolates memory. | `PageTable`, per-request mapping array, Copy-On-Write (CoW) handler. |
| `core/prefix_cache.py` | Reuses immutable prefix blocks across requests. | Hash tree/radix index, LRU eviction queue for unreferenced prefixes. |
| `core/scheduler.py` | Coordinates continuous batching, admission control, and preemption. | Waiting queue, active running batch, iteration-level execution loop. |
| `core/engine.py` | Executes forward passes across active sequences. | Dual-mode executor (PyTorch real weights + synthetic high-concurrency mode). |
| `core/profiler.py` | Measures latency distributions and memory utilization. | TTFT counter, ITL percentiles (p50, p95, p99), fragmentation calculator. |
| `kernels/paged_gather.py`| Gathers non-contiguous physical block tensors into dense attention inputs. | Vectorized PyTorch indexing vs custom CUDA gather kernel. |

---

## 3. Concrete State Machine & Lifecycle

### 3.1 Sequence States
```
 [Request Submitted] ──> WAITING ──(Blocks Available)──> RUNNING ──(Finished / EOS)──> COMPLETED
                            ▲                              │
                            │                        (Memory Full)
                            └─────── PREEMPTED <───────────┘
```

### 3.2 Copy-On-Write (CoW) Invariant
* Only completed, full prompt prefix blocks are marked **immutable** and registered in the `PrefixCache`.
* Sequences sharing an immutable prefix map to the identical physical block IDs with `ref_count > 1`.
* The active generation tail block is **always private** to the sequence.
* If a sequence attempts to modify an immutable shared block, the runtime allocates a fresh private block, duplicates past tokens, updates the page table, and decrements the shared block's reference counter.

---

## 4. Concrete Walkthrough Example (Alice & Bob)

* **Setup:** 16 tokens/block. System prompt = 32 tokens (takes 2 blocks: `Phys 14` & `Phys 03`).
* **$T_0$ (Alice arrives):** Alice submits prompt (32t prefix + 12t question = 44t). Allocates `Phys 14`, `Phys 03`, and private `Phys 29`.
* **$T_1$ (Bob arrives):** Bob submits prompt with **identical 32t system prompt** + 8t question = 40t.
  * **Prefix Cache Hit!** Bob's page table directly maps `Block 0 -> Phys 14` and `Block 1 -> Phys 03`.
  * `Phys 14` and `Phys 03` increment `ref_count = 2`.
  * Bob only allocates 1 private block (`Phys 07`) for his question. Zero extra VRAM for prefix; TTFT drops to 0ms!
* **$T_2$ (Incremental Generation):** Both Alice and Bob generate new tokens. Blocks allocate on demand as boundaries cross.
* **$T_3$ (Bob completes):** Bob's private block `Phys 07` returns to the free pool immediately. `Phys 14` & `Phys 03` decrement `ref_count` to 1.
* **$T_4$ (Alice completes):** Alice's private blocks are freed. The shared prefix blocks remain warm in the LRU cache ready for User C.
