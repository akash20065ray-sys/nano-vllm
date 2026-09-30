# MicroServe-LLM: 3-Week Implementation Roadmap & Milestones

## Overview
This roadmap enforces disciplined software engineering: **building the memory engine, scheduler, and benchmark suite first** before adding presentation layers.

---

## Week 1: Core Memory & Allocation Foundation

| Days | Focus Modules | Key Deliverables & Code | Testing & Validation Milestone |
| :--- | :--- | :--- | :--- |
| **Day 1-2** | `core/block_manager.py`<br/>`core/page_table.py` | • `PhysicalBlock` struct with atomic `ref_count`.<br/>• `BlockAllocator` with $O(1)$ free list pool.<br/>• `PageTable` with logical-to-physical block array. | **Unit Tests (`tests/test_allocator.py`):**<br/>Assert $O(1)$ allocation/free, verify free-pool integrity, test out-of-blocks error, assert double-free protection. |
| **Day 3-4** | `core/prefix_cache.py` | • Radix/hash prefix tree.<br/>• Immutable prefix registration.<br/>• Least-Recently-Used (LRU) cold cache eviction. | **Unit Tests (`tests/test_prefix_cache.py`):**<br/>Verify hash lookup hits, test LRU eviction order under memory limits, confirm ref-count increments on hit. |
| **Day 5-7** | `core/page_table.py` (CoW)<br/>`kernels/paged_gather.py` | • Copy-on-Write (CoW) page duplication logic.<br/>• Vectorized PyTorch paged gather (`torch.index_select`).<br/>• Custom CUDA/C++ gather kernel benchmarking. | **Integration Tests (`tests/test_cow.py`):**<br/>Confirm CoW triggers when mutating shared blocks; benchmark paged gather kernel latency and bandwidth. |

---

## Week 2: Serving Engine, Scheduler & Dual-Mode Execution

| Days | Focus Modules | Key Deliverables & Code | Testing & Validation Milestone |
| :--- | :--- | :--- | :--- |
| **Day 8-10** | `core/request.py`<br/>`core/scheduler.py` | • Sequence state machine (WAITING, RUNNING, PREEMPTED, COMPLETED).<br/>• Iteration-level continuous batching loop.<br/>• Dynamic admission control and priority queue. | **Unit Tests (`tests/test_scheduler.py`):**<br/>Simulate arrival of 10 mixed-length requests; verify continuous batching retires sequences immediately upon EOS. |
| **Day 11-12**| `core/engine.py` | • **Real LLM Mode:** Loads `Qwen2.5-0.5B` in FP16 (~1GB VRAM footprint).<br/>• **Simulation Mode:** Synthetic tensor generator for 128+ concurrent streams. | **Execution Tests:**<br/>Run 1 real user prompt; verify correct token streaming; run 64 simulated streams in under 5 seconds. |
| **Day 13-14**| `core/scheduler.py` (Preemption)<br/>`core/profiler.py` | • Preemption controller: gracefully pauses and re-queues sequences under 100% VRAM saturation.<br/>• Micro-profiler: TTFT, ITL percentiles, VRAM fragmentation %. | **Stress Tests:**<br/>Inject memory exhaustion; assert system preempts sequences without crashing with CUDA OOM; verify recovery. |

---

## Week 3: Empirical Benchmarking, Observability & Placement Polish

| Days | Focus Modules | Key Deliverables & Code | Testing & Validation Milestone |
| :--- | :--- | :--- | :--- |
| **Day 15-17**| `benchmark/runner.py`<br/>`benchmark/plot_results.py` | • Automated CLI benchmark harness.<br/>• Executes the 4 baselines & 5 ablation configurations.<br/>• Auto-generates publication-grade plots in `benchmark/results/`. | **Benchmark Command:**<br/>Execute `python benchmark/runner.py`; verify auto-generation of `latency.png`, `throughput.png`, `fragmentation.png`. |
| **Day 18-19**| `server/app.py`<br/>`dashboard/` | • FastAPI backend with Server-Sent Events (SSE).<br/>• Real-time 2D VRAM physical block matrix visualizer.<br/>• Sequence Diagnostics panel. | **Browser Verification:**<br/>Open `http://localhost:8000`; test live streaming text, watch memory blocks light up, test memory saturation simulation. |
| **Day 20-21**| `README.md`<br/>Placement Prep | • Professional README with "What We Learned" empirical findings.<br/>• Clean GitHub commit history following iterative feature milestones.<br/>• Rehearse 60-second elevator pitch and technical Q&A. | **Final Milestone:**<br/>Repository 100% complete, tested, documented, and ready to link on resume. |
