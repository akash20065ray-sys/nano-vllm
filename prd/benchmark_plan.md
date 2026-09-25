# MicroServe-LLM: Empirical Benchmarking & Evaluation Plan

## 1. Research Objectives
Rather than making unverified theoretical assertions, MicroServe-LLM implements an automated, multi-dimensional benchmarking harness (`python benchmark/runner.py`) designed to answer four empirical systems questions:
1. **Block Granularity:** What physical block size (4 vs 16 vs 64 tokens) balances internal fragmentation against page-table indexing overhead?
2. **Prefix-Sharing Threshold:** At what prefix sharing ratio (0% to 100%) does prefix caching yield statistically significant TTFT compression?
3. **Continuous Batching Gain:** How much does iteration-level scheduling increase request throughput compared to static batching under skewed request lengths?
4. **Preemption Resilience:** How does the scheduler handle 100% memory saturation without dropping requests or crashing with `CUDA: Out-Of-Memory`?

---

## 2. The 4 Comparative Baselines

| Baseline | Name | Configuration & Operational Characteristics |
| :--- | :--- | :--- |
| **Baseline 1** | **Contiguous Static** | Standard sequential serving pre-allocating full context window buffer ($L_{\max}$) contiguously. |
| **Baseline 2** | **Paged Allocation** | Physical 16-token block allocation without prefix sharing. |
| **Baseline 3** | **Paged + Prefix Cache** | Physical block allocation with hash-based prompt block reuse and LRU eviction. |
| **Baseline 4** | **Full MicroServe Engine** | Paged Allocation + Prefix Caching + Iteration-Level Continuous Batching + Preemption. |

---

## 3. The 5-Stage Ablation Study Matrix

To isolate the precise contribution of each architectural optimization:

| Configuration | Components Active | Research Question Isolated |
| :--- | :--- | :--- |
| **Baseline 1** | Contiguous Static Buffer | Baseline fragmentation and saturation ceiling under static memory buffers. |
| **Ablation A** | + Paged Memory Only | Isolated contribution of block paging on internal/external fragmentation. |
| **Ablation B** | + Prefix Caching | Isolated contribution of prompt sharing on TTFT and memory reduction. |
| **Ablation C** | + Continuous Batching | Isolated contribution of iteration scheduling on throughput and queue time. |
| **Full Engine** | + Preemption Controller | Isolated contribution of preemption on fault tolerance under 100% saturation. |

---

## 4. Test Dimensions & Workload Variations

1. **Concurrency Load:** 1, 2, 4, 8, 16, and 32 concurrent requests.
2. **Block Granularity:** 4, 8, 16, 32, and 64 tokens per block.
3. **Prefix Sharing Ratio:** 0%, 25%, 50%, 75%, 100% prompt token overlap.
4. **Sequence Length Distributions:**
   * *Uniform:* Requests of roughly equal length (e.g., 256 tokens).
   * *Skewed (Zipfian):* Heavy-tailed distribution (many short queries + a few very long generation requests).

---

## 5. Quantitative Output Metrics

* **TTFT (Time-To-First-Token):** Latency from arrival to the first generated token (ms).
* **ITL (Inter-Token Latency):** Generation latency between consecutive tokens (ms, reported as p50, p95, p99).
* **KV-Cache Fragmentation Rate (%):** Percentage of reserved memory holding unwritten or locked slots.
* **System Throughput:** Tokens Per Second (TPS) and Requests Per Second (RPS).
* **Preemption Frequency:** Number of graceful preemption events triggered under VRAM saturation.

---

## 6. Automated Failure Injection & Test Suite (`tests/`)

* `test_allocator.py`: Asserts $O(1)$ allocation/free, verifies free-pool integrity, prevents double-free bugs, and validates atomic reference counting.
* `test_page_table.py`: Verifies virtual-to-physical block mapping, asserts Copy-On-Write triggers when mutating shared blocks, and confirms ref-count decrements.
* `test_scheduler.py`: Tests continuous batching iteration loops, asserts preemption triggers when free blocks hit zero, and verifies request resumption when memory frees up.
