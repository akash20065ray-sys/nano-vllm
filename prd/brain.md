# MicroServe-LLM: The Brain (Core Concept & Problem Formulation)

## 1. What is the Core Concept?
**MicroServe-LLM** is an educational, research-grade systems implementation that applies classical **Operating Systems Virtual Memory Paging** to **GPU VRAM** for Large Language Model (LLM) serving.

Instead of pre-allocating contiguous memory buffers for worst-case sequence lengths, MicroServe:
1. Partitions GPU memory into small, uniform physical blocks (e.g., 16 tokens/block).
2. Maintains per-sequence **Page Tables** mapping logical token indices to arbitrary physical blocks in memory.
3. Allocates blocks dynamically on-demand as tokens generate.
4. Shares immutable prompt prefix blocks across concurrent sessions using **Atomic Reference Counting** and **Copy-On-Write (CoW)**.
5. Employs **Iteration-Level Continuous Batching** and **Preemption** to eliminate compute stranding and guarantee zero crashes under memory pressure.

---

## 2. The Core Problem in Deep Detail

### 2.1 Autoregressive Generation & The KV-Cache
When an LLM generates text, it does so sequentially token by token:
$$\text{Input: } x_1, \dots, x_t \implies \text{Predict: } x_{t+1}$$

To compute attention at step $t$, the Query vector of the new token must compute dot products with the Key (K) and Value (V) projections of all $t-1$ preceding tokens:
$$\text{Attention}(Q_t, K_{\le t}, V_{\le t}) = \text{softmax}\left(\frac{Q_t K_{\le t}^T}{\sqrt{d_k}}\right) V_{\le t}$$

* **Without KV Cache:** Computing step $t$ requires recomputing $K$ and $V$ for all past tokens from scratch. Across an entire sequence of length $N$, this produces an **$O(N^2)$ quadratic computational explosion**.
* **With KV Cache:** Past K and V tensors are cached in GPU VRAM. Computing new projections takes $O(1)$ work per token.
* **The Computational Nuance:** While projections are saved, the decode attention step still attends over the growing sequence ($O(t)$ memory loads). Thus, decoding is strictly **memory-bandwidth bound (GEMV)**: for every single token emitted, the entire accumulated KV cache must be streamed from VRAM into GPU SRAM.

### 2.2 The KV-Cache Memory Footprint Formula
$$\text{KV Memory per Token} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{bytes\_per\_element}$$

* For a compact 0.5B–1B model (e.g., 24 layers, 16 heads, 64 dim, FP16 = 2 bytes):  
  **$\approx 98.3 \text{ KB}$ per token.**
* For 100 concurrent requests of context length 1,024:  
  **$\approx 9.8 \text{ GB}$ of VRAM** solely for KV cache!

---

## 3. The Fatal Flaws of Contiguous Allocation

Standard deep learning frameworks allocate memory in **contiguous chunks**:
1. **Dynamic Length Ambiguity:** The server cannot predict how many tokens a user prompt will generate (10 tokens vs 2,000 tokens).
2. **Worst-Case Pre-Allocation:** Servers pre-allocate a static memory buffer sized for the maximum context window ($L_{\max}$).
3. **Internal Fragmentation:** If a user finishes in 60 tokens, 95%+ of the allocated buffer sits empty, locked, and unusable.
4. **External Fragmentation:** Over time, varied arrival and departure times cause memory checkerboarding. The allocator fails to find contiguous chunks, triggering premature `CUDA: Out-Of-Memory (OOM)` errors even when total free VRAM is abundant.
5. **Concurrency Starvation:** Memory-constrained devices cap out at 2 to 3 sequences, leaving GPU Tensor Cores starved.

---

## 4. The Paged Solution (The Hotel & Notebook Analogy)

* **Naive Contiguous Allocation (The Wasteful Hotel):** A hotel reserves a 50-bed penthouse for every guest upfront, even if they stay alone. After 4 guests, the hotel is "full," even though 90% of beds are empty.
* **Paged Allocation (The Loose-Leaf Notebook):** You hand a student 1 sheet of paper (16 lines). Only when line 16 is filled do you pull sheet 2 from anywhere in the drawer. A Page Table keeps track of which sheet belongs to whom.
* **Prefix Caching (The Shared Photocopy):** If 10 students share the same 2-page syllabus, they all read the same physical sheets in memory. Zero extra memory is allocated, and Time-To-First-Token drops to 0ms.
