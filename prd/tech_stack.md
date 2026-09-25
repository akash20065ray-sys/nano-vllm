# MicroServe-LLM: Technology Stack & Hardware Compatibility

## 1. Hardware Compatibility: Is This for RTX 3050 Only?

### **NO! It runs on ANY NVIDIA GPU and has automatic CPU fallback.**

The architecture of MicroServe-LLM is **completely hardware-agnostic**:
1. **Universal NVIDIA GPU Support:**
   * Works on any consumer or enterprise NVIDIA GPU: **RTX 3050, RTX 3060, RTX 4060, RTX 4090, A100, H100**.
   * Automatically queries `torch.cuda.is_available()`:
     * If an NVIDIA GPU is detected, physical KV-cache blocks are allocated directly in **CUDA VRAM**.
     * If no GPU is available (or when testing on a lightweight laptop), it gracefully falls back to **Host CPU RAM**.
2. **Why the RTX 3050 is Our Benchmark Proof-Point:**
   * We specifically benchmark on the **RTX 3050 (4GB VRAM)** because it represents a **memory-constrained environment**.
   * If an inference engine can eliminate fragmentation, support prefix caching, and avoid out-of-memory crashes on a 4GB laptop GPU, it will scale effortlessly to a 24GB or 80GB enterprise GPU.
   * This is a massive selling point in technical interviews: *"I proved memory efficiency on constrained consumer hardware."*

---

## 2. Hardware Compatibility Matrix

| Environment | KV Tensor Storage | Execution Mode | Max Tested Concurrency |
| :--- | :--- | :--- | :--- |
| **NVIDIA RTX 3050 (4GB VRAM)** *(Local Dev)* | CUDA VRAM (`cuda:0`) | Real Model (`Qwen2.5-0.5B`) + Sim Mode | 16–32 active streams |
| **NVIDIA RTX 4090 / A100 / H100** *(Enterprise)* | CUDA VRAM (`cuda:0`) | Real Model (`Qwen2.5-7B` / `Llama-3-8B`) | 128+ active streams |
| **CPU Fallback** *(Any PC / Mac / CI Server)* | Host RAM (`cpu`) | Simulation Mode + Tiny Real Model | 8–16 active streams |

---

## 3. Software Technology Stack

### 3.1 Core Systems & Runtime
* **Programming Language:** Python 3.13 (High-level memory orchestration, state machines, scheduling).
* **Deep Learning Runtime:** PyTorch 2.x (Tensor allocations, matrix multiplications, dynamic indexing).
* **CUDA Acceleration:** CUDA 12.x / Custom vectorized gather kernel (`kernels/paged_gather.py`).
* **Model Engine:** HuggingFace `transformers` / Local weights (`Qwen/Qwen2.5-0.5B-Instruct` or `HuggingFaceTB/SmolLM-360M-Instruct`).

### 3.2 Backend & Observability Server
* **API Framework:** `FastAPI` (Asynchronous HTTP endpoints + Server-Sent Events).
* **Server Protocol:** `Server-Sent Events (SSE)` for streaming token emissions to the browser in real-time.
* **ASGI Server:** `Uvicorn` (High-performance event loop).
* **System Telemetry:** `psutil` (Process RSS RAM, CPU load, and GPU memory tracking).

### 3.3 Dashboard & Presentation Layer
* **Structure:** Semantic HTML5.
* **Styling:** Vanilla CSS (Dark mode `#0b0d10`, NVIDIA emerald green `#76b900`, cyber-tech glassmorphism, responsive grid).
* **Logic:** Vanilla JavaScript (ES6+ WebSockets / SSE EventSource, dynamic DOM matrix updates).
* **Zero Bloat Principle:** No heavy React/Angular/Node dependencies. The dashboard runs directly from FastAPI static files in sub-10ms.

### 3.4 Benchmarking & Scientific Visualization
* **Data Processing:** `numpy`, `pandas`.
* **Plot Generation:** `matplotlib` (Auto-generates high-resolution PNG charts: latency distributions, throughput vs. concurrency, and fragmentation %).
* **Documentation & Whitepaper:** `reportlab`, `pymupdf` (Automated generation of publication-grade PDF specification).
