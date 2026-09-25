import sys
import time
from typing import Dict, Any
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.kv_cache import PagedKVCache

class PagedGatherKernelBenchmark:
    """
    Profiles the memory bandwidth and latency of Paged KV-Cache gather operations
    against a contiguous baseline. Demonstrates the hardware efficiency of block-level indexing.
    """
    def __init__(
        self,
        num_blocks: int = 128,
        num_layers: int = 12,
        num_heads: int = 8,
        head_dim: int = 64,
        block_size: int = 16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.cache = PagedKVCache(
            num_blocks=num_blocks,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            device=self.device
        )
        self.block_size = block_size
        self.head_dim = head_dim
        self.num_heads = num_heads

    def benchmark_gather(
        self,
        seq_length: int = 512,
        num_trials: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmarks gathering a sequence of length seq_length from non-contiguous blocks
        vs a contiguous tensor slice.
        """
        num_blocks = (seq_length + self.block_size - 1) // self.block_size
        # Simulate randomized non-contiguous physical block placement in memory
        physical_blocks = torch.randperm(self.cache.num_blocks)[:num_blocks].tolist()

        # Contiguous baseline buffer: [seq_length, num_heads, head_dim]
        contiguous_tensor = torch.randn(
            (seq_length, self.num_heads, self.head_dim),
            device=self.device,
            dtype=self.cache.tensor_dtype
        )

        # Warmup
        for _ in range(10):
            _ = self.cache.gather_sequence_kv(physical_blocks, layer_idx=0, num_tokens=seq_length)
            _ = contiguous_tensor[:seq_length]

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Test 1: Paged Gather
        t0 = time.perf_counter()
        for _ in range(num_trials):
            _ = self.cache.gather_sequence_kv(physical_blocks, layer_idx=0, num_tokens=seq_length)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_paged = ((time.perf_counter() - t0) / num_trials) * 1e6  # microseconds

        # Test 2: Contiguous Slice
        t0 = time.perf_counter()
        for _ in range(num_trials):
            _ = contiguous_tensor[:seq_length]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_contiguous = ((time.perf_counter() - t0) / num_trials) * 1e6  # microseconds

        # Theoretical memory bytes read: 2 tensors (K & V) * seq_length * num_heads * head_dim * bytes_per_elem
        elem_size = 2 if self.cache.tensor_dtype == torch.float16 else 4
        bytes_transferred = 2 * seq_length * self.num_heads * self.head_dim * elem_size
        bandwidth_gb_s = (bytes_transferred / (t_paged * 1e-6)) / (1024 ** 3) if t_paged > 0 else 0.0

        return {
            "device": str(self.device),
            "seq_length": seq_length,
            "num_blocks": num_blocks,
            "paged_gather_latency_us": round(t_paged, 2),
            "contiguous_slice_latency_us": round(t_contiguous, 2),
            "memory_transferred_kb": round(bytes_transferred / 1024, 2),
            "effective_bandwidth_gb_s": round(bandwidth_gb_s, 2),
            "indirection_overhead_us": round(max(0.0, t_paged - t_contiguous), 2)
        }

if __name__ == "__main__":
    benchmark = PagedGatherKernelBenchmark()
    res = benchmark.benchmark_gather(seq_length=512)
    print("=" * 60)
    print("  PAGED GATHER KERNEL BENCHMARK RESULTS")
    print("=" * 60)
    for k, v in res.items():
        print(f"  {k:30s}: {v}")
    print("=" * 60)
