import torch
from typing import List, Optional, Tuple, Dict, Any

class PagedKVCache:
    """
    Unified physical tensor pool managing Key and Value state buffers in GPU VRAM or CPU RAM.
    Provides vectorized block indexing and zero-copy block duplication for Copy-On-Write.
    """
    def __init__(
        self,
        num_blocks: int,
        num_layers: int = 12,
        num_heads: int = 8,
        head_dim: int = 64,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None
    ):
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Allocate 5D physical tensor pools: [num_blocks, num_layers, num_heads, block_size, head_dim]
        # In CPU fallback or tests, if float16 is unsupported on some CPU ops, use float32
        tensor_dtype = torch.float32 if self.device.type == "cpu" and dtype == torch.float16 else dtype
        self.tensor_dtype = tensor_dtype

        self.k_cache = torch.zeros(
            (num_blocks, num_layers, num_heads, block_size, head_dim),
            dtype=tensor_dtype,
            device=self.device
        )
        self.v_cache = torch.zeros(
            (num_blocks, num_layers, num_heads, block_size, head_dim),
            dtype=tensor_dtype,
            device=self.device
        )

    @property
    def total_memory_mb(self) -> float:
        """Calculates exact physical tensor pool memory in Megabytes"""
        elem_size = 2 if self.tensor_dtype == torch.float16 else 4
        total_elements = 2 * (self.num_blocks * self.num_layers * self.num_heads * self.block_size * self.head_dim)
        return round((total_elements * elem_size) / (1024 * 1024), 2)

    def write_kv(
        self,
        physical_block_id: int,
        offset: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor
    ):
        """
        Writes a single token's Key and Value vector into physical block at designated offset.
        k, v shape: [num_heads, head_dim]
        """
        self.k_cache[physical_block_id, layer_idx, :, offset, :] = k.to(device=self.device, dtype=self.tensor_dtype)
        self.v_cache[physical_block_id, layer_idx, :, offset, :] = v.to(device=self.device, dtype=self.tensor_dtype)

    def copy_physical_block(self, src_block_id: int, dst_block_id: int):
        """
        Zero-copy block duplication used by Copy-On-Write (CoW).
        Performs GPU-to-GPU memory transfer across all layers and heads.
        """
        self.k_cache[dst_block_id].copy_(self.k_cache[src_block_id])
        self.v_cache[dst_block_id].copy_(self.v_cache[src_block_id])

    def gather_sequence_kv(
        self,
        physical_block_ids: List[int],
        layer_idx: int,
        num_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized gather: transforms non-contiguous physical blocks into a continuous
        representation for attention computation.
        
        Returns:
            gathered_k, gathered_v of shape [num_tokens, num_heads, head_dim]
        """
        if not physical_block_ids or num_tokens == 0:
            empty_shape = (0, self.num_heads, self.head_dim)
            return (
                torch.empty(empty_shape, device=self.device, dtype=self.tensor_dtype),
                torch.empty(empty_shape, device=self.device, dtype=self.tensor_dtype)
            )

        # Vectorized gather across specified block indices
        block_tensor = torch.tensor(physical_block_ids, device=self.device, dtype=torch.long)
        
        # Slices: [num_blocks_in_seq, num_heads, block_size, head_dim]
        k_blocks = self.k_cache[block_tensor, layer_idx]
        v_blocks = self.v_cache[block_tensor, layer_idx]

        # Permute and reshape to [num_blocks * block_size, num_heads, head_dim]
        # Transpose: [num_blocks, block_size, num_heads, head_dim] -> Flatten -> Trim to num_tokens
        k_gathered = k_blocks.permute(0, 2, 1, 3).reshape(-1, self.num_heads, self.head_dim)[:num_tokens]
        v_gathered = v_blocks.permute(0, 2, 1, 3).reshape(-1, self.num_heads, self.head_dim)[:num_tokens]

        return k_gathered, v_gathered

    def reset_block(self, physical_block_id: int):
        """Zero out block memory upon release"""
        self.k_cache[physical_block_id].zero_()
        self.v_cache[physical_block_id].zero_()
