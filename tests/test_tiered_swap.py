import pytest
import torch
from core.block_manager import BlockAllocator, MemoryTier
from core.page_table import PageTable
from core.kv_cache import PagedKVCache

def test_tiered_block_allocators():
    """Verify separate GPU and Host CPU swap allocators maintain independent pools."""
    gpu_alloc = BlockAllocator(num_blocks=4, block_size=16, tier=MemoryTier.GPU)
    cpu_alloc = BlockAllocator(num_blocks=8, block_size=16, tier=MemoryTier.CPU)

    assert gpu_alloc.tier == MemoryTier.GPU
    assert cpu_alloc.tier == MemoryTier.CPU
    assert gpu_alloc.num_free_blocks == 4
    assert cpu_alloc.num_free_blocks == 8

    b_gpu = gpu_alloc.allocate()
    b_cpu = cpu_alloc.allocate()

    assert b_gpu.tier == MemoryTier.GPU
    assert b_cpu.tier == MemoryTier.CPU
    assert gpu_alloc.num_free_blocks == 3
    assert cpu_alloc.num_free_blocks == 7

def test_kv_cache_swap_out_and_in():
    """Verify asynchronous DMA copy between GPU VRAM and Host CPU RAM pools."""
    cache = PagedKVCache(
        num_blocks=4,
        num_cpu_blocks=8,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        block_size=16,
        dtype=torch.float32,
        device=torch.device("cpu")
    )

    # Generate dummy token vectors
    k_vec = torch.ones((4, 16), dtype=torch.float32) * 42.0
    v_vec = torch.ones((4, 16), dtype=torch.float32) * 84.0

    # Write to GPU block 1 at offset 5, layer 0
    cache.write_kv(physical_block_id=1, offset=5, layer_idx=0, k=k_vec, v=v_vec)

    # Swap Out: GPU block 1 -> CPU block 3
    cache.swap_out_block(gpu_block_id=1, cpu_block_id=3)

    # Verify CPU RAM buffer received identical tensors
    assert torch.allclose(cache.cpu_k_cache[3, 0, :, 5, :], k_vec)
    assert torch.allclose(cache.cpu_v_cache[3, 0, :, 5, :], v_vec)

    # Mutate original GPU block 1 to simulate reuse/zeroing
    cache.reset_block(1)
    assert not torch.allclose(cache.k_cache[1, 0, :, 5, :], k_vec)

    # Swap In: CPU block 3 -> GPU block 2
    cache.swap_in_block(cpu_block_id=3, gpu_block_id=2)

    # Verify new GPU block 2 has restored data
    assert torch.allclose(cache.k_cache[2, 0, :, 5, :], k_vec)
    assert torch.allclose(cache.v_cache[2, 0, :, 5, :], v_vec)

def test_page_table_swap_lifecycle():
    """Verify PageTable address re-mapping and allocator reclamation during tiered swap."""
    gpu_alloc = BlockAllocator(num_blocks=4, block_size=16, tier=MemoryTier.GPU)
    cpu_alloc = BlockAllocator(num_blocks=8, block_size=16, tier=MemoryTier.CPU)
    cache = PagedKVCache(
        num_blocks=4,
        num_cpu_blocks=8,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        block_size=16,
        dtype=torch.float32,
        device=torch.device("cpu")
    )

    pt = PageTable(seq_id=1, block_size=16)

    # Allocate 32 token slots (2 GPU blocks: 0 and 1)
    for _ in range(32):
        pt.append_slot(gpu_alloc)

    assert pt.num_blocks == 2
    assert pt.num_gpu_blocks == 2
    assert pt.num_cpu_blocks == 0
    assert not pt.is_swapped
    assert gpu_alloc.num_free_blocks == 2
    assert cpu_alloc.num_free_blocks == 8

    # Populate dummy data
    cache.k_cache[pt.logical_to_physical[0]].fill_(11.0)
    cache.k_cache[pt.logical_to_physical[1]].fill_(22.0)

    # Trigger Swap Out: Moves sequence from GPU VRAM to Host CPU RAM
    swapped = pt.swap_out(gpu_alloc, cpu_alloc, cache)
    assert len(swapped) == 2
    assert pt.is_swapped
    assert pt.num_gpu_blocks == 0
    assert pt.num_cpu_blocks == 2

    # GPU blocks must be completely reclaimed!
    assert gpu_alloc.num_free_blocks == 4
    # CPU blocks are now reserved
    assert cpu_alloc.num_free_blocks == 6

    # Verify CPU tensors received the data
    assert torch.allclose(cache.cpu_k_cache[pt.logical_to_physical[0]], torch.tensor(11.0))
    assert torch.allclose(cache.cpu_k_cache[pt.logical_to_physical[1]], torch.tensor(22.0))

    # Trigger Swap In: Re-schedules sequence back to GPU VRAM
    restored = pt.swap_in(gpu_alloc, cpu_alloc, cache)
    assert len(restored) == 2
    assert not pt.is_swapped
    assert pt.num_gpu_blocks == 2
    assert pt.num_cpu_blocks == 0

    # CPU blocks must be freed
    assert cpu_alloc.num_free_blocks == 8
    # GPU blocks are re-allocated
    assert gpu_alloc.num_free_blocks == 2

    # Clean up
    pt.free_all(gpu_alloc, cpu_alloc)
    assert gpu_alloc.num_free_blocks == 4
    assert cpu_alloc.num_free_blocks == 8

if __name__ == "__main__":
    pytest.main(["-v", __file__])
