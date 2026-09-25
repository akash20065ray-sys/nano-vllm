import pytest
import torch
from core.kv_cache import PagedKVCache

def test_paged_kv_cache_read_write():
    """Verify writing and gathering KV vectors across non-contiguous blocks."""
    cache = PagedKVCache(
        num_blocks=8,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        block_size=16,
        dtype=torch.float32,
        device=torch.device("cpu")
    )

    # Generate dummy K and V vectors for 1 token: [num_heads, head_dim]
    dummy_k = torch.ones((4, 16), dtype=torch.float32) * 5.0
    dummy_v = torch.ones((4, 16), dtype=torch.float32) * 9.0

    # Write to Block 3 at offset 7, Layer 0
    cache.write_kv(physical_block_id=3, offset=7, layer_idx=0, k=dummy_k, v=dummy_v)

    # Verify directly from tensor
    assert torch.allclose(cache.k_cache[3, 0, :, 7, :], dummy_k)
    assert torch.allclose(cache.v_cache[3, 0, :, 7, :], dummy_v)

def test_block_copy_on_write():
    """Verify zero-copy GPU/CPU block duplication for Copy-On-Write."""
    cache = PagedKVCache(
        num_blocks=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        block_size=16,
        dtype=torch.float32,
        device=torch.device("cpu")
    )

    # Populate Block 1 with distinct values
    cache.k_cache[1].fill_(42.0)
    cache.v_cache[1].fill_(99.0)

    # Perform CoW copy from Block 1 to Block 2
    cache.copy_physical_block(src_block_id=1, dst_block_id=2)

    # Verify Block 2 now matches Block 1
    assert torch.allclose(cache.k_cache[2], cache.k_cache[1])
    assert torch.allclose(cache.v_cache[2], cache.v_cache[1])

    # Mutate Block 2 and verify Block 1 remains untouched
    cache.k_cache[2].fill_(77.0)
    assert not torch.allclose(cache.k_cache[2], cache.k_cache[1])
    assert torch.allclose(cache.k_cache[1], torch.tensor(42.0))

def test_gather_sequence_kv_shape():
    """Verify gathered tensor matches exact sequence token length."""
    cache = PagedKVCache(
        num_blocks=8,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        block_size=16,
        dtype=torch.float32,
        device=torch.device("cpu")
    )

    # Gather across 3 physical blocks [5, 2, 7] for 35 tokens
    k_gathered, v_gathered = cache.gather_sequence_kv(
        physical_block_ids=[5, 2, 7],
        layer_idx=0,
        num_tokens=35
    )

    assert k_gathered.shape == (35, 4, 16)
    assert v_gathered.shape == (35, 4, 16)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
