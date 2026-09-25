import pytest
from core.block_manager import BlockAllocator
from core.prefix_cache import PrefixCache

def test_prefix_registration_and_match():
    """Verify registration and instant cache hit for identical prompt prefixes."""
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    cache = PrefixCache(block_size=16)

    # 32 tokens system prompt (2 blocks)
    sys_prompt = list(range(100, 132))
    b1 = allocator.allocate()
    b2 = allocator.allocate()

    reg_count = cache.register_prefix_blocks(sys_prompt, [b1.block_id, b2.block_id], allocator)
    assert reg_count == 2
    assert cache.hits == 0

    # User 1 submits same 32 tokens + 10 new tokens
    user_prompt = sys_prompt + list(range(200, 210))
    matched_blocks, matched_tokens = cache.match_prefix(user_prompt, allocator)

    assert matched_tokens == 32
    assert matched_blocks == [b1.block_id, b2.block_id]
    assert cache.hits == 1
    assert cache.misses == 0
    assert cache.hit_rate_pct == 100.0

def test_prefix_miss():
    """Verify cache miss on completely novel prompts."""
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    cache = PrefixCache(block_size=16)

    prompt = list(range(50, 90))
    matched_blocks, matched_tokens = cache.match_prefix(prompt, allocator)
    assert matched_tokens == 0
    assert matched_blocks == []
    assert cache.hits == 0
    assert cache.misses == 1

def test_lru_eviction():
    """Verify that oldest unreferenced prefix block is evicted first under pressure."""
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    cache = PrefixCache(block_size=16, max_cached_blocks=2)

    # Prefix A
    prompt_a = list(range(0, 16))
    b_a = allocator.allocate()
    cache.register_prefix_blocks(prompt_a, [b_a.block_id], allocator)
    b_a.decrement_ref()  # unreferenced

    # Prefix B
    prompt_b = list(range(16, 32))
    b_b = allocator.allocate()
    cache.register_prefix_blocks(prompt_b, [b_b.block_id], allocator)
    b_b.decrement_ref()  # unreferenced

    # Cache is at max (2 blocks). Registering Prefix C should evict oldest (A)
    prompt_c = list(range(32, 48))
    b_c = allocator.allocate()
    cache.register_prefix_blocks(prompt_c, [b_c.block_id], allocator)

    # Prefix A should have been evicted
    m_a, t_a = cache.match_prefix(prompt_a, allocator)
    assert t_a == 0

    # Prefix B and C should still be cached
    m_b, t_b = cache.match_prefix(prompt_b, allocator)
    assert t_b == 16

    m_c, t_c = cache.match_prefix(prompt_c, allocator)
    assert t_c == 16

if __name__ == "__main__":
    pytest.main(["-v", __file__])
