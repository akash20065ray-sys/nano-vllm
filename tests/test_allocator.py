import pytest
import time
from core.block_manager import BlockAllocator, PhysicalBlock
from core.page_table import PageTable

def test_allocation_and_free():
    """Verify O(1) allocation and deallocation lifecycle."""
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    assert allocator.num_free_blocks == 10
    assert allocator.num_allocated_blocks == 0
    assert allocator.utilization_pct == 0.0

    # Allocate 3 blocks
    b1 = allocator.allocate()
    b2 = allocator.allocate()
    b3 = allocator.allocate()

    assert allocator.num_free_blocks == 7
    assert allocator.num_allocated_blocks == 3
    assert b1.ref_count == 1
    assert b2.ref_count == 1

    # Free 1 block
    allocator.free(b2.block_id)
    assert allocator.num_free_blocks == 8
    assert allocator.num_allocated_blocks == 2

    # Free remaining
    allocator.free(b1.block_id)
    allocator.free(b3.block_id)
    assert allocator.num_free_blocks == 10
    assert allocator.num_allocated_blocks == 0

def test_out_of_blocks_error():
    """Verify memory exhaustion raises MemoryError without crashing."""
    allocator = BlockAllocator(num_blocks=3, block_size=16)
    b1 = allocator.allocate()
    b2 = allocator.allocate()
    b3 = allocator.allocate()

    with pytest.raises(MemoryError):
        allocator.allocate()

    # Freeing one block restores allocation capability
    allocator.free(b1.block_id)
    b4 = allocator.allocate()
    assert b4.block_id == b1.block_id

def test_double_free_protection():
    """Verify double-free calls do not corrupt the free pool."""
    allocator = BlockAllocator(num_blocks=5, block_size=16)
    b1 = allocator.allocate()
    bid = b1.block_id

    # First free
    allocator.free(bid)
    assert allocator.num_free_blocks == 5

    # Malicious/buggy second free call on already-free block
    allocator.free(bid)
    allocator.free(bid)
    assert allocator.num_free_blocks == 5  # No duplicates added!

def test_page_table_translation():
    """Verify virtual-to-physical address translation in O(1)."""
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    pt = PageTable(seq_id=1, block_size=16)

    # Append 35 token slots (should span across 3 blocks: 16 + 16 + 3)
    for _ in range(35):
        pt.append_slot(allocator)

    assert pt.num_blocks == 3
    assert pt.num_tokens == 35

    # Test address translation
    # Token 0 -> Block 0, Offset 0
    p0, off0 = pt.translate_token(0)
    assert p0 == pt.logical_to_physical[0]
    assert off0 == 0

    # Token 17 -> Block 1, Offset 1
    p17, off17 = pt.translate_token(17)
    assert p17 == pt.logical_to_physical[1]
    assert off17 == 1

    # Token 34 -> Block 2, Offset 2
    p34, off34 = pt.translate_token(34)
    assert p34 == pt.logical_to_physical[2]
    assert off34 == 2

    # Cleanup
    pt.free_all(allocator)
    assert allocator.num_free_blocks == 10

def test_copy_on_write():
    """Verify Copy-On-Write memory isolation on shared prefix blocks."""
    allocator = BlockAllocator(num_blocks=10, block_size=16)

    # Sequence A (Alice) gets a prefix block
    pt_a = PageTable(seq_id=1, block_size=16)
    blk = allocator.allocate()
    pt_a.logical_to_physical.append(blk.block_id)
    pt_a.num_tokens = 16

    # Sequence B (Bob) shares the same block (prefix hit)
    pt_b = PageTable(seq_id=2, block_size=16)
    pt_b.assign_prefix_block(blk.block_id, allocator)

    assert blk.ref_count == 2
    assert blk.is_shared is True

    # Bob attempts to mutate/write into the shared block 0
    cow_result = pt_b.trigger_copy_on_write(logical_block_idx=0, allocator=allocator)
    assert cow_result is not None
    old_id, new_id = cow_result

    # Bob now owns a new private block
    assert pt_b.logical_to_physical[0] == new_id
    assert new_id != old_id

    # Old block ref_count decremented; no longer shared
    assert blk.ref_count == 1
    assert blk.is_shared is False

    # Alice's mapping is completely unmutated
    assert pt_a.logical_to_physical[0] == old_id

    # Cleanup
    pt_a.free_all(allocator)
    pt_b.free_all(allocator)
    assert allocator.num_free_blocks == 10

def test_internal_fragmentation_math():
    """Verify internal fragmentation tracking."""
    allocator = BlockAllocator(num_blocks=5, block_size=16)
    pt = PageTable(seq_id=1, block_size=16)

    # 20 tokens -> takes 2 blocks (32 token capacity) -> 12 unused tokens
    for _ in range(20):
        pt.append_slot(allocator)

    stats = pt.get_fragmentation_stats()
    assert stats["num_tokens"] == 20
    assert stats["total_capacity_tokens"] == 32
    assert stats["internal_frag_tokens"] == 12
    assert stats["internal_frag_pct"] == 37.5

    pt.free_all(allocator)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
