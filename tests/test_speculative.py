import pytest
from core.block_manager import BlockAllocator, MemoryTier
from core.page_table import PageTable
from core.speculative import SpeculativeEngine, SpeculativeStepResult

def test_page_table_rollback_within_block():
    allocator = BlockAllocator(num_blocks=16, block_size=16)
    pt = PageTable(seq_id=1, block_size=16)

    # Allocate 10 tokens (1 block)
    for _ in range(10):
        pt.append_slot(allocator)

    assert pt.num_tokens == 10
    assert len(pt.logical_to_physical) == 1
    initial_bid = pt.logical_to_physical[0]

    # Rollback 3 slots -> 7 tokens remain, block not freed
    freed = pt.rollback_slots(num_slots=3, allocator=allocator)
    assert freed == []
    assert pt.num_tokens == 7
    assert len(pt.logical_to_physical) == 1
    assert pt.logical_to_physical[0] == initial_bid

def test_page_table_rollback_across_block_boundary():
    allocator = BlockAllocator(num_blocks=16, block_size=16)
    pt = PageTable(seq_id=1, block_size=16)

    # Allocate 34 tokens -> spans 3 blocks: block 0 (16), block 1 (16), block 2 (2)
    for _ in range(34):
        pt.append_slot(allocator)

    assert pt.num_tokens == 34
    assert len(pt.logical_to_physical) == 3
    free_count_before = len(allocator.free_blocks)

    # Rollback 4 slots -> 30 tokens remain (spans exactly 2 blocks)
    # The 3rd block should be abandoned and reclaimed
    freed = pt.rollback_slots(num_slots=4, allocator=allocator)
    assert len(freed) == 1
    assert pt.num_tokens == 30
    assert len(pt.logical_to_physical) == 2
    assert len(allocator.free_blocks) == free_count_before + 1

    # Rollback remaining 30 slots to 0 -> both remaining blocks should be freed
    freed_remaining = pt.rollback_slots(num_slots=30, allocator=allocator)
    assert len(freed_remaining) == 2
    assert pt.num_tokens == 0
    assert len(pt.logical_to_physical) == 0
    assert len(allocator.free_blocks) == 16

def test_page_table_rollback_invalid_raises():
    allocator = BlockAllocator(num_blocks=16, block_size=16)
    pt = PageTable(seq_id=1, block_size=16)

    for _ in range(5):
        pt.append_slot(allocator)

    with pytest.raises(ValueError):
        pt.rollback_slots(num_slots=10, allocator=allocator)

def test_speculative_engine_step():
    allocator = BlockAllocator(num_blocks=32, block_size=16)
    engine = SpeculativeEngine(allocator=allocator, block_size=16, k_draft=4)

    seq_id, pt = engine.create_sequence(prompt_tokens_count=8)
    assert pt.num_tokens == 8

    result = engine.execute_speculative_step(
        seq_id=seq_id,
        prompt_hint="virtual memory paging eliminates fragmentation",
        step_index=1,
        k=4
    )

    assert isinstance(result, SpeculativeStepResult)
    assert result.num_drafted == 4
    assert len(result.candidates) >= 1
    assert len(result.emitted_tokens) >= 1
    assert engine.total_target_passes == 1
    assert engine.total_draft_tokens == 4
    assert engine.empirical_acceptance_rate > 0.0
    assert engine.effective_speedup >= 1.0

def test_speculative_stream_generator():
    allocator = BlockAllocator(num_blocks=32, block_size=16)
    engine = SpeculativeEngine(allocator=allocator, block_size=16, k_draft=4)

    events = list(engine.generate_speculative_stream(
        prompt="Explain GPU memory coalescing",
        max_tokens=12,
        k_draft=4
    ))

    types = [e["type"] for e in events]
    assert "spec_start" in types
    assert "spec_step" in types
    assert "spec_done" in types

    done_event = events[-1]
    assert done_event["type"] == "spec_done"
    assert done_event["total_tokens"] >= 12
    assert done_event["target_forward_passes"] >= 1
    assert done_event["effective_tokens_per_pass"] > 1.0
