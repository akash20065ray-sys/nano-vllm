import pytest
from core.scheduler import ContinuousScheduler
from core.request import Sequence, SequenceStatus

def test_continuous_batching_lifecycle():
    """Verify that continuous batching retires sequences immediately upon reaching max tokens."""
    scheduler = ContinuousScheduler(
        num_gpu_blocks=32,
        num_cpu_blocks=64,
        block_size=16,
        max_batch_size=8
    )

    # Add 3 sequences with different target output lengths: 4, 8, 12 tokens
    s1 = Sequence(seq_id=1, prompt_tokens=[1, 2, 3], max_output_tokens=4)
    s2 = Sequence(seq_id=2, prompt_tokens=[4, 5, 6], max_output_tokens=8)
    s3 = Sequence(seq_id=3, prompt_tokens=[7, 8, 9], max_output_tokens=12)

    scheduler.add_request(s1)
    scheduler.add_request(s2)
    scheduler.add_request(s3)

    assert len(scheduler.waiting_queue) == 3
    assert len(scheduler.running_batch) == 0

    # Step 1: All 3 should be admitted into running batch
    out1 = scheduler.step()
    assert len(out1.running_seq_ids) == 3
    assert len(out1.waiting_seq_ids) == 0

    # Run until step 5: Sequence 1 must be completed and retired
    for _ in range(4):
        scheduler.step()

    assert s1.is_finished
    # Sequence 1 finished, Sequence 2 and 3 still running
    assert 1 not in [s.seq_id for s in scheduler.running_batch]
    assert 2 in [s.seq_id for s in scheduler.running_batch]
    assert 3 in [s.seq_id for s in scheduler.running_batch]

    # Run until step 13: All sequences must be completed
    while scheduler.has_unfinished_requests:
        scheduler.step()

    assert len(scheduler.finished_sequences) == 3
    assert len(scheduler.running_batch) == 0
    # Memory must be 100% cleanly reclaimed
    assert scheduler.gpu_allocator.num_free_blocks == 32

def test_tiered_preemption_and_recovery():
    """Verify that when GPU blocks run out, scheduler gracefully swaps to CPU RAM instead of crashing."""
    # Constrained GPU: only 4 blocks total, watermark=1
    scheduler = ContinuousScheduler(
        num_gpu_blocks=4,
        num_cpu_blocks=16,
        block_size=16,
        max_batch_size=4,
        watermark_blocks=1
    )

    # Add 2 sequences requiring 2 blocks each
    s1 = Sequence(seq_id=1, prompt_tokens=[10] * 20, max_output_tokens=6)
    s2 = Sequence(seq_id=2, prompt_tokens=[20] * 20, max_output_tokens=6)

    scheduler.add_request(s1)
    scheduler.add_request(s2)

    # Step 1: s1 admitted (takes 2 blocks). s2 cannot fit because watermark=1 (4 - 2 = 2, but needs 2 + 1 = 3)
    out1 = scheduler.step()
    assert 1 in out1.running_seq_ids

    # Step through s1 execution
    while not s1.is_finished:
        scheduler.step()

    # Once s1 completes, its 2 blocks are freed, allowing s2 to be admitted and execute without crash
    while scheduler.has_unfinished_requests:
        scheduler.step()

    assert s1.is_finished
    assert s2.is_finished
    assert len(scheduler.finished_sequences) == 2
    assert scheduler.gpu_allocator.num_free_blocks == 4

def test_swap_out_and_swap_in_under_saturation():
    """Verify active sequence swap-out to Host CPU RAM under memory pressure."""
    # 3 GPU blocks, 8 CPU blocks
    scheduler = ContinuousScheduler(
        num_gpu_blocks=3,
        num_cpu_blocks=8,
        block_size=16,
        max_batch_size=2,
        watermark_blocks=2
    )

    s1 = Sequence(seq_id=1, prompt_tokens=[1] * 16, max_output_tokens=4)
    s2 = Sequence(seq_id=2, prompt_tokens=[2] * 16, max_output_tokens=4)

    scheduler.add_request(s1)
    scheduler.step()

    # S1 is running and occupies 1 GPU block (2 free left)
    assert 1 in [s.seq_id for s in scheduler.running_batch]

    # Directly force preemption test: s1 swapped out
    pt1 = scheduler.page_tables[1]
    pt1.swap_out(scheduler.gpu_allocator, scheduler.cpu_allocator, scheduler.kv_cache)
    s1.mark_preempted()
    scheduler.swapped_queue.append(s1)
    scheduler.running_batch.remove(s1)

    assert pt1.is_swapped
    assert scheduler.cpu_allocator.num_allocated_blocks > 0
    assert scheduler.gpu_allocator.num_free_blocks == 3

    # Now step: Scheduler must recover s1 from CPU RAM to GPU VRAM
    out = scheduler.step()
    assert 1 in out.restored_seq_ids
    assert not pt1.is_swapped
    assert scheduler.cpu_allocator.num_allocated_blocks == 0

    while scheduler.has_unfinished_requests:
        scheduler.step()

    assert s1.is_finished
    assert scheduler.gpu_allocator.num_free_blocks == 3

if __name__ == "__main__":
    pytest.main(["-v", __file__])
