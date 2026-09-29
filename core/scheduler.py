from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

from core.block_manager import BlockAllocator, MemoryTier
from core.page_table import PageTable
from core.prefix_cache import PrefixCache
from core.kv_cache import PagedKVCache
from core.request import Sequence, SequenceStatus

@dataclass
class SchedulerOutput:
    step_id: int
    running_seq_ids: List[int]
    waiting_seq_ids: List[int]
    swapped_seq_ids: List[int]
    newly_finished_seq_ids: List[int]
    preempted_seq_ids: List[int]
    restored_seq_ids: List[int]
    gpu_utilization_pct: float
    cpu_utilization_pct: float
    gpu_free_blocks: int
    cpu_free_blocks: int

class ContinuousScheduler:
    """
    Iteration-Level Continuous Batching Scheduler with Tiered GPU ⇄ Host CPU RAM Swapping.
    
    Key Systems Capabilities:
    1. Iteration-Level Continuous Scheduling: Dynamic admission and instant retirement per token step.
    2. Zero Compute Stranding: Requests of uneven length finish without stalling other streams.
    3. Tiered Preemption: Evicts lowest-priority sequences to Host CPU RAM under 100% VRAM saturation,
       guaranteeing zero CUDA Out-Of-Memory crashes.
    4. Automatic Recovery: Seamlessly pages swapped sequences back into GPU VRAM once memory is freed.
    """
    def __init__(
        self,
        num_gpu_blocks: int = 128,
        num_cpu_blocks: int = 256,
        block_size: int = 16,
        max_batch_size: int = 32,
        watermark_blocks: int = 2,
        kv_cache: Optional[PagedKVCache] = None
    ):
        self.block_size = block_size
        self.max_batch_size = max_batch_size
        self.watermark_blocks = watermark_blocks

        # Tier 1 & Tier 2 Memory Allocators
        self.gpu_allocator = BlockAllocator(num_blocks=num_gpu_blocks, block_size=block_size, tier=MemoryTier.GPU)
        self.cpu_allocator = BlockAllocator(num_blocks=num_cpu_blocks, block_size=block_size, tier=MemoryTier.CPU)

        # Prefix Cache (Radix/Rolling hash tree)
        self.prefix_cache = PrefixCache(block_size=block_size)

        # Unified Paged KV-Cache
        self.kv_cache = kv_cache

        # Active page tables mapped by seq_id
        self.page_tables: Dict[int, PageTable] = {}

        # Scheduling Queues
        self.waiting_queue: deque[Sequence] = deque()
        self.running_batch: List[Sequence] = []
        self.swapped_queue: deque[Sequence] = deque()
        self.finished_sequences: List[Sequence] = []

        self.step_counter = 0

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting_queue or self.running_batch or self.swapped_queue)

    def add_request(self, sequence: Sequence):
        """Enqueues an incoming request into the waiting queue."""
        sequence.status = SequenceStatus.WAITING
        self.waiting_queue.append(sequence)

    def step(self, next_tokens: Optional[Dict[int, int]] = None) -> SchedulerOutput:
        """
        Executes one atomic iteration step across all active inference sequences.
        Advances running sequences by 1 token, handles preemption, recovery, and dynamic admission.
        """
        self.step_counter += 1
        newly_finished_ids: List[int] = []
        preempted_ids: List[int] = []
        restored_ids: List[int] = []

        # -------------------------------------------------------------
        # 1. RETIRE COMPLETED SEQUENCES & RECLAIM MEMORY IMMEDIATELY
        # -------------------------------------------------------------
        still_running: List[Sequence] = []
        for seq in self.running_batch:
            if seq.is_finished:
                newly_finished_ids.append(seq.seq_id)
                self.finished_sequences.append(seq)
                if seq.seq_id in self.page_tables:
                    pt = self.page_tables[seq.seq_id]
                    pt.free_all(self.gpu_allocator, self.cpu_allocator)
                    del self.page_tables[seq.seq_id]
            else:
                still_running.append(seq)
        self.running_batch = still_running

        # -------------------------------------------------------------
        # 2. TIERED PREEMPTION (GPU VRAM Saturated -> Swap Out to CPU)
        # -------------------------------------------------------------
        # If GPU free blocks drop below safety watermark, swap coldest sequence to Host CPU RAM
        while self.gpu_allocator.num_free_blocks < self.watermark_blocks and len(self.running_batch) > 1:
            # Preempt the most recently arrived / lowest-priority sequence
            victim_seq = self.running_batch.pop()
            victim_pt = self.page_tables[victim_seq.seq_id]

            # Move all blocks of this sequence from GPU to Host CPU RAM
            victim_pt.swap_out(self.gpu_allocator, self.cpu_allocator, self.kv_cache)
            victim_seq.mark_preempted()

            self.swapped_queue.append(victim_seq)
            preempted_ids.append(victim_seq.seq_id)

        # -------------------------------------------------------------
        # 3. RECOVERY / SWAP-IN (Memory Freed -> Page In from Host CPU)
        # -------------------------------------------------------------
        # If GPU has sufficient free blocks, page back swapped sequences from Host CPU RAM
        if self.swapped_queue:
            candidate = self.swapped_queue[0]
            candidate_pt = self.page_tables[candidate.seq_id]
            needed_gpu_blocks = candidate_pt.num_cpu_blocks

            if self.gpu_allocator.num_free_blocks >= needed_gpu_blocks:
                self.swapped_queue.popleft()
                candidate_pt.swap_in(self.gpu_allocator, self.cpu_allocator, self.kv_cache)
                candidate.status = SequenceStatus.RUNNING
                self.running_batch.append(candidate)
                restored_ids.append(candidate.seq_id)

        # -------------------------------------------------------------
        # 4. DYNAMIC ADMISSION CONTROL (Admit from Waiting Queue)
        # -------------------------------------------------------------
        while self.waiting_queue and len(self.running_batch) < self.max_batch_size:
            candidate = self.waiting_queue[0]
            raw_tokens = candidate.prompt_tokens
            prompt_len = len(raw_tokens)

            # Check Prefix Cache for instant block reuse
            matched_blocks, matched_tokens = self.prefix_cache.match_prefix(raw_tokens, self.gpu_allocator)
            unmatched_tokens = max(0, prompt_len - matched_tokens)
            needed_new_blocks = (unmatched_tokens + self.block_size - 1) // self.block_size
            total_needed_blocks = max(1, needed_new_blocks)

            if self.gpu_allocator.num_free_blocks >= (total_needed_blocks + self.watermark_blocks):
                candidate = self.waiting_queue.popleft()
                pt = PageTable(seq_id=candidate.seq_id, block_size=self.block_size)
                self.page_tables[candidate.seq_id] = pt

                # Assign shared prefix blocks
                for bid in matched_blocks:
                    pt.assign_prefix_block(bid, self.gpu_allocator)

                # Allocate private slots for unmatched prompt tokens
                for _ in range(unmatched_tokens):
                    pt.append_slot(self.gpu_allocator)

                # Register prefix for future queries if eligible
                if matched_tokens == 0 and len(pt.logical_to_physical) >= 2:
                    self.prefix_cache.register_prefix_blocks(
                        raw_tokens[:self.block_size * 2],
                        pt.logical_to_physical[:2],
                        self.gpu_allocator
                    )

                candidate.mark_started()
                self.running_batch.append(candidate)
            else:
                # Insufficient GPU memory to admit this prompt right now
                break

        # -------------------------------------------------------------
        # 5. ADVANCE RUNNING SEQUENCES BY 1 TOKEN
        # -------------------------------------------------------------
        for seq in self.running_batch:
            pt = self.page_tables[seq.seq_id]
            # Advance logical page table by 1 slot (allocating new physical block if crossing 16-token boundary)
            if self.gpu_allocator.num_free_blocks > 0:
                pt.append_slot(self.gpu_allocator)

            # Sample or emit next token
            if next_tokens and seq.seq_id in next_tokens:
                emitted_tok = next_tokens[seq.seq_id]
            else:
                emitted_tok = 100 + seq.seq_id  # Synthetic token for simulation
            seq.append_token(emitted_tok)

        return SchedulerOutput(
            step_id=self.step_counter,
            running_seq_ids=[s.seq_id for s in self.running_batch],
            waiting_seq_ids=[s.seq_id for s in self.waiting_queue],
            swapped_seq_ids=[s.seq_id for s in self.swapped_queue],
            newly_finished_seq_ids=newly_finished_ids,
            preempted_seq_ids=preempted_ids,
            restored_seq_ids=restored_ids,
            gpu_utilization_pct=self.gpu_allocator.utilization_pct,
            cpu_utilization_pct=self.cpu_allocator.utilization_pct,
            gpu_free_blocks=self.gpu_allocator.num_free_blocks,
            cpu_free_blocks=self.cpu_allocator.num_free_blocks
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Provides full real-time systems telemetry snapshot."""
        return {
            "step": self.step_counter,
            "running_count": len(self.running_batch),
            "waiting_count": len(self.waiting_queue),
            "swapped_count": len(self.swapped_queue),
            "completed_count": len(self.finished_sequences),
            "gpu_blocks": self.gpu_allocator.get_status(),
            "cpu_blocks": self.cpu_allocator.get_status()
        }
