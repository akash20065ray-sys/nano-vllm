from typing import List, Tuple, Dict, Any, Optional
from core.block_manager import BlockAllocator, PhysicalBlock, MemoryTier

class PageTable:
    """
    Manages the virtual-to-physical address translation for an individual inference sequence.
    Maps contiguous logical token space to arbitrary, non-contiguous physical blocks in GPU VRAM or Host RAM.
    Implements Copy-On-Write (CoW) page isolation and 2-Tier Swapping (GPU VRAM ⇄ Host CPU RAM).
    """
    def __init__(self, seq_id: int, block_size: int = 16):
        self.seq_id = seq_id
        self.block_size = block_size
        self.logical_to_physical: List[int] = []
        self.logical_to_tier: List[str] = []
        self.num_tokens: int = 0

    @property
    def num_blocks(self) -> int:
        return len(self.logical_to_physical)

    @property
    def is_swapped(self) -> bool:
        """Returns True if any logical block is currently swapped out to Host CPU RAM"""
        return any(t == "CPU" for t in self.logical_to_tier)

    @property
    def num_gpu_blocks(self) -> int:
        return sum(1 for t in self.logical_to_tier if t == "GPU")

    @property
    def num_cpu_blocks(self) -> int:
        return sum(1 for t in self.logical_to_tier if t == "CPU")

    def translate_token(self, token_idx: int) -> Tuple[int, int]:
        """
        Translates a virtual token index into (physical_block_id, block_offset) in O(1) time.
        Formula:
            logical_block = token_idx // block_size
            offset = token_idx % block_size
        """
        if token_idx < 0 or token_idx >= self.num_tokens:
            raise IndexError(
                f"Token index {token_idx} out of bounds for sequence {self.seq_id} (length {self.num_tokens})"
            )

        logical_block_idx = token_idx // self.block_size
        offset = token_idx % self.block_size
        physical_block_id = self.logical_to_physical[logical_block_idx]
        return physical_block_id, offset

    def append_slot(self, allocator: BlockAllocator) -> Tuple[int, int, bool]:
        """
        Advances the sequence by one token slot.
        If a 16-token physical block boundary is crossed, automatically allocates a new block.
        Returns: (physical_block_id, offset, is_new_block_allocated)
        """
        token_idx = self.num_tokens
        logical_block_idx = token_idx // self.block_size
        offset = token_idx % self.block_size

        is_new_block = False
        if logical_block_idx >= len(self.logical_to_physical):
            # Boundary crossed -> allocate a fresh physical block
            new_block = allocator.allocate()
            self.logical_to_physical.append(new_block.block_id)
            tier_val = new_block.tier.value if hasattr(new_block.tier, "value") else str(new_block.tier)
            self.logical_to_tier.append(tier_val)
            is_new_block = True

        physical_block_id = self.logical_to_physical[logical_block_idx]
        self.num_tokens += 1
        return physical_block_id, offset, is_new_block

    def assign_prefix_block(self, physical_block_id: int, allocator: BlockAllocator):
        """
        Assigns an existing physical block (from prefix cache) into this sequence's page table.
        Increments the physical block's atomic reference count.
        """
        block = allocator.blocks[physical_block_id]
        block.increment_ref()
        self.logical_to_physical.append(physical_block_id)
        tier_val = block.tier.value if hasattr(block.tier, "value") else str(block.tier)
        self.logical_to_tier.append(tier_val)
        self.num_tokens += self.block_size

    def trigger_copy_on_write(
        self, logical_block_idx: int, allocator: BlockAllocator
    ) -> Optional[Tuple[int, int]]:
        """
        Checks if the physical block at logical_block_idx is shared (ref_count > 1).
        If shared, allocates a new private block, re-points the page table,
        and decrements the old block's ref_count.
        
        Returns: (old_physical_block_id, new_physical_block_id) if CoW occurred, else None.
        """
        if logical_block_idx < 0 or logical_block_idx >= len(self.logical_to_physical):
            raise IndexError(f"Invalid logical_block_idx: {logical_block_idx}")

        old_phys_id = self.logical_to_physical[logical_block_idx]
        old_block = allocator.blocks[old_phys_id]

        if old_block.is_shared:
            # Shared block mutation detected -> Trigger Copy-On-Write
            new_block = allocator.allocate()
            new_phys_id = new_block.block_id

            # Re-map page table entry to new private block
            self.logical_to_physical[logical_block_idx] = new_phys_id
            tier_val = new_block.tier.value if hasattr(new_block.tier, "value") else str(new_block.tier)
            self.logical_to_tier[logical_block_idx] = tier_val

            # Decrement shared block reference count
            old_block.decrement_ref()

            return old_phys_id, new_phys_id

        return None

    def swap_out(
        self,
        gpu_allocator: BlockAllocator,
        cpu_allocator: BlockAllocator,
        kv_cache: Optional[Any] = None
    ) -> List[Tuple[int, int]]:
        """
        Paging out: Moves all GPU blocks owned by this sequence to Host CPU RAM.
        Returns list of (gpu_block_id, cpu_block_id) pairs.
        """
        swapped_pairs = []
        for logical_idx in range(len(self.logical_to_physical)):
            if self.logical_to_tier[logical_idx] == "GPU":
                gpu_bid = self.logical_to_physical[logical_idx]
                cpu_block = cpu_allocator.allocate()
                cpu_bid = cpu_block.block_id

                if kv_cache is not None:
                    kv_cache.swap_out_block(gpu_bid, cpu_bid)

                # Free GPU block and re-point page table entry
                gpu_allocator.free(gpu_bid)
                self.logical_to_physical[logical_idx] = cpu_bid
                self.logical_to_tier[logical_idx] = "CPU"
                swapped_pairs.append((gpu_bid, cpu_bid))

        return swapped_pairs

    def swap_in(
        self,
        gpu_allocator: BlockAllocator,
        cpu_allocator: BlockAllocator,
        kv_cache: Optional[Any] = None
    ) -> List[Tuple[int, int]]:
        """
        Paging in: Swaps all CPU blocks back to GPU VRAM when sequence is scheduled to run.
        Returns list of (cpu_block_id, gpu_block_id) pairs.
        """
        swapped_pairs = []
        for logical_idx in range(len(self.logical_to_physical)):
            if self.logical_to_tier[logical_idx] == "CPU":
                cpu_bid = self.logical_to_physical[logical_idx]
                gpu_block = gpu_allocator.allocate()
                gpu_bid = gpu_block.block_id

                if kv_cache is not None:
                    kv_cache.swap_in_block(cpu_bid, gpu_bid)

                # Free CPU block and re-point page table entry
                cpu_allocator.free(cpu_bid)
                self.logical_to_physical[logical_idx] = gpu_bid
                self.logical_to_tier[logical_idx] = "GPU"
                swapped_pairs.append((cpu_bid, gpu_bid))

        return swapped_pairs

    def free_all(self, allocator: BlockAllocator, cpu_allocator: Optional[BlockAllocator] = None):
        """
        Reclaims all physical memory blocks owned or referenced by this sequence.
        Returns blocks to the free pool if their ref_count drops to 0.
        """
        for i, phys_id in enumerate(self.logical_to_physical):
            tier = self.logical_to_tier[i] if i < len(self.logical_to_tier) else "GPU"
            if tier == "CPU" and cpu_allocator is not None:
                cpu_allocator.free(phys_id)
            else:
                allocator.free(phys_id)
        self.logical_to_physical.clear()
        self.logical_to_tier.clear()
        self.num_tokens = 0

    def get_fragmentation_stats(self) -> Dict[str, Any]:
        """
        Calculates exact internal memory fragmentation for this sequence.
        Internal fragmentation only occurs in the active trailing block.
        """
        if self.num_tokens == 0:
            return {"internal_frag_tokens": 0, "internal_frag_pct": 0.0}

        total_capacity_tokens = len(self.logical_to_physical) * self.block_size
        unused_tokens = total_capacity_tokens - self.num_tokens
        frag_pct = (unused_tokens / total_capacity_tokens) * 100.0 if total_capacity_tokens > 0 else 0.0

        return {
            "num_blocks": len(self.logical_to_physical),
            "num_tokens": self.num_tokens,
            "total_capacity_tokens": total_capacity_tokens,
            "internal_frag_tokens": unused_tokens,
            "internal_frag_pct": round(frag_pct, 2)
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq_id": self.seq_id,
            "num_tokens": self.num_tokens,
            "num_blocks": len(self.logical_to_physical),
            "logical_to_physical": list(self.logical_to_physical),
            "tiers": list(self.logical_to_tier),
            "is_swapped": self.is_swapped,
            "fragmentation": self.get_fragmentation_stats()
        }
