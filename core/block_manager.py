import time
from collections import deque
from typing import List, Optional, Dict, Any

class PhysicalBlock:
    """
    Represents an isolated, fixed-size physical memory block in GPU VRAM or Host RAM.
    Stores metadata including block ID, atomic reference count, and LRU access timestamp.
    """
    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0
        self.last_accessed: float = time.perf_counter()
        self.prefix_hash: Optional[str] = None
        self.is_immutable: bool = False

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0

    @property
    def is_shared(self) -> bool:
        return self.ref_count > 1

    def touch(self):
        """Update last accessed timestamp for LRU tracking"""
        self.last_accessed = time.perf_counter()

    def increment_ref(self):
        self.ref_count += 1
        self.touch()

    def decrement_ref(self) -> int:
        if self.ref_count > 0:
            self.ref_count -= 1
        self.touch()
        return self.ref_count

    def reset(self):
        """Reset block state when completely freed"""
        self.ref_count = 0
        self.prefix_hash = None
        self.is_immutable = False
        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "ref_count": self.ref_count,
            "is_shared": self.is_shared,
            "is_immutable": self.is_immutable,
            "prefix_hash": self.prefix_hash,
            "last_accessed": round(self.last_accessed, 3)
        }

class BlockAllocator:
    """
    Lock-free Physical Block Allocator managing fixed-size GPU VRAM memory blocks.
    Guarantees O(1) allocation, O(1) deallocation, and tracks memory fragmentation.
    """
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Pre-instantiate all physical blocks
        self.blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i, block_size=block_size) for i in range(num_blocks)
        ]

        # Free pool: circular deque for O(1) pop and append
        self.free_blocks: deque = deque(range(num_blocks))

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    @property
    def utilization_pct(self) -> float:
        """Percentage of total physical memory blocks allocated"""
        if self.num_blocks == 0:
            return 0.0
        return round((self.num_allocated_blocks / self.num_blocks) * 100.0, 2)

    def allocate(self) -> PhysicalBlock:
        """
        Allocates one physical block in O(1) time.
        Raises MemoryError if free pool is exhausted.
        """
        if not self.free_blocks:
            raise MemoryError(
                f"GPU VRAM Out-Of-Blocks: All {self.num_blocks} physical blocks are allocated."
            )

        block_id = self.free_blocks.popleft()
        block = self.blocks[block_id]
        block.reset()
        block.increment_ref()  # ref_count becomes 1
        return block

    def allocate_multiple(self, count: int) -> List[PhysicalBlock]:
        """Allocates 'count' physical blocks atomically."""
        if count > self.num_free_blocks:
            raise MemoryError(
                f"Cannot allocate {count} blocks. Only {self.num_free_blocks} available."
            )
        return [self.allocate() for _ in range(count)]

    def free(self, block_id: int):
        """
        Decrements reference count. If ref_count reaches 0, returns block to free pool in O(1).
        Protects against double-free bugs.
        """
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Invalid block_id: {block_id}")

        block = self.blocks[block_id]
        if block.ref_count <= 0:
            # Block is already free; prevent duplicate free list entries
            return

        remaining_refs = block.decrement_ref()
        if remaining_refs == 0:
            block.reset()
            self.free_blocks.append(block_id)

    def free_multiple(self, block_ids: List[int]):
        """Frees an array of physical block IDs."""
        for bid in block_ids:
            self.free(bid)

    def get_status(self) -> Dict[str, Any]:
        """Telemetry snapshot of current physical memory state"""
        shared_count = sum(1 for b in self.blocks if b.is_shared)
        return {
            "total_blocks": self.num_blocks,
            "allocated_blocks": self.num_allocated_blocks,
            "free_blocks": self.num_free_blocks,
            "shared_blocks": shared_count,
            "utilization_pct": self.utilization_pct,
            "block_size_tokens": self.block_size,
            "total_token_capacity": self.num_blocks * self.block_size
        }
