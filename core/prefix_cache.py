import hashlib
import time
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict, Any
from core.block_manager import BlockAllocator, PhysicalBlock

class PrefixCache:
    """
    High-performance hash-based prompt prefix cache with LRU eviction.
    Enables zero-copy, zero-compute sharing of common prompt prefixes (e.g. system prompts)
    across concurrent sessions.
    """
    def __init__(self, block_size: int = 16, max_cached_blocks: int = 256):
        self.block_size = block_size
        self.max_cached_blocks = max_cached_blocks

        # Hash -> physical_block_id mapping (OrderedDict acts as LRU cache)
        self.cache: OrderedDict[str, int] = OrderedDict()
        
        # Metrics
        self.hits = 0
        self.misses = 0

    @staticmethod
    def compute_block_hash(prev_hash: str, token_chunk: List[int]) -> str:
        """
        Computes a chained SHA-256 hash for a 16-token chunk.
        Chaining ensures prefix ordering: H_i = hash(H_{i-1} + tokens_i).
        """
        hasher = hashlib.sha256()
        hasher.update(prev_hash.encode("utf-8"))
        hasher.update(bytes(str(token_chunk), "utf-8"))
        return hasher.hexdigest()[:16]

    def match_prefix(self, prompt_tokens: List[int], allocator: BlockAllocator) -> Tuple[List[int], int]:
        """
        Scans prompt tokens in 16-token chunks against the cache.
        Returns:
            (matched_physical_block_ids, total_tokens_matched)
        """
        matched_blocks = []
        tokens_matched = 0
        prev_hash = "ROOT"

        num_full_blocks = len(prompt_tokens) // self.block_size
        for i in range(num_full_blocks):
            chunk = prompt_tokens[i * self.block_size : (i + 1) * self.block_size]
            b_hash = self.compute_block_hash(prev_hash, chunk)

            if b_hash in self.cache:
                phys_id = self.cache[b_hash]
                block = allocator.blocks[phys_id]

                # Ensure block is still valid and not corrupted
                if block.prefix_hash == b_hash:
                    # Move to end of OrderedDict for LRU tracking
                    self.cache.move_to_end(b_hash)
                    block.touch()
                    matched_blocks.append(phys_id)
                    tokens_matched += self.block_size
                    prev_hash = b_hash
                else:
                    # Stale entry
                    del self.cache[b_hash]
                    break
            else:
                break

        if matched_blocks:
            self.hits += 1
        else:
            self.misses += 1

        return matched_blocks, tokens_matched

    def register_prefix_blocks(
        self,
        prompt_tokens: List[int],
        physical_block_ids: List[int],
        allocator: BlockAllocator
    ) -> int:
        """
        Registers completed, full prefix blocks into the LRU cache.
        Marks blocks as immutable and assigns their prefix hashes.
        Returns number of newly registered blocks.
        """
        registered_count = 0
        prev_hash = "ROOT"
        num_blocks = min(len(prompt_tokens) // self.block_size, len(physical_block_ids))

        for i in range(num_blocks):
            chunk = prompt_tokens[i * self.block_size : (i + 1) * self.block_size]
            b_hash = self.compute_block_hash(prev_hash, chunk)
            phys_id = physical_block_ids[i]

            block = allocator.blocks[phys_id]
            block.prefix_hash = b_hash
            block.is_immutable = True
            block.touch()

            # Evict cold entries if cache exceeds capacity
            if len(self.cache) >= self.max_cached_blocks and b_hash not in self.cache:
                self.evict_coldest(allocator)

            self.cache[b_hash] = phys_id
            self.cache.move_to_end(b_hash)
            registered_count += 1
            prev_hash = b_hash

        return registered_count

    def evict_coldest(self, allocator: BlockAllocator) -> Optional[int]:
        """
        Evicts the oldest unreferenced prefix block (ref_count == 0) via LRU.
        Returns evicted physical block ID or None if no blocks can be evicted.
        """
        # Iterate from oldest to newest in OrderedDict
        for b_hash, phys_id in list(self.cache.items()):
            block = allocator.blocks[phys_id]
            # Only evict if no active sequence is currently using it
            if block.ref_count <= 0:
                del self.cache[b_hash]
                allocator.free(phys_id)
                return phys_id
        return None

    @property
    def hit_rate_pct(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return round((self.hits / total) * 100.0, 2)

    def get_status(self) -> Dict[str, Any]:
        return {
            "cached_prefix_blocks": len(self.cache),
            "max_capacity_blocks": self.max_cached_blocks,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": self.hit_rate_pct
        }
