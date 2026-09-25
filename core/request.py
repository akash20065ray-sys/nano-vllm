import time
from enum import Enum
from typing import List, Optional, Dict, Any

class SequenceStatus(str, Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    PREEMPTED = "PREEMPTED"
    COMPLETED = "COMPLETED"

class Sequence:
    """
    Represents an individual inference request throughout its lifecycle in the serving engine.
    Tracks token sequence, logical memory block ownership, timestamps, and latency metrics.
    """
    def __init__(
        self,
        seq_id: int,
        prompt_tokens: List[int],
        max_output_tokens: int = 64,
        arrival_time: Optional[float] = None
    ):
        self.seq_id = seq_id
        self.prompt_tokens = list(prompt_tokens)
        self.output_tokens: List[int] = []
        self.max_output_tokens = max_output_tokens
        self.status = SequenceStatus.WAITING

        # Timestamps for micro-profiling
        self.arrival_time = arrival_time if arrival_time is not None else time.perf_counter()
        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.finish_time: Optional[float] = None

        # Per-token generation latency history (Inter-Token Latency)
        self.inter_token_latencies: List[float] = []
        self._last_token_timestamp: Optional[float] = None

        # Memory mapping state
        self.logical_block_count: int = 0
        self.prefix_cached_blocks: int = 0
        self.preemption_count: int = 0

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def is_finished(self) -> bool:
        if self.status == SequenceStatus.COMPLETED:
            return True
        if len(self.output_tokens) >= self.max_output_tokens:
            return True
        if self.output_tokens and self.output_tokens[-1] == 2:
            return True
        return False

    def mark_started(self):
        self.status = SequenceStatus.RUNNING
        self.start_time = time.perf_counter()
        self._last_token_timestamp = self.start_time

    def append_token(self, token_id: int):
        now = time.perf_counter()
        if not self.output_tokens:
            self.first_token_time = now
        else:
            if self._last_token_timestamp is not None:
                self.inter_token_latencies.append((now - self._last_token_timestamp) * 1000.0)

        self._last_token_timestamp = now
        self.output_tokens.append(token_id)

        if self.is_finished:
            self.mark_completed()

    def mark_preempted(self):
        self.status = SequenceStatus.PREEMPTED
        self.preemption_count += 1
        self._last_token_timestamp = None

    def mark_completed(self):
        self.status = SequenceStatus.COMPLETED
        self.finish_time = time.perf_counter()

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time To First Token in milliseconds"""
        if self.first_token_time and self.start_time:
            return (self.first_token_time - self.start_time) * 1000.0
        return None

    @property
    def queue_latency_ms(self) -> Optional[float]:
        """Time spent waiting in queue prior to admission"""
        if self.start_time and self.arrival_time:
            return (self.start_time - self.arrival_time) * 1000.0
        return None

    @property
    def mean_itl_ms(self) -> Optional[float]:
        """Mean Inter-Token Latency in milliseconds"""
        if not self.inter_token_latencies:
            return None
        return sum(self.inter_token_latencies) / len(self.inter_token_latencies)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq_id": self.seq_id,
            "status": self.status.value,
            "prompt_len": len(self.prompt_tokens),
            "output_len": len(self.output_tokens),
            "total_len": self.total_tokens,
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
            "queue_latency_ms": round(self.queue_latency_ms, 2) if self.queue_latency_ms is not None else None,
            "mean_itl_ms": round(self.mean_itl_ms, 2) if self.mean_itl_ms is not None else None,
            "prefix_cached_blocks": self.prefix_cached_blocks,
            "preemption_count": self.preemption_count
        }
