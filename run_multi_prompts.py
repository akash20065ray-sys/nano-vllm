#!/usr/bin/env python3
"""
MicroServe-LLM / nano-vllm: Concurrent Multi-Prompt Continuous Batching Runner
Accepts multiple prompts from CLI or interactive input, submits them concurrently
into the iteration-level continuous scheduler, and displays real-time interleaved
streaming tokens with dynamic sequence retirement.

Usage:
  python run_multi_prompts.py
  python run_multi_prompts.py "Explain virtual memory paging" "What is GPU coalescing?" "Write quicksort in Python"
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Ensure repository root is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from core.request import Sequence, SequenceStatus
from core.scheduler import ContinuousScheduler

DEFAULT_PROMPTS = [
    "Explain how virtual memory paging eliminates external memory fragmentation.",
    "What is the difference between CPU cache lines and GPU memory coalescing?",
    "Write a high-performance Python function for binary search.",
    "Explain how Copy-On-Write enables zero-copy prompt prefix sharing across LLM requests."
]

def format_tokens_display(text: str, max_chars: int = 55) -> str:
    cleaned = text.replace("\n", " ").strip()
    return cleaned if len(cleaned) <= max_chars else cleaned[:max_chars - 3] + "..."

def run_concurrent_prompts(prompts: List[str], max_tokens_per_stream: int = 36):
    print("=" * 80)
    print("  MICROSERVE-LLM: CONCURRENT MULTI-PROMPT CONTINUOUS BATCHING RUNNER")
    print("=" * 80)
    print(f"-> Concurrency Load: {len(prompts)} Parallel Client Requests")
    print(f"-> Scheduling Mode : Iteration-Level Continuous Batching (O(1) Paged KV-Cache)")
    print(f"-> Max Tokens/Seq  : {max_tokens_per_stream} tokens\n")

    scheduler = ContinuousScheduler(
        num_gpu_blocks=128,
        num_cpu_blocks=256,
        block_size=16,
        max_batch_size=16
    )

    sequences: List[Sequence] = []
    stream_outputs: Dict[int, List[str]] = {}
    stream_prompts: Dict[int, str] = {}
    stream_expected_len: Dict[int, int] = {}
    stream_start_time: Dict[int, float] = {}
    stream_finish_time: Dict[int, float] = {}
    stream_ttft: Dict[int, float] = {}

    # Sample token vocabulary for simulated generation
    mock_responses = {
        0: "Virtual memory paging partitions logical address spaces into fixed-size physical pages, completely eliminating external memory fragmentation while page tables translate addresses dynamically.".split(),
        1: "GPU memory coalescing merges adjacent thread memory loads into single 128-byte transactions, maximizing DRAM bus bandwidth and avoiding compute stalls.".split(),
        2: "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1".split(),
        3: "Copy-On-Write allows multiple generation streams to share immutable prompt KV blocks with atomic reference counting, duplicating pages only when child sequences mutate attention slots.".split()
    }

    t0_global = time.perf_counter()

    for i, p in enumerate(prompts):
        seq_id = i + 1
        prompt_tokens = [ord(c) for c in p[:48]]
        # Assign varying output lengths to demonstrate uneven sequence completion
        expected_len = min(max_tokens_per_stream, len(mock_responses.get(i % len(mock_responses), [])) or 24)
        if i == 1:
            expected_len = 14  # Short sequence finishes early!
        elif i == 0:
            expected_len = 22  # Medium sequence
        else:
            expected_len = 30  # Long sequence

        seq = Sequence(
            seq_id=seq_id,
            prompt_tokens=prompt_tokens,
            max_output_tokens=expected_len,
            arrival_time=t0_global
        )
        sequences.append(seq)
        stream_outputs[seq_id] = []
        stream_prompts[seq_id] = p
        stream_expected_len[seq_id] = expected_len
        stream_start_time[seq_id] = t0_global

        scheduler.add_request(seq)
        print(f"  [Arrival] Req #{seq_id:02d} | Target Length: {expected_len:02d} tok | Prompt: \"{format_tokens_display(p)}\"")

    print("\n" + "-" * 80)
    print("  COMMENCING CONTINUOUS BATCHING EXECUTION LOOP")
    print("-" * 80)

    step_num = 0
    while scheduler.has_unfinished_requests:
        step_num += 1
        step_t0 = time.perf_counter()

        # Generate next token for active sequences
        next_tokens_map = {}
        for seq in scheduler.running_batch:
            resp_words = mock_responses.get((seq.seq_id - 1) % len(mock_responses), [])
            word_idx = len(seq.output_tokens)
            next_word = resp_words[word_idx] if word_idx < len(resp_words) else f"token_{word_idx}"
            next_tokens_map[seq.seq_id] = hash(next_word) % 50000
            stream_outputs[seq.seq_id].append(next_word)

            if seq.seq_id not in stream_ttft:
                stream_ttft[seq.seq_id] = (time.perf_counter() - stream_start_time[seq.seq_id]) * 1000.0

        output = scheduler.step(next_tokens=next_tokens_map)

        # Log newly finished sequences retiring immediately from batch
        if output.newly_finished_seq_ids:
            for fid in output.newly_finished_seq_ids:
                stream_finish_time[fid] = time.perf_counter()
                print(f"  -> [Step {step_num:02d}] Seq #{fid:02d} HIT EOS! -> RETIRED IMMEDIATELY -> VRAM Blocks Reclaimed! (Active batch: {len(output.running_seq_ids)})")

        # Brief delay to simulate GPU compute step
        time.sleep(0.04)

    total_elapsed = time.perf_counter() - t0_global
    total_tokens_all = sum(len(toks) for toks in stream_outputs.values())

    print("\n" + "=" * 80)
    print("  MULTI-STREAM GENERATION COMPLETED - RESPONSE INSPECTION")
    print("=" * 80)

    for seq in sequences:
        output_text = " ".join(stream_outputs[seq.seq_id])
        print(f"\n[Seq #{seq.seq_id:02d}] Prompt: \"{stream_prompts[seq.seq_id]}\"")
        print(f"  Response: {output_text}")

    print("\n" + "=" * 80)
    print("  QUANTITATIVE SCHEDULING & LATENCY REPORT")
    print("=" * 80)
    print(f"{'Seq ID':<8} | {'Prompt Tok':<12} | {'Gen Tok':<10} | {'TTFT (ms)':<12} | {'Duration (s)':<14} | {'Throughput':<12}")
    print("-" * 80)

    for seq in sequences:
        sid = seq.seq_id
        gen_count = len(stream_outputs[sid])
        prompt_count = len(seq.prompt_tokens)
        duration = stream_finish_time.get(sid, total_elapsed) - stream_start_time[sid]
        ttft = stream_ttft.get(sid, 0.85)
        tps = gen_count / max(0.001, duration)

        print(f"#{sid:<7} | {prompt_count:<12} | {gen_count:<10} | {ttft:<12.2f} | {duration:<14.3f} | {tps:<10.1f} tok/s")

    print("-" * 80)
    overall_tps = total_tokens_all / max(0.001, total_elapsed)
    print(f"Total Streamed Tokens : {total_tokens_all}")
    print(f"Total Wall Clock Time : {total_elapsed:.3f} s")
    print(f"System-Wide Throughput: {overall_tps:.2f} tokens/second")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="MicroServe-LLM Concurrent Multi-Prompt Runner")
    parser.add_argument("prompts", nargs="*", help="Optional list of prompts to run in concurrent continuous batch")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max new tokens to generate per stream")
    args = parser.parse_args()

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    run_concurrent_prompts(prompts, max_tokens_per_stream=args.max_tokens)

if __name__ == "__main__":
    main()
