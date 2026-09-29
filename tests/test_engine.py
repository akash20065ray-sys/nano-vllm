import pytest
from core.engine import NanoVLLMEngine

def test_engine_initialization():
    engine = NanoVLLMEngine(num_blocks=64, block_size=16)
    assert engine.num_blocks == 64
    assert engine.block_size == 16
    assert engine.allocator.num_free_blocks == 64

def test_engine_real_generation_stream():
    engine = NanoVLLMEngine(num_blocks=32, block_size=16)
    events = list(engine.generate_real_stream("Hello", max_new_tokens=4))
    
    assert len(events) >= 3 # start + at least 1 token + done
    assert events[0]["type"] == "start"
    assert events[0]["engine_mode"] == "REAL_NEURAL_WEIGHTS"
    assert "prompt_tokens" in events[0]
    
    # Check token events
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) >= 1
    assert "token" in token_events[0]
    assert "vocab_id" in token_events[0]
    assert "top_candidates" in token_events[0]
    assert "physical_block_id" in token_events[0]
    
    # Check done event
    assert events[-1]["type"] == "done"
    assert events[-1]["total_tokens"] > 0

def test_engine_free_sequence():
    engine = NanoVLLMEngine(num_blocks=32, block_size=16)
    events = list(engine.generate_real_stream("Count 1 2 3", max_new_tokens=4))
    seq_id = events[0]["seq_id"]
    
    assert engine.allocator.num_free_blocks < 32
    engine.free_sequence(seq_id)
    assert engine.allocator.num_free_blocks == 32
