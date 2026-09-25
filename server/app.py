import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.block_manager import BlockAllocator
from core.page_table import PageTable
from core.prefix_cache import PrefixCache

app = FastAPI(title="MicroServe-LLM Observability Server", version="1.0.0")

# Global systems state
TOTAL_BLOCKS = 128
BLOCK_SIZE = 16

allocator = BlockAllocator(num_blocks=TOTAL_BLOCKS, block_size=BLOCK_SIZE)
prefix_cache = PrefixCache(block_size=BLOCK_SIZE)
active_page_tables: Dict[int, PageTable] = {}
seq_id_counter = 1

# Mount dashboard static assets
DASHBOARD_DIR = ROOT_DIR / "dashboard"
app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_file = DASHBOARD_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

@app.get("/styles.css")
async def serve_styles():
    css_file = DASHBOARD_DIR / "styles.css"
    return HTMLResponse(content=css_file.read_text(encoding="utf-8"), media_type="text/css")

@app.get("/app.js")
async def serve_app_js():
    js_file = DASHBOARD_DIR / "app.js"
    return HTMLResponse(content=js_file.read_text(encoding="utf-8"), media_type="application/javascript")

@app.get("/api/health")
async def health_check():
    device_name = "CUDA:0 (RTX 3050)" if torch.cuda.is_available() else "Host CPU"
    return {
        "status": "ONLINE",
        "device": device_name,
        "cuda_available": torch.cuda.is_available(),
        "block_size": BLOCK_SIZE,
        "total_blocks": TOTAL_BLOCKS
    }

@app.get("/api/telemetry")
async def get_telemetry():
    alloc_status = allocator.get_status()
    cache_status = prefix_cache.get_status()
    return {
        "allocator": alloc_status,
        "prefix_cache": cache_status,
        "active_sequences": len(active_page_tables)
    }

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 48

@app.post("/api/generate")
async def generate_stream(req: PromptRequest):
    global seq_id_counter
    seq_id = seq_id_counter
    seq_id_counter += 1

    pt = PageTable(seq_id=seq_id, block_size=BLOCK_SIZE)
    active_page_tables[seq_id] = pt

    prompt_tokens = [ord(c) for c in req.prompt]
    
    # Check Prefix Cache
    matched_blocks, matched_tokens = prefix_cache.match_prefix(prompt_tokens, allocator)
    for bid in matched_blocks:
        pt.assign_prefix_block(bid, allocator)

    # Allocate remainder
    unmatched_tokens = len(prompt_tokens) - matched_tokens
    for _ in range(unmatched_tokens):
        pt.append_slot(allocator)

    # Register completed prefix blocks
    if matched_tokens == 0 and len(pt.logical_to_physical) >= 2:
        prefix_cache.register_prefix_blocks(prompt_tokens[:32], pt.logical_to_physical[:2], allocator)

    # Simulate SSE word-by-word streaming
    words = req.prompt.split() + ["is", "processed", "efficiently", "via", "MicroServe-LLM", "PagedAttention."]

    async def event_generator():
        for word in words:
            pt.append_slot(allocator)
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.04)
        yield "data: [DONE]\n\n"
        # Reclaim
        pt.free_all(allocator)
        active_page_tables.pop(seq_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/reclaim")
async def reclaim_all():
    for pt in list(active_page_tables.values()):
        pt.free_all(allocator)
    active_page_tables.clear()
    return {"status": "RECLAIMED", "free_blocks": allocator.num_free_blocks}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
