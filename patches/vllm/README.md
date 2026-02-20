# vLLM Patches for ContextPilot Integration

This directory contains patched vLLM files required for ContextPilot eviction sync.

## Installation

### Option 1: Automated (Recommended)

```bash
bash patches/vllm/apply_patch.sh
```

### Option 2: Manual Copy

```bash
# Find your vLLM installation path
VLLM_PATH=$(python -c "import vllm; print(vllm.__path__[0])")
echo "vLLM path: $VLLM_PATH"

# Backup original file
cp $VLLM_PATH/v1/core/block_pool.py $VLLM_PATH/v1/core/block_pool.py.bak

# Copy patched file
cp block_pool.py $VLLM_PATH/v1/core/
```

### Option 3: Symbolic Link (Development)

```bash
VLLM_PATH=$(python -c "import vllm; print(vllm.__path__[0])")
CONTEXTPILOT_PATCHES=$(pwd)

mv $VLLM_PATH/v1/core/block_pool.py $VLLM_PATH/v1/core/block_pool.py.bak
ln -s $CONTEXTPILOT_PATCHES/block_pool.py $VLLM_PATH/v1/core/
```

## What's Changed

### block_pool.py

- Added `create_contextpilot_eviction_callback()` factory (reads `CONTEXTPILOT_INDEX_URL` env var)
- Added `_block_to_requests` dict: tracks `block_hash → set(request_ids)`
- Added `_request_to_blocks` dict: tracks `request_id → set(block_hashes)`
- Modified `cache_full_blocks()` to record request-to-block mappings
- Modified `_maybe_evict_cached_block()` to detect fully-evicted requests
- Modified `get_new_blocks()` to batch and fire eviction callback
- Modified `evict_blocks()` and `reset_prefix_cache()` to fire callback
- Added helper methods: `get_tracked_request_ids()`, `is_request_in_cache()`, `set_eviction_callback()`

## Usage

After installing the patches, start vLLM with the `CONTEXTPILOT_INDEX_URL` environment variable:

```bash
# Start ContextPilot server first
python -m contextpilot.server.http_server --port 8765

# Start vLLM with ContextPilot integration
CONTEXTPILOT_INDEX_URL=http://localhost:8765 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```

When `CONTEXTPILOT_INDEX_URL` is not set, the patch has zero overhead — no tracking dicts are populated and no callbacks fire.

## Compatibility

These patches are compatible with vLLM version: **0.8.x** (v1 block manager architecture).

If you're using a different version, you may need to manually apply the changes.

## Reverting Changes

```bash
VLLM_PATH=$(python -c "import vllm; print(vllm.__path__[0])")
cp $VLLM_PATH/v1/core/block_pool.py.bak $VLLM_PATH/v1/core/block_pool.py
```
