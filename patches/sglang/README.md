# SGLang Patches for ContextPilot Integration

This directory contains patched SGLang files required for ContextPilot integration.

## Installation

### Option 1: Copy Files (Recommended)

Copy the patched files to your SGLang installation:

```bash
# Find your SGLang installation path
SGLANG_PATH=$(python -c "import sglang; print(sglang.__path__[0])")
echo "SGLang path: $SGLANG_PATH"

# Backup original files (recommended)
cp $SGLANG_PATH/srt/mem_cache/cache_init_params.py $SGLANG_PATH/srt/mem_cache/cache_init_params.py.bak
cp $SGLANG_PATH/srt/mem_cache/common.py $SGLANG_PATH/srt/mem_cache/common.py.bak
cp $SGLANG_PATH/srt/mem_cache/radix_cache.py $SGLANG_PATH/srt/mem_cache/radix_cache.py.bak

# Copy patched files
cp cache_init_params.py $SGLANG_PATH/srt/mem_cache/
cp common.py $SGLANG_PATH/srt/mem_cache/
cp radix_cache.py $SGLANG_PATH/srt/mem_cache/

echo "SGLang patched successfully!"
```

### Option 2: Use Symbolic Links (Development)

For development, you can symlink to the patched files:

```bash
SGLANG_PATH=$(python -c "import sglang; print(sglang.__path__[0])")
RAGBOOST_PATCHES=$(pwd)

# Backup and link
mv $SGLANG_PATH/srt/mem_cache/cache_init_params.py $SGLANG_PATH/srt/mem_cache/cache_init_params.py.bak
mv $SGLANG_PATH/srt/mem_cache/common.py $SGLANG_PATH/srt/mem_cache/common.py.bak  
mv $SGLANG_PATH/srt/mem_cache/radix_cache.py $SGLANG_PATH/srt/mem_cache/radix_cache.py.bak

ln -s $RAGBOOST_PATCHES/cache_init_params.py $SGLANG_PATH/srt/mem_cache/
ln -s $RAGBOOST_PATCHES/common.py $SGLANG_PATH/srt/mem_cache/
ln -s $RAGBOOST_PATCHES/radix_cache.py $SGLANG_PATH/srt/mem_cache/
```

## What's Changed

### 1. cache_init_params.py

- Added `EvictionCallback` type alias
- Added `eviction_callback` field to `CacheInitParams` dataclass
- Added `__post_init__` to auto-create callback from `RAGBOOST_INDEX_URL` env var

### 2. common.py

- Added `RAGBOOST_INDEX_URL` environment variable support
- Added `create_contextpilot_eviction_callback()` function
- Modified `evict_from_tree_cache()` to note callback is used

### 3. radix_cache.py

- Added `request_ids: set` to `TreeNode` for tracking request IDs
- Added `eviction_callback` parameter to `RadixCache`
- Added `_request_to_node` dict for request-to-node mapping
- Modified `insert()` to track request_id at leaf nodes
- Modified `evict()` to:
  - Collect evicted request IDs
  - Invoke callback with evicted IDs
- Added helper methods: `set_eviction_callback()`, `get_tracked_request_ids()`, `is_request_in_cache()`

## Usage

After installing the patches, start SGLang with the `RAGBOOST_INDEX_URL` environment variable:

```bash
# Start ContextPilot server first
python -m contextpilot.server.http_server --port 8765

# Start SGLang with ContextPilot integration
RAGBOOST_INDEX_URL=http://localhost:8765 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B \
    --port 30000
```

## Compatibility

These patches are compatible with SGLang version: **0.4.x** (tested with 0.4.6)

If you're using a different version, you may need to manually apply the changes.

## Reverting Changes

To revert to original SGLang:

```bash
SGLANG_PATH=$(python -c "import sglang; print(sglang.__path__[0])")

cp $SGLANG_PATH/srt/mem_cache/cache_init_params.py.bak $SGLANG_PATH/srt/mem_cache/cache_init_params.py
cp $SGLANG_PATH/srt/mem_cache/common.py.bak $SGLANG_PATH/srt/mem_cache/common.py
cp $SGLANG_PATH/srt/mem_cache/radix_cache.py.bak $SGLANG_PATH/srt/mem_cache/radix_cache.py
```
