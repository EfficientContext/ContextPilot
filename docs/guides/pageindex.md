# PageIndex Integration

[PageIndex](https://github.com/yinsicheng/PageIndex) builds a hierarchical tree from documents (e.g., PDFs, earnings reports). ContextPilot consumes that tree output to schedule multiple RAG queries for maximum KV-cache prefix sharing.

## How It Works

```
┌────────────┐      ┌──────────────┐      ┌──────────────┐      ┌────────────┐
│  Document  │ ──▸  │ PageIndex    │ ──▸  │ ContextPilot │ ──▸  │ LLM Engine │
│  (PDF)     │      │ (tree + IDs) │      │ (schedule)   │      │ (radix KV) │
└────────────┘      └──────────────┘      └──────────────┘      └────────────┘
```

1. **PageIndex** parses a document into a tree of titled, summarized nodes.
2. Per-query **tree search** (LLM or keyword) returns relevant node IDs.
3. **ContextPilot** takes the list of node-ID lists, clusters queries with overlapping nodes, reorders documents within each context so shared nodes form the longest common prefix, and schedules execution order.
4. The **LLM engine** (e.g., SGLang or vLLM with prefix caching) caches the shared prefix and reuses it across consecutive requests.

## Quick Start

### Demo (no API key)

```bash
python examples/pageindex_e2e_example.py
```

This runs 6 analyst queries against the bundled Disney Q1 FY25 earnings tree (41 nodes, `examples/data/disney_q1_fy25_tree.json`) and prints:

- Overlap analysis (which nodes are shared)
- Scheduled execution order with LCP bars
- Prefix sharing comparison: ContextPilot vs Naive vs Random

### Generate a Tree from Your Own Document

Use [PageIndex](https://github.com/yinsicheng/PageIndex) to build a tree from your PDF:

```bash
pip install pageindex
```

See the [PageIndex documentation](https://github.com/yinsicheng/PageIndex) for tree generation usage. The output JSON can be passed directly to the demo or the full pipeline.

### Full Pipeline (tree search + answer generation)

The full pipeline uses `PageIndexRetriever` for LLM-based tree search:

```bash
pip install openai
export OPENAI_API_KEY="your-key"

python examples/pageindex_e2e_example.py \
    --tree path/to/my_report_tree.json \
    -q "What was DTC revenue?" \
    -q "How did ESPN perform?" \
    -q "What is the FY25 CapEx guidance?"
```

## Python API

```python
import contextpilot as cp

# Each context = list of node IDs from PageIndex tree search.
# Important: doc order is typically NOT pre-sorted — shared nodes
# may appear anywhere in the list (just like real retrieval results).
contexts = [
    [8, 31, 2, 1],          # shared 1,2 buried at end
    [29, 5, 6, 3],          # no overlap with others
    [14, 12, 1, 10, 2],    # shared 1,2,10 scattered
    [20, 10, 2, 1],         # shared 1,2,10
    [15, 12, 1, 2],         # shared 1,2
    [17, 16, 2, 10, 1],    # shared 1,2,10 scattered
]

# One call: cluster, reorder, and schedule
engine = cp.ContextPilot(use_gpu=False)
reordered, order = engine.reorder(contexts)

# reordered[i]  = reordered doc IDs for the i-th scheduled query
# order[i]      = index into the original `contexts` list
```

### Using with the HTTP Server

```python
import requests

contexts = [[8, 31, 2, 1], [29, 5, 6, 3], [14, 12, 1, 10, 2], [20, 10, 2, 1], [15, 12, 1, 2], [17, 16, 2, 10, 1]]

# Stateless scheduling
resp = requests.post(
    "http://localhost:8765/reorder",
    json={"contexts": contexts}
).json()

scheduled_order = resp["original_indices"]
reordered = resp["reordered_contexts"]
```

## Expected Output

```
======================================================================
  PageIndex + ContextPilot Demo
  Document: q1-fy25-earnings.pdf
  Nodes: 41
======================================================================

  Queries (6):
    Revenue & EPS growth                -> nodes [8, 31, 2, 1]
    FY2025 outlook & CapEx              -> nodes [29, 5, 6, 3]
    Streaming (DTC) performance         -> nodes [14, 12, 1, 10, 2]
    ...

  Scheduled execution order:
    [2] Streaming (DTC) performance   docs=[1, 2, 12, 14, 10]  LCP=0   ← reordered
    [4] Content licensing results     docs=[1, 2, 12, 15]       LCP=3  █████████ ← reordered
    [3] Theme parks performance       docs=[1, 2, 10, 20]       LCP=2  ██████ ← reordered
    [5] ESPN & Sports results         docs=[1, 2, 10, 17, 16]   LCP=3  █████████ ← reordered
    [0] Revenue & EPS growth          docs=[1, 2, 8, 31]        LCP=2  ██████ ← reordered
    [1] FY2025 outlook & CapEx        docs=[29, 5, 6, 3]        LCP=0

  Prefix sharing (Longest Common Prefix):
    ContextPilot             10       16    38.5%       n/a
    Naive                     0       26     0.0%       n/a
    Random (avg)              0       26     0.0%   baseline
```
