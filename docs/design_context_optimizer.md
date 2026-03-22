# ContextPilot → Context Optimizer: Architecture Redesign

## TL;DR

把 ContextPilot 重构为 **Context Optimizer** — 类比 SQL Query Optimizer。
用户只管写 prompt，Optimizer 透明地处理一切（Dedup, Add, Repartition, Reorder）。

**用户代码变化**：

```python
# Before (当前 — 要理解 pipeline, retriever, optimizer config...)
from contextpilot.pipeline import RAGPipeline, RetrieverConfig, OptimizerConfig
pipeline = RAGPipeline(retriever="bm25", corpus_path="corpus.jsonl",
                       optimizer=OptimizerConfig(enabled=True, use_gpu=True))
results = pipeline.run(queries=["What is AI?"], generate_responses=True)

# After (目标 — 一行即用)
import contextpilot as cp
response = cp.openai.ChatCompletions.create(model="gpt-4", messages=messages)
# 或
messages = cp.optimize(messages)
```

---

## 1. SQL Query Optimizer 类比

```
 SQL World                          Context World
 ═══════════════════════════════    ═══════════════════════════════════

 SELECT * FROM t WHERE x=1         messages = [{role, content}, ...]
           │                                     │
           ▼                                     ▼
 ┌─────────────────────┐           ┌──────────────────────────────┐
 │     SQL Parser       │           │     Context Parser            │
 │  SQL → Parse Tree    │           │  Prompt → ContextBlockTree    │
 └────────┬────────────┘           │  (skill/memory/search/file)   │
          │                        └──────────────┬───────────────┘
          ▼                                       ▼
 ┌─────────────────────┐           ┌──────────────────────────────┐
 │    Catalog/Stats     │           │    Catalog / Usage Stats      │
 │  table size, indexes │           │  block freq, recency, tokens  │
 │  column cardinality  │           │  skill registry, cache state  │
 └────────┬────────────┘           └──────────────┬───────────────┘
          │                                       ▼
          ▼                        ┌──────────────────────────────┐
  ┌─────────────────────┐           │       4 Primitives (AI-free)   │
  │   Rewrite Rules      │           │  • Dedup       (Distinct δ)   │
  │  • predicate pushdown│           │  • Add         (Union ∪)      │
  │  • subquery unnest   │           │  • Repartition (Exchange ∥)   │
  │  • view merging      │           │  • Reorder     (Sort τ, LAST) │
  │  • constant folding  │           │                                │
  └────────┬────────────┘           └──────────────┬───────────────┘
          │                                       ▼
          ▼                        ┌──────────────────────────────┐
 ┌─────────────────────┐           │       Cost Model              │
 │     Cost Model       │           │  token_cost(block)            │
 │  row estimates       │           │  prefix_hit(add/reorder)      │
 │  I/O cost            │           │  latency_benefit(parallel)     │
 └────────┬────────────┘           └──────────────┬───────────────┘
          │                                       ▼
          ▼                        ┌──────────────────────────────┐
 ┌─────────────────────┐           │       Plan Generator          │
 │   Plan Generator     │           │  选最优 rule 组合             │
 │  join order, index   │           │  → OptimizationPlan           │
 └────────┬────────────┘           └──────────────┬───────────────┘
          │                                       ▼
          ▼                        ┌──────────────────────────────┐
 ┌─────────────────────┐           │       Executor                │
 │    Executor          │           │  apply plan → optimized body  │
 │  scan, join, sort    │           │  reorder, inject, trim        │
 └─────────────────────┘           └──────────────────────────────┘

 EXPLAIN SELECT ...                 co.explain(messages)
   → 显示执行计划                    → 显示优化计划
   → cost estimates                  → token savings / cache hit rate

 CREATE INDEX                       co.register_skill(path)
   → 建索引加速查询                  → 注册 skill 加速 predictive add

 ANALYZE table                      co.analyze()
   → 更新统计信息                    → 更新 block usage 统计
```

---

## 2. Architecture Diagram

```
                           ┌─────────────────────────────────────────┐
                           │          User Application               │
                           │                                         │
                           │  messages = [{role:"system", ...},      │
                           │              {role:"user", ...}]        │
                           └──────────────┬──────────────────────────┘
                                          │
                 ╔════════════════════════╤╧═══════════════════════════╗
                 ║     Integration Layer  │  (pick one, zero/one-line) ║
                 ╠═══════════╤═══════════╪═══════════╤════════════════╣
                 ║  Client   │  HTTP     │  SDK      │   Hook         ║
                 ║  Wrapper  │  Proxy    │  Direct   │   (engine)     ║
                 ║           │           │           │                ║
                 ║  cp.openai│ $cp serve │cp.optimize│ sglang/vllm    ║
                 ║  .create()│ :8765     │(messages) │ auto-patch     ║
                 ╚═══════╤═══╧═══════╤═══╧═════╤═════╧════════╤═══════╝
                         │           │         │              │
                         └─────────┬─┘─────────┘──────────────┘
                                   │
          ╔════════════════════════╧═══════════════════════════════════╗
          ║                 CONTEXT OPTIMIZER CORE                      ║
          ║                                                            ║
          ║  ┌──────────┐   ┌───────────┐   ┌───────────┐             ║
          ║  │ 1.PARSER │──▶│ 2.ANALYZER│──▶│ 3.PLANNER │             ║
          ║  └──────────┘   └───────────┘   └───────────┘             ║
          ║       │              │                │                    ║
          ║       │    ┌─────────┴──────────┐     │                    ║
          ║       │    │                    │     │                    ║
          ║       ▼    ▼                    ▼     ▼                    ║
          ║  ┌──────────┐  ┌────────────┐  ┌───────────┐              ║
          ║  │ CATALOG   │  │ COST MODEL │  │  REWRITE  │              ║
          ║  │           │  │            │  │  RULES    │              ║
           ║  │• skills   │  │• token_cost│  │           │              ║
           ║  │• memory   │  │• cache_hit │  │• dedup    │              ║
           ║  │• usage    │  │• latency   │  │• add      │              ║
           ║  │  stats    │  │  benefit   │  │• repartit.│              ║
           ║  │• block    │  │            │  │• reorder  │              ║
           ║  │  registry │  │            │  │  (LAST)   │              ║
           ║  └──────────┘  └────────────┘  └───────────┘              ║
           ║                                     │                      ║
          ║                    ┌─────────────────┘                      ║
          ║                    ▼                                        ║
          ║              ┌───────────┐                                  ║
          ║              │4.EXECUTOR │                                  ║
          ║              │  apply    │                                  ║
          ║              │  plan     │                                  ║
          ║              └─────┬─────┘                                  ║
          ╚════════════════════╧════════════════════════════════════════╝
                               │
          ╔════════════════════╧════════════════════════════════════════╗
          ║                 EXECUTION ENGINE                            ║
          ║                                                            ║
          ║  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐      ║
          ║  │ Cache Index   │  │ Scheduler   │  │ Eviction     │      ║
          ║  │ (prefix tree) │  │ (inter/     │  │ Manager      │      ║
          ║  │              │  │  intra)      │  │              │      ║
          ║  └──────────────┘  └─────────────┘  └──────────────┘      ║
          ╚════════════════════�════════════════════════════════════════╝
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LLM Backend       │
                    │  (OpenAI/Anthropic/ │
                    │   SGLang/vLLM/etc)  │
                    └─────────────────────┘
```

---

## 3. User-Facing API Design (即插即用)

### Level 0: Zero-Code (HTTP Proxy)

```bash
# 启动 proxy，指向你的 LLM backend
$ contextpilot serve --backend https://api.openai.com --port 8765

# 你的代码零修改，只改 base_url
client = OpenAI(base_url="http://localhost:8765/v1")
response = client.chat.completions.create(model="gpt-4", messages=messages)
# ContextOptimizer 自动 intercept → optimize → forward
```

### Level 1: Client Wrapper (一行改动)

```python
import contextpilot as cp

# 替换 openai 的 client — 其他代码零修改
client = cp.openai.Client(api_key="sk-...")

# 就这样，所有 chat.completions.create 自动优化
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages  # ← Optimizer 自动 dedup/add/reorder
)
```

```python
# Anthropic 同理
client = cp.anthropic.Client(api_key="sk-ant-...")
response = client.messages.create(model="claude-4-sonnet", messages=messages)
```

### Level 2: Explicit Optimize (两行)

```python
import contextpilot as cp

# 手动 optimize，用自己的 client 发送
optimized = cp.optimize(messages)
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=optimized
)
```

### Level 3: Full Control

```python
import contextpilot as cp

optimizer = cp.ContextOptimizer(
    rules=["dedup", "add", "repartition", "reorder"],
    token_budget=128_000,
    skill_dirs=["~/.config/opencode/skills/"],
)

# EXPLAIN — 看优化计划（不执行）
plan = optimizer.explain(messages)
print(plan)
# ╔══════════════════════════════════════════════╗
# ║ Context Optimization Plan                    ║
# ╠══════════════════════════════════════════════╣
# ║ Input:  47,832 tokens (12 blocks)            ║
# ║ Output: 31,206 tokens (9 blocks)             ║
# ║ Saving: 16,626 tokens (34.8%)                ║
# ╠══════════════════════════════════════════════╣
 # ║ Primitives applied:                           ║
 # ║  1. DEDUP       → 2 blocks removed           ║
 # ║     saved: 8,200 tokens                       ║
 # ║  2. REPARTITION → 2 parallel batches created  ║
 # ║     latency_benefit: 40% reduction            ║
 # ║  3. ADD         → SKIPPED (no prefix-hit      ║
 # ║     candidates found in BlockTree)            ║
 # ║  4. REORDER     → 3 blocks reordered (LAST)   ║
 # ║     cache_hit: 12,400 tokens shared           ║
# ╠══════════════════════════════════════════════╣
# ║ Block breakdown:                              ║
# ║  skill:whisper     8,400t  freq=12 ★★★★☆     ║
# ║  skill:notion      3,200t  freq=3  ★★☆☆☆     ║
# ║  memory:prefs        800t  freq=20 ★★★★★     ║
# ║  search:web        6,100t  freq=1  ★☆☆☆☆     ║
# ║  file:audio.py     2,400t  freq=4  ★★★☆☆     ║
# ╚══════════════════════════════════════════════╝

# 实际优化
optimized = optimizer.optimize(messages)

# 统计
print(optimizer.stats())
```

### Level 4: Batch (多 query 并发)

```python
import contextpilot as cp

# 多个 query 共享 context prefix
all_messages = [messages_1, messages_2, messages_3, ...]
optimized_batch, execution_order = cp.optimize_batch(all_messages)

# execution_order 告诉你最优执行顺序（最大化 cache 命中）
for i in execution_order:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=optimized_batch[i]
    )
```

---

## 4. Internal Module Structure (Refactored)

### Current → New 映射

```
contextpilot/                           contextpilot/
├── context_index/                      ├── engine/
│   ├── index_construction.py    ──→    │   ├── cache_index.py     (ContextIndex)
│   ├── tree_nodes.py            ──→    │   ├── tree.py            (ClusterNode, NodeManager)
│   ├── compute_distance_cpu.py  ──→    │   ├── distance.py        (CPU + GPU unified)
│   └── compute_distance_gpu.py  ──→    │   └── (merged into distance.py)
│                                       │
├── context_ordering/                   ├── engine/
│   ├── intra_ordering.py        ──→    │   ├── scheduler.py       (unified intra+inter)
│   └── inter_scheduler.py       ──→    │   └── (merged into scheduler.py)
│                                       │
├── server/                             ├── optimizer/
│   ├── intercept_parser.py      ──→    │   ├── parser.py          (ContextBlockParser)
│   ├── conversation_tracker.py  ──→    │   ├── rules/dedup.py     (dedup rule)
│   ├── metadata.py              ──→    │   ├── catalog.py         (usage stats + registry)
│   ├── live_index.py            ──→    │   └── planner.py         (plan generation)
│   ├── (eviction_heap removed)  ──→    │
│   └── http_server.py           ──→    ├── engine/
│                                       │   ├── live_index.py      (cache mgmt)
│                                       │   ├── eviction.py
│                                       │   └── metadata.py
│                                       │
├── pipeline/                           ├── proxy/
│   ├── rag_pipeline.py          ──→    │   ├── server.py          (HTTP proxy)
│   ├── multi_turn.py            ──→    │   └── interceptor.py     (request intercept)
│   └── components.py            ──→    │
│                                       │
├── retriever/                          ├── integrations/
│   ├── bm25.py                  ──→    │   ├── retrievers/        (moved, optional)
│   ├── faiss_embedding.py       ──→    │   ├── openai_client.py   (NEW: wrapper)
│   ├── mem0_retriever.py        ──→    │   ├── anthropic_client.py(NEW: wrapper)
│   └── pageindex_retriever.py   ──→    │   └── hooks/             (sglang, vllm, llama)
│                                       │
├── api.py                       ──→    ├── __init__.py            (public API)
├── __init__.py                  ──→    ├── optimizer/core.py      (ContextOptimizer class)
│                                       │
├── _sglang_hook.py              ──→    ├── integrations/hooks/
├── _vllm_hook.py                ──→    │
└── _llamacpp_hook.py            ──→    │
                                        │
                                        ├── optimizer/rules/       (NEW)
                                        │   ├── __init__.py
                                        │   ├── base.py            (Rule ABC)
                                        │   ├── reorder.py         (prefix sharing - LAST)
                                        │   ├── dedup.py           (cross-turn deletion)
                                        │   ├── add.py             (prefix-hit selection)
                                        │   └── repartition.py     (parallel topology)
                                        │
                                        ├── engine/                     # ★ Execution Engine
                                        │   ├── __init__.py
                                        │   ├── cache_index.py          # Hierarchical clustering
                                        │   ├── tree.py                 # ClusterNode, NodeManager
                                        │   ├── distance.py             # Distance (CPU/GPU)
                                        │   ├── scheduler.py            # Intra + Inter scheduling
                                        │   ├── live_index.py           # Live cache state
                                        │   ├── eviction.py             # Eviction logic
                                        │   └── metadata.py             # Node metadata
                                        │
                                        ├── proxy/                      # ★ HTTP Proxy mode
                                        │   ├── __init__.py
                                        │   ├── server.py               # FastAPI server
                                        │   └── interceptor.py          # Request intercept
                                        │
                                        ├── integrations/               # ★ 即插即用 wrappers
                                        │   ├── __init__.py
                                        │   ├── openai_client.py        # cp.openai.Client
                                        │   ├── anthropic_client.py     # cp.anthropic.Client
                                        │   ├── retrievers/             # BM25, FAISS, Mem0
                                        │   │   └── ...
                                        │   └── hooks/                  # sglang, vllm, llama
                                        │       └── ...
                                        │
                                        ├── pipeline/                   # RAG pipeline (保留)
                                        │   └── ...
                                        │
                                        └── _compat.py                  # 旧 API 兼容层
```

---

## 5. Core Class Design

### 5.1 ContextOptimizer (主入口)

```python
class ContextOptimizer:
    """
    Context Optimizer for LLM prompts.
    
    Analogous to a SQL Query Optimizer: sits between the user's prompt
    and the LLM, transparently optimizing context for maximum efficiency.
    """
    
    def __init__(
        self,
        rules: List[str] = None,       # ["dedup","add","repartition","reorder"]
        token_budget: int = None,       # max context tokens (None = no limit)
        skill_dirs: List[str] = None,   # skill directories to scan
        persist_path: str = None,       # usage stats persistence
        cache_backend: str = "memory",  # "memory" | "redis" | "file"
    ):
        self._parser = ContextParser()
        self._catalog = Catalog(skill_dirs, persist_path)
        self._cost_model = CostModel(token_budget)
        self._planner = Planner(rules)
        self._engine = CacheEngine()
    
    # ─── Primary API ─────────────────────────────────────────
    
    def optimize(self, messages: List[Dict]) -> List[Dict]:
        """Optimize messages. Drop-in: pass messages in, get optimized messages out."""
        # 1. Parser writes to BlockTree (sole INSERT authority)
        # Prefetch reads FROM BlockTree metadata (outside Optimizer, parallel)
        block_tree = self._parser.parse(messages)
        self._catalog.record(block_tree)
        
        # 2. Optimizer Core — ALL 4 primitives are READ-only on BlockTree
        # Dedup → Repartition → Add → Reorder (LAST)
        # Each primitive has applicable() check — may be skipped (no-op) if not beneficial
        # e.g. no duplicates → skip Dedup; all blocks dependent → skip Repartition;
        #      no prefix-hit candidates → skip Add; already optimal order → skip Reorder
        plan = self._planner.plan(block_tree, self._catalog, self._cost_model)
        return self._execute(plan, messages)
    
    def explain(self, messages: List[Dict]) -> OptimizationPlan:
        """Show optimization plan without executing (like SQL EXPLAIN)."""
        block_tree = self._parser.parse(messages)
        return self._planner.plan(block_tree, self._catalog, self._cost_model)
    
    def optimize_batch(
        self, 
        all_messages: List[List[Dict]]
    ) -> Tuple[List[List[Dict]], List[int]]:
        """Batch optimize + return execution order for max cache sharing."""
        ...
    
    # ─── Management API ──────────────────────────────────────
    
    def stats(self) -> Dict:
        """Usage statistics, cache hit rates, optimization history."""
        ...
    
    def analyze(self) -> None:
        """Refresh statistics (like SQL ANALYZE)."""
        ...
    
    def register_skill(self, path: str) -> None:
        """Register a new skill (like CREATE INDEX)."""
        ...
    
    def configure_rules(self, rules: List[str]) -> None:
        """Enable/disable optimization rules."""
        ...
```

### 5.2 Optimization Rules (Pluggable)

```python
class OptimizationRule(ABC):
    """Base class for all optimization rules. Like SQL rewrite rules."""
    
    name: str                    # "reorder", "dedup", etc.
    
    @abstractmethod
    def applicable(self, block_tree: ContextBlockTree, 
                   catalog: Catalog) -> bool:
        """Can this rule improve the plan? Quick check, no mutation."""
    
    @abstractmethod
    def estimate(self, block_tree: ContextBlockTree,
                 catalog: Catalog, cost_model: CostModel) -> RuleEstimate:
        """Estimate benefit/cost without applying. For plan selection."""
    
    @abstractmethod
    def apply(self, messages: List[Dict], 
              block_tree: ContextBlockTree) -> List[Dict]:
        """Apply the optimization. Returns new messages."""


# Concrete rules:

class DedupRule(OptimizationRule):
    """Remove documents seen in previous conversation turns (Deletion).
    Skip condition: Turn 1, or no content_hash collision with prior turns."""
    name = "dedup"
    # Operation: Distinct (δ) on BlockTree

class RepartitionRule(OptimizationRule):
    """Auto-discover independent chunks, split into parallel batches (Topology).
    Skip condition: all blocks have dependencies (no independent subtrees found)."""
    name = "repartition"
    # Operation: Exchange (∥) on BlockTree (Latency optimization)

class AddRule(OptimizationRule):
    """Query BlockTree for blocks that increase prefix hit ratio (READ).
    Solves: existing blocks can't hit a prefix no matter how reordered,
    but adding a block makes the match possible.
    Added blocks carry annotation: relevance='cache-only'.
    Runs BEFORE Reorder — expands the block set, then Reorder sorts it.
    Skip condition: no block in BlockTree can extend a cacheable prefix."""
    name = "add"
    # Operation: Union (∪) — adds to OUTPUT, reads from BlockTree

class ReorderRule(OptimizationRule):
    """Permute block order to maximize prefix hits (Permutation). MUST BE LAST.
    Runs on the expanded set (after Add), uses IntraContextOrderer.
    Skip condition: current order already maximizes prefix hit (e.g. single block)."""
    name = "reorder"
    # Operation: Sort (τ) on final block set
```

### 5.3 OptimizationPlan (EXPLAIN output)

```python
@dataclass
class OptimizationPlan:
    """The output of the Planner — describes what the Optimizer will do."""
    
    # Input analysis
    input_blocks: List[ContextBlock]
    input_tokens: int
    
    # Planned steps
    steps: List[PlanStep]
    
    # Expected output
    output_tokens: int
    token_savings: int
    cache_hit_estimate: float    # 0.0 ~ 1.0
    
    # Metadata
    rules_considered: List[str]
    rules_applied: List[str]
    planning_time_ms: float
    
    def __str__(self) -> str:
        """Pretty-print like SQL EXPLAIN output."""
        ...
    
    def to_markmap(self) -> str:
        """Export block tree as markmap-compatible markdown."""
        ...


@dataclass
class PlanStep:
    rule: str                    # "reorder", "dedup", etc.
    target_blocks: List[str]     # which blocks affected
    tokens_before: int
    tokens_after: int
    latency_benefit: float       # for repartition
    reason: str                  # human-readable explanation
```

---

## 6. Integration Layer: Client Wrappers

### 6.1 OpenAI Wrapper

```python
# contextpilot/integrations/openai_client.py

class Client:
    """Drop-in replacement for openai.Client with context optimization."""
    
    def __init__(self, api_key=None, optimizer=None, **kwargs):
        import openai
        self._client = openai.Client(api_key=api_key, **kwargs)
        self._optimizer = optimizer or _get_default_optimizer()
        self.chat = self._ChatNamespace(self._client, self._optimizer)
    
    class _ChatNamespace:
        def __init__(self, client, optimizer):
            self.completions = self._CompletionsNamespace(client, optimizer)
        
        class _CompletionsNamespace:
            def create(self, messages, **kwargs):
                optimized = self._optimizer.optimize(messages)
                return self._client.chat.completions.create(
                    messages=optimized, **kwargs
                )
```

### 6.2 简洁的 `__init__.py`

```python
# contextpilot/__init__.py

from .optimizer.core import ContextOptimizer
from . import integrations as openai      # cp.openai.Client(...)
from . import integrations as anthropic   # cp.anthropic.Client(...)

# Module-level convenience (uses singleton optimizer)
def optimize(messages, **kwargs):
    return _get_default().optimize(messages, **kwargs)

def explain(messages, **kwargs):
    return _get_default().explain(messages, **kwargs)

def optimize_batch(all_messages, **kwargs):
    return _get_default().optimize_batch(all_messages, **kwargs)
```

---

## 7. Data Flow (一次 optimize 调用)

```
messages = [
  {role: "system", content: "<skill_content name='whisper'>..."},
  {role: "user", content: [{type: "tool_result", content: "file contents..."}]},
  {role: "user", content: "Transcribe this audio file"}
]
      │
      ▼
 ┌─ 1. PARSE / PREFETCH (< 5ms) ──────────────────────┐
 │  (Parallel)                                          │
 │  ContextBlockTree (Append-only):                     │
 │    ├── skill:whisper (8400 tokens)                   │
 │    ├── file:/audio/test.wav (2400 tokens)            │
 │    └── user_query (12 tokens)                        │
 └──────────────────────────────────────────────────────┘
      │
      ▼
 ┌─ 2. ANALYZE (< 1ms) ───────────────────────────────┐
 │  Catalog lookup (AI-free):                           │
 │    skill:whisper     → freq=12, last=2min, hot ★★★★ │
 │    file:/audio/test.wav → freq=1, new                │
 └──────────────────────────────────────────────────────┘
      │
      ▼
 ┌─ 3. OPTIMIZE (4 Primitives, < 8ms total) ──────────┐
 │  Rules evaluated:                                    │
 │    ✅ dedup        → 1 block removed (Distinct)      │
 │    ✅ add          → prefix-hit blocks from tree (Union, cache-only) │
 │    ✅ repartition  → 2 parallel batches (Exchange)   │
 │    ✅ reorder      → permute for prefix (Sort, LAST) │
 │                                                      │
  │  Plan: dedup -> repartition -> add -> reorder          │
 │  Total Latency Budget: < 15ms                        │
 └──────────────────────────────────────────────────────┘
      │
      ▼
 ┌─ 4. EXECUTE (< 5ms) ───────────────────────────────┐
 │  Apply plan to BlockTree                             │
 │  Reconstruct messages for LLM backend                │
 └──────────────────────────────────────────────────────┘
      │
      ▼
 optimized_messages  → send to LLM
 (optimized KV layout)
```

---

## 8. Refactoring Plan

### Phase 0: 准备 (不改功能，只重组文件)

```
目标: 文件搬到新位置，旧 import 全部兼容
方法: 新文件 import from 旧位置，_compat.py 做旧 → 新转发

1. 创建新目录结构 (空 __init__.py)
2. engine/ — 移入 context_index/* + context_ordering/*
3. proxy/ — 移入 server/http_server.py + intercept_parser.py
4. integrations/ — 移入 retriever/* + hooks
5. _compat.py — 旧 import 路径全部转发
6. 跑通所有现有 tests
```

### Phase 1: Optimizer Core (新增，不改旧代码)

```
目标: ContextOptimizer 类 + 4 基础 primitives
特点: AI-free, < 15ms latency budget

1. optimizer/parser.py — ContextBlockParser (outside)
2. optimizer/prefetch.py — Predictive logic (outside, parallel with parser)
3. optimizer/catalog.py — Catalog + BlockUsageStore
4. optimizer/rules/dedup.py — Deletion filter
5. optimizer/rules/add.py — Cache-warm insertion
6. optimizer/rules/repartition.py — Topology optimization
7. optimizer/rules/reorder.py — Prefix sorting (LAST)
8. optimizer/core.py — ContextOptimizer 主类
9. optimizer/planner.py — 规则串行计划 (Phase 1,2,3)
```

### Phase 2: Client Wrappers (即插即用)

```
目标: cp.openai.Client, cp.anthropic.Client

1. integrations/openai_client.py — OpenAI wrapper
2. integrations/anthropic_client.py — Anthropic wrapper
3. 更新 __init__.py — import path
4. tests/test_wrappers.py
```

### Phase 3: BlockTree & Prefetch (新功能)

```
目标: append-only BlockTree, predictive prefetch

1. engine/tree.py — BlockTree (INSERT + READ only)
2. optimizer/prefetch.py — SkillPrefetcher 完善
3. integrations/hooks — vLLM/SGLang prefix cache reuse
4. explain() 输出完善 (primitive analogies)
```

### Phase 4: Proxy 重构

```
目标: proxy server 使用 ContextOptimizer 而不是直接调 engine

1. proxy/server.py — 重构 _intercept_and_forward 使用 optimizer.optimize()
2. proxy/interceptor.py — 从 intercept_parser.py 提取
3. CLI: contextpilot serve
4. 集成测试
```

---

## 9. Migration: 旧 API 完全兼容

```python
# _compat.py — 所有旧 import 都工作

# 旧:
from contextpilot import ContextPilot
from contextpilot.pipeline import RAGPipeline
from contextpilot.context_index import ContextIndex, IndexResult
from contextpilot.context_ordering import IntraContextOrderer
import contextpilot as cp; cp.optimize(docs, query)

# 全部通过 _compat.py 转发到新位置
# 旧 API 标记 DeprecationWarning，但功能不变
```

---

## 10. Why This Matters

```
                        Before                    After
                     (ContextPilot)          (Context Optimizer)
                    ──────────────          ──────────────────

 User effort         需要理解 pipeline,      cp.optimize(messages)
                     retriever, config       一行搞定

 Mental model        "KV cache prefix       "像 SQL Optimizer —
                      sharing library"        prompt 进去，
                                              优化后的 prompt 出来"

 Integration         自己写 pipeline         零代码 proxy 或
                     配 server               一行 client wrapper

优化效果             只有 reorder +          reorder + dedup +
                      basic dedup              add + repartition


 Observability       基本日志               EXPLAIN：完整优化计划
                                             markmap 可视化
                                             usage stats dashboard

 Extensibility       改核心代码              自定义 Rule 插件
                                             pluggable, 注册即用
```
