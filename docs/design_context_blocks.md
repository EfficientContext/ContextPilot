# Context Blocks: Prompt Decomposition, Usage Tracking & Predictive Prefetch

## 1. Problem Statement

当前 ContextPilot 的 intercept 路径将 prompt 视为扁平的 document 列表进行 clustering/reordering。但实际 agentic 场景中，一个 prompt 由**不同类型**的 context 组成：

```
prompt
├── skills/          # 技能指令（如 SKILL.md, slash commands）
│   ├── openai-whisper/SKILL.md
│   │   ├── §Installation    ← 上次用过
│   │   ├── §Usage           ← 高频使用
│   │   └── §Advanced        ← 从未使用
│   └── notion/SKILL.md
├── memory/          # 用户记忆、偏好、历史总结
├── searched/        # 实时检索结果（web search, grep）
└── file_based/      # 文件读取（Read tool results）
```

**目标**：
1. 将 prompt 解析为 typed context blocks，构建分解树
2. 细粒度追踪每个 block/section 的 `last_used_time` 和 `frequency`
3. 基于使用统计实现 predictive prefetch（类似 branch prediction）
4. 支持用户侧的 skill filter/merge 操作

**约束**：系统优化路径，不增加额外 LLM 调用，不引入高延迟操作。

---

## 2. Architecture Overview

```
User Input
    │
 ┌──┴──┐  parallel
Parser  Prefetch       ← BOTH outside Optimizer
 └──┬──┘
    ▼
 BlockTree             ← append-only: INSERT + READ only, no DELETE
    │
 ┌─ Optimizer ─────────────────┐
 │  Phase 1: {Dedup ∥ Add}     │  ALL READ-only
 │  Phase 2:  Repartition      │  READ
 │  Phase 3:  Reorder (LAST)   │  READ
 └─────────────────────────────┘
    │
 Optimized Output
```

### 数据流

```
Turn N request
  │
  ├─ 1. Parse: prompt → BlockTree（O(n) string scan，无 LLM）
  │     识别 skill_content tags, memory blocks, tool_results, file reads
  │
  ├─ 2. Prefetch: (parallel with Parse) 从 BlockTree metadata 选 block（O(k) lookup）
  │     读取 freq/co-occurrence → 选出值得放到输出的 block
  │
  ├─ 3. Track: 更新 block-level usage stats（O(1) dict update）
  │     last_used_time, frequency, section-level granularity
  │
  ├─ 4. Optimizer: {Dedup ∥ Add} → Repartition → Reorder（< 15ms, ALL READ-only）
  │     Dedup 过滤重复，Add 从 BlockTree 查 prefix-hit block（annotated cache-only），Reorder 优化 prefix
  │
  └─ 5. Forward: 优化后的 request → LLM backend
```

---

## 3. Module Design

### 3.1 `ContextBlockParser` — Prompt → Context Block Tree

**位置**: `contextpilot/server/context_block_parser.py` (新文件)

**职责**: 将 prompt 中的各类 context 识别并分类为 typed blocks，构建树形结构。

#### Block Types

| Type | 识别方式 | 来源 |
|------|---------|------|
| `skill` | `<skill_content name="...">` tags, SKILL.md 内容 | system prompt, tool_results |
| `memory` | Memory search results, `<memory>` tags, CLAUDE.md | system prompt, tool_results |
| `searched` | Web search / grep results (JSON with `results` array) | tool_results |
| `file_based` | File read tool results (Read tool, single-doc) | tool_results |

#### ContextBlockTree 结构

```python
@dataclass
class ContextBlock:
    """单个 context block。"""
    block_id: str              # 唯一标识，如 "skill:openai-whisper" 或 "file:/path/to.py"
    block_type: BlockType      # skill | memory | searched | file_based
    content_hash: str          # SHA-256[:16] 用于快速比较
    source_location: str       # 在 prompt 中的位置标识
    children: List['ContextBlock']  # 子 block（如 skill 的各 section）
    metadata: Dict[str, Any]   # 额外信息（文件路径、skill 名称等）
    char_count: int            # 字符数（用于评估 token 开销）

class BlockType(str, Enum):
    SKILL = "skill"
    MEMORY = "memory"
    SEARCHED = "searched"
    FILE_BASED = "file_based"

@dataclass  
class ContextBlockTree:
    """一次 request 的 context block 分解树。"""
    request_id: str
    timestamp: float
    blocks: List[ContextBlock]  # 顶层 blocks
    
    def to_markmap(self) -> str:
        """导出为 markmap 兼容的 markdown。"""
        ...
    
    def flat_blocks(self) -> List[ContextBlock]:
        """DFS 展平所有 blocks（含子 blocks）。"""
        ...
```

#### 解析策略（零 LLM，纯规则）

```python
def parse_context_blocks(body: Dict, api_format: str) -> ContextBlockTree:
    """
    从 request body 中解析 context blocks。
    
    策略（按优先级）：
    1. Skill detection:
       - <skill_content name="..."> ... </skill_content> 标签
       - 包含 SKILL.md 路径的 tool_result
       - 已知 skill 关键词匹配
    
    2. Memory detection:
       - <memory> 标签
       - CLAUDE.md / .cursorrules 等配置文件内容
       - mem0 search results
    
    3. Searched detection:
       - JSON tool_results with "results" array (web search, grep)
       - brave-search, google search 等 tool 名称匹配
    
    4. File-based detection:
       - Read tool results（单文件内容）
       - 文件路径特征匹配
    
    复杂度: O(n) where n = total content length
    """
```

#### Skill Section 分解（细粒度）

对于 skill 类型的 block，进一步按 markdown heading 分解为 sections：

```python
def _decompose_skill(content: str, skill_name: str) -> ContextBlock:
    """
    将 skill 内容按 markdown heading 分解为 section sub-blocks。
    
    Example:
      openai-whisper/SKILL.md
      ├── §Overview        (## Overview)
      ├── §Installation    (## Installation) 
      ├── §Basic Usage     (## Basic Usage)
      └── §Advanced        (## Advanced Options)
    
    每个 section 独立追踪使用情况。
    """
```

### 3.2 `BlockUsageStore` — Usage Tracking

**位置**: `contextpilot/server/block_usage.py` (新文件)

**职责**: 追踪每个 context block 的使用统计。

```python
@dataclass
class BlockUsageStats:
    """单个 block 的使用统计。"""
    block_id: str
    block_type: BlockType
    frequency: int = 0              # 总使用次数
    last_used_time: float = 0.0     # 最后使用时间戳
    first_seen_time: float = 0.0    # 首次出现时间
    avg_interval: float = 0.0       # 平均使用间隔
    co_occurrence: Dict[str, int] = field(default_factory=dict)
    # co_occurrence: 与其他 block 的共现次数
    # 例如 {"skill:notion": 5} 表示与 notion skill 共同出现 5 次


class BlockUsageStore:
    """
    Block 使用统计存储。
    
    设计原则：
    - 内存中维护，定期持久化到 JSON 文件
    - O(1) 查询和更新
    - 按 block_id 索引
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        self._stats: Dict[str, BlockUsageStats] = {}
        self._persist_path = persist_path  # 如 ~/.contextpilot/block_usage.json
    
    def record_usage(self, blocks: List[ContextBlock]) -> None:
        """
        记录一次 request 中所有 blocks 的使用。
        
        - 更新 frequency += 1
        - 更新 last_used_time = now
        - 更新 co_occurrence（同一 request 中的 blocks 互相计数）
        - 递归处理子 blocks
        
        复杂度: O(b²) where b = number of blocks per request（通常 < 20）
        """
    
    def get_hot_blocks(self, 
                       block_type: Optional[BlockType] = None,
                       top_k: int = 5,
                       recency_weight: float = 0.7,
                       frequency_weight: float = 0.3) -> List[BlockUsageStats]:
        """
        获取 "热" blocks，用于 prefetch 决策。
        
        Score = recency_weight * recency_score + frequency_weight * frequency_score
        
        recency_score = exp(-decay * (now - last_used_time))
        frequency_score = log(1 + frequency) / log(1 + max_frequency)
        """
    
    def get_cooccurrent_blocks(self, block_id: str, top_k: int = 3) -> List[str]:
        """获取与给定 block 最常共现的其他 blocks。"""
    
    def persist(self) -> None:
        """持久化到磁盘（JSON）。"""
    
    def load(self) -> None:
        """从磁盘加载。"""
```

### 3.3 `SkillPrefetcher` — Branch Prediction for Skills

**位置**: `contextpilot/server/skill_prefetcher.py` (新文件)

**职责**: 从 BlockTree 的 metadata（freq/co-occurrence）中读取统计信息，选出值得放到输出的 block。

> **Note**: SkillPrefetcher 运行在 Optimizer 外部，与 Parser 并行执行。它是 READ-only 的外部组件 — 从 BlockTree 读取 metadata，不写入 BlockTree。只有 Parser 有 BlockTree 的写入权限。

#### 核心思路

```
类比 CPU branch prediction：

CPU:   历史分支方向 → 预测下一次分支
我们:  历史 skill 使用 → 预测下次需要的 skill

命中: 省去 LLM 选择 skill 的步骤（省 1 次 LLM 调用 + 网络延迟）
未命中: fallback 到 LLM 选择 → grep skill 关键词
```

#### 预测策略

```python
class SkillPrefetcher:
    """
    Skill 预测与预取。
    
    三级 fallback 策略（低延迟优先）：
    
    Level 0 - 直接命中（< 1ms）:
        当前 request 已包含 skill content → 无需预测
    
    Level 1 - 统计预测（< 5ms）:
        usage_store.get_hot_blocks(type=SKILL) 
        + co_occurrence 分析
        → 注入 top-k 高概率 skills
    
    Level 2 - 关键词匹配（< 50ms）:
        从 user query 中提取关键词
        → grep 已知 skill 目录的描述/关键词
        → 注入匹配的 skills
    
    Level 3 - LLM Fallback（最后手段）:
        如果 Level 1+2 都没命中
        且 user query 明确需要某种能力
        → 让 LLM 输出需要的 skill 类型
        → grep skill 关键词
        （这是当前的默认行为，我们要尽量避免走到这一步）
    """
    
    def __init__(self, usage_store: BlockUsageStore, 
                 skill_registry: 'SkillRegistry'):
        self._usage_store = usage_store
        self._skill_registry = skill_registry
        self._prediction_history: List[PredictionResult] = []
    
    def predict(self, 
                current_blocks: ContextBlockTree,
                user_query: str = "") -> List[SkillPrediction]:
        """
        预测下一次需要的 skills。
        
        Returns:
            List of SkillPrediction(skill_id, confidence, reason)
            confidence > threshold → 自动注入
        """
    
    def evaluate_prediction(self, predicted: List[str], 
                           actual: List[str]) -> PredictionMetrics:
        """
        评估预测准确率（用于自适应调优）。
        
        Returns:
            PredictionMetrics(hit_rate, precision, recall, avg_latency_saved)
        """


@dataclass
class SkillPrediction:
    skill_id: str           # 如 "openai-whisper"
    confidence: float       # 0.0 ~ 1.0
    reason: str             # "high_frequency", "co_occurrence", "keyword_match"
    estimated_tokens: int   # 预估 token 开销
```

#### Skill Registry（轻量索引）

```python
class SkillRegistry:
    """
    已知 skills 的轻量索引。
    
    数据来源：
    - ~/.config/opencode/skills/ 目录扫描
    - 用户配置的外部 skill 目录
    
    索引内容（启动时构建，O(k) where k = skill count）：
    - skill_id → skill_path
    - skill_id → keywords（从 SKILL.md description 提取）
    - skill_id → section_headings（markdown heading 列表）
    """
    
    def __init__(self, skill_dirs: List[str]):
        self._skills: Dict[str, SkillInfo] = {}
        self._keyword_index: Dict[str, List[str]] = {}  # keyword → [skill_ids]
    
    def scan(self) -> None:
        """扫描 skill 目录，构建索引。"""
    
    def search_by_keywords(self, keywords: List[str]) -> List[str]:
        """关键词匹配搜索 skills。O(k * w)"""
    
    def get_skill_content(self, skill_id: str, 
                          sections: Optional[List[str]] = None) -> str:
        """
        获取 skill 内容。可选只获取特定 sections。
        
        sections=None → 全量
        sections=["Installation", "Usage"] → 只返回这两个 section
        """
```

### 3.4 `SkillFilter` — 用户侧 Skill 管理

**位置**: `contextpilot/server/skill_filter.py` (新文件)

**职责**: 允许用户管理（filter/merge）哪些 skills 参与 prefetch。

#### Filter 设计

```python
@dataclass
class SkillFilterConfig:
    """用户级 skill filter 配置。"""
    
    # Allowlist: 只允许这些 skills 被 prefetch（空 = 不限制）
    allowed_skills: Set[str] = field(default_factory=set)
    
    # Blocklist: 禁止这些 skills 被 prefetch
    blocked_skills: Set[str] = field(default_factory=set)
    
    # Section filter: 对特定 skill 只允许/禁止特定 sections
    # 如 {"openai-whisper": {"allowed": ["Usage", "Installation"]}}
    section_filters: Dict[str, Dict] = field(default_factory=dict)
    
    # Auto-prune: 自动移除超过 N 天未使用的 skills
    auto_prune_days: Optional[int] = None
    
    # Max prefetch: 最多同时 prefetch 的 skill 数量
    max_prefetch_skills: int = 3
    
    # Confidence threshold: prefetch 的置信度阈值
    confidence_threshold: float = 0.6


class SkillFilter:
    """
    Skill filter 与 merge 管理器。
    
    用户交互方式（通过 HTTP API 或 config 文件）：
    
    1. Filter:
       POST /skills/filter  {"block": ["notion"], "allow": ["openai-whisper"]}
       → 禁止 notion 被 prefetch，只允许 whisper
    
    2. Merge:
       POST /skills/merge   {"skills": ["skill-a", "skill-b"], "name": "merged-ab"}
       → 将多个 skill 合并为一个虚拟 skill（共享 cache prefix）
       → 合并后的 skill 作为整体被追踪和 prefetch
    
    3. Pin:
       POST /skills/pin     {"skills": ["openai-whisper"]}
       → 钉选：始终 prefetch，不受 confidence threshold 限制
    
    4. Stats:
       GET  /skills/stats
       → 返回所有 skill 的使用统计 + 预测命中率
    """
    
    def __init__(self, config: SkillFilterConfig, 
                 usage_store: BlockUsageStore):
        self._config = config
        self._usage_store = usage_store
        self._merged_skills: Dict[str, List[str]] = {}  # merged_name → [original_ids]
    
    def apply_filter(self, predictions: List[SkillPrediction]) -> List[SkillPrediction]:
        """
        应用 filter 规则到 prefetch 预测结果。
        
        流程：
        1. 移除 blocked_skills 中的预测
        2. 如果 allowed_skills 非空，只保留其中的
        3. 应用 section_filters（裁剪 skill 内容到允许的 sections）
        4. 应用 confidence_threshold
        5. 截断到 max_prefetch_skills
        """
    
    def merge_skills(self, skill_ids: List[str], merged_name: str) -> None:
        """
        合并多个 skills 为一个虚拟 skill。
        
        效果：
        - 合并后的 skill 的 usage stats = sum(individual stats)
        - prefetch 时作为整体 fetch（共享 cache prefix）
        - 在 ContextBlockTree 中显示为一个 merged block
        """
    
    def auto_prune(self) -> List[str]:
        """
        自动清理长期未使用的 skills。
        
        Returns:
            被清理的 skill_ids
        """
```

---

## 4. Integration Points

### 4.1 与现有 Intercept Pipeline 集成

```python
# contextpilot/server/http_server.py :: _intercept_and_forward()

async def _intercept_and_forward(request: Request, api_format: str):
    body = await request.json()
    
    # 1. Parse & Prefetch (Parallel)
    # Parser: prompt -> BlockTree (sole INSERT authority)
    # Prefetch: READ BlockTree metadata -> select blocks for output
    block_tree, prefetch_blocks = await gather(
        parse_context_blocks(body, api_format),
        prefetcher.select_from_tree(user_query=...)
    )
    
    # 2. Track usage stats
    usage_store.record_usage(block_tree.flat_blocks())
    
    # 3. Optimizer: ALL READ-only on BlockTree
    # - Dedup: 过滤重复 (READ)
    # - Add: 从 BlockTree 查 prefix-hit blocks, annotated cache-only (READ)
    # - Repartition: 拓扑变更 (READ)
    # - Reorder: 最终排序 (READ, LAST)
    optimized_body = optimizer.run(body, block_tree, prefetch_blocks)
    
    # 4. Forward optimized request
    return await forward(optimized_body)
```

### 4.2 新增 HTTP API Endpoints

```python
# GET /context-blocks/tree
# 返回最近一次 request 的 context block tree（markmap markdown 格式）

# GET /context-blocks/stats  
# 返回所有 block 的使用统计

# POST /skills/filter
# 设置 skill filter 规则

# POST /skills/merge
# 合并 skills

# POST /skills/pin
# 钉选 skills

# GET /skills/predictions
# 返回当前的 skill 预测结果和历史准确率

# GET /skills/registry
# 返回已知 skill 索引
```

### 4.3 Markmap Visualization

```python
# GET /context-blocks/markmap
# 返回可直接粘贴到 https://markmap.js.org/repl 的 markdown

# 示例输出:
"""
# Request ctx-20260309-001

## Skills (2)
### openai-whisper
- §Overview ✅ freq=12 last=2min ago
- §Installation ✅ freq=8 last=5min ago  
- §Usage ✅ freq=15 last=1min ago
- §Advanced ⬜ freq=0

### notion
- §API Setup ✅ freq=3 last=1h ago
- §Page Operations ⬜ freq=0

## Memory (1)
### User Preferences
- freq=20 last=30s ago

## Searched (3)
### Web: "whisper CLI usage"
- 5 results, freq=1
### Grep: "def transcribe"  
- 3 results, freq=2

## File-Based (2)
### /src/audio.py
- freq=4 last=10min ago
### /config/settings.json
- freq=1 last=1h ago
"""
```

---

## 5. Implementation Plan

### Phase 1: Context Block Parsing + Usage Tracking (Core)

| Task | 文件 | 估时 |
|------|------|------|
| `ContextBlock` / `BlockTree` 数据模型 (Append-only) | `context_block_parser.py` | 0.5d |
| Parser: skill/memory/searched/file_based 识别 | `context_block_parser.py` | 1d |
| Skill section 分解（markdown heading split） | `context_block_parser.py` | 0.5d |
| `BlockUsageStore` 统计引擎 | `block_usage.py` | 0.5d |
| 集成到 `_intercept_and_forward` | `http_server.py` | 0.5d |
| Markmap export | `context_block_parser.py` | 0.5d |
| 单元测试 | `tests/test_context_blocks.py` | 0.5d |

### Phase 2: Prefetch Component (External, Parallel with Parser)

| Task | 文件 | 估时 |
|------|------|------|
| `SkillRegistry` 目录扫描与索引 | `skill_prefetcher.py` | 0.5d |
| `SkillPrefetcher` 统计预测（Level 1） | `skill_prefetcher.py` | 0.5d |
| 关键词匹配预测（Level 2） | `skill_prefetcher.py` | 0.5d |
| Prefetch 结果并行注入 BlockTree | `http_server.py` | 0.5d |
| 预测准确率评估与自适应 | `skill_prefetcher.py` | 0.5d |
| 集成测试 | `tests/test_skill_prefetch.py` | 0.5d |

### Phase 3: Optimizer Primitives & User Management

| Task | 文件 | 估时 |
|------|------|------|
| Optimizer Primitives (Dedup, Add, Repartition, Reorder) | `optimizer.py` | 1.5d |
| `SkillFilterConfig` 数据模型 | `skill_filter.py` | 0.5d |
| Filter/Merge/Pin 逻辑 | `skill_filter.py` | 0.5d |
| HTTP API endpoints | `http_server.py` | 0.5d |
| Auto-prune | `skill_filter.py` | 0.5d |
| E2E 测试 | `tests/test_skill_filter.py` | 0.5d |

---

## 6. Performance Budget

| 操作 | 目标延迟 | 实现方式 |
|------|---------|---------|
| Context block parsing | < 5ms | 纯正则 + string scan，无 LLM |
| Usage tracking update | < 1ms | 内存 dict update |
| Skill prediction (Level 1) | < 5ms | 排序 + top-k 选取 |
| Optimizer: Dedup | < 2ms | Set logic, output filtering only |
| Optimizer: Add | < 2ms | Cache state lookup & insert |
| Optimizer: Repartition | < 3ms | Subtree grouping & split |
| Optimizer: Reorder | < 5ms | Topology sorting, optimized for prefix hits |
| Total Optimizer Latency | < 15ms | AI-free transformation pipeline |

**总额外延迟**: < 15ms（典型路径：parsing/prefetch 并行 + tracking + optimizer）

---

## 7. Key Design Decisions

### 7.1 为什么 Prefetch 放在 Optimizer 外部？
- **预测 vs 确定性变换**: Prefetch 是 speculative（投机性）的，基于历史概率预测需求；Optimizer 的 4 个 primitive 是确定性的集合变换。
- **并行化隐藏延迟**: Prefetch 与 Parser 并行运行。如果 Prefetch 放在 Optimizer 内部，则必须串行等待 Parser 完成，增加了关键路径延迟。
- **解耦**: Optimizer 只负责对已有的 BlockTree 状态进行最优表达，而不负责产生新的推测内容。

### 7.2 为什么 BlockTree 是 Append-only (无 DELETE)？
- **保留历史上下文**: 即使某个 Block 在当前请求中被 Dedup 过滤掉，它在树中的存在对于频率统计和共现分析（co-occurrence）依然重要。
- **写入权限唯一**: 只有 Parser 有 INSERT 权限。所有 Optimizer primitives 和 Prefetch 都是 READ-only。
- **简化追踪逻辑**: 避免了树节点删除导致的索引失效和引用断裂。
- **Dedup 的语义**: Dedup 作为 READ 操作，生成过滤后的输出视图，不修改 BlockTree。

### 7.3 Add primitive 与 Prefetch 的区别？
- **Add**: 位于 Optimizer 内部，READ from BlockTree，查找增加 **prefix hit** 的 block 加到输出。加的 block 不一定相关，附带 `relevance="cache-only"` annotation 告知 LLM 可忽略。确定性。
- **Prefetch**: 位于 Optimizer 外部，READ from BlockTree **metadata**（freq/co-occurrence），统计预测哪些 block 值得放到输出。Speculative。
- **共同点**: 都是 READ-only on BlockTree → 往输出加 block。区别在于决策依据（prefix matching vs 统计）和时机。

### 7.4 为什么不用 LLM 做 block classification？
- 每次 request 增加 1 次 LLM 调用 ≈ 100-500ms
- Block 类型可以通过结构化特征（tags, tool names, file paths）可靠识别
- 规则引擎 < 5ms，LLM ≈ 100x slower

### 7.5 为什么 section-level 而不是 token-level 追踪？
- Token-level 需要 LLM attention weights → 不可用（proxy 模式无法获取）
- Section-level（markdown heading）是 skill 的自然分界
- 粒度足够用于 prefetch 决策

### 7.6 Skill merge 的语义是什么？
- **场景**：用户总是同时用 skill-A 和 skill-B → merge 为 skill-AB
- **效果**：两个 skill 内容 concatenate 为一个 block → 共享 KV cache prefix
- **追踪**：merged block 作为一个整体统计 frequency/recency
- **可逆**：`POST /skills/unmerge` 拆回

### 7.7 Prefetch 的 "miss" 怎么处理？
- Prefetch 的 skill 没被用到 → 浪费了 prefill tokens
- 通过 `confidence_threshold` 控制误 prefetch 率
- `evaluate_prediction` 追踪历史准确率，自适应调整 threshold
- 最坏情况 = 多 prefill 了几千 tokens，不影响正确性

### 7.8 与现有 `_intercept_state.seen_doc_hashes` 的关系？
- 现有 dedup 是 document 粒度（hash 去重）
- Context block 是 semantic 粒度（skill/memory/searched/file）
- 两者正交：先 block 解析，再对每个 block 内部的 docs 做 dedup
- `seen_doc_hashes` 逻辑不变，block 是上层分类

---

## 8. Future Extensions

1. **Cross-user skill patterns**: 多用户共享匿名化的 skill usage patterns → collaborative filtering
2. **Adaptive section pruning**: 高频 section 保留全文，低频 section 自动压缩/摘要
3. **Skill versioning**: 当 SKILL.md 内容变化时，追踪版本差异对使用统计的影响
4. **Real-time markmap dashboard**: WebSocket 推送实时 context block tree 可视化
