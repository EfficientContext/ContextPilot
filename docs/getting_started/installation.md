# Installation

This guide covers installing ContextPilot and its dependencies.

**Requirements:** Python >= 3.10

---

## vLLM / SGLang

ContextPilot works with both CPU and GPU backends for building the context index. The `[gpu]` extra enables GPU-accelerated distance computation (via `cupy-cuda12x`) and is faster for large batches; without it, ContextPilot falls back to the CPU backend automatically.

**From PyPI** — the vLLM and SGLang hooks are installed automatically:
```bash
pip install contextpilot          # CPU index computation
pip install "contextpilot[gpu]"   # GPU index computation (CUDA 12.x)
```

**From source** — run `install_hook` manually after install, since editable installs do not copy the `.pth` file to site-packages:
```bash
git clone https://github.com/EfficientContext/ContextPilot.git
cd ContextPilot
pip install -e .                  # CPU
pip install -e ".[gpu]"           # GPU (CUDA 12.x)
python -m contextpilot.install_hook   # one-time: enables automatic vLLM / SGLang integration
```

The `install_hook` step writes a `.pth` file into your site-packages so the vLLM and SGLang hooks load automatically at Python startup — no code changes required. To uninstall: `python -m contextpilot.install_hook --remove`.

---

## Mac / Apple Silicon — llama.cpp

**From PyPI:**
```bash
pip install contextpilot
xcode-select --install    # one-time: provides clang++ to compile the native hook
```

**From source:**
```bash
git clone https://github.com/EfficientContext/ContextPilot.git
cd ContextPilot
pip install -e .
xcode-select --install    # one-time: provides clang++ to compile the native hook
```

> **Why `xcode-select`?** The llama.cpp integration uses a small C++ shared library injected into `llama-server` via `DYLD_INSERT_LIBRARIES`. It is compiled automatically on first use and requires `clang++` from Xcode Command Line Tools.

---

## Distributed Setup

If the ContextPilot index server and the inference engine run in **separate Python environments** (e.g., different virtualenvs or containers), the engine environment won't have the `contextpilot` package. Use the standalone hook instead:

```bash
# In the engine's Python environment (one command, no clone needed):
pip install requests
curl -sL https://raw.githubusercontent.com/EfficientContext/ContextPilot/main/contextpilot/install_standalone.py | python -
```

The installer downloads the hook from GitHub and installs it into site-packages. No `contextpilot` clone or install needed — just `requests` as a runtime dependency.

## Core Dependencies

| Package | Purpose |
|---------|--------|
| `fastapi[all]` | HTTP server |
| `aiohttp` | Async inference engine proxy |
| `scipy` | Hierarchical clustering |
| `transformers` | Tokenizer / chat templates |
| `cupy-cuda12x` | GPU distance computation (`[gpu]` extra only) |
| `elasticsearch` | BM25 retriever (optional) |
| `datasets` | Loading benchmark datasets |

## Verify Installation

```bash
python -c "import contextpilot; print('ContextPilot', contextpilot.__version__)"
```

Docker images are also available for both all-in-one and standalone deployment. See the [Docker guide](docker.md).
