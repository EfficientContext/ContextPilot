# Troubleshooting

Common issues and solutions for ContextPilot.

## Connection Issues

### "Connection refused" when connecting to SGLang

**Cause:** SGLang server is not running or wrong port.

**Solution:**
```bash
# Start SGLang server
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port 30000

# Verify it's running
curl http://localhost:30000/health
```

### "Index not initialized" from ContextPilot server

**Cause:** Trying to use stateful endpoints before building the index.

**Solution:**
- **Stateful mode:** Call `/build` endpoint first
- **Stateless mode:** Use `/schedule` endpoint (no build required)

---

## Performance Issues

### Slow distance computation

**Cause:** Using CPU for large batches.

**Solution:**
```python
# Use GPU for batches > 128 contexts
pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_gpu=True  # Enable GPU acceleration
)
```

Ensure CUDA is properly installed:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory issues with large batches

**Cause:** Too many documents retrieved per query.

**Solution:**
- Reduce `top_k` to retrieve fewer documents
- Process in smaller batches
- Use GPU with more VRAM

```python
# Reduce top_k
results = pipeline.run(queries, top_k=10)  # Instead of 20
```

---

## Import Errors

### "ModuleNotFoundError: No module named 'contextpilot'"

**Cause:** ContextPilot not installed or wrong environment.

**Solution:**
```bash
# Reinstall in development mode
cd ContextPilot
pip install -e .

# Verify installation
python -c "from contextpilot.pipeline import RAGPipeline; print('OK')"
```

### "ModuleNotFoundError: No module named 'faiss'"

**Cause:** FAISS not installed.

**Solution:**
```bash
# GPU version
conda install conda-forge::faiss-gpu

# CPU version
conda install conda-forge::faiss-cpu
```

---

## Server Issues

### ContextPilot server keeps timing out

**Cause:** Long computation time or network issues.

**Solution:**
```python
# Increase client timeout
from contextpilot.server.http_client import ContextPilotIndexClient

client = ContextPilotIndexClient(
    "http://localhost:8765",
    timeout=30.0  # Increase from default 1.0
)
```

### "Address already in use" when starting server

**Cause:** Port 8765 already in use.

**Solution:**
```bash
# Find and kill the process
lsof -i :8765
kill -9 <PID>

# Or use a different port
python -m contextpilot.server.http_server --port 8766
```

---

## Data Issues

### "KeyError: 'doc_id'" when loading corpus

**Cause:** Corpus file has wrong format.

**Solution:** Ensure corpus.jsonl has correct format:
```json
{"doc_id": 1, "text": "Document content here..."}
{"doc_id": 2, "text": "Another document..."}
```

### Empty generation results

**Cause:** SGLang server returned errors.

**Solution:**
- Check SGLang server logs
- Verify model is loaded correctly
- Check prompt format matches model's chat template

---

## Getting Help

If you're still having issues:

1. Check [GitHub Issues](https://github.com/SecretSettler/ContextPilot/issues)
2. Open a new issue with:
   - ContextPilot version
   - Python version
   - Full error traceback
   - Minimal reproduction code
3. Contact: ysc.jiang@ed.ac.uk
