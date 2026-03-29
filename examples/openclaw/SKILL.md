---
name: contextpilot
description: Optimize document ordering in LLM context for better retrieval performance
version: 1.0.0
triggers:
  - /contextpilot
  - /cp
---

# ContextPilot Integration

You have ContextPilot enabled as a transparent proxy. All your LLM API requests are automatically routed through ContextPilot, which reorders documents in your context window for optimal retrieval performance.

## What ContextPilot Does

When your request contains multiple documents (in system prompts or tool results), ContextPilot:

1. **Extracts** documents from XML tags (`<documents>`, `<files>`, etc.), numbered lists, separators, or markdown headers
2. **Clusters** documents by semantic similarity using hierarchical clustering
3. **Reorders** documents to minimize attention distance between related content
4. **Reconstructs** the request preserving the original format

## Supported Document Formats

- XML tags: `<documents><document>...</document></documents>`, `<files><file>...</file></files>`
- Numbered: `[1] doc [2] doc [3] doc`
- Separator: docs separated by `---` or `===`
- Markdown headers: sections split by `#` or `##` headers

## How to Verify

Check the `X-ContextPilot-Result` response header:

```
X-ContextPilot-Result: {"intercepted":true,"documents_reordered":true,"total_documents":5,"sources":{"system":1,"tool_results":1}}
```

## Configuration Headers

Send these headers with your API requests to control behavior:

| Header | Values | Default |
|--------|--------|---------|
| `X-ContextPilot-Enabled` | `true`/`false` | `true` |
| `X-ContextPilot-Mode` | `auto`/`xml_tag`/`numbered`/`separator`/`markdown_header` | `auto` |
| `X-ContextPilot-Scope` | `all`/`system`/`tool_results` | `all` |
| `X-ContextPilot-Alpha` | float | `0.001` |
