import { describe, expect, it } from "vitest";
import {
  injectAnthropicCacheControl,
  injectCacheControl,
  injectOpenAICacheControl,
} from "./cache-control.js";
import {
  buildToolNameMapOpenai,
  contentDefinedChunking,
  dedupChatCompletions,
  dedupResponsesApi,
  hashBlock,
} from "./dedup.js";
import {
  extractAllOpenai,
  extractDocuments,
  extractFromAnthropicMessages,
  extractFromAnthropicToolResults,
  extractFromOpenaiChat,
  extractFromOpenaiToolResults,
  extractSingleDocsFromOpenaiToolResults,
  getFormatHandler,
  parseInterceptHeaders,
  reconstructAnthropicToolResult,
  reconstructContent,
  reconstructOpenaiToolResult,
} from "./extract.js";
import { ReorderState, reorderDocuments } from "./reorder.js";

const DEFAULT_CONFIG = parseInterceptHeaders({});

const OPENAI_CHAT_BODY = {
  model: "claude-sonnet-4-6",
  messages: [
    {
      role: "system",
      content:
        "<documents><document>Doc A content here</document><document>Doc B content here</document><document>Doc C content here</document></documents>",
    },
    { role: "user", content: "What do these docs say?" },
  ],
};

const ANTHROPIC_MESSAGES_BODY = {
  model: "claude-sonnet-4-6",
  system:
    "<documents><document>Doc A content here</document><document>Doc B content here</document></documents>",
  messages: [{ role: "user", content: "Summarize the documents." }],
};

const LARGE_CONTENT = "x".repeat(600) + "\n".repeat(20) + "y".repeat(600);

const DEDUP_BODY = {
  messages: [
    {
      role: "assistant",
      content: "",
      tool_calls: [
        { id: "call_1", function: { name: "read_file", arguments: "{}" } },
        { id: "call_2", function: { name: "read_file", arguments: "{}" } },
      ],
    },
    { role: "tool", tool_call_id: "call_1", content: LARGE_CONTENT },
    { role: "tool", tool_call_id: "call_2", content: LARGE_CONTENT },
  ],
};

function makeLargeContent(prefix: string): string {
  return Array.from(
    { length: 20 },
    (_, i) => `${prefix} line ${i} ${"z".repeat(60)}`,
  ).join("\n");
}

describe("extract", () => {
  it("parseInterceptHeaders parses X-ContextPilot-* headers and defaults", () => {
    const parsed = parseInterceptHeaders({
      "X-ContextPilot-Enabled": "0",
      "x-contextpilot-mode": "xml_tag",
      "x-contextpilot-tag": "context",
      "x-contextpilot-separator": "===",
      "x-contextpilot-alpha": "0.05",
      "x-contextpilot-linkage": "single",
      "x-contextpilot-scope": "invalid",
    });

    expect(parsed).toEqual({
      enabled: false,
      mode: "xml_tag",
      tag: "context",
      separator: "===",
      alpha: 0.05,
      linkageMethod: "single",
      scope: "all",
    });

    const defaults = parseInterceptHeaders({});
    expect(defaults.enabled).toBe(true);
    expect(defaults.mode).toBe("auto");
    expect(defaults.tag).toBe("document");
    expect(defaults.separator).toBe("---");
    expect(defaults.alpha).toBe(0.001);
    expect(defaults.linkageMethod).toBe("average");
    expect(defaults.scope).toBe("all");
  });

  it("extractDocuments extracts XML-tagged documents", () => {
    const text =
      "<documents><document>A</document><document>B</document></documents>";
    const extraction = extractDocuments(text, DEFAULT_CONFIG);
    expect(extraction).not.toBeNull();
    expect(extraction?.mode).toBe("xml_tag");
    expect(extraction?.documents).toEqual(["A", "B"]);
    expect(extraction?.wrapperTag).toBe("documents");
    expect(extraction?.itemTag).toBe("document");
  });

  it("extractDocuments extracts numbered documents", () => {
    const extraction = extractDocuments(
      "[1] First doc\n[2] Second doc",
      parseInterceptHeaders({ "x-contextpilot-mode": "numbered" }),
    );
    expect(extraction).not.toBeNull();
    expect(extraction?.mode).toBe("numbered");
    expect(extraction?.documents).toEqual(["First doc", "Second doc"]);
  });

  it("extractDocuments extracts JSON results documents", () => {
    const extraction = extractDocuments(
      JSON.stringify({ results: [{ url: "a.com" }, { url: "b.com" }] }),
      parseInterceptHeaders({ "x-contextpilot-mode": "json_results" }),
    );
    expect(extraction).not.toBeNull();
    expect(extraction?.mode).toBe("json_results");
    expect(extraction?.documents).toEqual(["a.com", "b.com"]);
  });

  it("extractDocuments auto mode resolves XML > numbered > JSON", () => {
    const xml = extractDocuments(
      "<documents><document>[1] one</document><document>[2] two</document></documents>",
      DEFAULT_CONFIG,
    );
    expect(xml?.mode).toBe("xml_tag");

    const numbered = extractDocuments("[1] one\n[2] two", DEFAULT_CONFIG);
    expect(numbered?.mode).toBe("numbered");

    const json = extractDocuments(
      JSON.stringify({ results: [{ url: "one" }, { url: "two" }] }),
      DEFAULT_CONFIG,
    );
    expect(json?.mode).toBe("json_results");
  });

  it("extractDocuments returns null for fewer than two docs", () => {
    const numberedSingle = extractDocuments(
      "[1] Only one",
      parseInterceptHeaders({ "x-contextpilot-mode": "numbered" }),
    );
    expect(numberedSingle).toBeNull();

    const jsonSingle = extractDocuments(
      JSON.stringify({ results: [{ url: "only-one" }] }),
      parseInterceptHeaders({ "x-contextpilot-mode": "json_results" }),
    );
    expect(jsonSingle).toBeNull();
  });

  it("reconstructContent rebuilds XML while preserving tags", () => {
    const extraction = extractDocuments(
      "prefix<documents><document>A</document><document>B</document></documents>suffix",
      DEFAULT_CONFIG,
    );
    expect(extraction).not.toBeNull();
    if (!extraction) {
      throw new Error("expected extraction");
    }

    const rebuilt = reconstructContent(extraction, ["B", "A"]);
    expect(rebuilt).toContain("prefix");
    expect(rebuilt).toContain("suffix");
    expect(rebuilt).toContain("<documents>");
    expect(rebuilt).toContain("<document>B</document>");
    expect(rebuilt).toContain("<document>A</document>");
  });

  it("reconstructContent rebuilds numbered format", () => {
    const extraction = extractDocuments(
      "Lead\n[1] First\n[2] Second",
      parseInterceptHeaders({ "x-contextpilot-mode": "numbered" }),
    );
    expect(extraction).not.toBeNull();
    if (!extraction) {
      throw new Error("expected extraction");
    }

    const rebuilt = reconstructContent(extraction, ["Second", "First"]);
    expect(rebuilt).toContain("Lead");
    expect(rebuilt).toContain("[1] Second");
    expect(rebuilt).toContain("[2] First");
  });

  it("extractFromOpenaiChat extracts from system message", () => {
    const extraction = extractFromOpenaiChat(OPENAI_CHAT_BODY, DEFAULT_CONFIG);
    expect(extraction).not.toBeNull();
    expect(extraction?.[1]).toBe(0);
    expect(extraction?.[0].documents).toEqual([
      "Doc A content here",
      "Doc B content here",
      "Doc C content here",
    ]);
  });

  it("extractFromAnthropicMessages extracts from system string", () => {
    const extraction = extractFromAnthropicMessages(
      ANTHROPIC_MESSAGES_BODY,
      DEFAULT_CONFIG,
    );
    expect(extraction).not.toBeNull();
    expect(extraction?.documents).toEqual([
      "Doc A content here",
      "Doc B content here",
    ]);
  });

  it("extractFromOpenaiToolResults extracts tool-result documents", () => {
    const body = {
      messages: [
        { role: "tool", content: "<documents><document>A</document><document>B</document></documents>" },
      ],
    };
    const extractions = extractFromOpenaiToolResults(body, DEFAULT_CONFIG);
    expect(extractions).toHaveLength(1);
    expect(extractions[0]?.[0].documents).toEqual(["A", "B"]);
    expect(extractions[0]?.[1]).toEqual({
      msgIndex: 0,
      blockIndex: -1,
      innerBlockIndex: -1,
    });
  });

  it("extractFromAnthropicToolResults extracts tool_result blocks", () => {
    const body = {
      messages: [
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              content:
                "<documents><document>A</document><document>B</document></documents>",
            },
          ],
        },
      ],
    };
    const extractions = extractFromAnthropicToolResults(body, DEFAULT_CONFIG);
    expect(extractions).toHaveLength(1);
    expect(extractions[0]?.[0].documents).toEqual(["A", "B"]);
    expect(extractions[0]?.[1]).toEqual({
      msgIndex: 0,
      blockIndex: 0,
      innerBlockIndex: -1,
    });
  });

  it("FormatHandler OpenAI returns a working handler", () => {
    const handler = getFormatHandler("openai_chat");
    expect(handler.targetPath()).toBe("/v1/chat/completions");

    const body = structuredClone(OPENAI_CHAT_BODY);
    const all = handler.extractAll(body, DEFAULT_CONFIG);
    expect(all.systemExtraction).not.toBeNull();
    expect(all.hasExtractions).toBe(true);

    if (!all.systemExtraction) {
      throw new Error("expected system extraction");
    }

    handler.reconstructSystem(
      body,
      all.systemExtraction[0],
      ["Doc C content here", "Doc B content here", "Doc A content here"],
      all.systemExtraction[1],
    );
    expect(body.messages[0]?.content).toContain("Doc C content here");
  });

  it("FormatHandler Anthropic returns a working handler", () => {
    const handler = getFormatHandler("anthropic_messages");
    expect(handler.targetPath()).toBe("/v1/messages");

    const body = structuredClone(ANTHROPIC_MESSAGES_BODY);
    const all = handler.extractAll(body, DEFAULT_CONFIG);
    expect(all.systemExtraction).not.toBeNull();
    expect(all.hasExtractions).toBe(true);

    if (!all.systemExtraction) {
      throw new Error("expected system extraction");
    }

    handler.reconstructSystem(
      body,
      all.systemExtraction[0],
      ["Doc B content here", "Doc A content here"],
      all.systemExtraction[1],
    );
    expect(body.system).toContain("Doc B content here");
  });

  it("extractAllOpenai extracts from both system and tool results", () => {
    const body = {
      messages: [
        {
          role: "system",
          content:
            "<documents><document>Sys A</document><document>Sys B</document></documents>",
        },
        {
          role: "tool",
          content:
            "<documents><document>Tool A</document><document>Tool B</document></documents>",
        },
      ],
    };

    const all = extractAllOpenai(body, DEFAULT_CONFIG);
    expect(all.systemExtraction).not.toBeNull();
    expect(all.toolExtractions).toHaveLength(1);
    expect(all.totalDocuments).toBe(4);
  });

  it("extractSingleDocsFromOpenaiToolResults extracts single long docs", () => {
    const body = {
      messages: [
        {
          role: "tool",
          tool_call_id: "call_99",
          content: `Result:\n${"r".repeat(240)}`,
        },
      ],
    };

    const extracted = extractSingleDocsFromOpenaiToolResults(body, DEFAULT_CONFIG);
    expect(extracted).toHaveLength(1);
    expect(extracted[0]?.[0].toolCallId).toBe("call_99");
    expect(extracted[0]?.[0].content.length).toBeGreaterThanOrEqual(200);
    expect(extracted[0]?.[0].contentHash).toMatch(/^[0-9a-f]{64}$/);
  });

  it("reconstructOpenaiToolResult reconstructs a tool result in-place", () => {
    const body = {
      messages: [
        {
          role: "tool",
          content:
            "<documents><document>A</document><document>B</document></documents>",
        },
      ],
    };

    const extractions = extractFromOpenaiToolResults(body, DEFAULT_CONFIG);
    expect(extractions).toHaveLength(1);
    const first = extractions[0];
    if (!first) {
      throw new Error("expected extraction");
    }

    reconstructOpenaiToolResult(body, first[0], ["B", "A"], first[1]);
    expect(body.messages[0]?.content).toContain("<document>B</document>");
    expect(body.messages[0]?.content).toContain("<document>A</document>");
  });

  it("reconstructAnthropicToolResult reconstructs a tool result in-place", () => {
    const body = {
      messages: [
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              content:
                "<documents><document>A</document><document>B</document></documents>",
            },
          ],
        },
      ],
    };

    const extractions = extractFromAnthropicToolResults(body, DEFAULT_CONFIG);
    expect(extractions).toHaveLength(1);
    const first = extractions[0];
    if (!first) {
      throw new Error("expected extraction");
    }

    reconstructAnthropicToolResult(body, first[0], ["B", "A"], first[1]);
    expect(body.messages[0]?.content[0]?.content).toContain("<document>B</document>");
    expect(body.messages[0]?.content[0]?.content).toContain("<document>A</document>");
  });
});

describe("dedup", () => {
  it("contentDefinedChunking splits text into multiple blocks at boundaries", () => {
    const text = Array.from({ length: 12 }, (_, i) => `line-${i}`).join("\n");
    const blocks = contentDefinedChunking(text, 1);
    expect(blocks).toHaveLength(2);
    expect(blocks[0]?.split("\n")).toHaveLength(5);
    expect(blocks[1]?.split("\n")).toHaveLength(7);
  });

  it("contentDefinedChunking returns one block for short text", () => {
    const short = "a\nb\nc\nd\ne";
    const blocks = contentDefinedChunking(short);
    expect(blocks).toEqual([short]);
  });

  it("hashBlock is consistent and returns 20-char hex", () => {
    const h1 = hashBlock("  abc\n");
    const h2 = hashBlock("abc");
    expect(h1).toBe(h2);
    expect(h1).toMatch(/^[0-9a-f]{20}$/);
  });

  it("dedupChatCompletions returns zero savings with no duplicates", () => {
    const body = {
      messages: [
        {
          role: "assistant",
          tool_calls: [
            { id: "a", function: { name: "read_file" } },
            { id: "b", function: { name: "read_file" } },
          ],
        },
        { role: "tool", tool_call_id: "a", content: makeLargeContent("first") },
        { role: "tool", tool_call_id: "b", content: makeLargeContent("second") },
      ],
    };

    const before = body.messages[2]?.content;
    const result = dedupChatCompletions(body, { chunkModulus: 1 });
    expect(result.blocksDeduped).toBe(0);
    expect(result.charsSaved).toBe(0);
    expect(body.messages[2]?.content).toBe(before);
  });

  it("dedupChatCompletions dedups duplicate blocks and inserts references", () => {
    const body = structuredClone(DEDUP_BODY);
    const result = dedupChatCompletions(body, { chunkModulus: 1 });
    expect(result.blocksDeduped).toBeGreaterThan(0);
    expect(result.systemBlocksMatched).toBe(0);
    expect(result.charsSaved).toBeGreaterThan(0);
    expect(body.messages[2]?.content).toContain(
      "identical to earlier read_file result",
    );
  });

  it("dedupChatCompletions dedups tool content against pre-scanned system blocks", () => {
    const shared = makeLargeContent("shared");
    const body = {
      messages: [
        {
          role: "assistant",
          tool_calls: [{ id: "call_1", function: { name: "read_file" } }],
        },
        { role: "tool", tool_call_id: "call_1", content: shared },
      ],
    };

    const result = dedupChatCompletions(body, shared, { chunkModulus: 1 });
    expect(result.blocksDeduped).toBeGreaterThan(0);
    expect(result.systemBlocksMatched).toBeGreaterThan(0);
    expect(body.messages[1]?.content).toContain(
      "identical to earlier system prompt result",
    );
  });

  it("dedupChatCompletions dedups assistant fenced code blocks against seen tool blocks", () => {
    const shared = makeLargeContent("code-shared");
    const assistantWithCode = [
      "Here is the generated code:",
      "```ts",
      shared,
      "```",
    ].join("\n");

    const body = {
      messages: [
        {
          role: "assistant",
          tool_calls: [{ id: "call_1", function: { name: "read_file" } }],
        },
        { role: "tool", tool_call_id: "call_1", content: shared },
        { role: "assistant", content: assistantWithCode },
      ],
    };

    const result = dedupChatCompletions(body, { chunkModulus: 1 });
    expect(result.blocksDeduped).toBeGreaterThan(0);
    expect(result.systemBlocksMatched).toBe(0);
    expect(body.messages[2]?.content).toContain(
      "identical to earlier read_file result",
    );
  });

  it("dedupChatCompletions remains backward-compatible when systemContent is omitted", () => {
    const body = {
      messages: [
        {
          role: "assistant",
          tool_calls: [
            { id: "a", function: { name: "read_file" } },
            { id: "b", function: { name: "read_file" } },
          ],
        },
        { role: "tool", tool_call_id: "a", content: makeLargeContent("same") },
        { role: "tool", tool_call_id: "b", content: makeLargeContent("same") },
      ],
    };

    const result = dedupChatCompletions(body, { chunkModulus: 1 });
    expect(result.blocksDeduped).toBeGreaterThan(0);
    expect(result.systemBlocksMatched).toBe(0);
    expect(body.messages[2]?.content).toContain(
      "identical to earlier read_file result",
    );
  });

  it("dedupChatCompletions skips short content", () => {
    const short = "s".repeat(300);
    const body = {
      messages: [
        {
          role: "assistant",
          tool_calls: [
            { id: "a", function: { name: "search" } },
            { id: "b", function: { name: "search" } },
          ],
        },
        { role: "tool", tool_call_id: "a", content: short },
        { role: "tool", tool_call_id: "b", content: short },
      ],
    };

    const result = dedupChatCompletions(body);
    expect(result.blocksTotal).toBe(0);
    expect(result.blocksDeduped).toBe(0);
    expect(result.charsSaved).toBe(0);
    expect(body.messages[2]?.content).toBe(short);
  });

  it("dedupResponsesApi dedups duplicate function_call_output content", () => {
    const body = {
      input: [
        { type: "function_call", call_id: "r1", name: "search" },
        { type: "function_call", call_id: "r2", name: "search" },
        { type: "function_call_output", call_id: "r1", output: LARGE_CONTENT },
        { type: "function_call_output", call_id: "r2", output: LARGE_CONTENT },
      ],
    };

    const result = dedupResponsesApi(body, { chunkModulus: 1 });
    expect(result.blocksDeduped).toBeGreaterThan(0);
    expect(result.charsSaved).toBeGreaterThan(0);
    expect(body.input[3]?.output).toContain("identical to earlier search result");
  });

  it("buildToolNameMapOpenai maps tool_call_id to function name", () => {
    const mapping = buildToolNameMapOpenai([
      {
        role: "assistant",
        tool_calls: [
          { id: "id_1", function: { name: "read_file" } },
          { id: "id_2", function: { name: "search" } },
        ],
      },
      { role: "user" },
    ]);

    expect(mapping).toEqual({ id_1: "read_file", id_2: "search" });
  });
});

describe("cache-control", () => {
  it("injectAnthropicCacheControl converts string system into array with cache_control", () => {
    const body: Record<string, unknown> = { system: "system text", messages: [] };
    const result = injectAnthropicCacheControl(body);

    const system = result.system as Array<{
      type?: string;
      text?: string;
      cache_control?: { type: string };
    }>;
    expect(Array.isArray(system)).toBe(true);
    expect(system[0]).toEqual({
      type: "text",
      text: "system text",
      cache_control: { type: "ephemeral" },
    });
  });

  it("injectAnthropicCacheControl adds cache_control to last system block", () => {
    const body: Record<string, unknown> = {
      system: [
        { type: "text", text: "first" },
        { type: "text", text: "last" },
      ],
      messages: [],
    };
    const result = injectAnthropicCacheControl(body);
    const system = result.system as Array<{
      type?: string;
      text?: string;
      cache_control?: { type: string };
    }>;

    expect(system[0]?.cache_control).toBeUndefined();
    expect(system[1]?.cache_control).toEqual({ type: "ephemeral" });
  });

  it("injectAnthropicCacheControl adds cache_control to large tool_result blocks", () => {
    const body: Record<string, unknown> = {
      messages: [
        {
          role: "user",
          content: [
            { type: "tool_result", content: "x".repeat(1200) },
            {
              type: "tool_result",
              content: [
                { type: "text", text: "a".repeat(800) },
                { type: "text", text: "b".repeat(300) },
              ],
            },
          ],
        },
      ],
    };

    const result = injectAnthropicCacheControl(body);
    const messages = result.messages as Array<{
      role?: string;
      content?: Array<{
        type?: string;
        content?: string | Array<{ type?: string; text?: string; cache_control?: { type: string } }>;
        cache_control?: { type: string };
      }>;
    }>;

    const firstToolResult = messages[0]?.content?.[0];
    const secondToolResult = messages[0]?.content?.[1];
    const secondInner = secondToolResult?.content as Array<{
      type?: string;
      text?: string;
      cache_control?: { type: string };
    }>;

    expect(firstToolResult?.cache_control).toEqual({ type: "ephemeral" });
    expect(secondInner[0]?.cache_control).toBeUndefined();
    expect(secondInner[1]?.cache_control).toEqual({ type: "ephemeral" });
  });

  it("injectAnthropicCacheControl does not mutate original body", () => {
    const body: Record<string, unknown> = {
      system: "immutable",
      messages: [{ role: "user", content: [] }],
    };
    const snapshot = structuredClone(body);
    const result = injectAnthropicCacheControl(body);

    expect(body).toEqual(snapshot);
    expect(result).not.toBe(body);
  });

  it("injectOpenAICacheControl is a no-op", () => {
    const body: Record<string, unknown> = {
      messages: [{ role: "system", content: "keep" }],
    };
    const result = injectOpenAICacheControl(body);
    expect(result).toBe(body);
  });

  it("injectCacheControl dispatches by provider", () => {
    const anthropicBody: Record<string, unknown> = { system: "hello", messages: [] };
    const openaiBody: Record<string, unknown> = { messages: [] };

    const anthropicResult = injectCacheControl(anthropicBody, "anthropic");
    const openaiResult = injectCacheControl(openaiBody, "openai");

    expect(anthropicResult).not.toBe(anthropicBody);
    expect(Array.isArray(anthropicResult.system)).toBe(true);
    expect(openaiResult).toBe(openaiBody);
  });
});

describe("reorder", () => {
  it("ReorderState first call matches deterministic hash sort", () => {
    const docs = ["Doc C", "Doc A", "Doc B"];
    const state = new ReorderState();
    const [stateOrder] = state.reorder(docs);
    const [statelessOrder] = reorderDocuments(docs);
    expect(stateOrder).toEqual(statelessOrder);
  });

  it("ReorderState second call keeps known order and appends new docs", () => {
    const state = new ReorderState();
    const [first] = state.reorder(["alpha", "beta", "gamma"]);
    const [second] = state.reorder(["gamma", "alpha", "delta"]);

    const knownOrder = first.filter((doc) => doc === "gamma" || doc === "alpha");
    expect(second.slice(0, knownOrder.length)).toEqual(knownOrder);
    expect(second[second.length - 1]).toBe("delta");
  });

  it("ReorderState reset restores first-call behavior", () => {
    const docs = ["alpha", "beta", "gamma"];
    const state = new ReorderState();

    state.reorder(docs);
    state.reorder(["gamma", "alpha", "delta"]);
    state.reset();

    const [afterReset] = state.reorder(docs);
    const [expected] = reorderDocuments(docs);
    expect(afterReset).toEqual(expected);
  });

  it("reorderDocuments is deterministic and stateless", () => {
    const docs = ["one", "two", "three", "four"];
    const first = reorderDocuments(docs);
    const second = reorderDocuments(docs);
    expect(first).toEqual(second);
  });

  it("reorderDocuments returns correct originalOrder and newOrder mappings", () => {
    const docs = ["one", "two", "three", "four"];
    const [reordered, originalOrder, newOrder] = reorderDocuments(docs);

    expect(originalOrder).toHaveLength(docs.length);
    expect(newOrder).toHaveLength(docs.length);

    for (let newIndex = 0; newIndex < reordered.length; newIndex += 1) {
      const originalIndex = originalOrder[newIndex];
      expect(reordered[newIndex]).toBe(docs[originalIndex]);
    }

    for (let originalIndex = 0; originalIndex < docs.length; originalIndex += 1) {
      const mappedNewIndex = newOrder[originalIndex];
      expect(reordered[mappedNewIndex]).toBe(docs[originalIndex]);
    }
  });

  it("ReorderState preserves known-doc prefix stability across calls", () => {
    const state = new ReorderState();
    const knownDocs = ["alpha", "beta", "gamma"];

    const [first] = state.reorder(knownDocs);
    const [second] = state.reorder(["gamma", "beta", "alpha", "delta"]);
    const [third] = state.reorder(["alpha", "epsilon", "gamma", "beta", "zeta"]);

    const knownPrefix = first.filter((doc) =>
      knownDocs.includes(doc),
    );

    expect(second.slice(0, knownPrefix.length)).toEqual(knownPrefix);
    expect(third.slice(0, knownPrefix.length)).toEqual(knownPrefix);
  });
});
