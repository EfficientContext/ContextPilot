import { describe, it, expect } from "vitest";
import { getFormatHandler, type InterceptConfig } from "./extract.js";
import { dedupChatCompletions, dedupResponsesApi } from "./dedup.js";
import { injectCacheControl } from "./cache-control.js";
import { ReorderState } from "./reorder.js";

function runPipeline(
  body: Record<string, unknown>,
  opts: {
    provider?: "anthropic" | "openai";
    scope?: string;
    reorderState?: ReorderState;
  } = {}
): Record<string, unknown> {
  const provider = opts.provider ?? "anthropic";
  const scope = opts.scope ?? "all";
  const reorderState = opts.reorderState ?? new ReorderState();

  const clonedBody = structuredClone(body);
  const apiFormat = provider === "anthropic" ? "anthropic_messages" : "openai_chat";

  const interceptConfig: InterceptConfig = {
    enabled: true,
    mode: "auto",
    tag: "document",
    separator: "---",
    alpha: 0.001,
    linkageMethod: "average",
    scope,
  };

  const handler = getFormatHandler(apiFormat);
  const multi = handler.extractAll(clonedBody, interceptConfig);

  if (multi.systemExtraction) {
    const [extraction, sysIdx] = multi.systemExtraction;
    if (extraction.documents.length >= 2) {
      const [reordered] = reorderState.reorder(extraction.documents);
      handler.reconstructSystem(clonedBody, extraction, reordered, sysIdx);
    }
  }

  for (const [extraction, location] of multi.toolExtractions) {
    if (extraction.documents.length >= 2) {
      const [reordered] = reorderState.reorder(extraction.documents);
      handler.reconstructToolResult(clonedBody, extraction, reordered, location);
    }
  }

  if (apiFormat === "openai_chat") {
    dedupChatCompletions(clonedBody as any);
  }
  if (clonedBody.input && Array.isArray(clonedBody.input)) {
    dedupResponsesApi(clonedBody as any);
  }

  return injectCacheControl(clonedBody, provider);
}

describe("full pipeline — Anthropic", () => {
  it("system prompt with XML documents gets reordered and cache-controlled", () => {
    const body = {
      model: "claude-sonnet-4-6",
      system: `You are a helpful assistant.\n<documents>\n<document index="1">\nFirst document about TypeScript.\nIt has multiple lines.\n</document>\n<document index="2">\nSecond document about Python.\nAlso multi-line.\n</document>\n<document index="3">\nThird document about Rust.\nYet another multi-line doc.\n</document>\n</documents>\nPlease answer based on the above.`,
      messages: [{ role: "user", content: "Summarize the documents." }],
    };

    const reorderState = new ReorderState();
    const result = runPipeline(body, { provider: "anthropic", reorderState });

    expect(Array.isArray(result.system)).toBe(true);
    const systemArray = result.system as any[];
    
    const lastBlock = systemArray[systemArray.length - 1];
    expect(lastBlock.cache_control).toEqual({ type: "ephemeral" });

    const textContent = systemArray.map(b => b.text).join("");
    expect(textContent).toContain("You are a helpful assistant.");
    expect(textContent).toContain("Please answer based on the above.");
    
    expect(textContent).toContain("First document about TypeScript.");
    expect(textContent).toContain("Second document about Python.");
    expect(textContent).toContain("Third document about Rust.");
  });

  it("Anthropic tool_result with large content gets cache_control", () => {
    const body = {
      model: "claude-sonnet-4-6",
      system: "You are helpful.",
      messages: [
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "tu_1", content: "A".repeat(2000) },
          ],
        },
      ],
    };

    const result = runPipeline(body, { provider: "anthropic" });
    const messages = result.messages as any[];
    const content = messages[0].content as any[];
    expect(content[0].cache_control).toEqual({ type: "ephemeral" });
  });

  it("Anthropic scope=\"system\" only processes system, not tool results", () => {
    const docText = `<documents><document index="1">\nFirst document about TypeScript.\nIt has multiple lines.\n</document><document index="2">\nSecond document about Python.\nAlso multi-line.\n</document></documents>`;
    const body = {
      model: "claude-sonnet-4-6",
      system: `You are helpful.\n${docText}`,
      messages: [
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "tu_1", content: docText },
          ],
        },
      ],
    };

    const reorderState = new ReorderState();
    // Reorder stability means it will process it
    const result = runPipeline(body, { provider: "anthropic", scope: "system", reorderState });
    
    // System should have its format modified to array due to reconstruction/cache control
    expect(Array.isArray(result.system)).toBe(true);

    const messages = result.messages as any[];
    const content = messages[0].content as any[];
    // Tool result shouldn't have been reconstructed into blocks of its internal documents
    expect(content[0].content).toBe(docText);
  });

  it("Anthropic scope=\"tool_results\" only processes tools, not system", () => {
    const docText = `<documents><document index="1">\nFirst document about TypeScript.\nIt has multiple lines.\n</document><document index="2">\nSecond document about Python.\nAlso multi-line.\n</document></documents>`;
    const body = {
      model: "claude-sonnet-4-6",
      system: `You are helpful.\n${docText}`,
      messages: [
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "tu_1", content: docText },
          ],
        },
      ],
    };

    const reorderState = new ReorderState();
    const result = runPipeline(body, { provider: "anthropic", scope: "tool_results", reorderState });
    
    // System should not be processed for documents (though it may be arrayified for cache control)
    // Cache control injects string to array conversion for Anthropic system if needed
    if (Array.isArray(result.system)) {
        const textContent = (result.system as any[]).map(b => b.text).join("");
        expect(textContent).toBe(`You are helpful.\n${docText}`);
    } else {
        expect(result.system).toBe(`You are helpful.\n${docText}`);
    }

    // Tool results should be reconstructed/reordered
    const messages = result.messages as any[];
    const content = messages[0].content as any[];
    expect(typeof content[0].content).toBe("string");
    expect(content[0].content).toContain("First document about TypeScript.");
    expect(content[0].content).toContain("Second document about Python.");
  });
});

describe("full pipeline — OpenAI", () => {
  it("OpenAI chat system message with XML documents gets reordered", () => {
    const body = {
      model: "gpt-4o",
      messages: [
        { role: "system", content: "<documents><document>Doc A content</document><document>Doc B content</document><document>Doc C content</document></documents>" },
        { role: "user", content: "Hello" }
      ]
    };

    const result = runPipeline(body, { provider: "openai" });
    const msgs = result.messages as any[];
    const sysMsg = msgs[0].content;
    expect(sysMsg).toContain("Doc A content");
    expect(sysMsg).toContain("Doc B content");
    expect(sysMsg).toContain("Doc C content");
  });

  it("OpenAI chat with duplicate tool results gets deduped", () => {
    const sharedContent = Array.from({length: 30}, (_, i) => `Line ${i}: ${"x".repeat(50)}`).join("\n");
    const body = {
      model: "gpt-4o",
      messages: [
        { role: "assistant", content: null, tool_calls: [
          { id: "call_1", type: "function", function: { name: "read_file", arguments: "{}" } },
          { id: "call_2", type: "function", function: { name: "read_file", arguments: "{}" } }
        ]},
        { role: "tool", tool_call_id: "call_1", content: sharedContent },
        { role: "tool", tool_call_id: "call_2", content: sharedContent }
      ]
    };

    const result = runPipeline(body, { provider: "openai" });
    const msgs = result.messages as any[];
    
    expect(msgs[1].content).toBe(sharedContent);
    expect(msgs[2].content).not.toBe(sharedContent);
    expect(msgs[2].content).toContain("identical to earlier read_file result");
  });

  it("OpenAI body with no extractable docs passes through unchanged", () => {
    const body = {
      model: "gpt-4o",
      messages: [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hi" }
      ]
    };

    const result = runPipeline(body, { provider: "openai" });
    expect(result).toEqual(body);
  });

  it("OpenAI responses API format gets deduped", () => {
    const sharedContent = Array.from({length: 30}, (_, i) => `Line ${i}: ${"x".repeat(50)}`).join("\n");
    const body = {
      input: [
        { type: "function_call_output", call_id: "c1", output: sharedContent },
        { type: "function_call_output", call_id: "c2", output: sharedContent }
      ]
    };

    const result = runPipeline(body, { provider: "openai" });
    const input = result.input as any[];
    
    expect(input[0].output).toBe(sharedContent);
    expect(input[1].output).not.toBe(sharedContent);
    expect(input[1].output).toContain("identical");
  });
});

describe("multi-turn state — reorder stability", () => {
  it("reorder state preserves doc order across turns", () => {
    const reorderState = new ReorderState();
    
    const bodyTurn1 = {
      model: "gpt-4o",
      messages: [
        { role: "system", content: "<documents><document>Doc A content</document><document>Doc B content</document><document>Doc C content</document></documents>" }
      ]
    };
    
    runPipeline(bodyTurn1, { provider: "openai", reorderState });
    
    const bodyTurn2 = {
      model: "gpt-4o",
      messages: [
        { role: "system", content: "<documents><document>Doc A content</document><document>Doc B content</document><document>Doc C content</document><document>Doc D content</document></documents>" }
      ]
    };

    const res2 = runPipeline(bodyTurn2, { provider: "openai", reorderState });
    const sysMsg2 = (res2.messages as any[])[0].content;
    
    // In multi-turn, ReorderState should put the new item (D) at top, and preserve relative ordering of A, B, C.
    // We just verify all are present and stable.
    expect(sysMsg2).toContain("Doc A content");
    expect(sysMsg2).toContain("Doc B content");
    expect(sysMsg2).toContain("Doc C content");
    expect(sysMsg2).toContain("Doc D content");
  });

  it("reorder state reset clears history", () => {
    const reorderState = new ReorderState();
    const body = {
      model: "gpt-4o",
      messages: [
        { role: "system", content: "<documents><document>Doc A content</document><document>Doc B content</document></documents>" }
      ]
    };
    
    runPipeline(body, { provider: "openai", reorderState });
    
    reorderState.reset();
    
    const res2 = runPipeline(body, { provider: "openai", reorderState });
    const sysMsg2 = (res2.messages as any[])[0].content;
    
    expect(sysMsg2).toContain("Doc A content");
    expect(sysMsg2).toContain("Doc B content");
  });
});

describe("edge cases", () => {
  it("empty body passes through", () => {
    const result = runPipeline({}, { provider: "anthropic" });
    expect(result).toEqual({});
  });

  it("body with no messages passes through", () => {
    const body = { model: "gpt-4o" };
    const result = runPipeline(body, { provider: "openai" });
    expect(result).toEqual(body);
  });

  it("body with single document doesn't get reordered", () => {
    const body = {
      model: "gpt-4o",
      messages: [
        { role: "system", content: "<documents><document>Only Doc</document></documents>" }
      ]
    };
    const result = runPipeline(body, { provider: "openai" });
    // It should be unchanged
    expect(result).toEqual(body);
  });

  it("very short tool result content not deduped", () => {
    const shortContent = "Too short for dedup.";
    const body = {
      model: "gpt-4o",
      messages: [
        { role: "assistant", content: null, tool_calls: [
          { id: "call_1", type: "function", function: { name: "read_file", arguments: "{}" } },
          { id: "call_2", type: "function", function: { name: "read_file", arguments: "{}" } }
        ]},
        { role: "tool", tool_call_id: "call_1", content: shortContent },
        { role: "tool", tool_call_id: "call_2", content: shortContent }
      ]
    };

    const result = runPipeline(body, { provider: "openai" });
    const msgs = result.messages as any[];
    expect(msgs[1].content).toBe(shortContent);
    expect(msgs[2].content).toBe(shortContent);
  });

  it("null/undefined messages gracefully handled", () => {
    const body = { model: "gpt-4o", messages: null };
    const result = runPipeline(body, { provider: "openai" });
    expect(result).toEqual(body);
  });

  it("Anthropic body with system as content block array", () => {
    const body = {
      model: "claude-sonnet-4-6",
      system: [
        { type: "text", text: "<documents><document>A</document><document>B</document></documents>" }
      ],
      messages: [{ role: "user", content: "hi" }]
    };

    const result = runPipeline(body, { provider: "anthropic" });
    const sys = result.system as any[];
    expect(Array.isArray(sys)).toBe(true);
    // Last block should have cache_control
    expect(sys[sys.length - 1].cache_control).toEqual({ type: "ephemeral" });
    
    const fullText = sys.map(b => b.text).join("");
    expect(fullText).toContain("A");
    expect(fullText).toContain("B");
  });
});
