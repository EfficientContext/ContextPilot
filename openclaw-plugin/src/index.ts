import { Type } from "@sinclair/typebox";
import { delegateCompactionToRuntime } from "openclaw/plugin-sdk/core";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

import { injectCacheControl } from "./engine/cache-control.js";
import { dedupChatCompletions } from "./engine/dedup.js";
import { getFormatHandler, type InterceptConfig } from "./engine/extract.js";
import { ContextPilot } from "./engine/live-index.js";

type Scope = "all" | "system" | "tool_results";

function parseScope(value: unknown): Scope {
  if (value === "system" || value === "tool_results" || value === "all") {
    return value;
  }
  return "all";
}

function reorderWithEngine(engine: ContextPilot, docs: string[]): string[] {
  const [reordered] = engine.reorder(docs);
  if (!Array.isArray(reordered) || !Array.isArray(reordered[0])) {
    return docs;
  }
  const candidate = reordered[0];
  if (!candidate.every((entry) => typeof entry === "string")) {
    return docs;
  }
  return candidate as string[];
}

interface Message {
  role: string;
  content: unknown;
}

interface TextBlock {
  type?: string;
  text?: string;
}

interface ToolUseIdCarrier {
  tool_use_id?: unknown;
  toolUseId?: unknown;
}

function normalizeMessageContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }

  const parts: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") {
      continue;
    }
    const textBlock = block as TextBlock;
    if (textBlock.type === "text" && typeof textBlock.text === "string") {
      parts.push(textBlock.text);
    }
  }
  return parts.join("\n");
}

function extractToolUseId(message: Message, idx: number): string {
  const withToolUseId = message as Message & ToolUseIdCarrier;
  if (typeof withToolUseId.tool_use_id === "string" && withToolUseId.tool_use_id) {
    return withToolUseId.tool_use_id;
  }
  if (typeof withToolUseId.toolUseId === "string" && withToolUseId.toolUseId) {
    return withToolUseId.toolUseId;
  }
  return `tool_${idx}`;
}

export default definePluginEntry({
  id: "contextpilot",
  name: "ContextPilot",
  description: "Optimizes context via reordering, deduplication, and cache control injection.",
  register: (api) => {
    const config = {
      scope: parseScope(api.pluginConfig?.scope),
    };

    const engine = new ContextPilot(0.001, false, "average");

    let assembleCount = 0;
    let totalCharsSaved = 0;

    api.registerContextEngine("contextpilot", () => ({
      info: {
        id: "contextpilot",
        name: "ContextPilot",
        ownsCompaction: false,
      },

      async ingest() {
        return { ingested: true };
      },

      async assemble({ messages, system }: { messages: Message[]; system?: string }) {
        const interceptConfig: InterceptConfig = {
          enabled: true,
          mode: "auto",
          tag: "document",
          separator: "---",
          alpha: 0.001,
          linkageMethod: "average",
          scope: config.scope,
        };

        const convertedMessages = messages.map((msg, idx) => {
          if (msg.role === "toolResult") {
            const content = normalizeMessageContent(msg.content);
            return {
              role: "user",
              content: [{
                type: "tool_result",
                tool_use_id: extractToolUseId(msg, idx),
                content: content,
              }],
            };
          }
          return msg;
        });

        const convertedBody: Record<string, unknown> = {
          messages: convertedMessages,
          system: system,
        };

        const handler = getFormatHandler("anthropic_messages");
        const multi = handler.extractAll(convertedBody, interceptConfig);

        const reorderDocs = (docs: string[]): string[] => {
          if (docs.length < 2) {
            return docs;
          }
          return reorderWithEngine(engine, docs);
        };

        if (multi.systemExtraction) {
          const [extraction, sysIdx] = multi.systemExtraction;
          if (extraction.documents.length >= 2) {
            const reordered = reorderDocs(extraction.documents);
            handler.reconstructSystem(convertedBody, extraction, reordered, sysIdx);
          }
        }

        for (const [extraction, location] of multi.toolExtractions) {
          if (extraction.documents.length >= 2) {
            const reordered = reorderDocs(extraction.documents);
            handler.reconstructToolResult(convertedBody, extraction, reordered, location);
          }
        }

        const convertedMessageList = Array.isArray(convertedBody.messages)
          ? (convertedBody.messages as Array<{ content?: unknown }>)
          : [];

        const finalMessages = convertedMessageList.map((msg, idx) => {
          const original = messages[idx];
          if (original?.role === "toolResult") {
            const block = Array.isArray(msg.content)
              ? msg.content[0]
              : null;
            const extractedContent = block && typeof block === "object"
              ? (block as { content?: unknown }).content
              : undefined;

            if (Array.isArray(original.content)) {
              const newContentArray = original.content.map((entry) => {
                if (
                  entry
                  && typeof entry === "object"
                  && (entry as TextBlock).type === "text"
                  && typeof extractedContent === "string"
                ) {
                  return {
                    ...(entry as Record<string, unknown>),
                    text: extractedContent,
                  };
                }
                return entry;
              });
              return { ...original, content: newContentArray };
            } else if (typeof extractedContent === "string") {
              return { ...original, content: extractedContent };
            }
            return original;
          }
          return msg;
        });

        const finalBody: Record<string, unknown> = {
          messages: finalMessages,
          system: system,
        };

        const dedupResult = dedupChatCompletions(finalBody, system);
        totalCharsSaved += dedupResult.charsSaved;

        const optimizedBody = injectCacheControl(finalBody, "anthropic");

        assembleCount++;

        if (dedupResult.charsSaved > 0 || assembleCount % 5 === 0) {
          const estimatedTokensSaved = Math.round(totalCharsSaved / 4);
          const estimatedCostSaved = (estimatedTokensSaved * 0.003 / 1000).toFixed(4);
          console.error(`[ContextPilot] Stats: ${assembleCount} requests, ${totalCharsSaved.toLocaleString()} chars saved (~${estimatedTokensSaved.toLocaleString()} tokens, ~$${estimatedCostSaved})`);
        }

        return {
          messages: (optimizedBody.messages as Message[]) || messages,
          system: optimizedBody.system as string | undefined,
          estimatedTokens: 0,
        };
      },

      async compact(params) {
        return await delegateCompactionToRuntime(params);
      },
    }));

    api.registerTool({
      name: "contextpilot_status",
      description: "Report ContextPilot engine state",
      parameters: Type.Object({}),
      async execute(_toolCallId: string, _params: unknown) {
        const stats = engine.getStats();
        const lines = [
          "ContextPilot Engine Status:",
          `  Scope: ${config.scope}`,
          `  Contexts assembled: ${assembleCount}`,
          `  Total chars saved: ${totalCharsSaved.toLocaleString()}`,
          `  Live index: ${engine.isLive ? "active" : "warming"}`,
          `  Nodes: ${Number(stats.num_nodes ?? 0)}`,
          `  Active nodes: ${Number(stats.active_nodes ?? 0)}`,
          `  Requests tracked: ${Number(stats.num_requests ?? 0)}`,
          `  Total searches: ${Number(stats.total_searches ?? 0)}`,
          `  Total insertions: ${Number(stats.total_insertions ?? 0)}`,
          `  Total removals: ${Number(stats.total_removals ?? 0)}`,
          `  Avg search time (us): ${Number(stats.avg_search_time_us ?? 0).toFixed(2)}`,
        ];

        return {
          content: [
            {
              type: "text" as const,
              text: lines.join("\n"),
            },
          ],
        };
      },
    });
  },
});
