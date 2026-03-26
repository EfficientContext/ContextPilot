import { Type } from "@sinclair/typebox";
import {
  definePluginEntry,
  type ProviderResolveDynamicModelContext,
  type ProviderWrapStreamFnContext,
} from "openclaw/plugin-sdk/plugin-entry";
import { createProviderApiKeyAuthMethod } from "openclaw/plugin-sdk/provider-auth";

import { injectCacheControl } from "./engine/cache-control.js";
import { dedupChatCompletions, dedupResponsesApi } from "./engine/dedup.js";
import { getFormatHandler, type InterceptConfig } from "./engine/extract.js";
import { ContextPilotIndexClient } from "./engine/http-client.js";
import { ContextPilot } from "./engine/live-index.js";

const PROVIDER_ID = "contextpilot";
type BackendProvider = "anthropic" | "openai" | "sglang";

function parseBackendProvider(value: unknown): BackendProvider {
  if (value === "openai" || value === "sglang") {
    return value;
  }
  return "anthropic";
}

function parseScope(value: unknown): "all" | "system" | "tool_results" {
  if (value === "system" || value === "tool_results" || value === "all") {
    return value;
  }
  return "all";
}

function detectApiFormat(
  body: Record<string, unknown>,
  backendProvider: BackendProvider,
): "openai_chat" | "anthropic_messages" {
  if (backendProvider === "anthropic") {
    return "anthropic_messages";
  }
  if (backendProvider === "openai") {
    return "openai_chat";
  }
  return "system" in body ? "anthropic_messages" : "openai_chat";
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

async function reorderWithClient(
  client: ContextPilotIndexClient,
  docs: string[],
): Promise<string[]> {
  const encodedDocs = docs.map((doc) => Array.from(doc, (ch) => ch.charCodeAt(0)));
  const result = await client.reorder(encodedDocs, 0.001, false, "average");

  if (result === null) {
    return docs;
  }

  const [, originalIndices] = result;
  if (!Array.isArray(originalIndices) || originalIndices.length !== docs.length) {
    return docs;
  }

  const reordered = originalIndices.map((index) => {
    if (typeof index !== "number" || index < 0 || index >= docs.length) {
      return null;
    }
    return docs[index];
  });

  return reordered.includes(null) ? docs : (reordered as string[]);
}

function formatJson(value: unknown): string {
  return value === null || value === undefined ? "unavailable" : JSON.stringify(value);
}

export default definePluginEntry({
  id: "contextpilot",
  name: "ContextPilot",
  description: "Optimizes LLM requests in-process via extraction, dedup, caching, and reordering.",
  register: (api) => {
    const config = {
      backendProvider: parseBackendProvider(api.pluginConfig?.backendProvider),
      scope: parseScope(api.pluginConfig?.scope),
      indexServerUrl: String(api.pluginConfig?.indexServerUrl || "http://localhost:8765"),
    };

    const isSglang = config.backendProvider === "sglang";
    const engine = isSglang ? null : new ContextPilot(0.001, false, "average");
    const client = isSglang ? new ContextPilotIndexClient(config.indexServerUrl) : null;

    let requestCount = 0;
    let totalCharsSaved = 0;

    api.registerProvider({
      id: PROVIDER_ID,
      label: "ContextPilot",
      docsPath: "/providers/contextpilot",
      envVars: isSglang
        ? []
        : [config.backendProvider === "anthropic" ? "ANTHROPIC_API_KEY" : "OPENAI_API_KEY"],
      auth: isSglang
        ? []
        : [
          createProviderApiKeyAuthMethod({
            providerId: PROVIDER_ID,
            methodId: "api-key",
            label: config.backendProvider === "anthropic" ? "Anthropic API key" : "OpenAI API key",
            hint: "API key for the backend LLM provider",
            optionKey: config.backendProvider === "anthropic" ? "anthropicApiKey" : "openaiApiKey",
            flagName: config.backendProvider === "anthropic" ? "--anthropic-api-key" : "--openai-api-key",
            envVar: config.backendProvider === "anthropic" ? "ANTHROPIC_API_KEY" : "OPENAI_API_KEY",
            promptMessage: "Enter your API key",
            defaultModel:
              config.backendProvider === "anthropic"
                ? "contextpilot/claude-sonnet-4-6"
                : "contextpilot/gpt-4o",
          }),
        ],
      resolveDynamicModel: (ctx: ProviderResolveDynamicModelContext) => {
        if (config.backendProvider === "sglang") {
          return {
            id: ctx.modelId,
            name: ctx.modelId,
            provider: PROVIDER_ID,
            baseUrl: config.indexServerUrl,
            api: "openai-completions",
            reasoning: false,
            input: ["text", "image"] as Array<"text" | "image">,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 200000,
            maxTokens: 16384,
          };
        }

        const isAnthropic = config.backendProvider === "anthropic";
        return {
          id: ctx.modelId,
          name: ctx.modelId,
          provider: PROVIDER_ID,
          baseUrl: isAnthropic ? "https://api.anthropic.com/v1" : "https://api.openai.com/v1",
          api: isAnthropic ? "anthropic-messages" : "openai-completions",
          reasoning: false,
          input: ["text", "image"] as Array<"text" | "image">,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow: 200000,
          maxTokens: 16384,
        };
      },
      wrapStreamFn: (ctx: ProviderWrapStreamFnContext) => {
        const originalStreamFn = ctx.streamFn;
        if (!originalStreamFn) return undefined;

        return async (params) => {
          const request = params as { body?: unknown };
          if (!request.body) {
            return originalStreamFn(params);
          }

          const body = structuredClone(request.body) as Record<string, unknown>;
          const apiFormat = detectApiFormat(body, config.backendProvider);

          const interceptConfig: InterceptConfig = {
            enabled: true,
            mode: "auto",
            tag: "document",
            separator: "---",
            alpha: 0.001,
            linkageMethod: "average",
            scope: config.scope,
          };

          const handler = getFormatHandler(apiFormat);
          const multi = handler.extractAll(body, interceptConfig);

          const reorderDocs = async (docs: string[]): Promise<string[]> => {
            if (docs.length < 2) {
              return docs;
            }
            if (client) {
              return reorderWithClient(client, docs);
            }
            if (engine) {
              return reorderWithEngine(engine, docs);
            }
            return docs;
          };

          if (multi.systemExtraction) {
            const [extraction, sysIdx] = multi.systemExtraction;
            if (extraction.documents.length >= 2) {
              const reordered = await reorderDocs(extraction.documents);
              handler.reconstructSystem(body, extraction, reordered, sysIdx);
            }
          }

          for (const [extraction, location] of multi.toolExtractions) {
            if (extraction.documents.length >= 2) {
              const reordered = await reorderDocs(extraction.documents);
              handler.reconstructToolResult(body, extraction, reordered, location);
            }
          }

          if (apiFormat === "openai_chat") {
            const dedupResult = dedupChatCompletions(body);
            totalCharsSaved += dedupResult.charsSaved;
          }
          if (body.input && Array.isArray(body.input)) {
            const dedupResult = dedupResponsesApi(body);
            totalCharsSaved += dedupResult.charsSaved;
          }

          const optimizedBody = isSglang
            ? body
            : injectCacheControl(body, config.backendProvider === "anthropic" ? "anthropic" : "openai");

          requestCount++;

          return originalStreamFn({
            ...params,
            body: optimizedBody,
          });
        };
      },
      augmentModelCatalog: () => {
        if (config.backendProvider === "sglang") {
          return [
            { id: "default", name: "SGLang Default (ContextPilot)", provider: PROVIDER_ID },
          ];
        }

        const isAnthropic = config.backendProvider === "anthropic";
        if (isAnthropic) {
          return [
            { id: "claude-opus-4-6", name: "Claude Opus 4.6 (ContextPilot)", provider: PROVIDER_ID },
            {
              id: "claude-sonnet-4-6",
              name: "Claude Sonnet 4.6 (ContextPilot)",
              provider: PROVIDER_ID,
            },
          ];
        }
        return [
          { id: "gpt-4o", name: "GPT-4o (ContextPilot)", provider: PROVIDER_ID },
          { id: "gpt-4o-mini", name: "GPT-4o Mini (ContextPilot)", provider: PROVIDER_ID },
        ];
      },
    });

    api.registerTool({
      name: "contextpilot_status",
      description: "Report ContextPilot engine state",
      parameters: Type.Object({}),
      async execute(_toolCallId: string, _params: unknown) {
        const lines = [
          "ContextPilot Engine Status:",
          `  Backend: ${config.backendProvider}`,
          `  Scope: ${config.scope}`,
          `  Requests optimized: ${requestCount}`,
          `  Total chars saved: ${totalCharsSaved.toLocaleString()}`,
        ];

        if (engine) {
          const stats = engine.getStats();
          lines.push("  Mode: cloud-api (in-process ContextPilot engine)");
          lines.push(`  Live index: ${engine.isLive ? "active" : "warming"}`);
          lines.push(`  Nodes: ${Number(stats.num_nodes ?? 0)}`);
          lines.push(`  Active nodes: ${Number(stats.active_nodes ?? 0)}`);
          lines.push(`  Requests tracked: ${Number(stats.num_requests ?? 0)}`);
          lines.push(`  Total searches: ${Number(stats.total_searches ?? 0)}`);
          lines.push(`  Total insertions: ${Number(stats.total_insertions ?? 0)}`);
          lines.push(`  Total removals: ${Number(stats.total_removals ?? 0)}`);
          lines.push(`  Avg search time (us): ${Number(stats.avg_search_time_us ?? 0).toFixed(2)}`);
        }

        if (client) {
          const [health, remoteStats] = await Promise.all([client.health(), client.getStats()]);
          lines.push("  Mode: sglang (remote ContextPilot index)");
          lines.push(`  Index server URL: ${config.indexServerUrl}`);
          lines.push(`  Index server health: ${formatJson(health)}`);
          lines.push(`  Index server stats: ${formatJson(remoteStats)}`);
        }

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
