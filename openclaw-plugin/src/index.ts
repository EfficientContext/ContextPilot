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
import { ReorderState } from "./engine/reorder.js";

const PROVIDER_ID = "contextpilot";

export default definePluginEntry({
  id: "contextpilot",
  name: "ContextPilot",
  description: "Optimizes LLM requests in-process via extraction, dedup, caching, and reordering.",
  register: (api) => {
    const config = {
      backendProvider: api.pluginConfig?.backendProvider === "openai" ? "openai" : "anthropic",
      scope: ["system", "tool_results", "all"].includes(String(api.pluginConfig?.scope))
        ? String(api.pluginConfig?.scope)
        : "all",
    };

    const reorderState = new ReorderState();
    let requestCount = 0;
    let totalCharsSaved = 0;

    api.registerProvider({
      id: PROVIDER_ID,
      label: "ContextPilot",
      docsPath: "/providers/contextpilot",
      envVars: [config.backendProvider === "anthropic" ? "ANTHROPIC_API_KEY" : "OPENAI_API_KEY"],
      auth: [
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
          const apiFormat = config.backendProvider === "anthropic"
            ? "anthropic_messages"
            : "openai_chat";

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

          if (multi.systemExtraction) {
            const [extraction, sysIdx] = multi.systemExtraction;
            if (extraction.documents.length >= 2) {
              const [reordered] = reorderState.reorder(extraction.documents);
              handler.reconstructSystem(body, extraction, reordered, sysIdx);
            }
          }

          for (const [extraction, location] of multi.toolExtractions) {
            if (extraction.documents.length >= 2) {
              const [reordered] = reorderState.reorder(extraction.documents);
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

          const optimizedBody = injectCacheControl(
            body,
            config.backendProvider === "anthropic" ? "anthropic" : "openai",
          );

          requestCount++;

          return originalStreamFn({
            ...params,
            body: optimizedBody,
          });
        };
      },
      augmentModelCatalog: () => {
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
        return {
          content: [
            {
              type: "text" as const,
              text: [
                "ContextPilot Engine Status:",
                "  Mode: in-process (native TypeScript)",
                `  Requests optimized: ${requestCount}`,
                `  Total chars saved: ${totalCharsSaved.toLocaleString()}`,
                `  Backend: ${config.backendProvider}`,
                `  Scope: ${config.scope}`,
              ].join("\n"),
            },
          ],
        };
      },
    });
  },
});
