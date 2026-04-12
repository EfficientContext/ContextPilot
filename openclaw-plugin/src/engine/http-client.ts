type JsonObject = Record<string, unknown>;

function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function fetchJson(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<JsonObject | null> {
  try {
    const response = await fetch(url, {
      ...init,
      signal: AbortSignal.timeout(timeoutMs),
    });

    if (!response.ok) {
      return null;
    }

    const data: unknown = await response.json();
    return isJsonObject(data) ? data : null;
  } catch {
    return null;
  }
}

export class ContextPilotIndexClient {
  private readonly baseUrl: string;

  private readonly timeout: number;

  private readonly retryOnFailure: boolean;

  constructor(
    baseUrl: string = "http://localhost:8765",
    timeout: number = 1000,
    retryOnFailure: boolean = false,
  ) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.timeout = timeout;
    this.retryOnFailure = retryOnFailure;
  }

  private async _post(endpoint: string, jsonData: JsonObject): Promise<JsonObject | null> {
    const url = `${this.baseUrl}${endpoint}`;
    const attempt = () =>
      fetchJson(
        url,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(jsonData),
        },
        this.timeout,
      );

    const result = await attempt();
    if (result !== null || !this.retryOnFailure) {
      return result;
    }

    return attempt();
  }

  private async _get(endpoint: string): Promise<JsonObject | null> {
    const url = `${this.baseUrl}${endpoint}`;
    const attempt = () =>
      fetchJson(
        url,
        {
          method: "GET",
        },
        this.timeout,
      );

    const result = await attempt();
    if (result !== null || !this.retryOnFailure) {
      return result;
    }

    return attempt();
  }

  async evict(requestIds: string[]): Promise<JsonObject | null> {
    return this._post("/evict", { request_ids: requestIds });
  }

  async search(context: number[], updateAccess: boolean = true): Promise<JsonObject | null> {
    return this._post("/search", {
      context,
      update_access: updateAccess,
    });
  }

  async updateNode(searchPath: number[], tokenDelta: number): Promise<JsonObject | null> {
    return this._post("/update", {
      search_path: searchPath,
      token_delta: tokenDelta,
    });
  }

  async insert(
    context: number[],
    searchPath: number[],
    totalTokens: number = 0,
  ): Promise<JsonObject | null> {
    return this._post("/insert", {
      context,
      search_path: searchPath,
      total_tokens: totalTokens,
    });
  }

  async reorder(
    contexts: Array<Array<number | string>>,
    alpha: number = 0.001,
    useGpu: boolean = false,
    linkageMethod: string = "average",
    initialTokensPerContext: number = 0,
    deduplicate: boolean = false,
    parentRequestIds?: Array<string | null>,
    hintTemplate?: string,
  ): Promise<[Array<Array<number | string>>, number[]] | null> {
    const result = await this.reorderRaw(
      contexts,
      alpha,
      useGpu,
      linkageMethod,
      initialTokensPerContext,
      deduplicate,
      parentRequestIds,
      hintTemplate,
    );

    if (result === null) {
      return null;
    }

    const reorderedContexts = result.reordered_contexts;
    const originalIndices = result.original_indices;

    if (!Array.isArray(reorderedContexts) || !Array.isArray(originalIndices)) {
      return null;
    }

    if (!originalIndices.every((index) => typeof index === "number")) {
      return null;
    }

    return [reorderedContexts as Array<Array<number | string>>, originalIndices as number[]];
  }

  async reorderRaw(
    contexts: Array<Array<number | string>>,
    alpha: number = 0.001,
    useGpu: boolean = false,
    linkageMethod: string = "average",
    initialTokensPerContext: number = 0,
    deduplicate: boolean = false,
    parentRequestIds?: Array<string | null>,
    hintTemplate?: string,
  ): Promise<JsonObject | null> {
    const payload: JsonObject = {
      contexts,
      alpha,
      use_gpu: useGpu,
      linkage_method: linkageMethod,
      initial_tokens_per_context: initialTokensPerContext,
      deduplicate,
    };

    if (parentRequestIds !== undefined) {
      payload.parent_request_ids = parentRequestIds;
    }

    if (hintTemplate !== undefined) {
      payload.hint_template = hintTemplate;
    }

    return this._post("/reorder", payload);
  }

  async deduplicate(
    contexts: number[][],
    parentRequestIds: Array<string | null>,
    hintTemplate?: string,
  ): Promise<JsonObject | null> {
    const payload: JsonObject = {
      contexts,
      parent_request_ids: parentRequestIds,
    };

    if (hintTemplate !== undefined) {
      payload.hint_template = hintTemplate;
    }

    return this._post("/deduplicate", payload);
  }

  async reset(): Promise<JsonObject | null> {
    return this._post("/reset", {});
  }

  async getRequests(): Promise<JsonObject | null> {
    return this._get("/requests");
  }

  async getStats(): Promise<JsonObject | null> {
    return this._get("/stats");
  }

  async health(): Promise<JsonObject | null> {
    return this._get("/health");
  }

  async isReady(): Promise<boolean> {
    const health = await this.health();
    return health !== null && health.status === "ready";
  }
}

export async function evictRequests(
  requestIds: string[],
  serverUrl: string = "http://localhost:8765",
): Promise<JsonObject | null> {
  return fetchJson(
    `${serverUrl.replace(/\/+$/, "")}/evict`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ request_ids: requestIds }),
    },
    1000,
  );
}

export async function scheduleBatch(
  contexts: number[][],
  serverUrl: string = "http://localhost:8765",
  alpha: number = 0.001,
  useGpu: boolean = false,
  linkageMethod: string = "average",
  timeout: number = 30000,
): Promise<JsonObject | null> {
  return fetchJson(
    `${serverUrl.replace(/\/+$/, "")}/reorder`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        contexts,
        alpha,
        use_gpu: useGpu,
        linkage_method: linkageMethod,
      }),
    },
    timeout,
  );
}
