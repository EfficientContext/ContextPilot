export const MIN_CONTENT_LENGTH_FOR_CACHE = 1024;
export const CACHE_CONTROL_EPHEMERAL = { type: 'ephemeral' } as const;

type CacheControl = typeof CACHE_CONTROL_EPHEMERAL;

interface TextBlock extends Record<string, unknown> {
    type?: unknown;
    text?: unknown;
    cache_control?: CacheControl;
}

interface ToolResultBlock extends Record<string, unknown> {
    type?: unknown;
    content?: unknown;
    cache_control?: CacheControl;
}

interface MessageBlock extends Record<string, unknown> {
    role?: unknown;
    content?: unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}

function injectSystemCacheControl(
    body: Record<string, unknown>,
    cc: CacheControl
): Record<string, unknown> {
    const system = body.system;
    if (system === undefined || system === null) {
        return body;
    }

    if (typeof system === 'string') {
        body.system = [{ type: 'text', text: system, cache_control: cc }];
        return body;
    }

    if (Array.isArray(system) && system.length > 0) {
        const lastBlock = system[system.length - 1];
        if (isRecord(lastBlock)) {
            lastBlock.cache_control = cc;
        }
    }

    return body;
}

function maybeAddCacheControlToToolResult(block: ToolResultBlock, cc: CacheControl): void {
    const toolResultContent = block.content ?? '';

    if (typeof toolResultContent === 'string') {
        if (toolResultContent.length >= MIN_CONTENT_LENGTH_FOR_CACHE) {
            block.cache_control = cc;
        }
        return;
    }

    if (!Array.isArray(toolResultContent)) {
        return;
    }

    const totalChars = toolResultContent.reduce((sum, inner) => {
        if (!isRecord(inner) || inner.type !== 'text') {
            return sum;
        }
        return sum + (typeof inner.text === 'string' ? inner.text.length : 0);
    }, 0);

    if (totalChars < MIN_CONTENT_LENGTH_FOR_CACHE || toolResultContent.length === 0) {
        return;
    }

    let lastTextBlock: TextBlock | null = null;
    for (let i = toolResultContent.length - 1; i >= 0; i -= 1) {
        const inner = toolResultContent[i];
        if (isRecord(inner) && inner.type === 'text') {
            lastTextBlock = inner as TextBlock;
            break;
        }
    }

    if (lastTextBlock !== null) {
        lastTextBlock.cache_control = cc;
    }
}

function injectToolResultCacheControl(
    body: Record<string, unknown>,
    cc: CacheControl
): Record<string, unknown> {
    const messages = body.messages;
    if (!Array.isArray(messages) || messages.length === 0) {
        return body;
    }

    for (const msg of messages) {
        if (!isRecord(msg)) {
            continue;
        }

        const message = msg as MessageBlock;

        // Handle OpenClaw's toolResult role (content is the tool result itself)
        if (message.role === 'toolResult') {
            const toolResultContent = message.content ?? '';
            let totalChars = 0;

            if (typeof toolResultContent === 'string') {
                totalChars = toolResultContent.length;
            } else if (Array.isArray(toolResultContent)) {
                totalChars = toolResultContent.reduce((sum, inner) => {
                    if (isRecord(inner) && inner.type === 'text') {
                        return sum + (typeof inner.text === 'string' ? inner.text.length : 0);
                    }
                    return sum;
                }, 0);
            }

            if (totalChars >= MIN_CONTENT_LENGTH_FOR_CACHE) {
                (message as any).cache_control = cc;
            }
            continue;
        }

        // Handle Anthropic's user message with tool_result blocks
        if (message.role !== 'user' || !Array.isArray(message.content)) {
            continue;
        }

        for (const block of message.content) {
            if (!isRecord(block)) {
                continue;
            }
            if (block.type !== 'tool_result' && block.type !== 'toolResult') {
                continue;
            }
            maybeAddCacheControlToToolResult(block as ToolResultBlock, cc);
        }
    }

    return body;
}

export function injectAnthropicCacheControl(body: Record<string, unknown>): Record<string, unknown> {
    if (!body || typeof body !== 'object') {
        return body ?? {};
    }
    const copiedBody = structuredClone(body);
    injectSystemCacheControl(copiedBody, CACHE_CONTROL_EPHEMERAL);
    injectToolResultCacheControl(copiedBody, CACHE_CONTROL_EPHEMERAL);
    return copiedBody;
}

export function injectOpenAICacheControl(body: Record<string, unknown>): Record<string, unknown> {
    // OpenAI prompt caching is automatic and prefix-based, so no explicit
    // cache_control block injection is required at request construction time.
    return body;
}

export function injectCacheControl(
    body: Record<string, unknown>,
    provider: 'anthropic' | 'openai'
): Record<string, unknown> {
    if (provider === 'anthropic') {
        return injectAnthropicCacheControl(body);
    }
    return injectOpenAICacheControl(body);
}
