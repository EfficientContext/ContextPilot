import * as crypto from 'node:crypto';

export const MIN_BLOCK_CHARS = 80;
export const MIN_CONTENT_CHARS = 500;

export const CHUNK_MODULUS = 13;
export const CHUNK_MIN_LINES = 5;
export const CHUNK_MAX_LINES = 40;

export interface DedupResult {
    blocksDeduped: number;
    blocksTotal: number;
    charsBefore: number;
    charsAfter: number;
    charsSaved: number;
}

export interface DedupOptions {
    minBlockChars?: number;
    minContentChars?: number;
    chunkModulus?: number;
}

type SeenBlock = [number, string, number];

interface OpenAIToolCall {
    id?: string;
    function?: {
        name?: string;
    };
}

interface OpenAIAssistantMessage {
    role?: string;
    tool_calls?: OpenAIToolCall[];
}

interface OpenAIToolMessage {
    role?: string;
    content?: string;
    tool_call_id?: string;
    name?: string;
}

interface ChatCompletionsBody {
    messages?: OpenAIToolMessage[];
}

interface ResponsesFunctionCallItem {
    type?: string;
    call_id?: string;
    name?: string;
}

interface ResponsesFunctionCallOutputItem {
    type?: string;
    call_id?: string;
    output?: string;
}

interface ResponsesApiBody {
    input?: ResponsesFunctionCallOutputItem[];
}

function emptyDedupResult(): DedupResult {
    return {
        blocksDeduped: 0,
        blocksTotal: 0,
        charsBefore: 0,
        charsAfter: 0,
        charsSaved: 0
    };
}

export function hashString(str: string): number {
    let h = 5381;
    for (let i = 0; i < str.length; i++) {
        h = ((h << 5) + h + str.charCodeAt(i)) & 0xFFFFFFFF;
    }
    return h >>> 0;
}

export function buildToolNameMapOpenai(messages: OpenAIAssistantMessage[]): Record<string, string> {
    const mapping: Record<string, string> = {};
    for (const msg of messages) {
        if (!msg || typeof msg !== 'object' || msg.role !== 'assistant') {
            continue;
        }

        for (const tc of msg.tool_calls || []) {
            if (!tc || typeof tc !== 'object') {
                continue;
            }
            const tcId = tc.id || '';
            const fn = tc.function;
            if (fn && typeof fn === 'object' && fn.name) {
                mapping[tcId] = fn.name;
            }
        }
    }
    return mapping;
}

export function buildToolNameMapResponses(items: ResponsesFunctionCallItem[]): Record<string, string> {
    const mapping: Record<string, string> = {};
    for (const item of items) {
        if (item && typeof item === 'object' && item.type === 'function_call') {
            const callId = item.call_id || '';
            const name = item.name || '';
            if (callId && name) {
                mapping[callId] = name;
            }
        }
    }
    return mapping;
}

export function contentDefinedChunking(
    text: string,
    chunkModulus: number = CHUNK_MODULUS
): string[] {
    const lines = text.split('\n');
    if (lines.length <= CHUNK_MIN_LINES) {
        return [text];
    }

    const blocks: string[] = [];
    let current: string[] = [];

    for (const line of lines) {
        current.push(line);
        const lineHash = hashString(line.trim()) & 0xFFFFFFFF;
        const isBoundary = (
            lineHash % chunkModulus === 0 && current.length >= CHUNK_MIN_LINES
        ) || current.length >= CHUNK_MAX_LINES;

        if (isBoundary) {
            blocks.push(current.join('\n'));
            current = [];
        }
    }

    if (current.length > 0) {
        if (blocks.length > 0 && current.length < CHUNK_MIN_LINES) {
            blocks[blocks.length - 1] += `\n${current.join('\n')}`;
        } else {
            blocks.push(current.join('\n'));
        }
    }

    return blocks;
}

export function hashBlock(block: string): string {
    const normalized = block.trim();
    return crypto.createHash('sha256').update(normalized, 'utf8').digest('hex').slice(0, 20);
}

export function dedupChatCompletions(body: ChatCompletionsBody, opts: DedupOptions = {}): DedupResult {
    const minBlockChars = opts.minBlockChars ?? MIN_BLOCK_CHARS;
    const minContentChars = opts.minContentChars ?? MIN_CONTENT_CHARS;
    const chunkModulus = opts.chunkModulus ?? CHUNK_MODULUS;

    const messages = body?.messages;
    if (!Array.isArray(messages) || messages.length === 0) {
        return emptyDedupResult();
    }

    const toolNames = buildToolNameMapOpenai(messages);
    const seenBlocks = new Map<string, SeenBlock>();
    const result = emptyDedupResult();

    for (let idx = 0; idx < messages.length; idx++) {
        const msg = messages[idx];
        if (!msg || typeof msg !== 'object' || msg.role !== 'tool') {
            continue;
        }

        const content = msg.content || '';
        if (typeof content !== 'string' || content.length < minContentChars) {
            continue;
        }

        const toolCallId = msg.tool_call_id || '';
        const fnName = toolNames[toolCallId] || msg.name || 'tool';

        const blocks = contentDefinedChunking(content, chunkModulus);
        if (blocks.length < 2) {
            for (const block of blocks) {
                if (block.trim().length >= minBlockChars) {
                    const h = hashBlock(block);
                    if (!seenBlocks.has(h)) {
                        seenBlocks.set(h, [idx, fnName, 0]);
                    }
                }
            }
            continue;
        }

        const newBlocks: string[] = [];
        let dedupedInThis = 0;

        for (let blockIdx = 0; blockIdx < blocks.length; blockIdx++) {
            const block = blocks[blockIdx];
            if (block.trim().length < minBlockChars) {
                newBlocks.push(block);
                continue;
            }

            const h = hashBlock(block);
            result.blocksTotal += 1;

            const seen = seenBlocks.get(h);
            if (seen && seen[0] !== idx) {
                const origFn = seen[1];
                const firstLine = block.trim().split('\n')[0].slice(0, 80);
                const ref = `[... "${firstLine}" — identical to earlier ${origFn} result, see above ...]`;
                const charsSaved = block.length - ref.length;
                if (charsSaved > 0) {
                    newBlocks.push(ref);
                    dedupedInThis += 1;
                    result.blocksDeduped += 1;
                } else {
                    newBlocks.push(block);
                }
            } else {
                if (!seen) {
                    seenBlocks.set(h, [idx, fnName, blockIdx]);
                }
                newBlocks.push(block);
            }
        }

        if (dedupedInThis > 0) {
            const originalLen = content.length;
            const newContent = newBlocks.join('\n\n');
            msg.content = newContent;
            const newLen = newContent.length;
            result.charsBefore += originalLen;
            result.charsAfter += newLen;
            result.charsSaved += (originalLen - newLen);
        } else {
            for (let blockIdx = 0; blockIdx < blocks.length; blockIdx++) {
                const block = blocks[blockIdx];
                if (block.trim().length >= minBlockChars) {
                    const h = hashBlock(block);
                    if (!seenBlocks.has(h)) {
                        seenBlocks.set(h, [idx, fnName, blockIdx]);
                    }
                }
            }
        }
    }

    return result;
}

export function dedupResponsesApi(body: ResponsesApiBody, opts: DedupOptions = {}): DedupResult {
    const minBlockChars = opts.minBlockChars ?? MIN_BLOCK_CHARS;
    const minContentChars = opts.minContentChars ?? MIN_CONTENT_CHARS;
    const chunkModulus = opts.chunkModulus ?? CHUNK_MODULUS;

    const inputItems = body?.input;
    if (!Array.isArray(inputItems) || inputItems.length === 0) {
        return emptyDedupResult();
    }

    const fnNames = buildToolNameMapResponses(inputItems);
    const seenBlocks = new Map<string, SeenBlock>();
    const result = emptyDedupResult();

    for (let idx = 0; idx < inputItems.length; idx++) {
        const item = inputItems[idx];
        if (!item || typeof item !== 'object' || item.type !== 'function_call_output') {
            continue;
        }

        const output = item.output || '';
        if (typeof output !== 'string' || output.length < minContentChars) {
            continue;
        }

        const callId = item.call_id || '';
        const fnName = fnNames[callId] || callId || 'tool';

        const blocks = contentDefinedChunking(output, chunkModulus);
        if (blocks.length < 2) {
            for (const block of blocks) {
                if (block.trim().length >= minBlockChars) {
                    const h = hashBlock(block);
                    if (!seenBlocks.has(h)) {
                        seenBlocks.set(h, [idx, fnName, 0]);
                    }
                }
            }
            continue;
        }

        const newBlocks: string[] = [];
        let dedupedInThis = 0;

        for (let blockIdx = 0; blockIdx < blocks.length; blockIdx++) {
            const block = blocks[blockIdx];
            if (block.trim().length < minBlockChars) {
                newBlocks.push(block);
                continue;
            }

            const h = hashBlock(block);
            result.blocksTotal += 1;

            const seen = seenBlocks.get(h);
            if (seen && seen[0] !== idx) {
                const origFn = seen[1];
                const firstLine = block.trim().split('\n')[0].slice(0, 80);
                const ref = `[... "${firstLine}" — identical to earlier ${origFn} result, see above ...]`;
                const charsSaved = block.length - ref.length;
                if (charsSaved > 0) {
                    newBlocks.push(ref);
                    dedupedInThis += 1;
                    result.blocksDeduped += 1;
                } else {
                    newBlocks.push(block);
                }
            } else {
                if (!seen) {
                    seenBlocks.set(h, [idx, fnName, blockIdx]);
                }
                newBlocks.push(block);
            }
        }

        if (dedupedInThis > 0) {
            const originalLen = output.length;
            const newOutput = newBlocks.join('\n\n');
            item.output = newOutput;
            const newLen = newOutput.length;
            result.charsBefore += originalLen;
            result.charsAfter += newLen;
            result.charsSaved += (originalLen - newLen);
        } else {
            for (let blockIdx = 0; blockIdx < blocks.length; blockIdx++) {
                const block = blocks[blockIdx];
                if (block.trim().length >= minBlockChars) {
                    const h = hashBlock(block);
                    if (!seenBlocks.has(h)) {
                        seenBlocks.set(h, [idx, fnName, blockIdx]);
                    }
                }
            }
        }
    }

    return result;
}
