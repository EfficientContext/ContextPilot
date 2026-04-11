import * as crypto from 'node:crypto';

export const MIN_BLOCK_CHARS = 80;
export const MIN_CONTENT_CHARS = 500;

export const CHUNK_MODULUS = 13;
export const CHUNK_MIN_LINES = 5;
export const CHUNK_MAX_LINES = 40;

export interface DedupResult {
    blocksDeduped: number;
    blocksTotal: number;
    systemBlocksMatched: number;
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
    content?: unknown;
    tool_call_id?: string;
    name?: string;
}

interface OpenAIChatMessage extends OpenAIToolMessage {
    tool_calls?: OpenAIToolCall[];
}

interface TextContentBlock {
    type?: string;
    text?: string;
}

interface ChatCompletionsBody {
    messages?: OpenAIChatMessage[];
}

interface ResponsesFunctionCallItem {
    type?: string;
    call_id?: string;
    name?: string;
}

interface ResponsesFunctionCallOutputItem {
    type?: string;
    call_id?: string;
    output?: unknown;
}

interface ResponsesApiBody {
    input?: ResponsesFunctionCallOutputItem[];
}

const CODE_BLOCK_RE = /(```[\w]*\n)([\s\S]*?)(```)/g;

function emptyDedupResult(): DedupResult {
    return {
        blocksDeduped: 0,
        blocksTotal: 0,
        systemBlocksMatched: 0,
        charsBefore: 0,
        charsAfter: 0,
        charsSaved: 0
    };
}

export function hashString(str: string): number {
    let h = 5381;
    for (let i = 0; i < str.length; i++) {
        h = (Math.imul(h, 33) + str.charCodeAt(i)) | 0;
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

function normalizeDedupArgs(
    systemContentOrOpts?: string | DedupOptions,
    maybeOpts?: DedupOptions
): { systemContent: string | undefined; opts: DedupOptions } {
    if (typeof systemContentOrOpts === 'string') {
        return { systemContent: systemContentOrOpts, opts: maybeOpts ?? {} };
    }
    return { systemContent: undefined, opts: systemContentOrOpts ?? {} };
}

function mergePreSeen(
    seenBlocks: Map<string, SeenBlock>,
    preSeen?: Map<string, SeenBlock>
): void {
    if (!preSeen) {
        return;
    }

    for (const [hash, origin] of preSeen.entries()) {
        if (!seenBlocks.has(hash)) {
            seenBlocks.set(hash, origin);
        }
    }
}

function dedupText(
    text: string,
    seenBlocks: Map<string, SeenBlock>,
    msgIdx: number,
    fnName: string,
    result: DedupResult,
    minBlockChars: number,
    chunkModulus: number,
    preSeen?: Map<string, SeenBlock>
): string | undefined {
    mergePreSeen(seenBlocks, preSeen);

    const blocks = contentDefinedChunking(text, chunkModulus);
    if (blocks.length < 2) {
        for (const block of blocks) {
            if (block.trim().length >= minBlockChars) {
                const h = hashBlock(block);
                result.blocksTotal += 1;
                if (!seenBlocks.has(h)) {
                    seenBlocks.set(h, [msgIdx, fnName, 0]);
                }
            }
        }
        return undefined;
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
        if (seen && seen[0] !== msgIdx) {
            const [origMsgIdx, origFn] = seen;
            const firstLine = block.trim().split('\n')[0].slice(0, 80);
            const ref = `[..., "${firstLine}" — identical to earlier ${origFn} result, see above ...]`;
            const charsSaved = block.length - ref.length;
            if (charsSaved > 0) {
                if (origMsgIdx === -1) {
                    result.systemBlocksMatched += 1;
                }
                newBlocks.push(ref);
                dedupedInThis += 1;
                result.blocksDeduped += 1;
            } else {
                newBlocks.push(block);
            }
        } else {
            if (!seen) {
                seenBlocks.set(h, [msgIdx, fnName, blockIdx]);
            }
            newBlocks.push(block);
        }
    }

    if (dedupedInThis > 0) {
        return newBlocks.join('\n\n');
    }

    for (let blockIdx = 0; blockIdx < blocks.length; blockIdx++) {
        const block = blocks[blockIdx];
        if (block.trim().length >= minBlockChars) {
            const h = hashBlock(block);
            if (!seenBlocks.has(h)) {
                seenBlocks.set(h, [msgIdx, fnName, blockIdx]);
            }
        }
    }

    return undefined;
}

function prescanSystemBlocks(
    systemContent: string,
    minBlockChars: number,
    chunkModulus: number
): Map<string, SeenBlock> {
    const preSeen = new Map<string, SeenBlock>();
    if (typeof systemContent !== 'string' || !systemContent.trim()) {
        return preSeen;
    }

    const blocks = contentDefinedChunking(systemContent, chunkModulus);
    for (let blockIdx = 0; blockIdx < blocks.length; blockIdx++) {
        const block = blocks[blockIdx];
        if (block.trim().length < minBlockChars) {
            continue;
        }
        const h = hashBlock(block);
        if (!preSeen.has(h)) {
            preSeen.set(h, [-1, 'system prompt', blockIdx]);
        }
    }

    return preSeen;
}

export function dedupAssistantCodeBlocks(
    messages: OpenAIChatMessage[],
    seenBlocks: Map<string, SeenBlock>,
    result: DedupResult,
    minBlockChars: number,
    minContentChars: number,
    chunkModulus: number,
    preSeen?: Map<string, SeenBlock>
): void {
    for (let idx = 0; idx < messages.length; idx++) {
        const msg = messages[idx];
        if (!msg || typeof msg !== 'object' || msg.role !== 'assistant') {
            continue;
        }

        const rawContent = msg.content;
        let content = '';
        let isListContent = false;
        let textBlockIdx = -1;

        if (typeof rawContent === 'string') {
            content = rawContent;
        } else if (Array.isArray(rawContent)) {
            for (let blockIdx = 0; blockIdx < rawContent.length; blockIdx++) {
                const block = rawContent[blockIdx] as { type?: string; text?: string };
                if (block?.type !== 'text' || typeof block.text !== 'string') {
                    continue;
                }
                if (block.text.includes('```') && block.text.length > content.length) {
                    content = block.text;
                    textBlockIdx = blockIdx;
                    isListContent = true;
                }
            }

            if (!content) {
                continue;
            }
        } else {
            continue;
        }

        if (content.length < minContentChars) {
            continue;
        }

        const matches = Array.from(content.matchAll(CODE_BLOCK_RE));
        if (matches.length === 0) {
            continue;
        }

        let modified = false;
        let newContent = content;

        for (let i = matches.length - 1; i >= 0; i--) {
            const match = matches[i];
            const code = match[2] ?? '';
            if (code.trim().length < minBlockChars) {
                continue;
            }

            const dedupedCode = dedupText(
                code,
                seenBlocks,
                idx,
                'assistant',
                result,
                minBlockChars,
                chunkModulus,
                preSeen
            );

            if (dedupedCode === undefined) {
                continue;
            }

            const start = match.index + (match[1]?.length ?? 0);
            const end = start + code.length;
            const originalLen = end - start;

            newContent = `${newContent.slice(0, start)}${dedupedCode}${newContent.slice(end)}`;
            result.charsBefore += originalLen;
            result.charsAfter += dedupedCode.length;
            result.charsSaved += (originalLen - dedupedCode.length);
            modified = true;
        }

        if (!modified) {
            continue;
        }

        if (isListContent && Array.isArray(msg.content) && textBlockIdx >= 0) {
            const textBlock = msg.content[textBlockIdx] as { type?: string; text?: string };
            if (textBlock && textBlock.type === 'text') {
                textBlock.text = newContent;
            }
        } else {
            msg.content = newContent;
        }
    }
}

export function dedupChatCompletions(body: ChatCompletionsBody, opts?: DedupOptions): DedupResult;
export function dedupChatCompletions(
    body: ChatCompletionsBody,
    systemContent?: string,
    opts?: DedupOptions
): DedupResult;
export function dedupChatCompletions(
    body: ChatCompletionsBody,
    systemContentOrOpts?: string | DedupOptions,
    maybeOpts?: DedupOptions
): DedupResult {
    const normalized = normalizeDedupArgs(systemContentOrOpts, maybeOpts);
    const systemContent = normalized.systemContent;
    const opts = normalized.opts;
    const minBlockChars = opts.minBlockChars ?? MIN_BLOCK_CHARS;
    const minContentChars = opts.minContentChars ?? MIN_CONTENT_CHARS;
    const chunkModulus = opts.chunkModulus ?? CHUNK_MODULUS;

    const messages = body?.messages;
    if (!Array.isArray(messages) || messages.length === 0) {
        return emptyDedupResult();
    }

    const toolNames = buildToolNameMapOpenai(messages);
    const seenBlocks = new Map<string, SeenBlock>();
    const preSeen = systemContent
        ? prescanSystemBlocks(systemContent, minBlockChars, chunkModulus)
        : undefined;
    mergePreSeen(seenBlocks, preSeen);
    const result = emptyDedupResult();

    for (let idx = 0; idx < messages.length; idx++) {
        const msg = messages[idx];
        if (!msg || typeof msg !== 'object') {
            continue;
        }
        if (msg.role !== 'tool' && msg.role !== 'toolResult') {
            continue;
        }

        let content = msg.content ?? '';
        if (Array.isArray(content)) {
            const textParts: string[] = [];
            for (const block of content) {
                if (!block || typeof block !== 'object') {
                    continue;
                }
                const textBlock = block as TextContentBlock;
                if (textBlock.type === 'text' && typeof textBlock.text === 'string') {
                    textParts.push(textBlock.text);
                }
            }
            content = textParts.join('\n');
        }
        if (typeof content !== 'string' || content.length < minContentChars) {
            continue;
        }

        const toolCallId = msg.tool_call_id || '';
        const fnName = toolNames[toolCallId] || msg.name || 'tool';
        const dedupedContent = dedupText(
            content,
            seenBlocks,
            idx,
            fnName,
            result,
            minBlockChars,
            chunkModulus,
            preSeen
        );

        if (dedupedContent === undefined) {
            continue;
        }

        const originalLen = content.length;
        if (Array.isArray(msg.content)) {
            const textBlockIdx = msg.content.findIndex(
                (block) => !!block && typeof block === 'object' && (block as TextContentBlock).type === 'text'
            );
            if (textBlockIdx >= 0) {
                const textBlock = msg.content[textBlockIdx];
                if (textBlock && typeof textBlock === 'object') {
                    (textBlock as TextContentBlock).text = dedupedContent;
                }
            }
        } else {
            msg.content = dedupedContent;
        }

        const newLen = dedupedContent.length;
        result.charsBefore += originalLen;
        result.charsAfter += newLen;
        result.charsSaved += (originalLen - newLen);
    }

    dedupAssistantCodeBlocks(
        messages,
        seenBlocks,
        result,
        minBlockChars,
        minContentChars,
        chunkModulus,
        preSeen
    );

    return result;
}

export function dedupResponsesApi(body: ResponsesApiBody, opts?: DedupOptions): DedupResult;
export function dedupResponsesApi(
    body: ResponsesApiBody,
    systemContent?: string,
    opts?: DedupOptions
): DedupResult;
export function dedupResponsesApi(
    body: ResponsesApiBody,
    systemContentOrOpts?: string | DedupOptions,
    maybeOpts?: DedupOptions
): DedupResult {
    const normalized = normalizeDedupArgs(systemContentOrOpts, maybeOpts);
    const systemContent = normalized.systemContent;
    const opts = normalized.opts;
    const minBlockChars = opts.minBlockChars ?? MIN_BLOCK_CHARS;
    const minContentChars = opts.minContentChars ?? MIN_CONTENT_CHARS;
    const chunkModulus = opts.chunkModulus ?? CHUNK_MODULUS;

    const inputItems = body?.input;
    if (!Array.isArray(inputItems) || inputItems.length === 0) {
        return emptyDedupResult();
    }

    const fnNames = buildToolNameMapResponses(inputItems);
    const seenBlocks = new Map<string, SeenBlock>();
    const preSeen = systemContent
        ? prescanSystemBlocks(systemContent, minBlockChars, chunkModulus)
        : undefined;
    mergePreSeen(seenBlocks, preSeen);
    const result = emptyDedupResult();

    for (let idx = 0; idx < inputItems.length; idx++) {
        const item = inputItems[idx];
        if (!item || typeof item !== 'object' || item.type !== 'function_call_output') {
            continue;
        }

        const output = item.output ?? '';
        if (typeof output !== 'string' || output.length < minContentChars) {
            continue;
        }

        const callId = item.call_id || '';
        const fnName = fnNames[callId] || callId || 'tool';
        const dedupedOutput = dedupText(
            output,
            seenBlocks,
            idx,
            fnName,
            result,
            minBlockChars,
            chunkModulus,
            preSeen
        );

        if (dedupedOutput === undefined) {
            continue;
        }

        const originalLen = output.length;
        item.output = dedupedOutput;
        const newLen = dedupedOutput.length;
        result.charsBefore += originalLen;
        result.charsAfter += newLen;
        result.charsSaved += (originalLen - newLen);
    }

    return result;
}
