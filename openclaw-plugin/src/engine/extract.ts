import * as crypto from 'crypto';

const _KNOWN_WRAPPER_TAGS = new Set(["documents", "contexts", "docs", "passages", "references", "files"]);
const _KNOWN_ITEM_TAGS = new Set(["document", "context", "doc", "passage", "reference", "file"]);

const _NUMBERED_RE = /\[(\d+)\]\s*/;
const _SEPARATOR_PATTERNS = ["---", "==="];
const _SINGLE_DOC_MIN_CHARS = 200;

export interface InterceptConfig {
    enabled: boolean;
    mode: string;
    tag: string;
    separator: string;
    alpha: number;
    linkageMethod: string;
    scope: string;
}

export interface ExtractionResult {
    documents: string[];
    prefix: string;
    suffix: string;
    mode: string;
    wrapperTag: string;
    itemTag: string;
    separatorChar: string;
    originalContent: string;
    jsonItems: any[] | null;
}

export interface ToolResultLocation {
    msgIndex: number;
    blockIndex: number;      // -1 = content is string
    innerBlockIndex: number; // For Anthropic nested content blocks
}

export interface SingleDocExtraction {
    content: string;
    contentHash: string;
    toolCallId: string;
}

export class MultiExtractionResult {
    systemExtraction: [ExtractionResult, number] | null = null;
    toolExtractions: [ExtractionResult, ToolResultLocation][] = [];
    singleDocExtractions: [SingleDocExtraction, ToolResultLocation][] = [];

    get hasExtractions(): boolean {
        return (
            this.systemExtraction !== null ||
            this.toolExtractions.length > 0 ||
            this.singleDocExtractions.length > 0
        );
    }

    get totalDocuments(): number {
        let total = this.singleDocExtractions.length;
        if (this.systemExtraction) {
            total += this.systemExtraction[0].documents.length;
        }
        for (const [ext, _] of this.toolExtractions) {
            total += ext.documents.length;
        }
        return total;
    }
}

export function parseInterceptHeaders(headers: Record<string, string>): InterceptConfig {
    const get = (name: string, def: string = ""): string => {
        const key = `x-contextpilot-${name}`;
        for (const [k, v] of Object.entries(headers)) {
            if (k.toLowerCase() === key) {
                return v;
            }
        }
        return def;
    };

    const enabledStr = get("enabled", "true").toLowerCase();
    const enabled = !["false", "0", "no"].includes(enabledStr);

    let scope = get("scope", "all").toLowerCase();
    if (!["system", "tool_results", "all"].includes(scope)) {
        scope = "all";
    }

    return {
        enabled,
        mode: get("mode", "auto").toLowerCase(),
        tag: get("tag", "document").toLowerCase(),
        separator: get("separator", "---"),
        alpha: parseFloat(get("alpha", "0.001")) || 0.001,
        linkageMethod: get("linkage", "average"),
        scope
    };
}

// ── Document extraction ─────────────────────────────────────────────────────

function _escapeRegExp(string: string): string {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function extractXmlTags(text: string, config: InterceptConfig): ExtractionResult | null {
    let itemTagsToTry: string[] = [];
    let wrapperTagsToTry: string[] = [];

    if (config.mode === "xml_tag") {
        itemTagsToTry.push(config.tag);
        wrapperTagsToTry.push(config.tag + "s");
        for (const t of _KNOWN_ITEM_TAGS) {
            if (t !== config.tag) itemTagsToTry.push(t);
        }
        for (const t of _KNOWN_WRAPPER_TAGS) {
            if (t !== config.tag + "s") wrapperTagsToTry.push(t);
        }
    } else {
        itemTagsToTry = Array.from(_KNOWN_ITEM_TAGS);
        wrapperTagsToTry = Array.from(_KNOWN_WRAPPER_TAGS);
    }

    for (const wrapperTag of wrapperTagsToTry) {
        const wrapperPattern = new RegExp(`(<${wrapperTag}(?:\\s[^>]*)?>)(.*?)(</${wrapperTag}>)`, "s");
        const wrapperMatch = wrapperPattern.exec(text);
        if (!wrapperMatch) continue;

        const innerText = wrapperMatch[2];
        const prefix = text.substring(0, wrapperMatch.index);
        const suffix = text.substring(wrapperMatch.index + wrapperMatch[0].length);

        for (const itemTag of itemTagsToTry) {
            const itemPattern = new RegExp(`(<${itemTag}(?:\\s[^>]*)?>)(.*?)(</${itemTag}>)`, "gs");
            let items: string[] = [];
            while (true) {
                const itemMatch = itemPattern.exec(innerText);
                if (itemMatch === null) break;
                items.push(itemMatch[2].trim());
            }
            if (items.length > 0) {
                return {
                    documents: items,
                    prefix,
                    suffix,
                    mode: "xml_tag",
                    wrapperTag,
                    itemTag,
                    separatorChar: "",
                    originalContent: text,
                    jsonItems: null
                };
            }
        }
    }

    for (const itemTag of itemTagsToTry) {
        const itemPattern = new RegExp(`(<${itemTag}(?:\\s[^>]*)?>)(.*?)(</${itemTag}>)`, "gs");
        const items: RegExpExecArray[] = [];
        while (true) {
            const match = itemPattern.exec(text);
            if (match === null) break;
            items.push(match);
        }
        
        if (items.length >= 2) {
            const firstStart = items[0].index;
            const lastEnd = items[items.length - 1].index + items[items.length - 1][0].length;
            return {
                documents: items.map(m => m[2].trim()),
                prefix: text.substring(0, firstStart),
                suffix: text.substring(lastEnd),
                mode: "xml_tag",
                wrapperTag: "",
                itemTag,
                separatorChar: "",
                originalContent: text,
                jsonItems: null
            };
        }
    }

    return null;
}

export function extractNumbered(text: string, config: InterceptConfig): ExtractionResult | null {
    const splits = text.split(_NUMBERED_RE);
    if (splits.length < 4) {
        return null;
    }

    const prefix = splits[0];
    const documents: string[] = [];
    let i = 1;
    while (i + 1 < splits.length) {
        const docText = splits[i + 1].trim();
        if (docText) {
            documents.push(docText);
        }
        i += 2;
    }

    if (documents.length < 2) return null;

    return {
        documents,
        prefix,
        suffix: "",
        mode: "numbered",
        wrapperTag: "",
        itemTag: "",
        separatorChar: "",
        originalContent: text,
        jsonItems: null
    };
}

export function extractSeparator(text: string, config: InterceptConfig): ExtractionResult | null {
    let sep = config.separator;
    let parts: string[] = [];
    let documents: string[] = [];
    
    if (config.mode === "auto") {
        let found = false;
        for (const candidate of _SEPARATOR_PATTERNS) {
            const regex = new RegExp(`\\n${_escapeRegExp(candidate)}\\n`);
            parts = text.split(regex);
            if (parts.length >= 3) {
                sep = candidate;
                found = true;
                break;
            }
        }
        if (!found) return null;
        documents = parts.map(p => p.trim()).filter(p => p);
    } else {
        const regex = new RegExp(`\\n${_escapeRegExp(sep)}\\n`);
        parts = text.split(regex);
        documents = parts.map(p => p.trim()).filter(p => p);
    }

    if (documents.length < 2) return null;

    return {
        documents,
        prefix: "",
        suffix: "",
        mode: "separator",
        wrapperTag: "",
        itemTag: "",
        separatorChar: sep,
        originalContent: text,
        jsonItems: null
    };
}

export function extractMarkdownHeaders(text: string, config: InterceptConfig): ExtractionResult | null {
    const parts = text.split(/(?=^#{1,2}\s)/m);
    if (!parts || parts.length === 0) return null;

    let prefix = "";
    const sections: string[] = [];
    
    for (const part of parts) {
        const stripped = part.trim();
        if (!stripped) continue;
        
        if (/^#{1,2}\s/.test(stripped)) {
            sections.push(stripped);
        } else {
            prefix = part;
        }
    }

    if (sections.length < 2) return null;

    return {
        documents: sections,
        prefix,
        suffix: "",
        mode: "markdown_header",
        wrapperTag: "",
        itemTag: "",
        separatorChar: "",
        originalContent: text,
        jsonItems: null
    };
}

const _JSON_ID_KEYS = ["url", "path", "file", "filename", "uri", "href"];

function _extractJsonId(item: Record<string, unknown>): string | null {
    for (const key of _JSON_ID_KEYS) {
        const val = item[key];
        if (typeof val === "string" && val.trim()) {
            return val.trim();
        }
    }
    return null;
}

export function extractJsonResults(text: string, config: InterceptConfig): ExtractionResult | null {
    const stripped = text.trim();
    if (!stripped.startsWith("{")) return null;
    
    let obj: any;
    try {
        obj = JSON.parse(stripped);
    } catch (e) {
        return null;
    }

    if (typeof obj !== "object" || obj === null) return null;
    
    const results = obj.results;
    if (!Array.isArray(results) || results.length < 2) return null;

    const documents: string[] = [];
    for (const item of results) {
        if (typeof item === "object" && item !== null) {
            documents.push(_extractJsonId(item) ?? JSON.stringify(item));
        } else {
            documents.push(JSON.stringify(item));
        }
    }

    if (documents.length < 2) return null;

    return {
        documents,
        prefix: "",
        suffix: "",
        mode: "json_results",
        wrapperTag: "",
        itemTag: "",
        separatorChar: "",
        originalContent: text,
        jsonItems: results
    };
}

export function extractDocuments(text: string, config: InterceptConfig): ExtractionResult | null {
    if (config.mode === "xml_tag") {
        return extractXmlTags(text, config);
    }
    if (config.mode === "numbered") {
        return extractNumbered(text, config);
    }
    if (config.mode === "json_results") {
        return extractJsonResults(text, config);
    }
    if (config.mode === "separator") {
        return extractSeparator(text, config);
    }
    if (config.mode === "markdown_header") {
        return extractMarkdownHeaders(text, config);
    }

    return extractXmlTags(text, config)
        ?? extractNumbered(text, config)
        ?? extractJsonResults(text, config)
        ?? null;
}

// ── Reconstruction ───────────────────────────────────────────────────────────

export function reconstructContent(extraction: ExtractionResult, reorderedDocs: string[]): string {
    if (extraction.mode === "xml_tag") {
        return reconstructXml(extraction, reorderedDocs);
    }
    if (extraction.mode === "numbered") {
        return reconstructNumbered(extraction, reorderedDocs);
    }
    if (extraction.mode === "json_results") {
        return reconstructJsonResults(extraction, reorderedDocs);
    }
    if (extraction.mode === "separator") {
        return reconstructSeparator(extraction, reorderedDocs);
    }
    if (extraction.mode === "markdown_header") {
        return reconstructMarkdownHeaders(extraction, reorderedDocs);
    }
    return extraction.originalContent;
}

export function reconstructXml(extraction: ExtractionResult, reorderedDocs: string[]): string {
    const itemTag = extraction.itemTag;
    const items = reorderedDocs.map(doc => `<${itemTag}>${doc}</${itemTag}>`).join("\n");
    const block = extraction.wrapperTag
        ? `<${extraction.wrapperTag}>\n${items}\n</${extraction.wrapperTag}>`
        : items;
    return extraction.prefix + block + extraction.suffix;
}

export function reconstructNumbered(extraction: ExtractionResult, reorderedDocs: string[]): string {
    const parts = extraction.prefix ? [extraction.prefix] : [];
    for (let i = 0; i < reorderedDocs.length; i++) {
        parts.push(`[${i + 1}] ${reorderedDocs[i]}`);
    }
    let result = parts.length > 0 ? parts.join("\n") : "";
    if (extraction.suffix) {
        result += extraction.suffix;
    }
    return result;
}

export function reconstructJsonResults(extraction: ExtractionResult, reorderedDocs: string[]): string {
    const obj = JSON.parse(extraction.originalContent);
    if (extraction.jsonItems !== null) {
        const origDocs = extraction.documents;
        const docToIndices: Record<string, number[]> = {};
        for (let i = 0; i < origDocs.length; i++) {
            if (!docToIndices[origDocs[i]]) {
                docToIndices[origDocs[i]] = [];
            }
            docToIndices[origDocs[i]].push(i);
        }
        
        const used = new Set<number>();
        const reorderedItems: any[] = [];
        for (const doc of reorderedDocs) {
            const indices = docToIndices[doc] || [];
            for (const idx of indices) {
                if (!used.has(idx)) {
                    reorderedItems.push(extraction.jsonItems[idx]);
                    used.add(idx);
                    break;
                }
            }
        }
        obj.results = reorderedItems;
    } else {
        obj.results = reorderedDocs.map(doc => JSON.parse(doc));
    }
    return JSON.stringify(obj, null, 2);
}

export function reconstructSeparator(extraction: ExtractionResult, reorderedDocs: string[]): string {
    const sep = extraction.separatorChar || "---";
    return reorderedDocs.join(`\n${sep}\n`);
}

export function reconstructMarkdownHeaders(extraction: ExtractionResult, reorderedDocs: string[]): string {
    const parts: string[] = [];
    if (extraction.prefix.trim()) {
        parts.push(extraction.prefix.trimEnd());
    }
    parts.push(...reorderedDocs);
    return parts.join("\n\n");
}

// ── OpenAI Chat format ──────────────────────────────────────────────────────

export function extractFromOpenaiChat(body: any, config: InterceptConfig): [ExtractionResult, number] | null {
    const messages = body?.messages;
    if (!messages || !Array.isArray(messages)) return null;

    for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        if (msg?.role !== "system") continue;
        
        const content = msg.content || "";
        if (typeof content === "string") {
            const result = extractDocuments(content, config);
            if (result) return [result, i];
        } else if (Array.isArray(content)) {
            for (const block of content) {
                if (block && typeof block === "object" && block.type === "text") {
                    const result = extractDocuments(block.text || "", config);
                    if (result) return [result, i];
                }
            }
        }
    }
    return null;
}

export function reconstructOpenaiChat(
    body: any,
    extraction: ExtractionResult,
    reorderedDocs: string[],
    systemMsgIndex: number
): any {
    const newBody = structuredClone(body);
    const newContent = reconstructContent(extraction, reorderedDocs);
    const msg = newBody.messages[systemMsgIndex];

    if (typeof msg.content === "string") {
        msg.content = newContent;
    } else if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
            if (block && typeof block === "object" && block.type === "text") {
                if (extractDocuments(block.text || "", parseInterceptHeaders({}))) {
                    block.text = newContent;
                    break;
                }
            }
        }
    }
    return newBody;
}

// ── Anthropic Messages format ───────────────────────────────────────────────

export function extractFromAnthropicMessages(body: any, config: InterceptConfig): ExtractionResult | null {
    const system = body?.system;
    if (system === undefined || system === null) return null;

    if (typeof system === "string") {
        return extractDocuments(system, config);
    }
    if (Array.isArray(system)) {
        for (const block of system) {
            if (block && typeof block === "object" && block.type === "text") {
                const result = extractDocuments(block.text || "", config);
                if (result) return result;
            }
        }
    }
    return null;
}

export function reconstructAnthropicMessages(
    body: any,
    extraction: ExtractionResult,
    reorderedDocs: string[]
): any {
    const newBody = structuredClone(body);
    const newContent = reconstructContent(extraction, reorderedDocs);

    if (typeof newBody.system === "string") {
        newBody.system = newContent;
    } else if (Array.isArray(newBody.system)) {
        for (const block of newBody.system) {
            if (block && typeof block === "object" && block.type === "text") {
                if (extractDocuments(block.text || "", parseInterceptHeaders({}))) {
                    block.text = newContent;
                    break;
                }
            }
        }
    }
    return newBody;
}

// ── Tool result extraction ─────────────────────────────────────────────────

export function extractFromOpenaiToolResults(body: any, config: InterceptConfig): [ExtractionResult, ToolResultLocation][] {
    const messages = body?.messages;
    if (!messages || !Array.isArray(messages)) return [];

    const results: [ExtractionResult, ToolResultLocation][] = [];
    for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        if (msg?.role !== "tool" && msg?.role !== "toolResult") continue;
        
        const content = msg.content || "";
        if (typeof content === "string") {
            const extraction = extractDocuments(content, config);
            if (extraction && extraction.documents.length >= 2) {
                results.push([extraction, { msgIndex: i, blockIndex: -1, innerBlockIndex: -1 }]);
            }
        } else if (Array.isArray(content)) {
            for (let j = 0; j < content.length; j++) {
                const block = content[j];
                if (block && typeof block === "object" && block.type === "text") {
                    const extraction = extractDocuments(block.text || "", config);
                    if (extraction && extraction.documents.length >= 2) {
                        results.push([extraction, { msgIndex: i, blockIndex: j, innerBlockIndex: -1 }]);
                    }
                }
            }
        }
    }
    return results;
}

export function extractFromAnthropicToolResults(body: any, config: InterceptConfig): [ExtractionResult, ToolResultLocation][] {
    const messages = body?.messages;
    if (!messages || !Array.isArray(messages)) return [];

    const results: [ExtractionResult, ToolResultLocation][] = [];
    for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        if (msg?.role !== "user") continue;
        
        const content = msg.content;
        if (!Array.isArray(content)) continue;
        
        for (let j = 0; j < content.length; j++) {
            const block = content[j];
            if (!block || typeof block !== "object" || (block.type !== "tool_result" && block.type !== "toolResult")) continue;
            
            const trContent = block.content || "";
            if (typeof trContent === "string") {
                const extraction = extractDocuments(trContent, config);
                if (extraction && extraction.documents.length >= 2) {
                    results.push([extraction, { msgIndex: i, blockIndex: j, innerBlockIndex: -1 }]);
                }
            } else if (Array.isArray(trContent)) {
                for (let k = 0; k < trContent.length; k++) {
                    const inner = trContent[k];
                    if (inner && typeof inner === "object" && inner.type === "text") {
                        const extraction = extractDocuments(inner.text || "", config);
                        if (extraction && extraction.documents.length >= 2) {
                            results.push([extraction, { msgIndex: i, blockIndex: j, innerBlockIndex: k }]);
                        }
                    }
                }
            }
        }
    }
    return results;
}

// ── Tool result reconstruction ─────────────────────────────────────────────

export function reconstructOpenaiToolResult(
    body: any,
    extraction: ExtractionResult,
    reorderedDocs: string[],
    location: ToolResultLocation
): void {
    const newContent = reconstructContent(extraction, reorderedDocs);
    const msg = body.messages[location.msgIndex];
    if (location.blockIndex === -1) {
        msg.content = newContent;
    } else {
        msg.content[location.blockIndex].text = newContent;
    }
}

export function reconstructAnthropicToolResult(
    body: any,
    extraction: ExtractionResult,
    reorderedDocs: string[],
    location: ToolResultLocation
): void {
    const newContent = reconstructContent(extraction, reorderedDocs);
    const msg = body.messages[location.msgIndex];
    const block = msg.content[location.blockIndex];
    if (location.innerBlockIndex === -1) {
        block.content = newContent;
    } else {
        block.content[location.innerBlockIndex].text = newContent;
    }
}

// ── Aggregate extraction ───────────────────────────────────────────────────

export function extractAllOpenai(body: any, config: InterceptConfig): MultiExtractionResult {
    const result = new MultiExtractionResult();
    if (["system", "all"].includes(config.scope)) {
        const sysResult = extractFromOpenaiChat(body, config);
        if (sysResult) {
            result.systemExtraction = sysResult;
        }
    }
    if (["tool_results", "all"].includes(config.scope)) {
        result.toolExtractions = extractFromOpenaiToolResults(body, config);
        result.singleDocExtractions = extractSingleDocsFromOpenaiToolResults(body, config);
    }
    return result;
}

export function extractAllAnthropic(body: any, config: InterceptConfig): MultiExtractionResult {
    const result = new MultiExtractionResult();
    if (["system", "all"].includes(config.scope)) {
        const sysExtraction = extractFromAnthropicMessages(body, config);
        if (sysExtraction && sysExtraction.documents.length >= 2) {
            result.systemExtraction = [sysExtraction, -1];
        }
    }
    if (["tool_results", "all"].includes(config.scope)) {
        result.toolExtractions = extractFromAnthropicToolResults(body, config);
        result.singleDocExtractions = extractSingleDocsFromAnthropicToolResults(body, config);
    }
    return result;
}

// ── Single-document extraction (for cross-turn dedup) ─────────────────────

function _makeSingleDoc(content: string, toolCallId: string = ""): SingleDocExtraction {
    const stripped = content.trim();
    const contentHash = crypto.createHash("sha256").update(stripped).digest("hex");
    return {
        content: stripped,
        contentHash,
        toolCallId
    };
}

export function extractSingleDocsFromOpenaiToolResults(
    body: any, config: InterceptConfig
): [SingleDocExtraction, ToolResultLocation][] {
    const messages = body?.messages;
    if (!messages || !Array.isArray(messages)) return [];

    const results: [SingleDocExtraction, ToolResultLocation][] = [];
    for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        if (msg?.role !== "tool" && msg?.role !== "toolResult") continue;
        
        const toolCallId = msg.tool_call_id || "";
        const content = msg.content || "";

        if (typeof content === "string") {
            const extraction = extractDocuments(content, config);
            if (extraction && extraction.documents.length >= 2) continue;
            
            if (content.trim().length >= _SINGLE_DOC_MIN_CHARS) {
                results.push([
                    _makeSingleDoc(content, toolCallId),
                    { msgIndex: i, blockIndex: -1, innerBlockIndex: -1 }
                ]);
            }
        } else if (Array.isArray(content)) {
            for (let j = 0; j < content.length; j++) {
                const block = content[j];
                if (!block || typeof block !== "object" || block.type !== "text") continue;
                
                const text = block.text || "";
                const extraction = extractDocuments(text, config);
                if (extraction && extraction.documents.length >= 2) continue;
                
                if (text.trim().length >= _SINGLE_DOC_MIN_CHARS) {
                    results.push([
                        _makeSingleDoc(text, toolCallId),
                        { msgIndex: i, blockIndex: j, innerBlockIndex: -1 }
                    ]);
                }
            }
        }
    }
    return results;
}

export function extractSingleDocsFromAnthropicToolResults(
    body: any, config: InterceptConfig
): [SingleDocExtraction, ToolResultLocation][] {
    const messages = body?.messages;
    if (!messages || !Array.isArray(messages)) return [];

    const results: [SingleDocExtraction, ToolResultLocation][] = [];
    for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        if (msg?.role !== "user") continue;
        
        const content = msg.content;
        if (!Array.isArray(content)) continue;
        
        for (let j = 0; j < content.length; j++) {
            const block = content[j];
            if (!block || typeof block !== "object") continue;
            if (block.type !== "tool_result" && block.type !== "toolResult") continue;
            
            const toolUseId = block.tool_use_id || "";
            const trContent = block.content || "";

            if (typeof trContent === "string") {
                const extraction = extractDocuments(trContent, config);
                if (extraction && extraction.documents.length >= 2) continue;
                
                if (trContent.trim().length >= _SINGLE_DOC_MIN_CHARS) {
                    results.push([
                        _makeSingleDoc(trContent, toolUseId),
                        { msgIndex: i, blockIndex: j, innerBlockIndex: -1 }
                    ]);
                }
            } else if (Array.isArray(trContent)) {
                for (let k = 0; k < trContent.length; k++) {
                    const inner = trContent[k];
                    if (!inner || typeof inner !== "object" || inner.type !== "text") continue;
                    
                    const text = inner.text || "";
                    const extraction = extractDocuments(text, config);
                    if (extraction && extraction.documents.length >= 2) continue;
                    
                    if (text.trim().length >= _SINGLE_DOC_MIN_CHARS) {
                        results.push([
                            _makeSingleDoc(text, toolUseId),
                            { msgIndex: i, blockIndex: j, innerBlockIndex: k }
                        ]);
                    }
                }
            }
        }
    }
    return results;
}

// ── Single-document hint replacement ──────────────────────────────────────

export function replaceSingleDocOpenai(
    body: any, location: ToolResultLocation, hint: string
): void {
    const msg = body.messages[location.msgIndex];
    if (location.blockIndex === -1) {
        msg.content = hint;
    } else {
        msg.content[location.blockIndex].text = hint;
    }
}

export function replaceSingleDocAnthropic(
    body: any, location: ToolResultLocation, hint: string
): void {
    const msg = body.messages[location.msgIndex];
    const block = msg.content[location.blockIndex];
    if (location.innerBlockIndex === -1) {
        block.content = hint;
    } else {
        block.content[location.innerBlockIndex].text = hint;
    }
}

// ── Format handler abstraction ─────────────────────────────────────────────

export interface FormatHandler {
    extractAll(body: any, config: InterceptConfig): MultiExtractionResult;
    reconstructSystem(body: any, extraction: ExtractionResult, docs: string[], sysIdx: number): void;
    reconstructToolResult(body: any, extraction: ExtractionResult, docs: string[], location: ToolResultLocation): void;
    replaceSingleDoc(body: any, location: ToolResultLocation, hint: string): void;
    toolCallPresent(body: any, toolCallId: string): boolean;
    targetPath(): string;
    cacheSystem(body: any): any;
    restoreSystem(body: any, cached: any): void;
}

export class OpenAIChatHandler implements FormatHandler {
    extractAll(body: any, config: InterceptConfig): MultiExtractionResult {
        return extractAllOpenai(body, config);
    }

    reconstructSystem(body: any, extraction: ExtractionResult, docs: string[], sysIdx: number): void {
        const newContent = reconstructContent(extraction, docs);
        const msg = body.messages[sysIdx];
        if (typeof msg.content === "string") {
            msg.content = newContent;
        } else if (Array.isArray(msg.content)) {
            for (const block of msg.content) {
                if (block && typeof block === "object" && block.type === "text") {
                    if (extractDocuments(block.text || "", parseInterceptHeaders({}))) {
                        block.text = newContent;
                        break;
                    }
                }
            }
        }
    }

    reconstructToolResult(body: any, extraction: ExtractionResult, docs: string[], location: ToolResultLocation): void {
        reconstructOpenaiToolResult(body, extraction, docs, location);
    }

    replaceSingleDoc(body: any, location: ToolResultLocation, hint: string): void {
        replaceSingleDocOpenai(body, location, hint);
    }

    toolCallPresent(body: any, toolCallId: string): boolean {
        for (const msg of (body.messages || [])) {
            if (msg.role === "tool" || msg.role === "toolResult") {
                if (msg.tool_call_id === toolCallId) return true;
            }
        }
        return false;
    }

    targetPath(): string {
        return "/v1/chat/completions";
    }

    cacheSystem(_body: any): any {
        return null;
    }

    restoreSystem(_body: any, _cached: any): void {}
}

export class AnthropicMessagesHandler implements FormatHandler {
    extractAll(body: any, config: InterceptConfig): MultiExtractionResult {
        return extractAllAnthropic(body, config);
    }

    reconstructSystem(body: any, extraction: ExtractionResult, docs: string[], sysIdx: number): void {
        const newContent = reconstructContent(extraction, docs);
        if (typeof body.system === "string") {
            body.system = newContent;
        } else if (Array.isArray(body.system)) {
            for (const block of body.system) {
                if (block && typeof block === "object" && block.type === "text") {
                    if (extractDocuments(block.text || "", parseInterceptHeaders({}))) {
                        block.text = newContent;
                        break;
                    }
                }
            }
        }
    }

    reconstructToolResult(body: any, extraction: ExtractionResult, docs: string[], location: ToolResultLocation): void {
        reconstructAnthropicToolResult(body, extraction, docs, location);
    }

    replaceSingleDoc(body: any, location: ToolResultLocation, hint: string): void {
        replaceSingleDocAnthropic(body, location, hint);
    }

    toolCallPresent(body: any, toolCallId: string): boolean {
        for (const msg of (body.messages || [])) {
            if (msg.role === "user" && Array.isArray(msg.content)) {
                for (const block of msg.content) {
                    if (block && typeof block === "object" && 
                        (block.type === "tool_result" || block.type === "toolResult") && 
                        block.tool_use_id === toolCallId) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    targetPath(): string {
        return "/v1/messages";
    }

    cacheSystem(body: any): any {
        return structuredClone(body.system);
    }

    restoreSystem(body: any, cached: any): void {
        if (cached !== null && cached !== undefined) {
            body.system = structuredClone(cached);
        }
    }
}

const _FORMAT_HANDLERS: Record<string, FormatHandler> = {
    "openai_chat": new OpenAIChatHandler(),
    "anthropic_messages": new AnthropicMessagesHandler()
};

export function getFormatHandler(apiFormat: string): FormatHandler {
    return _FORMAT_HANDLERS[apiFormat] || _FORMAT_HANDLERS["openai_chat"];
}
