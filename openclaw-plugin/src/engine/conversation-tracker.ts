export interface DeduplicationResult {
    originalDocs: number[];
    overlappingDocs: number[];
    newDocs: number[];
    referenceHints: string[];
    deduplicatedDocs: number[];
    docSourceTurns: Map<number, string>;
    isNewConversation: boolean;
}

export interface RequestHistory {
    requestId: string;
    docs: number[];
    parentRequestId: string | null;
    turnNumber: number;
    timestamp: number;
}

export interface ConversationTrackerStats {
    totalRequests: number;
    totalDedupCalls: number;
    totalDocsDeduplicated: number;
    activeRequests: number;
}

export class ConversationTracker {
    private _requests: Map<string, RequestHistory>;
    private _hintTemplate: string;
    private _maxTrackedRequests: number;
    private _stats: {
        totalRequests: number;
        totalDedupCalls: number;
        totalDocsDeduplicated: number;
    };

    constructor(hintTemplate?: string, maxTrackedRequests: number = 256) {
        this._requests = new Map<string, RequestHistory>();
        this._hintTemplate =
            hintTemplate ?? "Please refer to [Doc {doc_id}] from the previous conversation turn.";
        this._maxTrackedRequests = maxTrackedRequests;
        this._stats = {
            totalRequests: 0,
            totalDedupCalls: 0,
            totalDocsDeduplicated: 0
        };
    }

    registerRequest(requestId: string, docs: number[], parentRequestId?: string | null): RequestHistory {
        let turnNumber = 1;
        if (parentRequestId && this._requests.has(parentRequestId)) {
            turnNumber = this._requests.get(parentRequestId)!.turnNumber + 1;
        }

        const history: RequestHistory = {
            requestId,
            docs: [...docs],
            parentRequestId: parentRequestId ?? null,
            turnNumber,
            timestamp: Date.now() / 1000
        };

        this._requests.set(requestId, history);
        this._stats.totalRequests += 1;

        // LRU eviction: remove oldest entries when over limit
        if (this._requests.size > this._maxTrackedRequests) {
            const oldest = this._requests.keys().next().value;
            if (oldest !== undefined) {
                this._requests.delete(oldest);
            }
        }

        return history;
    }

    getConversationChain(requestId: string): RequestHistory[] {
        const chain: RequestHistory[] = [];
        let currentId: string | null = requestId;

        while (currentId && this._requests.has(currentId)) {
            const history: RequestHistory = this._requests.get(currentId)!;
            chain.push(history);
            currentId = history.parentRequestId;
        }

        chain.reverse();
        return chain;
    }

    getAllPreviousDocs(parentRequestId: string): [Set<number>, Map<number, string>] {
        const allDocs = new Set<number>();
        const docSources = new Map<number, string>();

        const chain = this.getConversationChain(parentRequestId);

        for (const history of chain) {
            for (const docId of history.docs) {
                if (!allDocs.has(docId)) {
                    allDocs.add(docId);
                    docSources.set(docId, history.requestId);
                }
            }
        }

        return [allDocs, docSources];
    }

    deduplicate(
        requestId: string,
        docs: number[],
        parentRequestId?: string | null,
        hintTemplate?: string
    ): DeduplicationResult {
        this._stats.totalDedupCalls += 1;

        if (!parentRequestId || !this._requests.has(parentRequestId)) {
            this.registerRequest(requestId, docs, null);

            return {
                originalDocs: docs,
                overlappingDocs: [],
                newDocs: docs,
                referenceHints: [],
                deduplicatedDocs: docs,
                docSourceTurns: new Map<number, string>(),
                isNewConversation: true
            };
        }

        const [previousDocs, docSources] = this.getAllPreviousDocs(parentRequestId);

        const overlappingDocs: number[] = [];
        const newDocs: number[] = [];
        const docSourceTurns = new Map<number, string>();

        for (const docId of docs) {
            if (previousDocs.has(docId)) {
                overlappingDocs.push(docId);
                const sourceRequestId = docSources.get(docId);
                if (sourceRequestId !== undefined) {
                    docSourceTurns.set(docId, sourceRequestId);
                }
            } else {
                newDocs.push(docId);
            }
        }

        const template = hintTemplate ?? this._hintTemplate;
        const referenceHints: string[] = [];

        for (const docId of overlappingDocs) {
            const sourceRequest = docSources.get(docId);
            const sourceHistory = sourceRequest ? this._requests.get(sourceRequest) : undefined;
            const turnNumber = sourceHistory ? String(sourceHistory.turnNumber) : "previous";

            const hint = template
                .replaceAll("{doc_id}", String(docId))
                .replaceAll("{turn_number}", turnNumber)
                .replaceAll("{source_request}", sourceRequest ?? "previous");

            referenceHints.push(hint);
        }

        this.registerRequest(requestId, docs, parentRequestId);
        this._stats.totalDocsDeduplicated += overlappingDocs.length;

        return {
            originalDocs: docs,
            overlappingDocs,
            newDocs,
            referenceHints,
            deduplicatedDocs: newDocs,
            docSourceTurns,
            isNewConversation: false
        };
    }

    deduplicateBatch(
        requestIds: string[],
        docsList: number[][],
        parentRequestIds?: Array<string | null | undefined>,
        hintTemplate?: string
    ): DeduplicationResult[] {
        const effectiveParentRequestIds =
            parentRequestIds ?? new Array<string | null | undefined>(requestIds.length).fill(null);

        const results: DeduplicationResult[] = [];
        const n = Math.min(requestIds.length, docsList.length, effectiveParentRequestIds.length);

        for (let i = 0; i < n; i += 1) {
            const result = this.deduplicate(
                requestIds[i],
                docsList[i],
                effectiveParentRequestIds[i],
                hintTemplate
            );
            results.push(result);
        }

        return results;
    }

    removeRequest(requestId: string): boolean {
        return this._requests.delete(requestId);
    }

    clearConversation(requestId: string): number {
        const chain = this.getConversationChain(requestId);
        let count = 0;

        for (const history of chain) {
            if (this.removeRequest(history.requestId)) {
                count += 1;
            }
        }

        return count;
    }

    reset(): void {
        this._requests.clear();
        this._stats = {
            totalRequests: 0,
            totalDedupCalls: 0,
            totalDocsDeduplicated: 0
        };
    }

    getStats(): ConversationTrackerStats {
        return {
            ...this._stats,
            activeRequests: this._requests.size
        };
    }

    getRequestHistory(requestId: string): RequestHistory | null {
        return this._requests.get(requestId) ?? null;
    }
}
