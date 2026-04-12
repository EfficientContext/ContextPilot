import type { NodeMetadata } from "./metadata.js";

type HeapEntry = [number, number];

export interface EvictionHeapStats {
    size: number;
    total_tokens: number;
    max_tokens: number | null;
    utilization_pct: number;
    avg_tokens_per_node: number;
    oldest_access_time: number | null;
    newest_access_time: number | null;
    num_requests: number;
}

export class EvictionHeap {
    private _heap: HeapEntry[];
    private _metadata: Map<number, NodeMetadata>;
    private _requestToNode: Map<string, number>;
    private _inHeap: Map<number, boolean>;
    private _maxTokens: number | null;
    private _totalTokens: number;

    constructor(maxTokens?: number | null) {
        this._heap = [];
        this._metadata = new Map<number, NodeMetadata>();
        this._requestToNode = new Map<string, number>();
        this._inHeap = new Map<number, boolean>();
        this._maxTokens = maxTokens ?? null;
        this._totalTokens = 0;
    }

    get maxTokens(): number | null {
        return this._maxTokens;
    }

    set maxTokens(value: number | null) {
        this._maxTokens = value;
    }

    private _compare(a: HeapEntry, b: HeapEntry): number {
        if (a[0] !== b[0]) {
            return a[0] - b[0];
        }
        return a[1] - b[1];
    }

    private _swap(i: number, j: number): void {
        const tmp = this._heap[i];
        this._heap[i] = this._heap[j];
        this._heap[j] = tmp;
    }

    private _siftUp(index: number): void {
        let current = index;

        while (current > 0) {
            const parent = Math.floor((current - 1) / 2);
            if (this._compare(this._heap[current], this._heap[parent]) >= 0) {
                break;
            }

            this._swap(current, parent);
            current = parent;
        }
    }

    private _siftDown(index: number): void {
        const n = this._heap.length;
        let current = index;

        while (true) {
            const left = 2 * current + 1;
            const right = 2 * current + 2;
            let smallest = current;

            if (left < n && this._compare(this._heap[left], this._heap[smallest]) < 0) {
                smallest = left;
            }

            if (right < n && this._compare(this._heap[right], this._heap[smallest]) < 0) {
                smallest = right;
            }

            if (smallest === current) {
                break;
            }

            this._swap(current, smallest);
            current = smallest;
        }
    }

    private _heapPush(entry: HeapEntry): void {
        this._heap.push(entry);
        this._siftUp(this._heap.length - 1);
    }

    private _heapPop(): HeapEntry | null {
        if (this._heap.length === 0) {
            return null;
        }

        if (this._heap.length === 1) {
            return this._heap.pop() ?? null;
        }

        const min = this._heap[0];
        const last = this._heap.pop();
        if (last !== undefined) {
            this._heap[0] = last;
            this._siftDown(0);
        }
        return min;
    }

    push(metadata: NodeMetadata): void {
        const nodeId = metadata.nodeId;

        if (this._inHeap.get(nodeId) === true) {
            const oldMetadata = this._metadata.get(nodeId);
            if (oldMetadata) {
                this._totalTokens += metadata.extraTokens - oldMetadata.extraTokens;
            }
            this._metadata.set(nodeId, metadata);
            this.updateAccessTime(nodeId, metadata.lastAccessTime);
            return;
        }

        this._heapPush([metadata.lastAccessTime, nodeId]);
        this._metadata.set(nodeId, metadata);
        this._inHeap.set(nodeId, true);
        this._totalTokens += metadata.extraTokens;

        if (metadata.requestId) {
            this._requestToNode.set(metadata.requestId, nodeId);
        }
    }

    pop(): NodeMetadata | null {
        while (this._heap.length > 0) {
            const entry = this._heapPop();
            if (entry === null) {
                return null;
            }

            const [accessTime, nodeId] = entry;

            const metadata = this._metadata.get(nodeId);
            if (!metadata) {
                continue;
            }

            if (metadata.lastAccessTime === accessTime) {
                this._inHeap.set(nodeId, false);
                this._totalTokens -= metadata.extraTokens;
                return metadata;
            }
        }

        return null;
    }

    peek(): NodeMetadata | null {
        while (this._heap.length > 0) {
            const [accessTime, nodeId] = this._heap[0];

            const metadata = this._metadata.get(nodeId);
            if (!metadata) {
                this._heapPop();
                continue;
            }

            if (metadata.lastAccessTime === accessTime) {
                return metadata;
            }

            this._heapPop();
        }

        return null;
    }

    updateAccessTime(nodeId: number, newTime?: number): void {
        const metadata = this._metadata.get(nodeId);
        if (!metadata) {
            return;
        }

        metadata.lastAccessTime = newTime ?? Date.now() / 1000;
        this._heapPush([metadata.lastAccessTime, nodeId]);
    }

    remove(nodeId: number): void {
        const metadata = this._metadata.get(nodeId);

        if (metadata) {
            this._totalTokens -= metadata.extraTokens;

            if (metadata.requestId) {
                this._requestToNode.delete(metadata.requestId);
            }

            this._metadata.delete(nodeId);
        }

        this._inHeap.delete(nodeId);
    }

    getNodeByRequestId(requestId: string): NodeMetadata | null {
        const nodeId = this._requestToNode.get(requestId);
        if (nodeId !== undefined) {
            return this._metadata.get(nodeId) ?? null;
        }
        return null;
    }

    updateTokensForRequest(requestId: string, inputTokens: number, outputTokens: number): boolean {
        const metadata = this.getNodeByRequestId(requestId);
        if (metadata === null) {
            return false;
        }

        const delta = (inputTokens + outputTokens) - metadata.totalTokens;
        metadata.totalTokens = inputTokens + outputTokens;
        metadata.extraTokens = Math.max(0, metadata.extraTokens + delta);
        metadata.updateAccessTime();

        this._totalTokens += delta;
        this._heapPush([metadata.lastAccessTime, metadata.nodeId]);

        return true;
    }

    needsEviction(): boolean {
        if (this._maxTokens === null) {
            return false;
        }
        return this._totalTokens > this._maxTokens;
    }

    tokensToEvict(): number {
        if (this._maxTokens === null || this._totalTokens <= this._maxTokens) {
            return 0;
        }
        return this._totalTokens - this._maxTokens;
    }

    getMetadata(nodeId: number): NodeMetadata | null {
        return this._metadata.get(nodeId) ?? null;
    }

    isEmpty(): boolean {
        return this.peek() === null;
    }

    size(): number {
        return this._metadata.size;
    }

    totalTokens(): number {
        return this._totalTokens;
    }

    getAllRequestIds(): Set<string> {
        return new Set(this._requestToNode.keys());
    }

    getStats(): EvictionHeapStats {
        if (this._metadata.size === 0) {
            return {
                size: 0,
                total_tokens: 0,
                max_tokens: this._maxTokens,
                utilization_pct: 0,
                avg_tokens_per_node: 0,
                oldest_access_time: null,
                newest_access_time: null,
                num_requests: 0
            };
        }

        const accessTimes = Array.from(this._metadata.values(), (m) => m.lastAccessTime);
        const utilization = this._maxTokens ? (this._totalTokens / this._maxTokens) * 100 : 0;

        return {
            size: this._metadata.size,
            total_tokens: this._totalTokens,
            max_tokens: this._maxTokens,
            utilization_pct: utilization,
            avg_tokens_per_node: this._totalTokens / this._metadata.size,
            oldest_access_time: Math.min(...accessTimes),
            newest_access_time: Math.max(...accessTimes),
            num_requests: this._requestToNode.size
        };
    }

    toString(): string {
        return `EvictionHeap(size=${this._metadata.size}, total_tokens=${this._totalTokens}, max_tokens=${this._maxTokens})`;
    }
}
