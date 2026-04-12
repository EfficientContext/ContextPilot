export interface NodeMetadataInit {
    totalTokens?: number;
    extraTokens?: number;
    lastAccessTime?: number;
    searchPath?: number[];
    isActive?: boolean;
    isLeaf?: boolean;
    docIds?: number[] | null;
    requestId?: string | null;
}

export class NodeMetadata {
    nodeId: number;
    totalTokens: number;
    extraTokens: number;
    lastAccessTime: number;
    searchPath: number[];
    isActive: boolean;
    isLeaf: boolean;
    docIds: number[] | null;
    requestId: string | null;

    constructor(nodeId: number, init: NodeMetadataInit = {}) {
        this.nodeId = nodeId;
        this.totalTokens = init.totalTokens ?? 0;
        this.extraTokens = init.extraTokens ?? 0;
        this.lastAccessTime = init.lastAccessTime ?? Date.now() / 1000;
        this.searchPath = init.searchPath ?? [];
        this.isActive = init.isActive ?? true;
        this.isLeaf = init.isLeaf ?? false;
        this.docIds = init.docIds ?? null;
        this.requestId = init.requestId ?? null;
    }

    updateAccessTime(): void {
        this.lastAccessTime = Date.now() / 1000;
    }

    addTokens(delta: number): void {
        this.totalTokens += delta;
        this.extraTokens += delta;
        this.updateAccessTime();
    }

    removeTokens(delta: number): number {
        if (delta <= 0) {
            return 0;
        }

        let tokensRemoved = Math.min(delta, this.extraTokens);
        this.extraTokens -= tokensRemoved;
        this.totalTokens -= tokensRemoved;

        const remaining = delta - tokensRemoved;
        if (remaining > 0) {
            const actualRemoved = Math.min(remaining, this.totalTokens);
            this.totalTokens -= actualRemoved;
            tokensRemoved += actualRemoved;
        }

        return tokensRemoved;
    }

    isEmpty(): boolean {
        return this.totalTokens <= 0;
    }

    lessThan(other: NodeMetadata): boolean {
        return this.lastAccessTime < other.lastAccessTime;
    }

    toString(): string {
        const req = this.requestId ? `, request_id=${this.requestId}` : "";
        return (
            `NodeMetadata(id=${this.nodeId}, ` +
            `total_tokens=${this.totalTokens}, ` +
            `extra_tokens=${this.extraTokens}, ` +
            `is_leaf=${this.isLeaf}${req}, ` +
            `active=${this.isActive})`
        );
    }
}
