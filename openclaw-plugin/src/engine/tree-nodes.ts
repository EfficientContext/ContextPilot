export class ClusterNode {
    nodeId: number;
    content: Set<number>;
    originalIndices: Set<number>;
    distance: number;
    children: number[];
    parent: number | null;
    frequency: number;
    mergeDistance: number;
    searchPath: number[];

    constructor(
        nodeId: number,
        content: Set<number>,
        originalIndices: Set<number> = new Set([nodeId]),
        distance: number = 0.0,
        children: number[] = [],
        parent: number | null = null,
        frequency: number = 1
    ) {
        this.nodeId = nodeId;
        this.content = content instanceof Set ? new Set(content) : new Set(content);
        this.originalIndices = originalIndices;
        this.distance = distance;
        this.children = children;
        this.parent = parent;
        this.frequency = frequency;
        this.mergeDistance = distance;
        this.searchPath = [];
    }

    get isLeaf(): boolean {
        return !Array.isArray(this.children) || this.children.length === 0;
    }

    get isRoot(): boolean {
        return this.parent === null;
    }

    get isEmpty(): boolean {
        return this.content.size === 0;
    }

    get docIds(): number[] {
        return Array.from(this.content).sort((a, b) => a - b);
    }

    set docIds(value: number[]) {
        this.content = new Set(value);
    }

    addChild(childId: number): void {
        // Defensive: ensure children is an array
        if (!Array.isArray(this.children)) {
            this.children = [];
        }
        if (!this.children.includes(childId) && childId !== this.nodeId) {
            this.children.push(childId);
        }
    }

    removeChild(childId: number): void {
        const idx = this.children.indexOf(childId);
        if (idx !== -1) {
            this.children.splice(idx, 1);
        }
    }

    updateFrequency(additionalFrequency: number): void {
        this.frequency += additionalFrequency;
    }

    mergeWith(otherNode: ClusterNode): void {
        this.content = new Set(Array.from(this.content).filter((v) => otherNode.content.has(v)));
        this.originalIndices = new Set([...this.originalIndices, ...otherNode.originalIndices]);
        this.frequency += otherNode.frequency;
    }

    getDepth(): number {
        return this.searchPath.length;
    }
}

export interface NodeStats {
    totalNodes: number;
    leafNodes: number;
    rootNodes: number;
    internalNodes: number;
}

export class NodeManager {
    clusterNodes: Map<number, ClusterNode>;
    uniqueNodes: Map<number, ClusterNode>;
    redirects: Map<number, number>;
    contentToNodeId: Map<string, number>;

    constructor() {
        this.clusterNodes = new Map<number, ClusterNode>();
        this.uniqueNodes = new Map<number, ClusterNode>();
        this.redirects = new Map<number, number>();
        this.contentToNodeId = new Map<string, number>();
    }

    private contentKey(content: Set<number>): string {
        return Array.from(content).sort((a, b) => a - b).join(',');
    }

    createLeafNode(nodeId: number, promptContent: Iterable<number>): ClusterNode {
        const contentSet = promptContent instanceof Set ? new Set(promptContent) : new Set(promptContent);
        const key = this.contentKey(contentSet);

        const canonicalId = this.contentToNodeId.get(key);
        if (canonicalId !== undefined) {
            const canonicalNode = this.uniqueNodes.get(canonicalId);
            if (!canonicalNode) {
                throw new Error(`Missing canonical leaf node for id ${canonicalId}`);
            }

            canonicalNode.updateFrequency(1);
            canonicalNode.originalIndices.add(nodeId);

            this.redirects.set(nodeId, canonicalId);
            this.clusterNodes.set(nodeId, canonicalNode);
            return canonicalNode;
        }

        const node = new ClusterNode(nodeId, contentSet);
        this.clusterNodes.set(nodeId, node);
        this.uniqueNodes.set(nodeId, node);
        this.contentToNodeId.set(key, nodeId);
        return node;
    }

    createInternalNode(
        nodeId: number,
        child1Id: number,
        child2Id: number,
        distance: number
    ): ClusterNode {
        const canonicalChild1Id = this.redirects.get(child1Id) ?? child1Id;
        const canonicalChild2Id = this.redirects.get(child2Id) ?? child2Id;

        if (canonicalChild1Id === canonicalChild2Id) {
            this.redirects.set(nodeId, canonicalChild1Id);
            const canonicalNode = this.uniqueNodes.get(canonicalChild1Id);
            if (!canonicalNode) {
                throw new Error(`Missing canonical child node for id ${canonicalChild1Id}`);
            }
            this.clusterNodes.set(nodeId, canonicalNode);
            return canonicalNode;
        }

        const child1 = this.uniqueNodes.get(canonicalChild1Id);
        const child2 = this.uniqueNodes.get(canonicalChild2Id);
        if (!child1 || !child2) {
            throw new Error(
                `Missing child nodes for internal node ${nodeId}: ${canonicalChild1Id}, ${canonicalChild2Id}`
            );
        }

        const intersectionContent = new Set(
            Array.from(child1.content).filter((v) => child2.content.has(v))
        );
        const key = this.contentKey(intersectionContent);

        const existingId = this.contentToNodeId.get(key);
        if (existingId !== undefined && intersectionContent.size > 0) {
            if (existingId !== canonicalChild1Id && existingId !== canonicalChild2Id) {
                const existingNode = this.uniqueNodes.get(existingId);
                if (!existingNode) {
                    throw new Error(`Missing existing node for id ${existingId}`);
                }

                existingNode.addChild(canonicalChild1Id);
                existingNode.addChild(canonicalChild2Id);
                existingNode.frequency = Math.max(
                    existingNode.frequency,
                    child1.frequency + child2.frequency
                );
                existingNode.originalIndices = new Set([
                    ...existingNode.originalIndices,
                    ...child1.originalIndices,
                    ...child2.originalIndices
                ]);

                child1.parent = existingId;
                child2.parent = existingId;

                this.redirects.set(nodeId, existingId);
                this.clusterNodes.set(nodeId, existingNode);
                return existingNode;
            }
        }

        const combinedIndices = new Set([...child1.originalIndices, ...child2.originalIndices]);
        const node = new ClusterNode(
            nodeId,
            intersectionContent,
            combinedIndices,
            distance,
            [canonicalChild1Id, canonicalChild2Id],
            null,
            child1.frequency + child2.frequency
        );

        this.clusterNodes.set(nodeId, node);
        this.uniqueNodes.set(nodeId, node);

        if (intersectionContent.size > 0) {
            this.contentToNodeId.set(key, nodeId);
        }

        child1.parent = nodeId;
        child2.parent = nodeId;

        return node;
    }

    cleanupEmptyNodes(): void {
        const emptyNodeIds = Array.from(this.uniqueNodes.entries())
            .filter(([_, node]) => node.isEmpty)
            .map(([nodeId]) => nodeId);

        if (emptyNodeIds.length === 0) {
            return;
        }

        const sortedEmptyIds = emptyNodeIds.sort((a, b) => b - a);

        for (const emptyId of sortedEmptyIds) {
            const emptyNode = this.uniqueNodes.get(emptyId);
            if (!emptyNode) {
                continue;
            }

            const parentId = emptyNode.parent;
            const childrenIds = [...emptyNode.children];

            if (parentId !== null) {
                const parentNode = this.uniqueNodes.get(parentId);
                if (parentNode) {
                    parentNode.removeChild(emptyId);
                    for (const childId of childrenIds) {
                        if (this.uniqueNodes.has(childId)) {
                            parentNode.addChild(childId);
                        }
                    }
                }
            }

            for (const childId of childrenIds) {
                const childNode = this.uniqueNodes.get(childId);
                if (childNode) {
                    childNode.parent = parentId;
                }
            }

            this.uniqueNodes.delete(emptyId);
        }

        for (const node of this.uniqueNodes.values()) {
            if (node.parent !== null && !this.uniqueNodes.has(node.parent)) {
                node.parent = null;
            }
        }
    }

    getNodeStats(): NodeStats {
        const totalNodes = this.uniqueNodes.size;
        let leafNodes = 0;
        let rootNodes = 0;

        for (const node of this.uniqueNodes.values()) {
            if (node.isLeaf) {
                leafNodes += 1;
            }
            if (node.isRoot) {
                rootNodes += 1;
            }
        }

        return {
            totalNodes,
            leafNodes,
            rootNodes,
            internalNodes: totalNodes - leafNodes
        };
    }

    updateSearchPaths(): void {
        const rootNodes = Array.from(this.uniqueNodes.values()).filter((node) => node.isRoot);

        if (rootNodes.length === 0) {
            return;
        }

        if (rootNodes.length === 1) {
            const root = rootNodes[0];
            root.searchPath = [];
            this._updatePathsFromNode(root);
            return;
        }

        const currentMaxId = Math.max(...Array.from(this.uniqueNodes.keys()));
        const virtualRootId = currentMaxId + 1;
        const virtualRoot = new ClusterNode(
            virtualRootId,
            new Set<number>(),
            new Set<number>(),
            0.0,
            rootNodes.map((node) => node.nodeId),
            null,
            rootNodes.reduce((sum, node) => sum + node.frequency, 0)
        );
        virtualRoot.searchPath = [];

        this.uniqueNodes.set(virtualRootId, virtualRoot);

        for (const node of rootNodes) {
            node.parent = virtualRootId;
        }

        this._updatePathsFromNode(virtualRoot);
    }

    _updatePathsFromNode(node: ClusterNode): void {
        for (let childIndex = 0; childIndex < node.children.length; childIndex += 1) {
            const childId = node.children[childIndex];
            const childNode = this.uniqueNodes.get(childId);
            if (!childNode) {
                continue;
            }

            childNode.searchPath = [...node.searchPath, childIndex];
            this._updatePathsFromNode(childNode);
        }
    }
}
