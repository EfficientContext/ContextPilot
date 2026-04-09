import type { ClusterNode } from './tree-nodes.js';

export class IntraContextOrderer {
    reorderContexts(originalContexts: number[][], uniqueNodes: Map<number, ClusterNode>): number[][] {
        let rootNode: ClusterNode | null = null;
        for (const node of uniqueNodes.values()) {
            if (node.isRoot) {
                rootNode = node;
                break;
            }
        }

        if (!rootNode) {
            return originalContexts;
        }

        for (const node of uniqueNodes.values()) {
            if (node.isLeaf && node.originalIndices.size > 0) {
                const firstIdx = Math.min(...node.originalIndices);
                if (firstIdx < originalContexts.length) {
                    this._setNodeDocs(node, [...originalContexts[firstIdx]]);
                }
            }
        }

        const queue: number[] = [rootNode.nodeId];
        const visited = new Set<number>();

        while (queue.length > 0) {
            const nodeId = queue.shift()!;
            if (visited.has(nodeId) || !uniqueNodes.has(nodeId)) {
                continue;
            }

            visited.add(nodeId);
            const node = uniqueNodes.get(nodeId)!;

            if (!node.isRoot && node.parent !== null) {
                const parentNode = uniqueNodes.get(node.parent);
                if (parentNode) {
                    const parentDocs = this._getNodeDocs(parentNode);
                    const nodeDocs = this._getNodeDocs(node);
                    if (parentDocs.length > 0 && nodeDocs.length > 0) {
                        this._setNodeDocs(node, this._reorderWithParentPrefix(nodeDocs, parentDocs));
                    }
                }
            }

            for (const childId of node.children) {
                if (uniqueNodes.has(childId)) {
                    queue.push(childId);
                }
            }
        }

        const reorderedContexts: number[][] = [];
        for (let i = 0; i < originalContexts.length; i += 1) {
            const leafNode = this._findLeafNode(i, uniqueNodes);
            if (leafNode) {
                const leafDocs = this._getNodeDocs(leafNode);
                if (leafDocs.length > 0) {
                    reorderedContexts.push(leafDocs);
                    continue;
                }
            }

            reorderedContexts.push([...originalContexts[i]]);
        }

        return reorderedContexts;
    }

    _updateTreeAndReorderNodes(uniqueNodes: Map<number, ClusterNode>, reorderedContexts: number[][]): void {
        let rootNode: ClusterNode | null = null;
        for (const node of uniqueNodes.values()) {
            if (node.isRoot) {
                rootNode = node;
                break;
            }
        }

        for (const node of uniqueNodes.values()) {
            if (node.isLeaf && node.originalIndices.size > 0) {
                const firstIdx = Math.min(...node.originalIndices);
                if (firstIdx < reorderedContexts.length) {
                    this._setNodeDocs(node, [...reorderedContexts[firstIdx]]);
                }
            }
        }

        if (!rootNode) {
            return;
        }

        const queue: Array<[number, boolean]> = [];
        for (const childId of rootNode.children) {
            if (uniqueNodes.has(childId)) {
                queue.push([childId, true]);
            }
        }

        while (queue.length > 0) {
            const [nodeId, isChildOfRoot] = queue.shift()!;
            const node = uniqueNodes.get(nodeId);
            if (!node) {
                continue;
            }

            if (!isChildOfRoot && node.parent !== null) {
                const parentNode = uniqueNodes.get(node.parent);
                if (parentNode) {
                    const parentDocs = this._getNodeDocs(parentNode);
                    const nodeDocs = this._getNodeDocs(node);
                    if (parentDocs.length > 0 && nodeDocs.length > 0) {
                        this._setNodeDocs(node, this._reorderWithParentPrefix(nodeDocs, parentDocs));
                    }
                }
            }

            for (const childId of node.children) {
                if (uniqueNodes.has(childId)) {
                    queue.push([childId, false]);
                }
            }
        }
    }

    _reorderWithParentPrefix(nodeDocs: number[], parentDocs: number[]): number[] {
        if (parentDocs.length === 0) {
            return nodeDocs;
        }

        const result = [...parentDocs];
        const parentSet = new Set(parentDocs);

        for (const doc of nodeDocs) {
            if (!parentSet.has(doc)) {
                result.push(doc);
            }
        }

        return result;
    }

    _reorderContextWithTreePrefix(
        contextIndex: number,
        originalContext: number[],
        uniqueNodes: Map<number, ClusterNode>
    ): number[] {
        const leafNode = this._findLeafNode(contextIndex, uniqueNodes);
        if (!leafNode) {
            return [...originalContext];
        }

        const prefixDocs: number[] = [];
        const visited = new Set<number>();
        let currentNode: ClusterNode | undefined = leafNode;

        const ancestors: ClusterNode[] = [];
        while (currentNode && !currentNode.isRoot) {
            if (visited.has(currentNode.nodeId)) {
                break;
            }

            visited.add(currentNode.nodeId);
            ancestors.push(currentNode);

            if (currentNode.parent !== null && uniqueNodes.has(currentNode.parent)) {
                currentNode = uniqueNodes.get(currentNode.parent);
            } else {
                break;
            }
        }

        ancestors.reverse();

        const seenDocs = new Set<number>();
        for (const ancestor of ancestors) {
            const ancestorDocs = this._getNodeDocs(ancestor);
            for (const doc of ancestorDocs) {
                if (!seenDocs.has(doc)) {
                    prefixDocs.push(doc);
                    seenDocs.add(doc);
                }
            }
        }

        const result = [...prefixDocs];
        for (const doc of originalContext) {
            if (!seenDocs.has(doc)) {
                result.push(doc);
                seenDocs.add(doc);
            }
        }

        return result;
    }

    extractSearchPaths(uniqueNodes: Map<number, ClusterNode>, numContexts: number): number[][] {
        const searchPaths: number[][] = Array.from({ length: numContexts }, () => []);

        const contextToLeaf = new Map<number, number>();
        for (const [nodeId, node] of uniqueNodes.entries()) {
            if (!node.isLeaf) {
                continue;
            }

            for (const origIdx of node.originalIndices) {
                contextToLeaf.set(origIdx, nodeId);
            }
        }

        for (let contextIdx = 0; contextIdx < numContexts; contextIdx += 1) {
            const leafId = contextToLeaf.get(contextIdx);
            if (leafId === undefined) continue;

            const childIndices: number[] = [];
            let currentId: number | null = leafId;
            const visited = new Set<number>();

            while (currentId !== null) {
                if (visited.has(currentId)) {
                    break;
                }
                visited.add(currentId);

                const currentNode = uniqueNodes.get(currentId);
                if (!currentNode) {
                    break;
                }

                if (currentNode.parent !== null) {
                    const parentNode = uniqueNodes.get(currentNode.parent);
                    if (parentNode) {
                        const childIndex = parentNode.children.indexOf(currentId);
                        if (childIndex !== -1) {
                            childIndices.push(childIndex);
                        }
                    }
                }

                currentId = currentNode.parent;
            }

            searchPaths[contextIdx] = childIndices.reverse();
        }

        return searchPaths;
    }

    _reorderSingleContext(
        contextIndex: number,
        originalContext: number[],
        uniqueNodes: Map<number, ClusterNode>
    ): number[] {
        const originalSet = new Set(originalContext);

        const leafNode = this._findLeafNode(contextIndex, uniqueNodes);
        if (!leafNode) {
            return [...originalContext];
        }

        if (leafNode.isRoot) {
            return Array.from(leafNode.content).sort((a, b) => a - b);
        }

        if (leafNode.frequency > 1) {
            const prefixContent = leafNode.content;
            const prefixList = Array.from(prefixContent).sort((a, b) => a - b);
            const remainingList = Array.from(originalSet)
                .filter((value) => !prefixContent.has(value))
                .sort((a, b) => a - b);
            return [...prefixList, ...remainingList];
        }

        const bestNode = this._findBestAncestor(leafNode, uniqueNodes);
        if (!bestNode) {
            return [...originalContext];
        }

        const prefixContent = bestNode.content;
        const prefixList = Array.from(prefixContent).sort((a, b) => a - b);
        const remainingList = Array.from(originalSet)
            .filter((value) => !prefixContent.has(value))
            .sort((a, b) => a - b);
        return [...prefixList, ...remainingList];
    }

    _findLeafNode(contextIndex: number, uniqueNodes: Map<number, ClusterNode>): ClusterNode | null {
        for (const node of uniqueNodes.values()) {
            if (node.isLeaf && node.originalIndices.has(contextIndex)) {
                return node;
            }
        }

        return null;
    }

    _findBestAncestor(startNode: ClusterNode, uniqueNodes: Map<number, ClusterNode>): ClusterNode | null {
        let currentNode: ClusterNode = startNode;

        while (currentNode.parent !== null) {
            const parentId = currentNode.parent;
            const parentNode = uniqueNodes.get(parentId);
            if (!parentNode) {
                return null;
            }

            if (parentNode.frequency > 1 && !parentNode.isEmpty) {
                return parentNode;
            }

            currentNode = parentNode;
        }

        return null;
    }

    reorderPrompts(originalPrompts: number[][], uniqueNodes: Map<number, ClusterNode>): number[][] {
        return this.reorderContexts(originalPrompts, uniqueNodes);
    }

    _reorderSinglePrompt(
        promptIndex: number,
        originalPrompt: number[],
        uniqueNodes: Map<number, ClusterNode>
    ): number[] {
        return this._reorderSingleContext(promptIndex, originalPrompt, uniqueNodes);
    }

    private _getNodeDocs(node: ClusterNode): number[] {
        return Array.from(node.content);
    }

    private _setNodeDocs(node: ClusterNode, docs: number[]): void {
        node.content = new Set(docs);
    }
}
