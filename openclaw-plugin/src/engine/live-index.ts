import { ContextIndex, IndexResult } from './index-construction.js';
import { ClusterNode, NodeManager } from './tree-nodes.js';
import { NodeMetadata } from './metadata.js';
import { InterContextScheduler } from './inter-scheduler.js';
import { IntraContextOrderer } from './intra-ordering.js';
import { computeDistanceSingle, computeDistancesBatch } from './compute-distance.js';
import { ConversationTracker, type DeduplicationResult } from './conversation-tracker.js';
import { EvictionHeap } from './eviction-heap.js';
import crypto from 'crypto';

export function computePrefixLength(list1: number[], list2: number[]): number {
    let length = 0;
    const minLen = Math.min(list1.length, list2.length);
    for (let i = 0; i < minLen; i++) {
        if (list1[i] === list2[i]) {
            length++;
        } else {
            break;
        }
    }
    return length;
}

export class ContextPilot extends ContextIndex {
    metadata: Map<number, NodeMetadata> = new Map();
    interScheduler = new InterContextScheduler();
    
    protected _requestToNode: Map<string, number | null> = new Map();
    protected _nextRequestCounter: number = 0;
    
    protected _conversations: Map<string, { seenDocs: Set<any>; turnCount: number }> = new Map();
    protected _hasExplicitConversation: boolean = false;
    
    isLive: boolean = false;
    initialResult: any = null;
    scheduledResult: any = null;
    
    nodes: Map<number, ClusterNode> = new Map();
    rootId: number | null = null;
    nextNodeId: number = 0;
    
    liveStats = {
        totalSearches: 0,
        totalInsertions: 0,
        totalEvictions: 0,
        totalSearchTimeUs: 0,
        totalTraversalTimeUs: 0,
        totalRemovals: 0
    };
    
    static readonly _DEFAULT_CONVERSATION = "_default";

    constructor(alpha: number = 0.001, useGpu: boolean = false, linkageMethod: string = "average", batchSize: number = 10000) {
        super(alpha, useGpu, linkageMethod, batchSize);
    }

    getAllRequestIds(): Set<string> {
        return new Set(this._requestToNode.keys());
    }

    reset(): void {
        this.metadata.clear();
        this._requestToNode.clear();
        this._nextRequestCounter = 0;
        this.isLive = false;
        this.initialResult = null;
        this.scheduledResult = null;
        this.nodes.clear();
        this.rootId = null;
        this.nextNodeId = 0;
        this.liveStats = {
            totalSearches: 0,
            totalInsertions: 0,
            totalEvictions: 0,
            totalSearchTimeUs: 0,
            totalTraversalTimeUs: 0,
            totalRemovals: 0
        };
    }

    buildAndSchedule(contexts: number[][], initialTokensPerContext: number = 0): any {
        this.initialResult = this.fitTransform(contexts);
        
        const [scheduledReordered, scheduledOriginals, finalMapping, groups] = 
            this.interScheduler.scheduleContexts(this.initialResult);
            
        this.scheduledResult = {
            reordered_contexts: scheduledReordered,
            original_indices: finalMapping,
            scheduled_originals: scheduledOriginals,
            groups: groups,
            clustering_result: this.initialResult
        };
        
        const [requestIdMapping, requestIdsOrdered] = this._initializeLiveMetadata(
            initialTokensPerContext,
            contexts.length
        );
        
        this.scheduledResult['request_id_mapping'] = requestIdMapping;
        this.scheduledResult['request_ids'] = requestIdsOrdered;
        
        this.isLive = true;
        
        return this.scheduledResult;
    }

    reorder(contexts: any, initialTokensPerContext: number = 0, conversationId?: string): [any[], number[]] {
        if (contexts && !Array.isArray(contexts[0])) {
            contexts = [contexts];
        }

        const result = this.buildIncremental(contexts, initialTokensPerContext);
        const reordered = result.reordered_contexts;

        const cid = conversationId || ContextPilot._DEFAULT_CONVERSATION;
        if (conversationId !== undefined && conversationId !== null) {
            this._hasExplicitConversation = true;
        }

        let conv = this._conversations.get(cid);
        if (!conv) {
            conv = { seenDocs: new Set(), turnCount: 0 };
            this._conversations.set(cid, conv);
        }

        for (const ctx of reordered) {
            for (const doc of ctx) {
                conv.seenDocs.add(doc);
            }
        }
        conv.turnCount += 1;

        return [reordered, result.original_indices];
    }

    optimize(docs: string[], query: string, conversationId?: string, systemInstruction?: string): any[] {
        const [reordered, _indices] = this.reorder(docs, 0, conversationId);
        const reorderedDocs = reordered[0];
        
        const systemContent = [systemInstruction, ...reorderedDocs].filter(Boolean).join("\n\n");
        
        return [
            { role: "system", content: systemContent },
            { role: "user", content: query }
        ];
    }

    optimizeBatch(allDocs: string[][], allQueries: string[], systemInstruction?: string): [any[][], number[]] {
        if (allDocs.length !== allQueries.length) {
            throw new Error(`all_docs (${allDocs.length}) and all_queries (${allQueries.length}) must have the same length.`);
        }

        const [reorderedContexts, order] = this.reorder(allDocs);
        const messagesBatch: any[][] = [];

        for (let i = 0; i < reorderedContexts.length; i++) {
            const ctx = reorderedContexts[i];
            const origIdx = order[i];
            
            const systemContent = [systemInstruction, ...ctx].filter(Boolean).join("\n\n");
            messagesBatch.push([
                { role: "system", content: systemContent },
                { role: "user", content: allQueries[origIdx] }
            ]);
        }

        return [messagesBatch, order];
    }

    deduplicate(contexts: any[][], conversationId: string, hintTemplate?: string): any[] {
        if (!conversationId) {
            throw new Error("conversation_id is required for .deduplicate().");
        }

        const template = hintTemplate || "Please refer to [Doc {doc_id}] from the previous conversation.";

        if (!this._conversations.has(conversationId)) {
            throw new Error(`No prior .reorder() call found for conversation_id='${conversationId}'.`);
        }

        const conv = this._conversations.get(conversationId)!;
        const seen = conv.seenDocs;
        const results: any[] = [];

        for (const ctx of contexts) {
            const overlapping = ctx.filter(d => seen.has(d));
            const newDocs = ctx.filter(d => !seen.has(d));
            const hints = overlapping.map(d => template.replace("{doc_id}", String(d)));

            results.push({
                new_docs: newDocs,
                overlapping_docs: overlapping,
                reference_hints: hints,
                deduplicated_docs: newDocs
            });

            for (const d of ctx) {
                seen.add(d);
            }
        }

        conv.turnCount += 1;
        return results;
    }

    buildIncremental(contexts: any[][], initialTokensPerContext: number = 0): any {
        // @ts-ignore - Assuming inherited from ContextIndex
        const convertedContexts = this._convertToInt ? this._convertToInt(contexts) : contexts;

        if (!this.isLive) {
            const result = this.buildAndSchedule(convertedContexts, initialTokensPerContext);
            const reordered = result.reordered_contexts || convertedContexts;
            // @ts-ignore
            const stringReordered = this._convertToStr ? this._convertToStr(reordered) : reordered;
            
            return {
                request_ids: result.request_ids || [],
                reordered_contexts: stringReordered,
                matched_count: 0,
                inserted_count: convertedContexts.length,
                merged_count: 0,
                original_indices: result.original_indices || Array.from({ length: convertedContexts.length }, (_, i) => i),
                groups: result.groups || []
            };
        }

        const matchedContexts: any[] = [];
        const unmatchedContexts: any[] = [];

        const searchResults = this.searchBatch(convertedContexts);

        for (let i = 0; i < convertedContexts.length; i++) {
            const context = convertedContexts[i];
            let [searchPath, matchedNodeId, overlapCount, hasPrefix] = searchResults[i];

            if (overlapCount > 0 && matchedNodeId >= 0 && matchedNodeId !== this.rootId) {
                const matchedNode = this.nodes.get(matchedNodeId);
                let nodeDocs: number[] | null = null;
                
                if (this.metadata.has(matchedNodeId) && this.metadata.get(matchedNodeId)!.docIds) {
                    nodeDocs = this.metadata.get(matchedNodeId)!.docIds as number[];
                } else if (matchedNode && matchedNode.docIds) {
                    nodeDocs = matchedNode.docIds as number[];
                }

                let reordered = context;
                if (nodeDocs) {
                    reordered = this._reorderWithPrefix(context, nodeDocs);
                } else {
                    hasPrefix = true;
                }
                
                matchedContexts.push([i, reordered, searchPath, hasPrefix]);
            } else {
                unmatchedContexts.push([i, context]);
            }
        }

        const requestIds: (string | null)[] = new Array(convertedContexts.length).fill(null);
        const reorderedContexts: any[] = new Array(convertedContexts.length).fill(null);
        const contextInfo: any[] = [];

        for (const [origIdx, reordered, searchPath, hasPrefix] of matchedContexts) {
            const matchedNode = this.traverse(searchPath);
            let newNodeId: number, newSearchPath: number[], requestId: string;

            if (hasPrefix && matchedNode && matchedNode.isLeaf) {
                [newNodeId, newSearchPath, requestId] = this._splitLeafAndInsert(
                    reordered, matchedNode, searchPath, initialTokensPerContext
                );
            } else if (hasPrefix) {
                [newNodeId, newSearchPath, requestId] = this.insert(
                    reordered, searchPath, initialTokensPerContext
                );
            } else {
                const insertPath = searchPath.length > 0 ? searchPath.slice(0, -1) : searchPath;
                [newNodeId, newSearchPath, requestId] = this.insert(
                    reordered, insertPath, initialTokensPerContext
                );
            }
            
            requestIds[origIdx] = requestId;
            reorderedContexts[origIdx] = reordered;
            contextInfo.push([origIdx, requestId, newSearchPath]);
        }

        let mergedCount = 0;
        if (unmatchedContexts.length > 0) {
            const unmatchedOnly = unmatchedContexts.map(x => x[1]);
            
            const tempIndex = new ContextPilot(
                this.alpha,
                // @ts-ignore
                this.useGpu,
                // @ts-ignore
                this.linkageMethod,
                // @ts-ignore
                this.batchSize
            );
            
            const tempResult = tempIndex.fitTransform(unmatchedOnly);
            
            const [mergedRequestIds, mergedSearchPaths] = this._mergeIndex(
                tempResult,
                unmatchedContexts,
                initialTokensPerContext
            );

            for (let i = 0; i < unmatchedContexts.length; i++) {
                const [origIdx, origContext] = unmatchedContexts[i];
                requestIds[origIdx] = mergedRequestIds[i];
                
                if (tempResult.reordered_contexts && i < tempResult.reordered_contexts.length) {
                    reorderedContexts[origIdx] = tempResult.reordered_contexts[i];
                } else {
                    reorderedContexts[origIdx] = origContext;
                }
                
                contextInfo.push([origIdx, mergedRequestIds[i], mergedSearchPaths[i]]);
            }
            
            mergedCount = unmatchedContexts.length;
        }

        const scheduledOrder = this._scheduleIncremental(contextInfo);
        const groups = this._groupByPathPrefix(contextInfo);

        // @ts-ignore
        const finalReorderedStr = this._convertToStr ? this._convertToStr(reorderedContexts) : reorderedContexts;

        return {
            request_ids: requestIds,
            reordered_contexts: finalReorderedStr,
            matched_count: matchedContexts.length,
            inserted_count: convertedContexts.length,
            merged_count: mergedCount,
            original_indices: scheduledOrder,
            groups: groups
        };
    }

    _reorderWithPrefix(context: number[], prefix: number[]): number[] {
        const contextSet = new Set(context);
        const result: number[] = [];
        const prefixUsed = new Set<number>();

        for (const elem of prefix) {
            if (contextSet.has(elem) && !prefixUsed.has(elem)) {
                result.push(elem);
                prefixUsed.add(elem);
            }
        }

        for (const elem of context) {
            if (!prefixUsed.has(elem)) {
                result.push(elem);
            }
        }

        return result;
    }

    _mergeIndex(tempResult: any, unmatchedInfo: any[], initialTokens: number): [string[], number[][]] {
        const requestIds: string[] = [];
        const searchPaths: number[][] = [];
        
        const uniqueNodes = tempResult.unique_nodes || tempResult.uniqueNodes;
        let tempRoot: any = null;
        
        if (uniqueNodes) {
            for (const node of uniqueNodes.values()) {
                if (node.isRoot) {
                    tempRoot = node;
                    break;
                }
            }
        }

        const fallbackInsert = () => {
            for (const [origIdx, context] of unmatchedInfo) {
                const [newNodeId, newPath, reqId] = this.insert(context, [], initialTokens);
                requestIds.push(reqId);
                searchPaths.push(newPath);
            }
        };

        if (!tempRoot || this.rootId === null) {
            fallbackInsert();
            return [requestIds, searchPaths];
        }

        const globalRoot = this.nodes.get(this.rootId);
        if (!globalRoot) {
            fallbackInsert();
            return [requestIds, searchPaths];
        }

        const nodeIdMap = new Map<number, number>();
        const baseChildIdx = globalRoot.children.length;

        for (let childIdx = 0; childIdx < tempRoot.children.length; childIdx++) {
            const tempChildId = tempRoot.children[childIdx];
            const newChildIdx = baseChildIdx + childIdx;
            this._copySubtree(
                uniqueNodes,
                tempChildId,
                this.rootId,
                nodeIdMap,
                initialTokens,
                [newChildIdx]
            );
        }

        for (let i = 0; i < unmatchedInfo.length; i++) {
            const [origIdx, context] = unmatchedInfo[i];
            let tempLeafId: number | null = null;
            
            for (const [nodeId, node] of uniqueNodes.entries()) {
                if (node.isLeaf && node.originalIndices && node.originalIndices.has(i)) {
                    tempLeafId = nodeId;
                    break;
                }
            }

            if (tempLeafId !== null && nodeIdMap.has(tempLeafId)) {
                const newNodeId = nodeIdMap.get(tempLeafId)!;
                if (this.metadata.has(newNodeId)) {
                    const meta = this.metadata.get(newNodeId)!;
                    requestIds.push(meta.requestId!);
                    searchPaths.push(meta.searchPath);
                    continue;
                }
            }

            const [newNodeId, newPath, reqId] = this.insert(context, [], initialTokens);
            requestIds.push(reqId);
            searchPaths.push(newPath);
        }

        return [requestIds, searchPaths];
    }

    _copySubtree(sourceNodes: Map<number, any>, sourceNodeId: number, parentId: number, 
                 nodeIdMap: Map<number, number>, initialTokens: number, searchPath: number[]): void {
        const sourceNode = sourceNodes.get(sourceNodeId);
        if (!sourceNode) return;

        const newNodeId = this.nextNodeId++;
        const content = sourceNode.docIds ? [...sourceNode.docIds] : (sourceNode.content ? [...sourceNode.content] : []);
        const originalIndices = sourceNode.originalIndices ? new Set(sourceNode.originalIndices) : new Set<number>();
        
        const newNode = new ClusterNode(
            newNodeId,
            content,
            [],
            parentId,
            originalIndices
        );
        
        if (sourceNode.docIds) {
            newNode.docIds = [...sourceNode.docIds];
        }

        this.nodes.set(newNodeId, newNode);
        nodeIdMap.set(sourceNodeId, newNodeId);

        const parentNode = this.nodes.get(parentId);
        if (parentNode) {
            parentNode.addChild(newNodeId);
        }

        const isLeaf = sourceNode.isLeaf || sourceNode.is_leaf;
        const requestId = isLeaf ? `req-${crypto.randomUUID().replace(/-/g, '').substring(0, 12)}` : null;

        const parentTokens = this.metadata.has(parentId) ? this.metadata.get(parentId)!.totalTokens : 0;
        
        const metadata = new NodeMetadata(
            newNodeId,
            isLeaf ? initialTokens : 0,
            isLeaf ? Math.max(0, initialTokens - parentTokens) : 0,
            searchPath,
            sourceNode.docIds ? [...sourceNode.docIds] : null,
            isLeaf,
            requestId
        );
        
        this.metadata.set(newNodeId, metadata);

        if (isLeaf && requestId) {
            this._requestToNode.set(requestId, newNodeId);
        }

        if (sourceNode.children) {
            for (let childIdx = 0; childIdx < sourceNode.children.length; childIdx++) {
                const childId = sourceNode.children[childIdx];
                const childSearchPath = [...searchPath, childIdx];
                this._copySubtree(
                    sourceNodes, childId, newNodeId,
                    nodeIdMap, initialTokens, childSearchPath
                );
            }
        }
    }

    _scheduleIncremental(contextInfo: any[]): number[] {
        const groups = new Map<number, any[]>();

        for (const [ctxIdx, reqId, path] of contextInfo) {
            const groupKey = path && path.length > 0 ? path[0] : -1;
            if (!groups.has(groupKey)) {
                groups.set(groupKey, []);
            }
            groups.get(groupKey)!.push({ ctxIdx, len: path ? path.length : 0 });
        }

        const scheduled: number[] = [];
        const sortedKeys = Array.from(groups.keys()).sort((a, b) => a - b);

        for (const groupKey of sortedKeys) {
            const items = groups.get(groupKey)!;
            items.sort((a, b) => b.len - a.len);
            scheduled.push(...items.map(item => item.ctxIdx));
        }

        return scheduled;
    }

    _groupByPathPrefix(contextInfo: any[]): [number, number[]][] {
        const groups = new Map<number, number[]>();

        for (const [ctxIdx, reqId, path] of contextInfo) {
            const groupKey = path && path.length > 0 ? path[0] : -1;
            if (!groups.has(groupKey)) {
                groups.set(groupKey, []);
            }
            groups.get(groupKey)!.push(ctxIdx);
        }

        const result: [number, number[]][] = [];
        for (const [groupKey, indices] of groups.entries()) {
            result.push([indices.length, indices]);
        }

        result.sort((a, b) => b[0] - a[0]);
        return result;
    }

    scheduleOnly(contexts: number[][]): any {
        const result = this.fitTransform(contexts);
        
        const [scheduledReordered, scheduledOriginals, finalMapping, groups] = 
            this.interScheduler.scheduleContexts(result);
            
        return {
            reordered_contexts: scheduledReordered,
            original_indices: finalMapping,
            scheduled_originals: scheduledOriginals,
            groups: groups,
            stats: {
                total_nodes: result.stats?.total_nodes || result.stats?.totalNodes,
                leaf_nodes: result.stats?.leaf_nodes || result.stats?.leafNodes,
                num_contexts: contexts.length,
                num_groups: groups.length
            }
        };
    }

    _initializeLiveMetadata(initialTokensPerContext: number, numInputContexts?: number): [Record<string, number>, (string | null)[]] {
        if (!this.initialResult) {
            throw new Error("Must call fitTransform() before initializing metadata");
        }

        const uniqueNodes = this.initialResult.unique_nodes || this.initialResult.uniqueNodes;
        const reorderedContexts = this.initialResult.reordered_contexts || this.initialResult.reorderedContexts;
        const requestIdMapping: Record<string, number> = {};

        this.nodes = uniqueNodes;

        for (const [nodeId, node] of uniqueNodes.entries()) {
            if (node.isRoot || node.is_root) {
                this.rootId = nodeId;
                break;
            }
        }

        this.nextNodeId = uniqueNodes.size > 0 ? Math.max(...Array.from(uniqueNodes.keys())) + 1 : 0;
        let leafCounter = 0;
        const originalIndexToRequestId = new Map<number, string>();

        for (const [nodeId, node] of uniqueNodes.entries()) {
            const searchPath = this._computeSearchPath(nodeId);
            const isLeaf = node.isLeaf || node.is_leaf;
            
            let totalTokens = 0;
            let requestId: string | null = null;

            if (isLeaf) {
                totalTokens = initialTokensPerContext;
                requestId = `req-${crypto.randomUUID().replace(/-/g, '').substring(0, 12)}`;
                leafCounter++;

                if (node.originalIndices || node.original_indices) {
                    const indices = node.originalIndices || node.original_indices;
                    for (const origIdx of indices) {
                        originalIndexToRequestId.set(origIdx, requestId);
                    }
                }
            }

            let parentTokens = 0;
            if (node.parent !== null && this.metadata.has(node.parent)) {
                parentTokens = this.metadata.get(node.parent)!.totalTokens;
            }
            const extraTokens = Math.max(0, totalTokens - parentTokens);

            let leafDocIds: number[] | null = null;
            if (isLeaf && (node.originalIndices || node.original_indices)) {
                const indices = Array.from((node.originalIndices || node.original_indices) as Set<number>);
                if (indices.length > 0) {
                    const firstOrigIdx = Math.min(...indices);
                    if (reorderedContexts && firstOrigIdx < reorderedContexts.length) {
                        leafDocIds = reorderedContexts[firstOrigIdx];
                    } else {
                        leafDocIds = node.docIds || node.doc_ids;
                    }
                }
            } else {
                leafDocIds = node.docIds || node.doc_ids;
            }

            const metadata = new NodeMetadata(
                nodeId,
                totalTokens,
                extraTokens,
                searchPath,
                leafDocIds,
                isLeaf,
                requestId
            );

            this.metadata.set(nodeId, metadata);

            if (isLeaf && requestId) {
                this._requestToNode.set(requestId, nodeId);
                requestIdMapping[requestId] = nodeId;
            }
        }

        this.nextNodeId = this.nodes.size > 0 ? Math.max(...Array.from(this.nodes.keys())) + 1 : 0;
        this._nextRequestCounter = leafCounter;

        const numContexts = numInputContexts !== undefined ? numInputContexts : originalIndexToRequestId.size;
        const requestIdsOrdered: (string | null)[] = [];
        
        for (let i = 0; i < numContexts; i++) {
            requestIdsOrdered.push(originalIndexToRequestId.get(i) || null);
        }

        return [requestIdMapping, requestIdsOrdered];
    }

    trackRequest(requestId: string): void {
        if (!this._requestToNode.has(requestId)) {
            this._requestToNode.set(requestId, null);
        }
    }

    removeRequests(requestIds: Set<string>): any {
        const evictedNodes: number[] = [];
        const notFound: string[] = [];

        for (const requestId of requestIds) {
            if (!this._requestToNode.has(requestId)) {
                notFound.push(requestId);
                continue;
            }

            const nodeId = this._requestToNode.get(requestId);
            this._requestToNode.delete(requestId);

            if (nodeId !== null && nodeId !== undefined) {
                evictedNodes.push(nodeId);
                this._removeNodeAndPrune(nodeId);
            }
        }

        this.liveStats.totalEvictions += evictedNodes.length;

        const arrayReqs = Array.from(requestIds);
        return {
            removed_count: evictedNodes.length,
            evicted_node_ids: evictedNodes,
            evicted_request_ids: arrayReqs.filter(id => !notFound.includes(id)),
            not_found: notFound,
            nodes_remaining: this.nodes.size,
            requests_remaining: this._requestToNode.size
        };
    }

    removeRequestById(requestId: string): boolean {
        const result = this.removeRequests(new Set([requestId]));
        return result.evicted_node_ids.length > 0;
    }

    getRequestNode(requestId: string): number | null {
        return this._requestToNode.get(requestId) ?? null;
    }

    _collectAllNodeDocs(): [number[], number[][], Record<number, number[]>] {
        const nodeIds: number[] = [];
        const nodeDocsList: number[][] = [];
        const nodeIdToPath: Record<number, number[]> = {};

        if (this.rootId === null) return [nodeIds, nodeDocsList, nodeIdToPath];

        const queue: [number, number[]][] = [[this.rootId, []]];

        while (queue.length > 0) {
            const [nodeId, path] = queue.shift()!;

            if (!this.nodes.has(nodeId)) continue;

            const node = this.nodes.get(nodeId)!;
            const nodeMeta = this.metadata.get(nodeId);

            let docs: number[] | null = null;
            if (nodeMeta && nodeMeta.docIds) {
                docs = nodeMeta.docIds;
            } else if (node.docIds) {
                docs = node.docIds;
            }

            if (docs) {
                nodeIds.push(nodeId);
                nodeDocsList.push(docs);
                nodeIdToPath[nodeId] = path;
            }

            if (!node.isLeaf && node.children) {
                for (let idx = 0; idx < node.children.length; idx++) {
                    queue.push([node.children[idx], [...path, idx]]);
                }
            }
        }

        return [nodeIds, nodeDocsList, nodeIdToPath];
    }

    _getNodeDocs(nodeId: number): number[] | null {
        const meta = this.metadata.get(nodeId);
        if (meta && meta.docIds) return meta.docIds;
        const node = this.nodes.get(nodeId);
        if (node && node.docIds) return node.docIds;
        return null;
    }

    _searchSingleHierarchical(context: number[]): [number[], number, number, boolean] {
        const contextSet = new Set(context);
        let currentId = this.rootId;
        let currentPath: number[] = [];

        while (true) {
            if (currentId === null) return [[], -1, 0, false];
            const currentNode = this.nodes.get(currentId);
            
            if (!currentNode || currentNode.isLeaf || !currentNode.children || currentNode.children.length === 0) {
                const docs = this._getNodeDocs(currentId);
                if (docs && currentId !== this.rootId) {
                    const overlap = Array.from(contextSet).filter(x => new Set(docs).has(x)).length;
                    const hasPrefix = overlap > 0 ? contextSet.has(docs[0]) : false;
                    return [currentPath, currentId, overlap, hasPrefix];
                }
                return [[], -1, 0, false];
            }

            const childIds: number[] = [];
            const childDocsList: number[][] = [];
            const childIndices: number[] = [];

            for (let idx = 0; idx < currentNode.children.length; idx++) {
                const childId = currentNode.children[idx];
                const docs = this._getNodeDocs(childId);
                if (docs) {
                    childIds.push(childId);
                    childDocsList.push(docs);
                    childIndices.push(idx);
                }
            }

            if (childIds.length === 0) return [[], -1, 0, false];

            const distances = computeDistancesBatch([context], childDocsList, this.alpha);
            
            let bestJ = -1;
            let bestDistance = Infinity;
            let bestOverlap = 0;

            for (let j = 0; j < childIds.length; j++) {
                const docs = childDocsList[j];
                const overlap = Array.from(contextSet).filter(x => new Set(docs).has(x)).length;
                if (overlap === 0) continue;
                
                const dist = Array.isArray(distances[0]) ? distances[0][j] : distances[j];
                
                if (dist < bestDistance) {
                    bestDistance = dist;
                    bestOverlap = overlap;
                    bestJ = j;
                }
            }

            if (bestJ < 0) {
                if (currentId !== this.rootId) {
                    const docs = this._getNodeDocs(currentId);
                    if (docs) {
                        const overlap = Array.from(contextSet).filter(x => new Set(docs).has(x)).length;
                        return [currentPath, currentId, overlap, true];
                    }
                }
                return [[], -1, 0, false];
            }

            const bestChildId = childIds[bestJ];
            const bestChildIdx = childIndices[bestJ];
            const bestDocs = childDocsList[bestJ];
            const childPath = [...currentPath, bestChildIdx];

            if (contextSet.has(bestDocs[0])) {
                const bestChildNode = this.nodes.get(bestChildId);
                if (bestChildNode && !bestChildNode.isLeaf && bestChildNode.children && bestChildNode.children.length > 0) {
                    currentId = bestChildId;
                    currentPath = childPath;
                    continue;
                } else {
                    return [childPath, bestChildId, bestOverlap, true];
                }
            } else {
                return [childPath, bestChildId, bestOverlap, false];
            }
        }
    }

    searchBatch(contexts: number[][]): [number[], number, number, boolean][] {
        const startTime = globalThis.performance ? globalThis.performance.now() : Date.now();

        if (this.rootId === null || contexts.length === 0) {
            return contexts.map(() => [[], -1, 0, false]);
        }

        const results = contexts.map(ctx => this._searchSingleHierarchical(ctx));

        const endTime = globalThis.performance ? globalThis.performance.now() : Date.now();
        const elapsedUs = (endTime - startTime) * 1000;
        
        this.liveStats.totalSearches += contexts.length;
        this.liveStats.totalSearchTimeUs += elapsedUs;

        return results;
    }

    search(context: number[], updateAccess: boolean = true): [number[], number, number, boolean] {
        const results = this.searchBatch([context]);
        const [searchPath, nodeId, overlap, hasPrefix] = results[0];

        if (updateAccess && nodeId >= 0 && this.metadata.has(nodeId)) {
            this.metadata.get(nodeId)!.updateAccessTime();
        }

        return [searchPath, nodeId, overlap, hasPrefix];
    }

    traverse(searchPath: number[]): ClusterNode | null {
        const startTime = globalThis.performance ? globalThis.performance.now() : Date.now();

        if (this.rootId === null) return null;

        let currentId = this.rootId;

        for (const childIdx of searchPath) {
            if (!this.nodes.has(currentId)) return null;

            const currentNode = this.nodes.get(currentId)!;

            if (!currentNode.children || childIdx >= currentNode.children.length) {
                return null;
            }

            currentId = currentNode.children[childIdx];
        }

        const endTime = globalThis.performance ? globalThis.performance.now() : Date.now();
        const elapsedUs = (endTime - startTime) * 1000;
        this.liveStats.totalTraversalTimeUs += elapsedUs;

        return this.nodes.get(currentId) || null;
    }

    insert(context: number[], searchPath: number[], totalTokens: number = 0): [number, number[], string] {
        const startTime = globalThis.performance ? globalThis.performance.now() : Date.now();

        let matchedNode = this.traverse(searchPath);

        if (!matchedNode) {
            matchedNode = this.nodes.get(this.rootId!)!;
            searchPath = [];
        }

        let newNodeId: number, newSearchPath: number[], requestId: string;

        if (matchedNode.isLeaf) {
            [newNodeId, newSearchPath, requestId] = this._insertAtLeaf(
                context, matchedNode, searchPath, totalTokens
            );
        } else {
            [newNodeId, newSearchPath, requestId] = this._insertAtInternal(
                context, matchedNode, searchPath, totalTokens
            );
        }

        const endTime = globalThis.performance ? globalThis.performance.now() : Date.now();
        this.liveStats.totalInsertions += 1;

        return [newNodeId, newSearchPath, requestId];
    }

    _insertAtInternal(context: number[], parentNode: ClusterNode, searchPath: number[], totalTokens: number): [number, number[], string] {
        const requestId = `req-${crypto.randomUUID().replace(/-/g, '').substring(0, 12)}`;
        
        const newNodeId = this.nextNodeId++;
        const newNode = new ClusterNode(
            newNodeId,
            context,
            [],
            parentNode.nodeId,
            new Set([newNodeId])
        );

        this.nodes.set(newNodeId, newNode);
        parentNode.addChild(newNodeId);

        const parentTokens = this.metadata.has(parentNode.nodeId) ? this.metadata.get(parentNode.nodeId)!.totalTokens : 0;
        const newSearchPath = [...searchPath, parentNode.children.length - 1];

        const metadata = new NodeMetadata(
            newNodeId,
            totalTokens,
            Math.max(0, totalTokens - parentTokens),
            newSearchPath,
            context,
            true,
            requestId
        );

        this.metadata.set(newNodeId, metadata);
        this._requestToNode.set(requestId, newNodeId);

        return [newNodeId, newSearchPath, requestId];
    }

    _insertAtLeaf(context: number[], leafNode: ClusterNode, searchPath: number[], totalTokens: number): [number, number[], string] {
        const requestId = `req-${crypto.randomUUID().replace(/-/g, '').substring(0, 12)}`;
        
        let parentNode: ClusterNode;
        let parentSearchPath: number[];

        if (leafNode.parent === null) {
            parentNode = this.nodes.get(this.rootId!)!;
            parentSearchPath = [];
        } else {
            parentNode = this.nodes.get(leafNode.parent)!;
            parentSearchPath = searchPath.length > 0 ? searchPath.slice(0, -1) : [];
        }

        const newLeafId = this.nextNodeId++;
        const newLeaf = new ClusterNode(
            newLeafId,
            context,
            [],
            parentNode.nodeId,
            new Set([newLeafId])
        );

        this.nodes.set(newLeafId, newLeaf);
        parentNode.addChild(newLeafId);

        const newSearchPath = [...parentSearchPath, parentNode.children.length - 1];
        const parentTokens = this.metadata.has(parentNode.nodeId) ? this.metadata.get(parentNode.nodeId)!.totalTokens : 0;

        const newMetadata = new NodeMetadata(
            newLeafId,
            totalTokens,
            Math.max(0, totalTokens - parentTokens),
            newSearchPath,
            context,
            true,
            requestId
        );

        this.metadata.set(newLeafId, newMetadata);
        this._requestToNode.set(requestId, newLeafId);

        return [newLeafId, newSearchPath, requestId];
    }

    _splitLeafAndInsert(context: number[], leafNode: ClusterNode, searchPath: number[], totalTokens: number): [number, number[], string] {
        const matchedDocs = this._getNodeDocs(leafNode.nodeId);

        if (!matchedDocs) {
            return this._insertAtLeaf(context, leafNode, searchPath, totalTokens);
        }

        const sharedPrefix: number[] = [];
        for (let i = 0; i < Math.min(matchedDocs.length, context.length); i++) {
            if (matchedDocs[i] === context[i]) {
                sharedPrefix.push(matchedDocs[i]);
            } else {
                break;
            }
        }

        if (sharedPrefix.length === 0) {
            return this._insertAtLeaf(context, leafNode, searchPath, totalTokens);
        }

        if (sharedPrefix.length === matchedDocs.length && new Set(matchedDocs).size === new Set(context).size && 
            [...new Set(matchedDocs)].every(d => new Set(context).has(d))) {
            return this._insertAtLeaf(context, leafNode, searchPath, totalTokens);
        }

        let parentId = leafNode.parent;
        if (parentId === null) {
            parentId = this.rootId!;
        }
        const parentNode = this.nodes.get(parentId)!;
        const parentSearchPath = searchPath.length > 0 ? searchPath.slice(0, -1) : [];

        const leafChildIdx = parentNode.children.indexOf(leafNode.nodeId);

        const newInternalId = this.nextNodeId++;
        const allContent = new Set([...leafNode.content, ...context]);
        
        const newInternal = new ClusterNode(
            newInternalId,
            Array.from(allContent),
            [leafNode.nodeId],
            parentId,
            new Set()
        );
        newInternal.docIds = [...sharedPrefix];

        this.nodes.set(newInternalId, newInternal);

        parentNode.children[leafChildIdx] = newInternalId;
        leafNode.parent = newInternalId;

        const parentTokens = this.metadata.has(parentId) ? this.metadata.get(parentId)!.totalTokens : 0;
        const leafMeta = this.metadata.get(leafNode.nodeId);
        const leafTotal = leafMeta ? leafMeta.totalTokens : 0;

        let internalTokens = parentTokens;
        if (matchedDocs && matchedDocs.length > 0) {
            const prefixRatio = sharedPrefix.length / matchedDocs.length;
            internalTokens = Math.floor(parentTokens + (leafTotal - parentTokens) * prefixRatio);
        }

        const internalPath = [...parentSearchPath, leafChildIdx];

        const internalMeta = new NodeMetadata(
            newInternalId,
            internalTokens,
            Math.max(0, internalTokens - parentTokens),
            internalPath,
            [...sharedPrefix],
            false,
            null
        );
        this.metadata.set(newInternalId, internalMeta);

        if (leafMeta) {
            leafMeta.extraTokens = Math.max(0, leafTotal - internalTokens);
            leafMeta.searchPath = [...internalPath, 0];
        }

        const requestId = `req-${crypto.randomUUID().replace(/-/g, '').substring(0, 12)}`;
        const newLeafId = this.nextNodeId++;

        const newLeaf = new ClusterNode(
            newLeafId,
            context,
            [],
            newInternalId,
            new Set([newLeafId])
        );
        newLeaf.docIds = [...context];

        this.nodes.set(newLeafId, newLeaf);
        newInternal.addChild(newLeafId);

        const newLeafPath = [...internalPath, 1];

        const newLeafMeta = new NodeMetadata(
            newLeafId,
            totalTokens,
            Math.max(0, totalTokens - internalTokens),
            newLeafPath,
            [...context],
            true,
            requestId
        );

        this.metadata.set(newLeafId, newLeafMeta);
        this._requestToNode.set(requestId, newLeafId);

        return [newLeafId, newLeafPath, requestId];
    }

    updateNode(searchPath: number[], tokenDelta: number): boolean {
        const node = this.traverse(searchPath);
        
        if (!node || !this.metadata.has(node.nodeId)) {
            return false;
        }

        const metadata = this.metadata.get(node.nodeId)!;

        if (tokenDelta > 0) {
            metadata.addTokens(tokenDelta);
        } else {
            metadata.removeTokens(Math.abs(tokenDelta));
        }

        return true;
    }

    _removeNode(nodeId: number): void {
        this._removeNodeAndPrune(nodeId);
    }

    _removeNodeAndPrune(nodeId: number): number {
        if (!this.nodes.has(nodeId)) {
            return 0;
        }

        let nodesPruned = 0;
        const node = this.nodes.get(nodeId)!;
        const parentId = node.parent;

        if (parentId !== null && this.nodes.has(parentId)) {
            const parent = this.nodes.get(parentId)!;
            const idx = parent.children.indexOf(nodeId);
            if (idx > -1) {
                parent.children.splice(idx, 1);
            }

            if (parent.children.length === 0 && !parent.isRoot) {
                nodesPruned += 1;
                nodesPruned += this._removeNodeAndPrune(parentId);
            }
        }

        this.nodes.delete(nodeId);

        if (this.metadata.has(nodeId)) {
            this.metadata.delete(nodeId);
        }

        return nodesPruned;
    }

    _computeSearchPath(nodeId: number): number[] {
        if (nodeId === this.rootId) return [];

        const path: number[] = [];
        let currentId: number | null = nodeId;
        const visited = new Set<number>();

        while (currentId !== this.rootId && currentId !== null) {
            if (visited.has(currentId)) break;
            visited.add(currentId);

            const node = this.nodes.get(currentId);
            if (!node || node.parent === null) break;

            const parent = this.nodes.get(node.parent);
            if (!parent) break;

            const childIdx = parent.children.indexOf(currentId);
            if (childIdx === -1) break;

            path.push(childIdx);
            currentId = node.parent;
        }

        return path.reverse();
    }

    _findCommonPrefix(list1: number[], list2: number[]): number[] {
        const prefix: number[] = [];
        const minLen = Math.min(list1.length, list2.length);
        for (let i = 0; i < minLen; i++) {
            if (list1[i] === list2[i]) {
                prefix.push(list1[i]);
            } else {
                break;
            }
        }
        return prefix;
    }

    getStats(): any {
        const avgSearchTime = this.liveStats.totalSearches > 0 
            ? this.liveStats.totalSearchTimeUs / this.liveStats.totalSearches 
            : 0;

        let totalTokens = 0;
        for (const meta of this.metadata.values()) {
            totalTokens += meta.extraTokens;
        }

        return {
            num_nodes: this.nodes.size,
            active_nodes: this.metadata.size,
            total_tokens: totalTokens,
            num_requests: this._requestToNode.size,
            total_searches: this.liveStats.totalSearches,
            total_insertions: this.liveStats.totalInsertions,
            total_removals: this.liveStats.totalRemovals,
            avg_search_time_us: avgSearchTime
        };
    }
}
