import { ClusterNode, NodeManager, NodeStats } from './tree-nodes.js';
import { IntraContextOrderer } from './intra-ordering.js';
import { computeDistanceMatrixCpu } from './compute-distance.js';

export function linkage(
  condensedDistances: Float64Array,
  n: number,
  method: "single" | "complete" | "average" = "average"
): number[][] {
  const dist: number[][] = Array.from({length: n}, () => new Array(n).fill(Infinity));
  for (let i = 0; i < n; i++) dist[i][i] = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const idx = n * i - (i * (i + 1)) / 2 + j - i - 1;
      dist[i][j] = condensedDistances[idx];
      dist[j][i] = condensedDistances[idx];
    }
  }
  
  const active = new Set(Array.from({length: n}, (_, i) => i));
  const sizes = new Array(2 * n - 1).fill(1);
  const result: number[][] = [];
  
  for (let step = 0; step < n - 1; step++) {
    let minDist = Infinity;
    let minI = -1, minJ = -1;
    
    for (const i of active) {
      for (const j of active) {
        if (j <= i) continue;
        if (dist[i][j] < minDist) {
          minDist = dist[i][j];
          minI = i;
          minJ = j;
        }
      }
    }
    
    const newClusterId = n + step;
    const sizeNew = sizes[minI] + sizes[minJ];
    sizes[newClusterId] = sizeNew;
    
    result.push([minI, minJ, minDist, sizeNew]);
    
    while (dist.length <= newClusterId) {
      dist.push(new Array(dist[0]?.length ?? 0).fill(Infinity));
    }
    for (const row of dist) {
      while (row.length <= newClusterId) row.push(Infinity);
    }
    dist[newClusterId][newClusterId] = 0;
    
    for (const k of active) {
      if (k === minI || k === minJ) continue;
      let newDist: number;
      if (method === "single") {
        newDist = Math.min(dist[minI][k], dist[minJ][k]);
      } else if (method === "complete") {
        newDist = Math.max(dist[minI][k], dist[minJ][k]);
      } else { // average (UPGMA)
        newDist = (dist[minI][k] * sizes[minI] + dist[minJ][k] * sizes[minJ]) / sizeNew;
      }
      dist[newClusterId][k] = newDist;
      dist[k][newClusterId] = newDist;
    }
    
    active.delete(minI);
    active.delete(minJ);
    active.add(newClusterId);
  }
  
  return result;
}

export class IndexResult {
    linkageMatrix: number[][];
    clusterNodes: Map<number, ClusterNode>;
    uniqueNodes: Map<number, ClusterNode>;
    reorderedContexts: (number[] | string[])[];
    originalContexts: (number[] | string[])[];
    stats: NodeStats;
    searchPaths: number[][] | null;
    
    // Legacy attributes for backward compatibility
    reorderedPrompts: (number[] | string[])[];
    originalPrompts: (number[] | string[])[];

    constructor(
        linkageMatrix: number[][],
        clusterNodes: Map<number, ClusterNode>,
        uniqueNodes: Map<number, ClusterNode>,
        reorderedContexts: (number[] | string[])[],
        originalContexts: (number[] | string[])[],
        stats: NodeStats,
        searchPaths: number[][] | null = null
    ) {
        this.linkageMatrix = linkageMatrix;
        this.clusterNodes = clusterNodes;
        this.uniqueNodes = uniqueNodes;
        this.reorderedContexts = reorderedContexts;
        this.originalContexts = originalContexts;
        this.stats = stats;
        this.searchPaths = searchPaths;

        this.reorderedPrompts = this.reorderedContexts;
        this.originalPrompts = this.originalContexts;
    }

    printTree(): void {
        console.log("\n--- Unique Cluster Tree Nodes ---");
        const sortedKeys = Array.from(this.uniqueNodes.keys()).sort((a, b) => a - b);
        for (const nodeId of sortedKeys) {
            const node = this.uniqueNodes.get(nodeId);
            if (!node) continue;
            console.log(`ClusterNode ${nodeId}`);
            console.log(`  Content: [${node.docIds.join(', ')}]`);
            console.log(`  Original indices: [${Array.from(node.originalIndices).sort((a, b) => a - b).join(', ')}]`);
            if (node.searchPath && node.searchPath.length > 0) {
                const pathStr = "[" + node.searchPath.join("][") + "]";
                console.log(`  Search path (child indices from root): ${pathStr}`);
            } else {
                console.log(`  Search path: (root node)`);
            }
            if (!node.isLeaf) {
                console.log(`  Children: [${node.children.join(', ')}]`);
                console.log(`  Merge distance: ${node.mergeDistance.toFixed(4)}`);
            }
            console.log("-".repeat(40));
        }
    }
}

export interface ContextIndexOptions {
    linkageMethod?: "single" | "complete" | "average";
    useGpu?: boolean;
    alpha?: number;
    numWorkers?: number | null;
    batchSize?: number;
}

export class ContextIndex {
    linkageMethod: "single" | "complete" | "average";
    useGpu: boolean;
    alpha: number;
    numWorkers: number | null;
    batchSize: number;

    nodeManager: NodeManager;
    contextOrderer: IntraContextOrderer;

    _strToId: Map<string, number>;
    _idToStr: Map<number, string>;
    _nextStrId: number;
    _isStringInput: boolean;

    constructor(options: ContextIndexOptions = {}) {
        this.linkageMethod = options.linkageMethod || "average";
        this.useGpu = false;
        this.alpha = options.alpha !== undefined ? options.alpha : 0.001;
        this.numWorkers = options.numWorkers || null;
        this.batchSize = options.batchSize || 1000;

        this.nodeManager = new NodeManager();
        this.contextOrderer = new IntraContextOrderer();

        this._strToId = new Map<string, number>();
        this._idToStr = new Map<number, string>();
        this._nextStrId = 0;
        this._isStringInput = false;
    }

    _convertToInt(contexts: (number[] | string[])[]): number[][] {
        if (!contexts || contexts.length === 0 || !contexts[0] || contexts[0].length === 0) {
            return contexts as number[][];
        }
        if (typeof contexts[0][0] === "string") {
            this._isStringInput = true;
            const converted: number[][] = [];
            for (const ctx of contexts as string[][]) {
                const convertedCtx: number[] = [];
                for (const item of ctx) {
                    let sid = this._strToId.get(item);
                    if (sid === undefined) {
                        sid = this._nextStrId;
                        this._strToId.set(item, sid);
                        this._idToStr.set(sid, item);
                        this._nextStrId += 1;
                    }
                    convertedCtx.push(sid);
                }
                converted.push(convertedCtx);
            }
            return converted;
        }
        return contexts as number[][];
    }

    _convertToStr(contexts: number[][]): string[][] {
        if (!this._isStringInput || !contexts || contexts.length === 0) {
            return contexts as any;
        }
        if (contexts[0] && typeof contexts[0][0] === "string") {
            return contexts as any;
        }
        const result: string[][] = [];
        for (const ctx of contexts) {
            const strCtx: string[] = [];
            for (const i of ctx) {
                strCtx.push(this._idToStr.get(i) as string);
            }
            result.push(strCtx);
        }
        return result;
    }

    fitTransform(contexts: (number[] | string[])[]): IndexResult {
        const intContexts = this._convertToInt(contexts);
        const n = intContexts.length;

        if (n < 2) {
            return this._handleSinglePrompt(intContexts);
        }

        const condensedDistances = this._computeDistanceMatrix(intContexts);
        const linkageMatrix = linkage(condensedDistances, n, this.linkageMethod);

        this._buildTree(intContexts, linkageMatrix);
        
        this.nodeManager.cleanupEmptyNodes();
        this.nodeManager.updateSearchPaths();

        const reorderedContexts = this.contextOrderer.reorderContexts(
            intContexts,
            this.nodeManager.uniqueNodes
        );

        const searchPaths = this.contextOrderer.extractSearchPaths(
            this.nodeManager.uniqueNodes,
            intContexts.length
        );

        const stats = this.nodeManager.getNodeStats();

        return new IndexResult(
            linkageMatrix,
            this.nodeManager.clusterNodes,
            this.nodeManager.uniqueNodes,
            reorderedContexts,
            intContexts,
            stats,
            searchPaths
        );
    }

    _computeDistanceMatrix(contexts: number[][]): Float64Array {
        return computeDistanceMatrixCpu(contexts, this.alpha);
    }

    _handleSinglePrompt(contexts: number[][]): IndexResult {
        for (let i = 0; i < contexts.length; i++) {
            const prompt = contexts[i];
            const node = this.nodeManager.createLeafNode(i, prompt);
            node.docIds = [...prompt];
        }

        const leafIds = Array.from(this.nodeManager.uniqueNodes.keys());
        const virtualRootId = leafIds.length > 0 ? Math.max(...leafIds) + 1 : 0;
        
        let freqSum = 0;
        for (const nid of leafIds) {
            const n = this.nodeManager.uniqueNodes.get(nid);
            if (n) freqSum += n.frequency;
        }

        const virtualRoot = new ClusterNode(
            virtualRootId,
            new Set<number>(),
            new Set<number>(),
            0.0,
            leafIds,
            null,
            freqSum
        );
        this.nodeManager.uniqueNodes.set(virtualRootId, virtualRoot);

        for (const nid of leafIds) {
            const n = this.nodeManager.uniqueNodes.get(nid);
            if (n) {
                n.parent = virtualRootId;
            }
        }

        this.nodeManager.updateSearchPaths();

        const searchPaths = this.contextOrderer.extractSearchPaths(
            this.nodeManager.uniqueNodes,
            contexts.length
        );

        const reorderedContexts = contexts.map(c => [...c]);

        return new IndexResult(
            [],
            this.nodeManager.clusterNodes,
            this.nodeManager.uniqueNodes,
            reorderedContexts,
            contexts,
            this.nodeManager.getNodeStats(),
            searchPaths
        );
    }

    _buildTree(contexts: number[][], linkageMatrix: number[][]): void {
        const n = contexts.length;

        for (let i = 0; i < n; i++) {
            this.nodeManager.createLeafNode(i, contexts[i]);
        }

        for (let i = 0; i < linkageMatrix.length; i++) {
            const [idx1, idx2, distance] = linkageMatrix[i];
            const newNodeId = n + i;
            this.nodeManager.createInternalNode(
                newNodeId,
                Math.floor(idx1),
                Math.floor(idx2),
                distance
            );
        }
    }
}

export function buildContextIndex(
    contexts: (number[] | string[])[],
    options: ContextIndexOptions = {}
): IndexResult {
    const indexer = new ContextIndex(options);
    const result = indexer.fitTransform(contexts);

    if (indexer._isStringInput) {
        result.reorderedContexts = indexer._convertToStr(result.reorderedContexts as number[][]);
        result.originalContexts = indexer._convertToStr(result.originalContexts as number[][]);
        result.reorderedPrompts = result.reorderedContexts;
        result.originalPrompts = result.originalContexts;
    }

    return result;
}
