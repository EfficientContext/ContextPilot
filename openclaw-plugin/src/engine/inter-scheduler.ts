import type { ClusterNode } from './tree-nodes.js';

export interface ClusteringResult {
    reorderedPrompts: number[][];
    originalPrompts: number[][];
    searchPaths: number[][];
}

export class InterContextScheduler {
    scheduleContexts(
        clusteringResult: ClusteringResult
    ): [number[][], number[][], number[], Array<[number, number[]]>] {
        const { reorderedPrompts, originalPrompts, searchPaths } = clusteringResult;

        const groupsByRoot = this._groupByRootPrefix(searchPaths);
        const sortedGroups = this._sortGroupsByPathLength(groupsByRoot, searchPaths);

        const allGroupsWithInfo: Array<[number, number[]]> = [];
        for (const groupIndices of sortedGroups) {
            allGroupsWithInfo.push([0, groupIndices]);
        }

        allGroupsWithInfo.sort((a, b) => {
            const sizeDiff = b[1].length - a[1].length;
            if (sizeDiff !== 0) {
                return sizeDiff;
            }

            const aFirst = a[1].length > 0 ? a[1][0] : Number.POSITIVE_INFINITY;
            const bFirst = b[1].length > 0 ? b[1][0] : Number.POSITIVE_INFINITY;
            return aFirst - bFirst;
        });

        const finalIndexMapping = allGroupsWithInfo.flatMap(([, group]) => group);

        const scheduledReordered = finalIndexMapping.map((idx) => reorderedPrompts[idx]);
        const scheduledOriginals = finalIndexMapping.map((idx) => originalPrompts[idx]);

        return [scheduledReordered, scheduledOriginals, finalIndexMapping, allGroupsWithInfo];
    }

    _groupByRootPrefix(searchPaths: number[][]): Map<number, number[]> {
        const groups = new Map<number, number[]>();

        for (let contextIdx = 0; contextIdx < searchPaths.length; contextIdx += 1) {
            const path = searchPaths[contextIdx];
            const groupKey = path.length >= 1 ? path[0] : -1;

            const existing = groups.get(groupKey);
            if (existing) {
                existing.push(contextIdx);
            } else {
                groups.set(groupKey, [contextIdx]);
            }
        }

        return groups;
    }

    _sortGroupsByPathLength(
        groupsByRoot: Map<number, number[]>,
        searchPaths: number[][]
    ): number[][] {
        const sortedGroups: number[][] = [];

        for (const groupIndices of groupsByRoot.values()) {
            const sortedGroup = [...groupIndices].sort((a, b) => {
                const lengthDiff = searchPaths[b].length - searchPaths[a].length;
                if (lengthDiff !== 0) {
                    return lengthDiff;
                }

                const lexCompare = this._compareNumberArrays(searchPaths[a], searchPaths[b]);
                if (lexCompare !== 0) {
                    return lexCompare;
                }

                return a - b;
            });

            sortedGroups.push(sortedGroup);
        }

        return sortedGroups;
    }

    reorderPrompts(
        clusteringResult: ClusteringResult
    ): [number[][], number[][], number[], Array<[number, number[]]>] {
        return this.scheduleContexts(clusteringResult);
    }

    _reorderSinglePrompt(
        promptIndex: number,
        originalPrompt: number[],
        uniqueNodes: Map<number, ClusterNode>
    ): number[] {
        void promptIndex;
        void uniqueNodes;
        return [...originalPrompt];
    }

    private _compareNumberArrays(a: number[], b: number[]): number {
        const minLength = Math.min(a.length, b.length);
        for (let i = 0; i < minLength; i += 1) {
            if (a[i] !== b[i]) {
                return a[i] - b[i];
            }
        }
        return a.length - b.length;
    }
}
