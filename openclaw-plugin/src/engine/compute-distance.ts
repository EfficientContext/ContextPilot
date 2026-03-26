export interface PreparedContextsCpu {
    chunkIds: number[];
    originalPositions: number[];
    lengths: number[];
    offsets: number[];
}

export function computeDistanceSingle(
    contextA: number[],
    contextB: number[],
    alpha: number = 0.001
): number {
    if (contextA.length === 0 || contextB.length === 0) {
        return 1.0;
    }

    const posA = new Map<number, number>();
    const posB = new Map<number, number>();

    for (let pos = 0; pos < contextA.length; pos += 1) {
        posA.set(contextA[pos], pos);
    }
    for (let pos = 0; pos < contextB.length; pos += 1) {
        posB.set(contextB[pos], pos);
    }

    let intersectionSize = 0;
    let positionDiffSum = 0;

    for (const [docId, aPos] of posA) {
        const bPos = posB.get(docId);
        if (bPos === undefined) {
            continue;
        }

        intersectionSize += 1;
        positionDiffSum += Math.abs(aPos - bPos);
    }

    if (intersectionSize === 0) {
        return 1.0;
    }

    const maxSize = Math.max(contextA.length, contextB.length);
    const overlapTerm = 1.0 - intersectionSize / maxSize;
    const positionTerm = alpha * (positionDiffSum / intersectionSize);

    return overlapTerm + positionTerm;
}

export function computeDistancesBatch(
    queries: number[][],
    targets: number[][],
    alpha: number = 0.001
): number[][] {
    const nQueries = queries.length;
    const nTargets = targets.length;

    if (nQueries === 0 || nTargets === 0) {
        return Array.from({ length: nQueries }, () => new Array<number>(nTargets).fill(0));
    }

    const totalPairs = nQueries * nTargets;
    const distances: number[][] = Array.from(
        { length: nQueries },
        () => new Array<number>(nTargets).fill(1.0)
    );

    if (totalPairs < 1000) {
        for (let i = 0; i < nQueries; i += 1) {
            for (let j = 0; j < nTargets; j += 1) {
                distances[i][j] = computeDistanceSingle(queries[i], targets[j], alpha);
            }
        }
        return distances;
    }

    for (let i = 0; i < nQueries; i += 1) {
        for (let j = 0; j < nTargets; j += 1) {
            distances[i][j] = computeDistanceSingle(queries[i], targets[j], alpha);
        }
    }

    return distances;
}

export function prepareContextsForCpu(contexts: number[][]): PreparedContextsCpu {
    const n = contexts.length;
    const sortedData: Array<Array<[number, number]>> = new Array(n);
    const lengths: number[] = new Array(n).fill(0);

    for (let idx = 0; idx < n; idx += 1) {
        const ctx = contexts[idx];
        if (ctx.length === 0) {
            sortedData[idx] = [];
            lengths[idx] = 0;
            continue;
        }

        const pairs: Array<[number, number]> = new Array(ctx.length);
        for (let origPos = 0; origPos < ctx.length; origPos += 1) {
            pairs[origPos] = [ctx[origPos], origPos];
        }
        pairs.sort((a, b) => a[0] - b[0]);

        sortedData[idx] = pairs;
        lengths[idx] = pairs.length;
    }

    const offsets: number[] = new Array(n + 1).fill(0);
    for (let i = 0; i < n; i += 1) {
        offsets[i + 1] = offsets[i] + lengths[i];
    }

    const totalElements = offsets[n];
    const chunkIds: number[] = new Array(totalElements).fill(0);
    const originalPositions: number[] = new Array(totalElements).fill(0);

    for (let i = 0; i < n; i += 1) {
        const pairs = sortedData[i];
        const start = offsets[i];
        for (let j = 0; j < pairs.length; j += 1) {
            const [chunkId, origPos] = pairs[j];
            chunkIds[start + j] = chunkId;
            originalPositions[start + j] = origPos;
        }
    }

    return {
        chunkIds,
        originalPositions,
        lengths,
        offsets
    };
}

export function computeDistanceOptimized(
    chunkIds: number[],
    originalPositions: number[],
    lengths: number[],
    offsets: number[],
    i: number,
    j: number,
    alpha: number
): number {
    const lenI = lengths[i];
    const lenJ = lengths[j];

    if (lenI === 0 || lenJ === 0) {
        return 1.0;
    }

    const offsetI = offsets[i];
    const offsetJ = offsets[j];
    const endI = offsetI + lenI;
    const endJ = offsetJ + lenJ;

    let intersectionSize = 0;
    let positionDiffSum = 0;

    let pi = offsetI;
    let pj = offsetJ;

    while (pi < endI && pj < endJ) {
        const chunkI = chunkIds[pi];
        const chunkJ = chunkIds[pj];

        if (chunkI === chunkJ) {
            intersectionSize += 1;
            positionDiffSum += Math.abs(originalPositions[pi] - originalPositions[pj]);
            pi += 1;
            pj += 1;
        } else if (chunkI < chunkJ) {
            pi += 1;
        } else {
            pj += 1;
        }
    }

    const maxSize = Math.max(lenI, lenJ);
    const overlapTerm = 1.0 - intersectionSize / maxSize;

    let positionTerm = 0.0;
    if (intersectionSize !== 0) {
        const avgPosDiff = positionDiffSum / intersectionSize;
        positionTerm = alpha * avgPosDiff;
    }

    return overlapTerm + positionTerm;
}

export function computeDistanceMatrixCpu(
    contexts: number[][],
    alpha: number = 0.001
): Float64Array {
    const n = contexts.length;
    const numPairs = (n * (n - 1)) / 2;

    if (numPairs === 0) {
        return new Float64Array(0);
    }

    const { chunkIds, originalPositions, lengths, offsets } = prepareContextsForCpu(contexts);
    const condensedDistances = new Float64Array(numPairs);

    for (let i = 0; i < n; i += 1) {
        for (let j = i + 1; j < n; j += 1) {
            const dist = computeDistanceOptimized(
                chunkIds,
                originalPositions,
                lengths,
                offsets,
                i,
                j,
                alpha
            );

            const condensedIdx = n * i - (i * (i + 1)) / 2 + j - i - 1;
            condensedDistances[condensedIdx] = dist;
        }
    }

    return condensedDistances;
}
