import * as crypto from 'node:crypto';

interface IndexedDoc {
    doc: string;
    hash: string;
    originalIndex: number;
    previousPosition: number;
}

function hashDoc(doc: string): string {
    return crypto.createHash('sha256').update(doc.trim()).digest('hex').slice(0, 16);
}

function buildIndexMappings(entries: IndexedDoc[], total: number): [number[], number[]] {
    const originalOrder = entries.map((entry) => entry.originalIndex);

    const newOrder = new Array<number>(total);
    for (let newIndex = 0; newIndex < entries.length; newIndex += 1) {
        newOrder[entries[newIndex].originalIndex] = newIndex;
    }

    return [originalOrder, newOrder];
}

function indexDocuments(docs: string[]): IndexedDoc[] {
    return docs.map((doc, originalIndex) => ({
        doc,
        hash: hashDoc(doc),
        originalIndex,
        previousPosition: Number.POSITIVE_INFINITY
    }));
}

export function reorderDocuments(docs: string[]): [string[], number[], number[]] {
    const indexed = indexDocuments(docs);
    indexed.sort((a, b) => {
        const byHash = a.hash.localeCompare(b.hash);
        if (byHash !== 0) {
            return byHash;
        }
        return a.originalIndex - b.originalIndex;
    });

    const reorderedDocs = indexed.map((entry) => entry.doc);
    const [originalOrder, newOrder] = buildIndexMappings(indexed, docs.length);
    return [reorderedDocs, originalOrder, newOrder];
}

export class ReorderState {
    private previousOrder: string[] = [];

    private hashToDoc: Map<string, string> = new Map();

    reorder(docs: string[]): [string[], number[], number[]] {
        const indexed = indexDocuments(docs);
        const previousPositions = new Map<string, number>();

        for (let i = 0; i < this.previousOrder.length; i += 1) {
            const hash = this.previousOrder[i];
            if (!previousPositions.has(hash)) {
                previousPositions.set(hash, i);
            }
        }

        const known: IndexedDoc[] = [];
        const unknown: IndexedDoc[] = [];

        for (const entry of indexed) {
            const previousPosition = previousPositions.get(entry.hash);
            if (previousPosition === undefined) {
                unknown.push(entry);
                continue;
            }

            known.push({ ...entry, previousPosition });
        }

        known.sort((a, b) => {
            if (a.previousPosition !== b.previousPosition) {
                return a.previousPosition - b.previousPosition;
            }
            return a.originalIndex - b.originalIndex;
        });

        unknown.sort((a, b) => {
            const byHash = a.hash.localeCompare(b.hash);
            if (byHash !== 0) {
                return byHash;
            }
            return a.originalIndex - b.originalIndex;
        });

        const reordered = [...known, ...unknown];

        this.previousOrder = reordered.map((entry) => entry.hash);
        for (const entry of reordered) {
            this.hashToDoc.set(entry.hash, entry.doc);
        }

        const reorderedDocs = reordered.map((entry) => entry.doc);
        const [originalOrder, newOrder] = buildIndexMappings(reordered, docs.length);
        return [reorderedDocs, originalOrder, newOrder];
    }

    reset(): void {
        this.previousOrder = [];
        this.hashToDoc.clear();
    }
}
