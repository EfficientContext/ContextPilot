#!/usr/bin/env npx tsx
/**
 * E2E test for ContextPilot plugin
 *
 * Run: npx tsx test-e2e.ts
 * Requires: ANTHROPIC_API_KEY in environment
 */

import { ContextPilot } from './src/engine/live-index.js';
import { getFormatHandler, type InterceptConfig } from './src/engine/extract.js';
import { injectCacheControl } from './src/engine/cache-control.js';
import { dedupChatCompletions } from './src/engine/dedup.js';

const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
if (!ANTHROPIC_API_KEY) {
  console.error('Error: ANTHROPIC_API_KEY not set');
  process.exit(1);
}

// Simulated system prompt with multiple documents (like Claude Code's context)
const systemPromptWithDocs = `You are a helpful coding assistant.

<documents>
<document>
# File: src/index.ts
export function main() {
  console.log("Hello world");
  const result = processData(getData());
  return result;
}

function getData() {
  return { items: [1, 2, 3, 4, 5] };
}

function processData(data: { items: number[] }) {
  return data.items.map(x => x * 2);
}
</document>
<document>
# File: src/utils.ts
export function formatOutput(data: number[]): string {
  return data.join(', ');
}

export function validateInput(input: unknown): boolean {
  return Array.isArray(input) && input.every(x => typeof x === 'number');
}

export function calculateSum(numbers: number[]): number {
  return numbers.reduce((a, b) => a + b, 0);
}
</document>
<document>
# File: README.md
# My Project

This is a sample project demonstrating the ContextPilot optimization.

## Installation
npm install

## Usage
npm start

## Features
- Data processing
- Input validation
- Output formatting
</document>
</documents>

Answer questions about the code above.`;

// Build Anthropic Messages API request body
const requestBody = {
  model: 'claude-sonnet-4-6',
  max_tokens: 256,
  system: systemPromptWithDocs,
  messages: [
    {
      role: 'user',
      content: 'What does the main function do? Be brief.'
    }
  ]
};

async function runTest() {
  console.log('=== ContextPilot E2E Test ===\n');

  // 1. Initialize engine
  const engine = new ContextPilot(0.001, false, 'average');
  console.log('1. Engine initialized');

  // 2. Extract documents
  const interceptConfig: InterceptConfig = {
    enabled: true,
    mode: 'auto',
    tag: 'document',
    separator: '---',
    alpha: 0.001,
    linkageMethod: 'average',
    scope: 'all'
  };

  const body = structuredClone(requestBody);
  const handler = getFormatHandler('anthropic_messages');
  const multi = handler.extractAll(body, interceptConfig);

  console.log(`2. Extracted ${multi.totalDocuments} documents from system prompt`);

  // 3. Reorder documents
  if (multi.systemExtraction) {
    const [extraction, sysIdx] = multi.systemExtraction;
    console.log(`   Original order: ${extraction.documents.map((_, i) => i).join(', ')}`);

    if (extraction.documents.length >= 2) {
      const [reordered] = engine.reorder(extraction.documents);
      if (Array.isArray(reordered) && Array.isArray(reordered[0])) {
        const reorderedDocs = reordered[0] as string[];
        handler.reconstructSystem(body, extraction, reorderedDocs, sysIdx);
        console.log('3. Documents reordered for prefix cache optimization');
      }
    }
  }

  // 4. Inject cache control
  const optimizedBody = injectCacheControl(body, 'anthropic');
  console.log('4. Cache control markers injected');

  // 5. Show optimization summary
  const originalLen = JSON.stringify(requestBody).length;
  const optimizedLen = JSON.stringify(optimizedBody).length;
  console.log(`\n=== Optimization Summary ===`);
  console.log(`Original request size: ${originalLen} chars`);
  console.log(`Optimized request size: ${optimizedLen} chars`);
  console.log(`Cache control added: ${JSON.stringify(optimizedBody).includes('cache_control')}`);

  // 6. Make real API call
  console.log('\n=== Making API Call ===');
  console.log('Calling Anthropic API with optimized request...\n');

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
      'anthropic-beta': 'prompt-caching-2024-07-31'
    },
    body: JSON.stringify(optimizedBody)
  });

  const result = await response.json();

  if (result.error) {
    console.error('API Error:', result.error);
    process.exit(1);
  }

  console.log('=== Response ===');
  console.log('Model:', result.model);
  console.log('Stop reason:', result.stop_reason);
  console.log('\nAssistant:', result.content?.[0]?.text || '(no text)');

  console.log('\n=== Usage ===');
  console.log('Input tokens:', result.usage?.input_tokens);
  console.log('Output tokens:', result.usage?.output_tokens);
  if (result.usage?.cache_creation_input_tokens) {
    console.log('Cache creation tokens:', result.usage.cache_creation_input_tokens);
  }
  if (result.usage?.cache_read_input_tokens) {
    console.log('Cache read tokens:', result.usage.cache_read_input_tokens);
  }

  console.log('\n=== Engine Stats ===');
  const stats = engine.getStats();
  console.log('Nodes:', stats.num_nodes);
  console.log('Active nodes:', stats.active_nodes);
  console.log('Total insertions:', stats.total_insertions);

  console.log('\n✓ E2E test complete');
}

runTest().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
