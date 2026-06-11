import { describe, expect, it, vi } from "vitest"

// Mock the external dependencies before importing the plugin
vi.mock("../../openclaw-plugin/src/engine/dedup.js", () => ({
  dedupChatCompletions: vi.fn(() => ({ charsSaved: 0, blocksDeduped: 0, blocksTotal: 0, systemBlocksMatched: 0 })),
}))

vi.mock("../../openclaw-plugin/src/engine/live-index.js", () => ({
  ContextPilot: vi.fn(() => { throw new Error("no reorder in test") }),
}))

vi.mock("node:fs", () => ({ appendFileSync: vi.fn() }))


import pluginDefault, { ContextPilotPlugin } from "./index.js"
import { dedupChatCompletions } from "../../openclaw-plugin/src/engine/dedup.js"

// ── Helpers ─────────────────────────────────────────────────────────────

interface OpenCodeMessage {
  info: { id: string; role: string; sessionID: string }
  parts: Array<{
    id: string; sessionID: string; messageID: string; type: string;
    callID?: string; tool?: string;
    state?: { status: string; output?: string };
    text?: string;
  }>
}

function makeToolMessage(id: string, parts: Array<{ partId: string; callID: string; tool: string; output: string }>): OpenCodeMessage {
  return {
    info: { id, role: "assistant", sessionID: "s1" },
    parts: parts.map((p) => ({
      id: p.partId,
      sessionID: "s1",
      messageID: id,
      type: "tool",
      callID: p.callID,
      tool: p.tool,
      state: { status: "completed", output: p.output },
    })),
  }
}

function makeTextMessage(id: string, role: string, text: string): OpenCodeMessage {
  return {
    info: { id, role, sessionID: "s1" },
    parts: [{ id: `${id}-p1`, sessionID: "s1", messageID: id, type: "text", text }],
  }
}

const LONG_OUTPUT = "x".repeat(200)

// ── Tests ───────────────────────────────────────────────────────────────

describe("plugin export format", () => {
  it("default export has id 'contextpilot' and server function", () => {
    expect(pluginDefault.id).toBe("contextpilot")
    expect(typeof pluginDefault.server).toBe("function")
  })
})

describe("plugin initialization", () => {
  it("server() returns hooks with transform and contextpilot_status tool", async () => {
    const hooks = await ContextPilotPlugin()
    expect(hooks["experimental.chat.messages.transform"]).toBeDefined()
    expect(typeof hooks["experimental.chat.messages.transform"]).toBe("function")
    expect(hooks.tool).toBeDefined()
    expect(hooks.tool!.contextpilot_status).toBeDefined()
  })
})

describe("single-doc cross-turn dedup", () => {
  it("replaces duplicate tool output with a hint on second occurrence", async () => {
    const hooks = await ContextPilotPlugin()
    const transform = hooks["experimental.chat.messages.transform"]!

    const msg1 = makeToolMessage("m1", [{ partId: "p1", callID: "c1", tool: "read_file", output: LONG_OUTPUT }])
    const msg2 = makeToolMessage("m2", [{ partId: "p2", callID: "c2", tool: "read_file", output: LONG_OUTPUT }])
    const messages = [msg1, msg2] as any

    await transform({} as any, { messages })

    expect(msg1.parts[0]!.state!.output).toBe(LONG_OUTPUT)
    expect(msg2.parts[0]!.state!.output).toContain("Duplicate")
    expect(msg2.parts[0]!.state!.output).toContain("c1")
  })
})

describe("no dedup for short outputs", () => {
  it("outputs under 100 chars are not deduped", async () => {
    const hooks = await ContextPilotPlugin()
    const transform = hooks["experimental.chat.messages.transform"]!

    const shortOutput = "short"
    const msg1 = makeToolMessage("m1", [{ partId: "p1", callID: "c1", tool: "read_file", output: shortOutput }])
    const msg2 = makeToolMessage("m2", [{ partId: "p2", callID: "c2", tool: "read_file", output: shortOutput }])
    const messages = [msg1, msg2] as any

    await transform({} as any, { messages })

    expect(msg1.parts[0]!.state!.output).toBe(shortOutput)
    expect(msg2.parts[0]!.state!.output).toBe(shortOutput)
  })
})

describe("no dedup on first occurrence", () => {
  it("first time seeing content, output is unchanged", async () => {
    const hooks = await ContextPilotPlugin()
    const transform = hooks["experimental.chat.messages.transform"]!

    const msg = makeToolMessage("m1", [{ partId: "p1", callID: "c1", tool: "read_file", output: LONG_OUTPUT }])
    const messages = [msg] as any

    await transform({} as any, { messages })

    expect(msg.parts[0]!.state!.output).toBe(LONG_OUTPUT)
  })
})

describe("block-level dedup", () => {
  it("fires dedupChatCompletions and saves chars when blocks are shared", async () => {
    const mockDedup = vi.mocked(dedupChatCompletions)
    mockDedup.mockReturnValueOnce({ charsSaved: 500, blocksDeduped: 2, blocksTotal: 4, systemBlocksMatched: 0 } as any)

    const hooks = await ContextPilotPlugin()
    const transform = hooks["experimental.chat.messages.transform"]!

    const msg1 = makeToolMessage("m1", [{ partId: "p1", callID: "c1", tool: "read_file", output: "unique-a-" + "z".repeat(200) }])
    const msg2 = makeToolMessage("m2", [{ partId: "p2", callID: "c2", tool: "read_file", output: "unique-b-" + "z".repeat(200) }])
    const messages = [msg1, msg2] as any

    await transform({} as any, { messages })

    expect(mockDedup).toHaveBeenCalled()
  })
})

describe("stats tracking", () => {
  it("contextpilot_status returns correct cumulative stats after optimization", async () => {
    const mockDedup = vi.mocked(dedupChatCompletions)
    mockDedup.mockReturnValue({ charsSaved: 0, blocksDeduped: 0, blocksTotal: 0, systemBlocksMatched: 0 } as any)

    const hooks = await ContextPilotPlugin()
    const transform = hooks["experimental.chat.messages.transform"]!
    const statusTool = hooks.tool!.contextpilot_status

    // Run a transform with a duplicate to accumulate stats
    const msg1 = makeToolMessage("m1", [{ partId: "p1", callID: "c1", tool: "read_file", output: LONG_OUTPUT }])
    const msg2 = makeToolMessage("m2", [{ partId: "p2", callID: "c2", tool: "read_file", output: LONG_OUTPUT }])
    const messages = [msg1, msg2] as any

    await transform({} as any, { messages })

    const result = await (statusTool as any).execute({})
    expect(result).toContain("Turns optimized: 1")
    expect(result).toContain("Docs deduped: 1")
    expect(result).toContain("Tracked hashes: 1")
    expect(result).toContain("Reorder: dedup-only")
  })
})

describe("transform hook error handling", () => {
  it("bad input does not crash the transform", async () => {
    const hooks = await ContextPilotPlugin()
    const transform = hooks["experimental.chat.messages.transform"]!

    // null messages
    await expect(transform({} as any, { messages: null } as any)).resolves.toBeUndefined()

    // messages with missing parts
    await expect(transform({} as any, { messages: [{ info: { id: "x", role: "user", sessionID: "s" } }] } as any)).resolves.toBeUndefined()

    // completely invalid input
    await expect(transform({} as any, {} as any)).resolves.toBeUndefined()
  })
})
