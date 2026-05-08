import type { Plugin } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin"
import { dedupChatCompletions } from "../../openclaw-plugin/src/engine/dedup.js"
import { ContextPilot } from "../../openclaw-plugin/src/engine/live-index.js"
import * as crypto from "node:crypto"
import * as fs from "node:fs"
import * as path from "node:path"

const LOG_DIR = path.join(process.env.XDG_DATA_HOME || path.join(process.env.HOME || "/tmp", ".local/share"), "opencode/log")
const LOG_FILE = path.join(LOG_DIR, "contextpilot.log")
function log(msg: string) {
  try { fs.appendFileSync(LOG_FILE, `${new Date().toISOString()} ${msg}\n`) } catch {}
}

// ── Types mirroring OpenCode's message format ────────────────────────────

interface OpenCodeMessage {
  info: { id: string; role: string; sessionID: string; [k: string]: unknown }
  parts: OpenCodePart[]
}

type OpenCodePart = {
  id: string
  sessionID: string
  messageID: string
  type: string
  [k: string]: unknown
}

interface ToolPart extends OpenCodePart {
  type: "tool"
  callID: string
  tool: string
  state: {
    status: string
    input?: Record<string, unknown>
    output?: string
    title?: string
    metadata?: Record<string, unknown>
    time?: { start: number; end?: number }
    [k: string]: unknown
  }
}

interface TextPart extends OpenCodePart {
  type: "text"
  text: string
}

// ── Helpers ──────────────────────────────────────────────────────────────

function hashText(text: string): string {
  return crypto.createHash("sha256").update(text, "utf8").digest("hex").slice(0, 16)
}

function isToolPart(p: OpenCodePart): p is ToolPart {
  return p.type === "tool"
}

function isCompletedToolPart(p: OpenCodePart): p is ToolPart {
  return isToolPart(p) && (p as ToolPart).state?.status === "completed"
}

function getToolOutput(p: ToolPart): string {
  return typeof p.state?.output === "string" ? p.state.output : ""
}

// ── Convert OpenCode messages to OpenAI format for the pipeline ─────────

function toOpenAIMessages(messages: OpenCodeMessage[]): { role: string; content: string; tool_call_id?: string }[] {
  const result: { role: string; content: string; tool_call_id?: string }[] = []

  for (const msg of messages) {
    const role = msg.info.role

    // Collect text parts
    const textParts = msg.parts.filter((p): p is TextPart => p.type === "text")
    const toolParts = msg.parts.filter(isCompletedToolPart) as ToolPart[]

    if (textParts.length > 0) {
      result.push({
        role: role === "user" ? "user" : role === "assistant" ? "assistant" : "system",
        content: textParts.map((p) => (p as TextPart).text).join("\n"),
      })
    }

    // Tool results become role=tool messages
    for (const tp of toolParts) {
      result.push({
        role: "tool",
        content: getToolOutput(tp),
        tool_call_id: tp.callID || tp.id,
      })
    }
  }

  return result
}

// Map optimized OpenAI content back to OpenCode parts
function applyOptimizedContent(
  messages: OpenCodeMessage[],
  optimizedOpenAI: { role: string; content: string; tool_call_id?: string }[],
): void {
  // Build a lookup: tool_call_id → optimized content
  const optimizedToolContent = new Map<string, string>()
  for (const msg of optimizedOpenAI) {
    if (msg.role === "tool" && msg.tool_call_id) {
      optimizedToolContent.set(msg.tool_call_id, msg.content)
    }
  }

  // Apply back to OpenCode parts
  for (const msg of messages) {
    for (const part of msg.parts) {
      if (isCompletedToolPart(part)) {
        const tp = part as ToolPart
        const key = tp.callID || tp.id
        const optimized = optimizedToolContent.get(key)
        if (optimized !== undefined && optimized !== getToolOutput(tp)) {
          tp.state.output = optimized
        }
      }
    }
  }
}

// ── Session state ────────────────────────────────────────────────────────

class SessionState {
  private singleDocHashes = new Map<string, string>() // content_hash → part_id
  private optimizeCount = 0
  totalCharsSaved = 0
  totalDocsDeduped = 0

  private engine: ContextPilot | null = null
  private hasReorder = false

  constructor() {
    try {
      this.engine = new ContextPilot(0.001, false, "average")
      this.hasReorder = true
    } catch {
      this.hasReorder = false
    }
  }

  private reorderDocs(docs: string[]): string[] {
    if (!this.hasReorder || !this.engine || docs.length < 2) return docs
    try {
      const [reordered] = this.engine.reorder(docs)
      if (Array.isArray(reordered) && Array.isArray(reordered[0])) {
        const candidate = reordered[0] as string[]
        if (candidate.every((s) => typeof s === "string")) return candidate
      }
    } catch { /* graceful degradation */ }
    return docs
  }

  optimize(messages: OpenCodeMessage[]): void {
    this.optimizeCount++

    let charsSaved = 0

    // ── Single-doc cross-turn dedup ──────────────────────────────────
    for (const msg of messages) {
      for (const part of msg.parts) {
        if (!isCompletedToolPart(part)) continue
        const tp = part as ToolPart
        const output = getToolOutput(tp)
        if (output.length < 100) continue

        const contentHash = hashText(output)
        const partId = tp.callID || tp.id

        if (this.singleDocHashes.has(contentHash)) {
          const prevId = this.singleDocHashes.get(contentHash)!
          if (partId !== prevId) {
            // Check the previous part still exists
            const prevExists = messages.some((m) =>
              m.parts.some((p) => isCompletedToolPart(p) && ((p as ToolPart).callID || p.id) === prevId),
            )
            if (prevExists) {
              const saved = output.length
              tp.state.output = `[Duplicate — identical to previous tool result (${prevId}). Refer to the earlier result above.]`
              charsSaved += saved - tp.state.output.length
              this.totalDocsDeduped++
            }
          }
        } else {
          this.singleDocHashes.set(contentHash, partId)
        }
      }
    }

    // ── Block-level dedup via OpenAI conversion ─────────────────────
    const postDedup = toOpenAIMessages(messages)
    const systemContent = postDedup.find((m) => m.role === "system")?.content
    const body = { messages: postDedup }
    const dedupResult = dedupChatCompletions(body, systemContent)

    if (dedupResult.charsSaved > 0) {
      charsSaved += dedupResult.charsSaved
      applyOptimizedContent(messages, body.messages as any)
    }

    this.totalCharsSaved += charsSaved

    log(`[ContextPilot] Turn ${this.optimizeCount}: saved ${charsSaved} chars (~${Math.round(charsSaved / 4)} tokens) | docs deduped: ${this.totalDocsDeduped} | tracked: ${this.singleDocHashes.size} | cumulative: ${this.totalCharsSaved} chars (~${Math.round(this.totalCharsSaved / 4)} tokens)`)
  }

  getStats() {
    return {
      turns: this.optimizeCount,
      totalCharsSaved: this.totalCharsSaved,
      estimatedTokensSaved: Math.round(this.totalCharsSaved / 4),
      docsDeduped: this.totalDocsDeduped,
      trackedHashes: this.singleDocHashes.size,
      reorderAvailable: this.hasReorder,
    }
  }
}

// ── Plugin export ────────────────────────────────────────────────────────

export const ContextPilotPlugin: Plugin = async () => {
  const state = new SessionState()
  log("[ContextPilot] Plugin loaded successfully")

  return {
    "experimental.chat.messages.transform": async (_input, output) => {
      try {
        const msgs = output.messages as unknown as OpenCodeMessage[]
        log(`[ContextPilot] Transform called — ${msgs.length} messages, ${msgs.reduce((n, m) => n + m.parts.length, 0)} parts`)
        state.optimize(msgs)
      } catch (e) {
        log(`[ContextPilot] Transform error: ${e}`)
      }
    },
    tool: {
      contextpilot_status: tool({
        description: "Show ContextPilot cumulative token savings and dedup statistics",
        args: {},
        async execute() {
          const stats = state.getStats()
          return [
            "ContextPilot Status:",
            `  Turns optimized: ${stats.turns}`,
            `  Chars saved: ${stats.totalCharsSaved.toLocaleString()}`,
            `  Tokens saved: ~${stats.estimatedTokensSaved.toLocaleString()}`,
            `  Docs deduped: ${stats.docsDeduped}`,
            `  Tracked hashes: ${stats.trackedHashes}`,
            `  Reorder: ${stats.reorderAvailable ? "active" : "dedup-only"}`,
          ].join("\n")
        },
      }),
    },
  }
}

export default {
  id: "contextpilot",
  server: ContextPilotPlugin,
}
