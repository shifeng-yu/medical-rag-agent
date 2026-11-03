import { afterEach, describe, expect, it, vi } from 'vitest'
import { streamChat } from '../src/api/sse'

/**
 * 用最小可读流替身喂数据：不依赖 Response/ReadableStream 在 jsdom 下是否存在，
 * 只实现 streamChat 真正用到的 getReader().read()。
 */
function readerFrom(text, chunkSize = 7) {
  const bytes = new TextEncoder().encode(text)
  let offset = 0
  return {
    getReader() {
      return {
        async read() {
          if (offset >= bytes.length) return { done: true, value: undefined }
          const value = bytes.slice(offset, offset + chunkSize)
          offset += chunkSize
          return { done: false, value }
        },
      }
    },
  }
}

function stubFetch(payload) {
  vi.stubGlobal('fetch', vi.fn(async () => payload))
}

const FRAMES = [
  'event: meta\ndata: {"session_id":"s1"}\n\n',
  ': ping\n\n',
  'event: risk\ndata: {"level":"low","mode":"token"}\n\n',
  'event: delta\ndata: {"text":"你"}\n\n',
  'event: delta\ndata: {"text":"好"}\n\n',
  'event: replace\ndata: {"text":"整段替换","reason":"judge_retry"}\n\n',
  'event: final\ndata: {"answer":"整段替换","session_id":"s1","message_id":"7","outcome":"ok","sources":[]}\n\n',
]

afterEach(() => vi.unstubAllGlobals())

describe('SSE 帧解析', () => {
  it('能处理 CRLF、任意切片、心跳、replace 与 final', async () => {
    // 一帧被切成 7 字节的多段，并模拟代理把 \n 改写成 \r\n
    const body = FRAMES.join('').replace(/\n/g, '\r\n')
    stubFetch({ ok: true, body: readerFrom(body), status: 200 })

    const events = []
    await streamChat({ query: 'q', sessionId: null, onEvent: (e) => events.push(e) })

    expect(events.map((e) => e.event)).toEqual([
      'meta', 'risk', 'delta', 'delta', 'replace', 'final',
    ])
    expect(
      events.filter((e) => e.event === 'delta').map((e) => e.data.text).join(''),
    ).toBe('你好')
    expect(events.at(-1).data.message_id).toBe('7')
  })

  it('非 200 时抛出后端 detail', async () => {
    stubFetch({
      ok: false,
      status: 429,
      json: async () => ({ detail: '限流：10 次/分钟' }),
    })

    await expect(
      streamChat({ query: 'q', onEvent: () => {} }),
    ).rejects.toThrow('限流：10 次/分钟')
  })
})
