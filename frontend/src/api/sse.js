/**
 * SSE 客户端：fetch + ReadableStream 手写帧解析。
 *
 * 为什么不用 EventSource：
 *   - 原生 EventSource 只支持 GET，塞不进 {query, session_id} 请求体；
 *   - EventSource 断线会自动重连，问诊场景重连会重复提问，反而危险。
 *
 * 帧格式（服务端 src/core/stream_protocol.py 唯一来源）：
 *   event: delta\n
 *   data: {"text":"..."}\n
 *   \n
 * 心跳是 `: ping\n\n` 注释帧，解析时直接忽略。
 */

export async function streamChat({ query, sessionId, onEvent, signal }) {
  const resp = await fetch('/api/v1/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ query, session_id: sessionId || null }),
    signal,
  })

  if (!resp.ok) {
    let detail = ''
    try {
      const data = await resp.json()
      detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
    } catch {
      detail = await resp.text().catch(() => '')
    }
    throw new Error(detail || `HTTP ${resp.status}`)
  }
  if (!resp.body) throw new Error('当前浏览器不支持流式响应（ReadableStream）')

  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const flush = (final = false) => {
    // 兼容代理把 \n 改写成 \r\n 的情况
    buffer = buffer.replace(/\r\n/g, '\n')
    let idx
    while ((idx = buffer.indexOf('\n\n')) >= 0) {
      const raw = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      const event = parseFrame(raw)
      if (event) onEvent(event)
    }
    if (final && buffer.trim()) {
      const event = parseFrame(buffer)
      if (event) onEvent(event)
      buffer = ''
    }
  }

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    flush()
  }
  buffer += decoder.decode()
  flush(true)
}

function parseFrame(raw) {
  if (!raw) return null
  const lines = raw.split('\n')
  // 整帧都是注释（心跳）
  if (lines.every((line) => !line || line.startsWith(':'))) return null

  let event = 'message'
  const dataLines = []
  for (const line of lines) {
    if (!line || line.startsWith(':')) continue
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''))
  }
  if (!dataLines.length) return null

  const text = dataLines.join('\n')
  try {
    return { event, data: JSON.parse(text) }
  } catch {
    return { event, data: { raw: text } }
  }
}
