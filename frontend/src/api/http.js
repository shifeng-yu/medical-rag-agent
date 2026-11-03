// 统一 HTTP 封装：错误信息优先取后端的 detail（FastAPI 的 400/404 都带人话）
export async function request(url, { method = 'GET', body, headers, signal } = {}) {
  const init = { method, signal, headers: { ...(headers || {}) } }

  if (body instanceof FormData) {
    init.body = body // 让浏览器自己带 boundary，不能手写 Content-Type
  } else if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json'
    init.body = JSON.stringify(body)
  }

  const resp = await fetch(url, init)
  if (!resp.ok) {
    let detail = ''
    try {
      const data = await resp.json()
      detail = data.detail ?? data.message ?? ''
      if (Array.isArray(detail)) {
        detail = detail.map((d) => d.msg || JSON.stringify(d)).join('；')
      }
    } catch {
      detail = await resp.text().catch(() => '')
    }
    throw new Error(detail || `HTTP ${resp.status}`)
  }
  if (resp.status === 204) return null
  const contentType = resp.headers.get('content-type') || ''
  return contentType.includes('application/json') ? resp.json() : resp.text()
}
