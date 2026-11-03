import { request } from './http'
import { streamChat } from './sse'

export const api = {
  // 问诊
  chatStream: streamChat,

  // 反馈
  sendFeedback: (payload) =>
    request('/api/v1/feedback', { method: 'POST', body: payload }),
  feedbackStats: (days = 7) =>
    request(`/api/v1/feedback/stats?days=${encodeURIComponent(days)}`),

  // 会话复盘
  listSessions: (limit = 30, offset = 0) =>
    request(`/api/v1/sessions?limit=${limit}&offset=${offset}`),
  sessionDetail: (id) =>
    request(`/api/v1/sessions/${encodeURIComponent(id)}`),
  deleteSession: (id) =>
    request(`/api/v1/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' }),

  // 知识库
  listDocuments: (collection = 'local_kb') =>
    request(`/api/v1/documents?collection=${encodeURIComponent(collection)}`),
  uploadDocuments: (formData) =>
    request('/api/v1/documents/upload', { method: 'POST', body: formData }),
  previewChunks: (formData) =>
    request('/api/v1/documents/preview', { method: 'POST', body: formData }),
  rebuildDocument: (docId, department, collection = 'local_kb') =>
    request(
      `/api/v1/documents/${encodeURIComponent(docId)}/rebuild?department=${encodeURIComponent(
        department,
      )}&collection=${encodeURIComponent(collection)}`,
      { method: 'POST' },
    ),
  rebuildCollection: (collection = 'local_kb') =>
    request(`/api/v1/documents/rebuild?collection=${encodeURIComponent(collection)}`, {
      method: 'POST',
    }),
  deleteDocument: (docId, department, collection = 'local_kb') =>
    request(
      `/api/v1/documents/${encodeURIComponent(docId)}?department=${encodeURIComponent(
        department,
      )}&collection=${encodeURIComponent(collection)}`,
      { method: 'DELETE' },
    ),
  getTask: (taskId) =>
    request(`/api/v1/documents/tasks/${encodeURIComponent(taskId)}`),
}
