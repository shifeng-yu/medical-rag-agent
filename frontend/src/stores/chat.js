import { reactive } from 'vue'
import { api } from '../api/client'

/**
 * 会话状态：一个 reactive 对象就够了，没必要上 Pinia（只有一个 store，
 * 也没有跨模块共享的复杂派生状态）。
 *
 * 消息形状（assistant）：
 *   { key, role, content, streaming, sources[], judge, messageId,
 *     feedback, stages[], risk, outcome, stageText, error }
 * messageId 是后端 chat_message.id，反馈必须靠它定位到"哪一次回答"。
 */

let seq = 0
const uid = () => `m${Date.now().toString(36)}${seq++}`

const SESSION_KEY = 'medrag.sessionId'

export const chat = reactive({
  sessionId: localStorage.getItem(SESSION_KEY) || '',
  messages: [],
  streaming: false,
  risk: null,
  activeMessage: null,
  error: '',
  abort: null,
})

export function newSession() {
  chat.sessionId = ''
  chat.messages = []
  chat.risk = null
  chat.error = ''
  chat.activeMessage = null
  localStorage.removeItem(SESSION_KEY)
}

export function focusSources(msg) {
  chat.activeMessage = msg
}

export function closeSources() {
  chat.activeMessage = null
}

export async function loadHistory() {
  if (!chat.sessionId) return
  try {
    const detail = await api.sessionDetail(chat.sessionId)
    chat.messages = (detail.messages || []).map((m) => ({
      key: uid(),
      role: m.role,
      content: m.content,
      streaming: false,
      sources: m.sources || [],
      judge: m.judge_layer
        ? { layer: m.judge_layer, scores: m.judge_scores || null }
        : null,
      messageId: m.id ? String(m.id) : '',
      feedback: m.feedback || null,
      stages: [],
      risk: null,
      outcome: m.outcome || '',
      stageText: '',
      error: '',
      restored: true,
    }))
    const lastAssistant = [...chat.messages].reverse().find((m) => m.role === 'assistant')
    chat.activeMessage = lastAssistant && lastAssistant.sources.length ? lastAssistant : null
  } catch (e) {
    // 会话被清理（404）→ 回到新会话，避免一直报错
    if (String(e.message).includes('404')) newSession()
  }
}

export async function ask(text) {
  const query = text.trim()
  if (!query || chat.streaming) return

  chat.error = ''
  const userMsg = { key: uid(), role: 'user', content: query }
  const botMsg = reactive({
    key: uid(),
    role: 'assistant',
    content: '',
    streaming: true,
    sources: [],
    judge: null,
    messageId: '',
    feedback: null,
    stages: [],
    risk: null,
    outcome: '',
    stageText: '正在建立连接…',
    error: '',
  })
  chat.messages.push(userMsg, botMsg)
  chat.streaming = true

  const controller = new AbortController()
  chat.abort = () => controller.abort()

  try {
    await api.chatStream({
      query,
      sessionId: chat.sessionId,
      signal: controller.signal,
      onEvent: (evt) => applyEvent(botMsg, evt),
    })
  } catch (e) {
    if (e.name !== 'AbortError') {
      botMsg.error = e.message || '请求失败'
      chat.error = botMsg.error
    } else {
      botMsg.error = '已停止生成'
    }
  } finally {
    botMsg.streaming = false
    botMsg.stageText = ''
    chat.streaming = false
    chat.abort = null
    if (!botMsg.content && !botMsg.error) botMsg.content = '（本次没有返回内容）'
  }
}

const NODE_LABELS = {
  safety: ['安全校验中…', '安全校验'],
  classify: ['问题分类中…', '问题分类'],
  retrieve: ['检索知识库…', '双源检索'],
  rerank: ['重排序中…', '重排'],
  generate: ['生成回答中…', '生成'],
  compliance: ['合规检查中…', '合规检查'],
}

function applyEvent(msg, evt) {
  const data = evt.data || {}
  switch (evt.event) {
    case 'meta':
      if (data.session_id) {
        chat.sessionId = data.session_id
        localStorage.setItem(SESSION_KEY, data.session_id)
      }
      break

    case 'risk':
      msg.risk = data
      chat.risk = data
      break

    case 'stage': {
      const [startLabel, doneLabel] = NODE_LABELS[data.node] || [data.node, data.node]
      if (data.status === 'start') {
        msg.stageText = startLabel
      } else if (data.status === 'retry') {
        // 不清空旧内容：重生成期间保留上一版，等 replace 到了再整段覆盖，
        // 避免屏幕先变空、再等十几秒才出现新答案
        msg.stageText = `校验未通过，第 ${data.attempt} 次重生成…`
      } else {
        msg.stageText = ''
        msg.stages.push({
          node: data.node,
          label: data.ms != null ? `${doneLabel} ${Math.round(data.ms)}ms` : doneLabel,
        })
      }
      break
    }

    case 'sources':
      msg.sources = data.sources || []
      if (msg.sources.length && !chat.activeMessage) chat.activeMessage = msg
      break

    case 'delta':
      msg.content += data.text || ''
      break

    case 'replace':
      // 校验不过重生成 / 合规拦截 / 最终拒答：整段覆盖，不能追加
      msg.content = data.text || ''
      break

    case 'judge':
      msg.judge = {
        layer: data.layer,
        scores: data.scores || null,
        passed: data.passed,
        reason: data.reason || '',
        retry: data.retry,
      }
      break

    case 'final':
      // final 是权威结果：用它对齐流式过程中可能被 replace 过的内容
      if (data.answer) msg.content = data.answer
      msg.messageId = data.message_id || ''
      msg.outcome = data.outcome || ''
      if (data.judge_result) {
        msg.judge = { ...(msg.judge || {}), ...data.judge_result }
      }
      if (Array.isArray(data.sources) && data.sources.length) msg.sources = data.sources
      if (data.session_id) {
        chat.sessionId = data.session_id
        localStorage.setItem(SESSION_KEY, data.session_id)
      }
      msg.streaming = false
      msg.stageText = ''
      break

    case 'error':
      msg.error = data.message || data.code || '流式异常'
      break

    default:
      break
  }
}

export async function submitFeedback(msg, rating, tags = [], comment = '') {
  if (!msg.messageId) return
  msg.feedback = await api.sendFeedback({
    message_id: Number(msg.messageId),
    rating,
    reason_tags: tags,
    comment,
  })
}
