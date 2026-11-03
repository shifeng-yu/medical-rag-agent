<script setup>
import { computed, onMounted, ref } from 'vue'
import { api } from '../../api/client'

const sessions = ref([])
const total = ref(0)
const active = ref(null)
const detail = ref(null)
const busy = ref(false)
const error = ref('')

onMounted(load)

async function load() {
  error.value = ''
  try {
    const data = await api.listSessions(50, 0)
    sessions.value = data.items || []
    total.value = data.total || 0
  } catch (e) {
    error.value = e.message
  }
}

async function open(item) {
  active.value = item.session_id
  detail.value = null
  try {
    detail.value = await api.sessionDetail(item.session_id)
  } catch (e) {
    error.value = e.message
  }
}

async function remove(item) {
  if (!window.confirm(`删除会话 ${item.session_id}？消息、来源、反馈会一起清掉。`)) return
  busy.value = true
  try {
    await api.deleteSession(item.session_id)
    if (active.value === item.session_id) {
      active.value = null
      detail.value = null
    }
    await load()
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = false
  }
}

function outcomeTone(outcome) {
  if (!outcome || outcome === 'ok') return 'ok'
  if (outcome === 'degraded' || outcome === 'emergency_blocked' || outcome === 'content_blocked') {
    return 'warn'
  }
  return 'danger'
}

const assistantCount = computed(
  () => (detail.value?.messages || []).filter((m) => m.role === 'assistant').length,
)
</script>

<template>
  <div class="grid-2" style="align-items: start; grid-template-columns: 380px minmax(0, 1fr)">
    <div class="card">
      <div class="card-head">
        <h3>会话列表</h3>
        <span class="small muted">共 {{ total }} 个</span>
      </div>
      <table class="table">
        <thead>
          <tr>
            <th>会话</th>
            <th style="width: 70px">出口</th>
            <th style="width: 70px">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="item in sessions"
            :key="item.session_id"
            class="session-row"
            :class="{ on: active === item.session_id }"
            @click="open(item)"
          >
            <td>
              <div style="font-weight: 600">{{ item.title || '（未命名会话）' }}</div>
              <div class="small muted mono">{{ item.session_id.slice(0, 12) }}</div>
              <div class="small muted">
                {{ item.last_active_at }} · {{ item.turn_count }} 轮
              </div>
            </td>
            <td>
              <span class="badge" :class="outcomeTone(item.last_outcome)">
                {{ item.last_outcome || '—' }}
              </span>
            </td>
            <td>
              <button class="btn sm danger" :disabled="busy" @click.stop="remove(item)">
                删除
              </button>
            </td>
          </tr>
          <tr v-if="!sessions.length">
            <td colspan="3" class="empty">还没有会话记录</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-head">
        <h3>会话回放</h3>
        <span v-if="detail" class="small muted">
          {{ detail.messages.length }} 条消息 · {{ assistantCount }} 次回答
        </span>
      </div>
      <div class="card-body" style="max-height: 68vh; overflow-y: auto">
        <div v-if="!detail" class="empty">左侧选一个会话</div>
        <div v-else class="trace">
          <div class="row wrap" style="margin-bottom: 4px">
            <span class="badge mono">{{ detail.session_id }}</span>
            <span class="badge">{{ detail.last_outcome || '—' }}</span>
            <span class="badge">{{ detail.turn_count }} 轮</span>
            <span class="badge">{{ detail.created_at }}</span>
          </div>
          <div v-if="detail.summary" class="small muted">
            压缩摘要：{{ detail.summary }}
          </div>

          <div
            v-for="m in detail.messages"
            :key="m.id"
            class="trace-msg"
            :class="{ user: m.role === 'user' }"
          >
            <div class="trace-head">
              <span class="badge" :class="m.role === 'user' ? '' : 'brand'">
                {{ m.role === 'user' ? '用户' : '助手' }}
              </span>
              <span class="badge mono">#{{ m.id }}</span>
              <span v-if="m.outcome" class="badge" :class="outcomeTone(m.outcome)">
                {{ m.outcome }}
              </span>
              <span v-if="m.judge_layer" class="badge">{{ m.judge_layer }}</span>
              <span v-if="m.latency_ms" class="badge">{{ Math.round(m.latency_ms) }} ms</span>
              <span v-if="m.feedback" class="badge" :class="m.feedback.rating > 0 ? 'ok' : 'danger'">
                反馈 {{ m.feedback.rating > 0 ? '👍' : '👎' }}
                {{ (m.feedback.reason_tags || []).join('/') }}
              </span>
            </div>

            <div class="trace-content">{{ m.content }}</div>

            <div v-if="m.sources && m.sources.length" style="margin-top: 8px">
              <div class="small muted">命中 {{ m.sources.length }} 条来源</div>
              <div v-for="s in m.sources" :key="s.chunk_id" class="chunk">
                <div class="chunk-head">
                  <span class="badge">{{ s.source === 'pubmed' ? 'PubMed' : '本地库' }}</span>
                  <strong style="font-size: 12px">{{ s.title }}</strong>
                  <span class="badge mono">{{ Number(s.score || 0).toFixed(3) }}</span>
                </div>
                <div class="small">{{ (s.snippet || '').slice(0, 220) }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div v-if="error" class="toast err">{{ error }}</div>
  </div>
</template>
