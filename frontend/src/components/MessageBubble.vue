<script setup>
import { computed } from 'vue'
import { focusSources } from '../stores/chat'
import FeedbackBar from './FeedbackBar.vue'

const props = defineProps({
  msg: { type: Object, required: true },
})

const isUser = computed(() => props.msg.role === 'user')
const sourceCount = computed(() => (props.msg.sources || []).length)

const OUTCOME_LABELS = {
  emergency_blocked: '急症拦截',
  content_blocked: '内容拦截',
  degraded: '检索降级',
  error: '系统异常',
  overload: '过载降级',
  timeout: '超时降级',
}

const outcomeLabel = computed(() => OUTCOME_LABELS[props.msg.outcome] || '')

const judge = computed(() => {
  const j = props.msg.judge
  if (!j || !j.layer || j.layer === 'ok') return null
  const map = {
    both: ['双校验通过', 'ok'],
    rule_only: ['规则通过', 'ok'],
    rule: ['规则未通过', 'danger'],
    judge: ['模型校验未通过', 'danger'],
    ablation_off: ['校验已关闭', 'warn'],
    emergency_blocked: ['急症拦截', 'warn'],
    content_blocked: ['内容拦截', 'warn'],
    degraded: ['降级输出', 'warn'],
    error: ['异常出口', 'danger'],
    overload: ['过载出口', 'warn'],
    timeout: ['超时出口', 'warn'],
  }
  const [label, tone] = map[j.layer] || [j.layer, '']
  const scores = j.scores
    ? Object.entries(j.scores)
        .map(([k, v]) => `${shortKey(k)} ${v}`)
        .join(' · ')
    : ''
  return { label, tone, scores }
})

const onSources = () => focusSources(props.msg)

function shortKey(key) {
  if (key.startsWith('fact')) return '事实'
  if (key.startsWith('logic')) return '逻辑'
  if (key.startsWith('answer') || key.startsWith('useful')) return '有用'
  return key
}
</script>

<template>
  <div class="msg" :class="isUser ? 'user' : 'assistant'">
    <div class="avatar">{{ isUser ? '我' : '医' }}</div>

    <div class="bubble-wrap">
      <div v-if="!isUser && msg.stageText" class="stage-line">
        <span class="spinner"></span>{{ msg.stageText }}
      </div>

      <div
        class="bubble"
        :class="{ error: !!msg.error }"
      >
        <template v-if="msg.content">{{ msg.content }}</template>
        <span v-else-if="msg.streaming" class="muted">正在生成…</span>
        <span v-else-if="!msg.error" class="muted">（空回答）</span>
        <span v-if="msg.streaming && msg.content" class="caret"></span>
        <div v-if="msg.error" style="margin-top: 6px">{{ msg.error }}</div>
      </div>

      <div class="msg-meta">
        <span v-if="outcomeLabel" class="badge warn">{{ outcomeLabel }}</span>
        <span v-if="judge" class="badge" :class="judge.tone" :title="judge.scores">
          {{ judge.label }}
        </span>
        <button
          v-if="!isUser && sourceCount"
          class="btn sm ghost"
          @click="onSources"
        >
          引用 {{ sourceCount }} 条
        </button>
        <span v-if="msg.risk && !isUser && !msg.streaming" class="badge">
          {{ msg.risk.level === 'high' ? '高风控链路' : '常规链路' }}
        </span>
      </div>

      <FeedbackBar
        v-if="!isUser && !msg.streaming && msg.content && !msg.error"
        :msg="msg"
      />
    </div>
  </div>
</template>
