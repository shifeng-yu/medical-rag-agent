<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  msg: { type: Object, required: true },
})
const emit = defineEmits(['close'])

const expanded = ref({})

const sources = computed(() => props.msg.sources || [])
const judgeScores = computed(() => {
  const scores = props.msg.judge?.scores
  if (!scores) return []
  return Object.entries(scores).map(([key, value]) => ({
    key: label(key),
    value,
  }))
})

function label(key) {
  if (key.startsWith('fact')) return '事实一致性'
  if (key.startsWith('logic')) return '逻辑连贯'
  if (key.startsWith('answer') || key.startsWith('useful')) return '回答有用性'
  return key
}

function toggle(index) {
  expanded.value = { ...expanded.value, [index]: !expanded.value[index] }
}

function sourceName(source) {
  return source === 'pubmed' ? 'PubMed 文献' : '本地知识库'
}
</script>

<template>
  <aside class="sources-drawer">
    <div class="drawer-head">
      <div>
        <div style="font-weight: 600">引用溯源</div>
        <div class="small muted">命中 {{ sources.length }} 条 · 点开看原文片段</div>
      </div>
      <div class="spacer"></div>
      <button class="btn sm ghost" title="收起" @click="emit('close')">✕</button>
    </div>

    <div class="drawer-body">
      <div v-if="!sources.length" class="empty">
        本次回答没有命中知识库片段。
      </div>

      <article v-for="(s, i) in sources" :key="s.chunk_id || i" class="source-card">
        <div class="source-title">{{ s.title || '（无标题片段）' }}</div>

        <div class="source-meta">
          <span class="badge" :class="s.source === 'pubmed' ? 'warn' : 'brand'">
            {{ sourceName(s.source) }}
          </span>
          <span v-if="s.department" class="badge">{{ s.department }}</span>
          <span v-if="s.authority_label" class="badge ok">{{ s.authority_label }}</span>
          <span class="badge mono">score {{ Number(s.score || 0).toFixed(3) }}</span>
        </div>

        <div class="snippet" :class="{ open: expanded[i] }">{{ s.snippet || '（无原文）' }}</div>

        <div class="row" style="margin-top: 6px">
          <button class="btn sm ghost snippet-toggle" @click="toggle(i)">
            {{ expanded[i] ? '收起' : '展开全文' }}
          </button>
          <div class="spacer"></div>
          <span class="badge mono" :title="`doc_id=${s.doc_id} chunk_id=${s.chunk_id}`">
            {{ (s.chunk_id || '--------').slice(0, 8) }}
          </span>
        </div>
      </article>

      <div v-if="judgeScores.length" class="source-card">
        <div class="source-title">LLM-Judge 打分</div>
        <div class="row wrap" style="margin-top: 4px">
          <span v-for="item in judgeScores" :key="item.key" class="badge">
            {{ item.key }} {{ item.value }}
          </span>
        </div>
        <div v-if="msg.judge?.retry" class="small muted" style="margin-top: 6px">
          重试 {{ msg.judge.retry }} 次
        </div>
      </div>
    </div>
  </aside>
</template>
