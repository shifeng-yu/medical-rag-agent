<script setup>
import { nextTick, onMounted, ref, watch } from 'vue'
import { ask, chat, closeSources, loadHistory, newSession } from '../stores/chat'
import MessageBubble from '../components/MessageBubble.vue'
import SourceDrawer from '../components/SourceDrawer.vue'

const draft = ref('')
const scroller = ref(null)
const inputEl = ref(null)

const SAMPLES = [
  '最近总是失眠，白天没精神，怎么办？',
  '冠心病人平时运动要注意什么？',
  '体检发现血脂偏高，饮食上怎么调整？',
  '布洛芬一次吃几片，一天最多几次？',
]

onMounted(async () => {
  await loadHistory()
  scrollToBottom()
  inputEl.value?.focus()
})

// 流式阶段正文与进度都会变，用长度做轻量依赖，避免 deep watch 整棵树
watch(
  () => chat.messages.map((m) => `${m.content.length}:${m.stageText || ''}`).join('|'),
  () => scrollToBottom(),
)

function scrollToBottom() {
  nextTick(() => {
    const el = scroller.value
    if (el) el.scrollTop = el.scrollHeight
  })
}

async function send() {
  const text = draft.value.trim()
  if (!text || chat.streaming) return
  draft.value = ''
  await ask(text)
}

// 中文输入法：回车用于选词时不能发送（isComposing / 229）
function onEnter(event) {
  if (event.isComposing || event.keyCode === 229) return
  if (event.shiftKey) return
  event.preventDefault()
  send()
}

function autoGrow(event) {
  const el = event.target
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, 160)}px`
}

function useSample(text) {
  draft.value = text
  inputEl.value?.focus()
}

function stop() {
  chat.abort?.()
}

function reset() {
  if (chat.streaming) return
  newSession()
  inputEl.value?.focus()
}

function shortId(id) {
  return (id || '').slice(0, 8)
}
</script>

<template>
  <div class="chat-layout" :class="{ 'no-drawer': !chat.activeMessage }">
    <section class="chat-main">
      <div class="chat-head">
        <div>
          <div class="chat-title">问诊对话</div>
          <div class="chat-sub">
            {{ chat.sessionId ? `会话 ${shortId(chat.sessionId)}` : '新会话' }}
            · SSE 逐字输出 · 右侧可查命中原文
          </div>
        </div>
        <div class="spacer"></div>
        <span v-if="chat.risk" class="badge" :class="chat.risk.level === 'high' ? 'warn' : 'ok'">
          {{ chat.risk.level === 'high' ? '高风控 · 先校验后回放' : '常规 · 边生成边推' }}
        </span>
        <button class="btn sm ghost" :disabled="chat.streaming" @click="reset">新建会话</button>
      </div>

      <div ref="scroller" class="messages">
        <div class="messages-inner">
          <div v-if="!chat.messages.length" class="welcome">
            <h1>把症状说清楚一点，我来查资料</h1>
            <p>
              系统只做医学科普参考，不提供诊断、处方与剂量建议；急症会直接提示就医。
            </p>
            <div class="samples">
              <button v-for="s in SAMPLES" :key="s" class="sample-chip" @click="useSample(s)">
                {{ s }}
              </button>
            </div>
          </div>

          <MessageBubble v-for="m in chat.messages" :key="m.key" :msg="m" />
        </div>
      </div>

      <div class="composer">
        <div class="composer-inner">
          <div class="composer-box">
            <textarea
              ref="inputEl"
              v-model="draft"
              rows="1"
              :disabled="chat.streaming"
              placeholder="描述症状、持续时间、伴随表现，例如「胸闷 2 小时，伴出汗」"
              @input="autoGrow"
              @keydown.enter="onEnter"
            ></textarea>
            <button v-if="chat.streaming" class="btn danger" @click="stop">停止</button>
            <button v-else class="btn primary" :disabled="!draft.trim()" @click="send">
              发送
            </button>
          </div>
          <div class="composer-foot">
            <span>Enter 发送 · Shift+Enter 换行</span>
            <div class="spacer"></div>
            <span>本产品仅提供医学科普参考，不构成诊疗建议</span>
          </div>
        </div>
      </div>
    </section>

    <SourceDrawer
      v-if="chat.activeMessage"
      :msg="chat.activeMessage"
      @close="closeSources"
    />
  </div>
</template>
