<script setup>
import { computed, ref } from 'vue'
import { submitFeedback } from '../stores/chat'

const props = defineProps({
  msg: { type: Object, required: true },
})

const REASONS = ['答非所问', '引用不相关', '内容太笼统', '可能不安全']

const open = ref(false)
const tags = ref([])
const comment = ref('')
const busy = ref(false)
const error = ref('')

const rating = computed(() => props.msg.feedback?.rating || 0)
const disabled = computed(() => !props.msg.messageId || busy.value)

async function rate(value) {
  if (disabled.value) return
  if (value === -1) {
    open.value = !open.value
    return
  }
  await send(1, [], '')
}

async function send(value, reasonTags, text) {
  busy.value = true
  error.value = ''
  try {
    await submitFeedback(props.msg, value, reasonTags, text)
    open.value = false
    tags.value = []
    comment.value = ''
  } catch (e) {
    error.value = e.message || '反馈提交失败'
  } finally {
    busy.value = false
  }
}

function toggleTag(tag) {
  const idx = tags.value.indexOf(tag)
  if (idx >= 0) tags.value.splice(idx, 1)
  else tags.value.push(tag)
}

function submitNegative() {
  send(-1, [...tags.value], comment.value.trim())
}
</script>

<template>
  <div style="margin-top: 8px">
    <div class="row">
      <div class="feedback-bar">
        <button
          class="fb-btn"
          :class="{ on: rating === 1 }"
          :disabled="disabled"
          title="回答有用"
          @click="rate(1)"
        >
          👍 有用
        </button>
        <button
          class="fb-btn"
          :class="{ on: rating === -1 }"
          :disabled="disabled"
          title="回答没用"
          @click="rate(-1)"
        >
          👎 没用
        </button>
      </div>
      <span v-if="msg.feedback" class="small muted">
        已记录 · {{ msg.feedback.created_at || '' }}
      </span>
      <span v-else-if="!msg.messageId" class="small muted">未落库，无法反馈</span>
    </div>

    <div v-if="open" class="fb-panel">
      <div class="small muted">哪里不对？（可多选）</div>
      <div class="fb-tags">
        <button
          v-for="tag in REASONS"
          :key="tag"
          class="fb-tag"
          :class="{ on: tags.includes(tag) }"
          @click="toggleTag(tag)"
        >
          {{ tag }}
        </button>
      </div>
      <textarea
        v-model="comment"
        class="textarea"
        style="min-height: 60px"
        maxlength="1000"
        placeholder="补充说明（可选）"
      ></textarea>
      <div class="row" style="margin-top: 8px">
        <span class="small muted">{{ comment.length }}/1000</span>
        <div class="spacer"></div>
        <button class="btn sm ghost" @click="open = false">取消</button>
        <button class="btn sm danger" :disabled="busy" @click="submitNegative">
          提交
        </button>
      </div>
      <div v-if="error" class="small" style="color: var(--danger); margin-top: 6px">
        {{ error }}
      </div>
    </div>
  </div>
</template>
