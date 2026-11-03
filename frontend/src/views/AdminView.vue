<script setup>
import { ref } from 'vue'
import DocumentsPanel from '../components/admin/DocumentsPanel.vue'
import SessionsPanel from '../components/admin/SessionsPanel.vue'
import FeedbackPanel from '../components/admin/FeedbackPanel.vue'

const tab = ref('docs')
</script>

<template>
  <div class="admin">
    <div class="tabs">
      <button :class="{ on: tab === 'docs' }" @click="tab = 'docs'">知识库</button>
      <button :class="{ on: tab === 'sessions' }" @click="tab = 'sessions'">会话复盘</button>
      <button :class="{ on: tab === 'feedback' }" @click="tab = 'feedback'">反馈看板</button>
    </div>

    <DocumentsPanel v-if="tab === 'docs'" />
    <SessionsPanel v-else-if="tab === 'sessions'" />
    <FeedbackPanel v-else />
  </div>
</template>
