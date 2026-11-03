<script setup>
import { onMounted, ref } from 'vue'
import { request } from './api/http'

const online = ref(false)
const version = ref('')

onMounted(async () => {
  try {
    const health = await request('/api/v1/health')
    online.value = health.status === 'ok'
    version.value = health.version || ''
  } catch {
    online.value = false
  }
})
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand-mark">
        <span class="brand-dot">医</span>
        <span>全科医疗问诊 · RAG 控制台</span>
      </div>

      <nav class="nav">
        <router-link to="/" exact-active-class="active">问诊</router-link>
        <router-link to="/admin" active-class="active">管理台</router-link>
      </nav>

      <div class="spacer"></div>

      <div class="status" :class="{ off: !online }">
        <i></i>{{ online ? '服务在线' : '服务未就绪' }}
      </div>
      <span v-if="version" class="badge mono">v{{ version }}</span>
    </header>

    <main class="page">
      <router-view />
    </main>
  </div>
</template>
