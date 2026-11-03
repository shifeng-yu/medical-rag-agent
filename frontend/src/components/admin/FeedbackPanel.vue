<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { api } from '../../api/client'

const days = ref(7)
const stats = ref(null)
const error = ref('')
const loading = ref(false)

onMounted(load)
watch(days, load)

async function load() {
  loading.value = true
  error.value = ''
  try {
    stats.value = await api.feedbackStats(days.value)
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

const maxReason = computed(() => {
  const rows = stats.value?.top_reasons || []
  return rows.reduce((max, r) => Math.max(max, r.count), 0) || 1
})

const maxDay = computed(() => {
  const rows = stats.value?.by_day || []
  return rows.reduce((max, r) => Math.max(max, r.positive + r.negative), 0) || 1
})

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`
}
</script>

<template>
  <div class="trace" style="gap: 16px">
    <div class="row">
      <div class="tabs" style="margin: 0">
        <button :class="{ on: days === 7 }" @click="days = 7">近 7 天</button>
        <button :class="{ on: days === 30 }" @click="days = 30">近 30 天</button>
      </div>
      <div class="spacer"></div>
      <button class="btn ghost" :disabled="loading" @click="load">刷新</button>
    </div>

    <div class="grid-3">
      <div class="stat">
        <div class="k">反馈总数</div>
        <div class="v">{{ stats?.total ?? 0 }}</div>
        <div class="s">同一条回答重复提交只算一次</div>
      </div>
      <div class="stat">
        <div class="k">好评率</div>
        <div class="v">{{ pct(stats?.positive_rate) }}</div>
        <div class="s">👍 {{ stats?.positive ?? 0 }} · 👎 {{ stats?.negative ?? 0 }}</div>
      </div>
      <div class="stat">
        <div class="k">负反馈</div>
        <div class="v">{{ stats?.negative ?? 0 }}</div>
        <div class="s">按原因标签归因</div>
      </div>
    </div>

    <div class="grid-2" style="align-items: start">
      <div class="card">
        <div class="card-head"><h3>负反馈原因分布</h3></div>
        <div class="card-body">
          <div v-if="!(stats?.top_reasons || []).length" class="empty">
            窗口内还没有带标签的负反馈
          </div>
          <div v-for="row in stats?.top_reasons || []" :key="row.tag" class="bar-row">
            <div class="name">{{ row.tag }}</div>
            <div class="bar-track">
              <i :style="{ width: `${(row.count / maxReason) * 100}%` }"></i>
            </div>
            <div class="val">{{ row.count }}</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-head"><h3>按天趋势</h3></div>
        <div class="card-body">
          <div v-if="!(stats?.by_day || []).length" class="empty">窗口内暂无反馈</div>
          <div v-for="row in stats?.by_day || []" :key="row.date" class="bar-row">
            <div class="name mono">{{ row.date }}</div>
            <div class="bar-track">
              <i :style="{ width: `${((row.positive + row.negative) / maxDay) * 100}%` }"></i>
            </div>
            <div class="val">👍{{ row.positive }} 👎{{ row.negative }}</div>
          </div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <h3>怎么用这些数据</h3>
      </div>
      <div class="card-body small muted" style="line-height: 1.9">
        负反馈集中在「引用不相关」→ 先看该回答的 sources 与 chunk 命中，多半是检索或分块问题；
        集中在「答非所问」→ 看问题分类是否把本地库/PubMed 路由错；
        集中在「可能不安全」→ 检查该问题是否触发了高风控链路与 LLM-Judge。
        每条反馈都能通过会话回放定位到 message_id 与命中块，不需要翻日志。
      </div>
    </div>

    <div v-if="error" class="toast err">{{ error }}</div>
  </div>
</template>
