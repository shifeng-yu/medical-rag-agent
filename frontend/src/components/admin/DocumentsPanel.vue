<script setup>
import { computed, onMounted, ref } from 'vue'
import { api } from '../../api/client'

const department = ref('神经内科')
const picked = ref([])
const dragOver = ref(false)
const docs = ref([])
const tasks = ref([])
const task = ref(null)
const busy = ref(false)
const error = ref('')
const toast = ref('')

const preview = ref(null)
const previewOpen = ref(false)
const fileInput = ref(null)

const COLLECTION = 'local_kb'

onMounted(refresh)

async function refresh() {
  try {
    const data = await api.listDocuments(COLLECTION)
    docs.value = data.files || []
    tasks.value = data.tasks || []
  } catch (e) {
    error.value = e.message
  }
}

function onPick(event) {
  picked.value = Array.from(event.target.files || [])
}

function onDrop(event) {
  dragOver.value = false
  picked.value = Array.from(event.dataTransfer.files || [])
}

function formData(maxChunks) {
  const fd = new FormData()
  for (const file of picked.value) fd.append('files', file)
  fd.append('department', department.value)
  fd.append('collection', COLLECTION)
  if (maxChunks != null) fd.append('max_chunks', String(maxChunks))
  return fd
}

async function upload() {
  if (!picked.value.length) return
  busy.value = true
  error.value = ''
  task.value = null
  try {
    const accepted = await api.uploadDocuments(formData())
    toast.value = `已受理 ${accepted.accepted_files} 个文件，任务 ${accepted.task_id}`
    picked.value = []
    if (fileInput.value) fileInput.value.value = ''
    await follow(accepted.task_id)
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = false
    setTimeout(() => (toast.value = ''), 4000)
  }
}

async function doPreview() {
  if (!picked.value.length) return
  busy.value = true
  error.value = ''
  try {
    preview.value = await api.previewChunks(formData(20))
    previewOpen.value = true
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = false
  }
}

function confirmUpload() {
  previewOpen.value = false
  upload()
}

async function follow(taskId) {
  for (let i = 0; i < 200; i += 1) {
    const current = await api.getTask(taskId)
    task.value = current
    if (current.status === 'done' || current.status === 'error') {
      await refresh()
      if (current.status === 'error') error.value = current.errors?.join('；') || '任务失败'
      return current
    }
    await new Promise((resolve) => setTimeout(resolve, 800))
  }
  return task.value
}

async function rebuildDoc(row) {
  busy.value = true
  error.value = ''
  try {
    const started = await api.rebuildDocument(row.doc_id, row.department, COLLECTION)
    task.value = started
    await follow(started.task_id)
    toast.value = `已重建 ${row.doc_id}`
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = false
    setTimeout(() => (toast.value = ''), 4000)
  }
}

async function rebuildAll() {
  if (!window.confirm('整库重建会逐文件重切重嵌，耗时较长，确认继续？')) return
  busy.value = true
  error.value = ''
  try {
    const started = await api.rebuildCollection(COLLECTION)
    task.value = started
    await follow(started.task_id)
    toast.value = '整库重建完成'
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = false
    setTimeout(() => (toast.value = ''), 4000)
  }
}

async function remove(row) {
  if (!window.confirm(`删除 ${row.doc_id}（向量 + 文件 + 元数据）？`)) return
  busy.value = true
  error.value = ''
  try {
    const res = await api.deleteDocument(row.doc_id, row.department, COLLECTION)
    toast.value = `已删除向量 ${res.deleted_vectors} 条 / 文件 ${res.deleted_files} 个`
    await refresh()
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = false
    setTimeout(() => (toast.value = ''), 4000)
  }
}

const summary = computed(() => {
  const indexed = docs.value.filter((d) => d.indexed).length
  const chunks = docs.value.reduce((sum, d) => sum + (d.chunks || 0), 0)
  return { total: docs.value.length, indexed, chunks }
})

function fmtSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

const progressText = computed(() => {
  if (!task.value) return ''
  const t = task.value
  if (t.status === 'processing') return `任务 ${t.task_id} 处理中…`
  if (t.status === 'done') return `任务 ${t.task_id} 完成：${t.chunks} 块，失败 ${t.failed} 个文件`
  return `任务 ${t.task_id} 失败`
})
</script>

<template>
  <div class="trace" style="gap: 16px">
    <div class="grid-3">
      <div class="stat">
        <div class="k">目录文档</div>
        <div class="v">{{ summary.total }}</div>
        <div class="s">local_kb · 磁盘为源真</div>
      </div>
      <div class="stat">
        <div class="k">已入库</div>
        <div class="v">{{ summary.indexed }}</div>
        <div class="s">kb_document 有记录</div>
      </div>
      <div class="stat">
        <div class="k">向量块总数</div>
        <div class="v">{{ summary.chunks }}</div>
        <div class="s">按文档累加</div>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <h3>上传与切分预览</h3>
        <span class="badge">{{ COLLECTION }}</span>
      </div>
      <div class="card-body">
        <div class="grid-2" style="align-items: start">
          <div class="field">
            <label>科室 / 主题（决定落盘目录与检索元数据）</label>
            <input v-model="department" class="input" maxlength="32" />
          </div>
          <div class="field">
            <label>选择文件（.md / .txt / .json，单批 ≤20 个，单个 ≤50MB）</label>
            <input
              ref="fileInput"
              type="file"
              class="input"
              multiple
              accept=".md,.txt,.json"
              @change="onPick"
            />
          </div>
        </div>

        <div
          class="dropzone"
          :class="{ on: dragOver }"
          style="margin-top: 12px"
          @dragover.prevent="dragOver = true"
          @dragleave.prevent="dragOver = false"
          @drop.prevent="onDrop"
        >
          拖拽文件到这里（或使用上方文件选择）
          <div v-if="picked.length" class="small" style="margin-top: 6px">
            已选 {{ picked.length }} 个：{{ picked.map((f) => f.name).join('、') }}
          </div>
        </div>

        <div class="row" style="margin-top: 12px">
          <button class="btn primary" :disabled="!picked.length || busy" @click="upload">
            上传并入库
          </button>
          <button class="btn" :disabled="!picked.length || busy" @click="doPreview">
            先预览切分（不落库）
          </button>
          <div class="spacer"></div>
          <button class="btn ghost" :disabled="busy" @click="refresh">刷新</button>
          <button class="btn danger" :disabled="busy" @click="rebuildAll">整库重建</button>
        </div>

        <div v-if="task" style="margin-top: 14px">
          <div class="progress">
            <i :style="{ width: task.status === 'done' ? '100%' : '55%' }"></i>
          </div>
          <div class="small muted" style="margin-top: 6px">{{ progressText }}</div>
        </div>

        <div v-if="error" class="small" style="color: var(--danger); margin-top: 10px">
          {{ error }}
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <h3>文档清单</h3>
        <span class="small muted">分块数来自入库记录，未入库显示 —</span>
      </div>
      <table class="table">
        <thead>
          <tr>
            <th>路径</th>
            <th style="width: 90px">科室</th>
            <th style="width: 80px">大小</th>
            <th style="width: 80px">分块</th>
            <th style="width: 110px">状态</th>
            <th style="width: 150px">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in docs" :key="row.path">
            <td>
              <div class="mono small">{{ row.path }}</div>
              <div class="small muted">{{ row.modified }}</div>
            </td>
            <td>{{ row.department }}</td>
            <td class="small">{{ fmtSize(row.size) }}</td>
            <td class="small">{{ row.indexed ? row.chunks : '—' }}</td>
            <td>
              <span class="badge" :class="row.indexed ? 'ok' : 'warn'">
                {{ row.indexed ? '已入库' : '未入库' }}
              </span>
            </td>
            <td class="actions">
              <button class="btn sm ghost" :disabled="busy" @click="rebuildDoc(row)">重建</button>
              <button class="btn sm danger" :disabled="busy" @click="remove(row)">删除</button>
            </td>
          </tr>
          <tr v-if="!docs.length">
            <td colspan="6" class="empty">目录为空，先上传一篇文档</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-head"><h3>近期任务</h3></div>
      <table class="table">
        <thead>
          <tr>
            <th style="width: 140px">任务</th>
            <th style="width: 120px">类型</th>
            <th style="width: 100px">执行器</th>
            <th style="width: 80px">文件</th>
            <th style="width: 80px">分块</th>
            <th style="width: 90px">状态</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="t in tasks" :key="t.task_id">
            <td class="mono small">{{ t.task_id }}</td>
            <td class="small">{{ t.kind }}</td>
            <td class="small">{{ t.executor }}</td>
            <td class="small">{{ t.files }}</td>
            <td class="small">{{ t.chunks }}</td>
            <td>
              <span
                class="badge"
                :class="t.status === 'done' ? 'ok' : t.status === 'error' ? 'danger' : 'warn'"
              >
                {{ t.status }}
              </span>
            </td>
          </tr>
          <tr v-if="!tasks.length">
            <td colspan="6" class="empty">暂无任务</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <div v-if="previewOpen && preview" class="modal-mask" @click.self="previewOpen = false">
    <div class="modal">
      <div class="modal-head">
        <div>
          <div style="font-weight: 600">切分预览 · dry-run</div>
          <div class="small muted">
            共 {{ preview.total_chunks }} 块 · 不落盘、不入库 · 参数 max 800 / min 100 / overlap 50
          </div>
        </div>
        <button class="btn sm ghost" @click="previewOpen = false">✕</button>
      </div>
      <div class="modal-body">
        <div v-for="file in preview.files" :key="file.name" style="margin-bottom: 18px">
          <div class="row" style="margin-bottom: 8px">
            <strong>{{ file.name }}</strong>
            <span class="badge">{{ file.department }}</span>
            <span class="badge">{{ file.chunk_count }} 块</span>
            <span class="badge">{{ file.chars }} 字</span>
            <span v-if="file.truncated" class="badge warn">
              仅显示前 {{ file.returned }} 块
            </span>
          </div>
          <div v-for="chunk in file.chunks" :key="chunk.index" class="chunk">
            <div class="chunk-head">
              <span class="badge mono">#{{ chunk.index + 1 }}</span>
              <span class="badge">{{ chunk.tokens }} tokens</span>
            </div>
            <div class="chunk-text">{{ chunk.text }}</div>
          </div>
        </div>
      </div>
      <div class="modal-foot">
        <button class="btn" @click="previewOpen = false">关闭</button>
        <button class="btn primary" :disabled="busy" @click="confirmUpload">确认入库</button>
      </div>
    </div>
  </div>

  <div v-if="toast" class="toast">{{ toast }}</div>
</template>
