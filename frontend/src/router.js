import { createRouter, createWebHashHistory } from 'vue-router'
import ChatView from './views/ChatView.vue'
import AdminView from './views/AdminView.vue'

// 用 hash 路由：FastAPI 只需托管 index.html，不需要为 SPA 配置
// 「非 /api 路径一律回 index」的兜底规则，私有化部署少一处坑。
export const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', name: 'chat', component: ChatView },
    { path: '/admin', name: 'admin', component: AdminView },
    { path: '/:pathMatch(.*)*', redirect: '/' },
  ],
})
