import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发时 vite dev 代理 /api 到后端（SSE 也能透传，http-proxy 按流式转发）；
// 生产构建产物 frontend/dist 由 FastAPI 直接托管，不依赖任何 CDN。
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // 单文件依赖很少，拆 vendor 反而多一次请求；这里只调告警阈值
    chunkSizeWarningLimit: 900,
  },
})
