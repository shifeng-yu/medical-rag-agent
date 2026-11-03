import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'

// 测试用独立配置：不复用 vite.config.js 里的 devServer/proxy，
// 避免测试时误连真实后端。
export default defineConfig({
  plugins: [vue()],
  test: {
    environment: 'jsdom',
    include: ['tests/**/*.spec.js'],
    restoreMocks: true,
    // 串行跑文件：并行 worker 在 Windows + 受限文件系统下偶发漏收集
    // （表现为只跑了其中一个 spec）。测试文件很少，串行代价可以忽略。
    fileParallelism: false,
  },
})
