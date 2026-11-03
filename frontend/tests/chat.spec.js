import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { nextTick } from 'vue'

// 只替换网络层：组件、store、SSE 状态机都跑真实代码
vi.mock('../src/api/client', () => ({
  api: {
    chatStream: vi.fn(),
    sessionDetail: vi.fn(),
    sendFeedback: vi.fn(),
  },
}))

import { api } from '../src/api/client'
import ChatView from '../src/views/ChatView.vue'
import { newSession } from '../src/stores/chat'

const SOURCE = {
  title: '失眠指南',
  source: 'local_kb',
  score: 0.9,
  publish_time: '2026-01',
  department: '神经内科',
  authority_tier: 'tier_1',
  authority_label: '临床指南',
  doc_id: 'insomnia',
  chunk_id: 'abc123def456',
  snippet: '失眠与作息、压力相关，建议规律作息。',
}

const FINAL = {
  answer: '建议就医',
  sources: [SOURCE],
  judge_result: { layer: 'both', scores: { factual_consistency: 9 } },
  latency_ms: 12,
  session_id: 's1',
  outcome: 'ok',
  message_id: '7',
}

function streamOf(events) {
  api.chatStream.mockImplementation(async ({ onEvent }) => {
    for (const event of events) onEvent(event)
  })
}

async function send(wrapper, text) {
  await wrapper.find('textarea').setValue(text)
  await nextTick()
  const button = wrapper.find('.composer-box .btn.primary')
  expect(button.attributes('disabled')).toBeUndefined()
  await button.trigger('click')
  await flushPromises()
  await nextTick()
}

beforeEach(() => {
  newSession()
  localStorage.clear()
  api.chatStream.mockReset()
  api.sendFeedback.mockReset()
  api.sessionDetail.mockRejectedValue(new Error('404'))
  api.sendFeedback.mockResolvedValue({
    id: 1,
    message_id: 7,
    session_id: 's1',
    rating: -1,
    reason_tags: ['答非所问'],
    comment: '',
    created_at: '2026-11-03T12:00:00',
  })
})

describe('问诊交互', () => {
  it('输入并点击发送后，流式增量显示在气泡里', async () => {
    streamOf([
      { event: 'meta', data: { session_id: 's1' } },
      { event: 'risk', data: { level: 'low', mode: 'token' } },
      { event: 'sources', data: { sources: [SOURCE], count: 1 } },
      { event: 'delta', data: { text: '建议' } },
      { event: 'delta', data: { text: '就医' } },
      { event: 'final', data: FINAL },
    ])

    const wrapper = mount(ChatView)
    await send(wrapper, '最近总是失眠怎么办')

    expect(wrapper.findAll('.msg')).toHaveLength(2)
    expect(wrapper.find('.msg.user .bubble').text()).toBe('最近总是失眠怎么办')
    expect(wrapper.find('.msg.assistant .bubble').text()).toBe('建议就医')

    // 有来源时右侧溯源栏自动打开，且能看到命中原文
    const drawer = wrapper.find('.sources-drawer')
    expect(drawer.exists()).toBe(true)
    expect(drawer.text()).toContain('失眠指南')
    expect(drawer.text()).toContain('失眠与作息')
    expect(api.chatStream).toHaveBeenCalledWith(
      expect.objectContaining({ query: '最近总是失眠怎么办' }),
    )
  })

  it('收到 replace 时整段覆盖，不追加', async () => {
    streamOf([
      { event: 'delta', data: { text: '旧答案' } },
      { event: 'replace', data: { text: '新答案', reason: 'judge_retry' } },
      { event: 'final', data: { ...FINAL, answer: '新答案', sources: [] } },
    ])

    const wrapper = mount(ChatView)
    await send(wrapper, '最近总是失眠怎么办')

    const bubble = wrapper.find('.msg.assistant .bubble')
    expect(bubble.text()).toBe('新答案')
    expect(bubble.text()).not.toContain('旧答案')
  })

  it('点👎选原因后提交，反馈带上 message_id 与标签', async () => {
    streamOf([
      { event: 'sources', data: { sources: [SOURCE], count: 1 } },
      { event: 'delta', data: { text: '建议就医' } },
      { event: 'final', data: FINAL },
    ])

    const wrapper = mount(ChatView)
    await send(wrapper, '最近总是失眠怎么办')

    const dislike = wrapper
      .findAll('.fb-btn')
      .find((b) => b.text().includes('没用'))
    await dislike.trigger('click')
    expect(wrapper.find('.fb-panel').exists()).toBe(true)

    await wrapper.findAll('.fb-tag')[0].trigger('click')
    await wrapper.find('.fb-panel .btn.danger').trigger('click')
    await flushPromises()

    expect(api.sendFeedback).toHaveBeenCalledTimes(1)
    expect(api.sendFeedback.mock.calls[0][0]).toMatchObject({
      message_id: 7,
      rating: -1,
      reason_tags: ['答非所问'],
    })
    expect(wrapper.text()).toContain('已记录')
  })

  it('回答未落库（message_id 为空）时反馈按钮禁用', async () => {
    streamOf([
      { event: 'delta', data: { text: '降级回答' } },
      { event: 'final', data: { ...FINAL, answer: '降级回答', sources: [], message_id: '' } },
    ])

    const wrapper = mount(ChatView)
    await send(wrapper, '最近总是失眠怎么办')

    const dislike = wrapper
      .findAll('.fb-btn')
      .find((b) => b.text().includes('没用'))
    expect(dislike.attributes('disabled')).toBeDefined()
  })

  it('流式请求失败时把错误写进气泡，不静默', async () => {
    api.chatStream.mockRejectedValue(new Error('服务不可用'))

    const wrapper = mount(ChatView)
    await send(wrapper, '最近总是失眠怎么办')

    expect(wrapper.find('.msg.assistant .bubble').text()).toContain('服务不可用')
  })
})
