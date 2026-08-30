# Qwen3.8-27B Thinking / Reasoning Effort 使用指南

> 日期：2026-08-30 | 配置：NVFP4 TP=4 + DFlash n=7，vLLM `--reasoning-parser qwen3`
> 结论：**reasoning parser 工作正常，支持三档 effort。** 之前误判为"坏了"是因为读错了字段名。

---

## 0. 核心结论

| 问题 | 答案 |
|------|------|
| 支持 thinking effort 分档吗？ | ✅ **支持三档：`low` / `medium` / `xhigh`**（传 `high` 自动映射为 `xhigh`） |
| reasoning parser 坏了吗？ | ❌ 没坏。正常提取思考内容 |
| 思考内容在哪个字段？ | **`message.reasoning`**（不是旧的 `reasoning_content`！） |
| 流式支持吗？ | ✅ 流式下 `delta.reasoning` 正常分离 |
| 能关闭吗？ | ✅ `enable_thinking: false` 完全关闭 |

**关键坑**：这个 vLLM nightly 用 OpenAI 新标准字段名 **`reasoning`**，不是旧的 `reasoning_content`。读旧字段会一直得到空值，误以为 parser 坏了。

---

## 1. 三档 effort 实测差异（农夫过河难题）

| effort | reasoning_tokens | 说明 |
|--------|---:|------|
| `low` | 728 | 思考发散、啰嗦 |
| `medium` | 435 | 适中 |
| `xhigh` | 322 | **思考更聚焦高效，直奔答案** |

**注意**：effort 高 ≠ token 多。Qwen 设计上 `xhigh` 思考更精准，reasoning_tokens 反而可能更少——是"越高越聚焦"，不是"越高越长"。

---

## 2. 使用方式

### 开启 thinking + 指定 effort
```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "证明根号2是无理数"}],
  "chat_template_kwargs": {"reasoning_effort": "xhigh"},
  "max_tokens": 2000
}
```
思考内容在响应的 `choices[0].message.reasoning`，最终答案在 `choices[0].message.content`。

### 关闭 thinking（Codex/agent 推荐）
```json
{
  "model": "qwen3.8-27b",
  "messages": [...],
  "chat_template_kwargs": {"enable_thinking": false}
}
```
`reasoning_tokens=0`，响应快、干净，只有 `content`。

### 流式
```json
{..., "stream": true, "chat_template_kwargs": {"reasoning_effort": "low"}}
```
- 思考阶段：`delta.reasoning` 逐块返回
- 答案阶段：`delta.content` 逐块返回
- 客户端可分别渲染（思考折叠 / 答案正常显示）

---

## 3. 对 Codex agent 的建议

**代码 agent 通常应关闭 thinking**：
- agent 要的是快速、确定的工具调用和代码输出。
- 思考过程增加延迟、消耗 token、可能干扰工具调用解析。
- **推荐 Codex 默认传 `enable_thinking: false`**。

**何时开 thinking**：
- 复杂算法设计、疑难 bug 推理、需要多步逻辑的任务。
- 这类任务可临时传 `reasoning_effort: medium/xhigh`。

**混合策略**（最优）：
- 简单编辑/工具调用 → `enable_thinking: false`（快）
- 复杂推理任务 → `reasoning_effort: xhigh`（准）
- 同一个服务同时支持，由每个请求自己指定。

---

## 4. 字段对照（避免踩坑）

| 用途 | 正确字段/参数 | 常见错误 |
|------|------|------|
| 读思考内容 | `message.reasoning` | ~~`message.reasoning_content`~~（旧版，此服务返回空） |
| 读思考 token 数 | `usage.completion_tokens_details.reasoning_tokens` | — |
| 设置 effort | `chat_template_kwargs.reasoning_effort` = low/medium/xhigh | 顶层 `reasoning_effort`（被忽略） |
| 开关思考 | `chat_template_kwargs.enable_thinking` = true/false | — |
| 流式思考 | `delta.reasoning` | ~~`delta.reasoning_content`~~ |

**重要**：effort 和 enable_thinking 都要放在 `chat_template_kwargs` 里，放在请求顶层会被 vLLM 忽略（不报错但无效）。

---

## 5. 验证命令

```bash
# 测 effort（看 reasoning 字段和 reasoning_tokens）
curl -s http://<NODE0_IP>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen3.8-27b",
  "messages":[{"role":"user","content":"证明根号2是无理数"}],
  "chat_template_kwargs":{"reasoning_effort":"xhigh"},
  "max_tokens":2000
}' | python3 -c "import json,sys; d=json.load(sys.stdin); m=d['choices'][0]['message']; print('reasoning:', (m.get('reasoning') or '')[:100]); print('reasoning_tokens:', d['usage']['completion_tokens_details']['reasoning_tokens'])"

# 关闭 thinking（agent 场景）
curl -s ... -d '{..., "chat_template_kwargs":{"enable_thinking":false}}'
```
