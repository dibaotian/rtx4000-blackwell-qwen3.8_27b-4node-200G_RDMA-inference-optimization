# Agent（Codex 等）接入指南 — Qwen3.8-27B vLLM 服务

> 更新：2026-08-30 | 服务：NVFP4 TP=4 + DFlash n=7，OpenAI 兼容
> 本文档面向对接 Codex / 其他 OpenAI 兼容 agent 的工程人员。所有信息均实测。

---

## 1. 快速接入（TL;DR）

```
Base URL:  http://<NODE0_IP>:8000/v1
Model:     qwen3.8-27b        ← 必须精确匹配，见 §5
API Key:   (当前为空，内网无鉴权；对外必须设，见 §7)
最大上下文: 131072 tokens (128K)
```

OpenAI 兼容，直接把 agent 的 `base_url` 指过来即可。支持：Chat Completions、工具调用（function calling）、流式、thinking 控制。

### 最小调用示例
```bash
curl http://<NODE0_IP>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "写一个Python快速排序"}],
    "max_tokens": 512,
    "temperature": 0
  }'
```

---

## 2. 端点（OpenAI 兼容）

| 端点 | 用途 | 状态 |
|------|------|:---:|
| `GET /v1/models` | 列出模型 | ✅ |
| `POST /v1/chat/completions` | 对话（agent 主用） | ✅ |
| `POST /v1/completions` | 文本补全 | ✅ |
| `GET /health` | 健康检查 | ✅ |
| `GET /metrics` | Prometheus 指标 | ✅ |

---

## 3. 工具调用（Function Calling）— agent 核心

服务已启用 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`，用标准 OpenAI `tools` 格式。**实测通过**：单轮、流式、多轮工具对话全部正常。

### 3.1 定义工具并调用
```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "北京天气怎么样"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "查询城市天气",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto",
  "temperature": 0
}
```
响应里 `choices[0].message.tool_calls[0]`：
```json
{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"北京\"}"}}
```

### 3.2 多轮工具对话（agent 拿到工具结果后继续）
把工具结果作为 `role: "tool"` 消息回传：
```json
{
  "messages": [
    {"role": "user", "content": "北京天气如何"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_1", "type": "function",
       "function": {"name": "get_weather", "arguments": "{\"city\":\"北京\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "晴，25度"}
  ]
}
```
模型会基于工具结果生成最终回复。

### 3.3 流式工具调用
加 `"stream": true`，工具调用参数通过 `delta.tool_calls` 逐块返回，客户端需累积拼接 `arguments`。

---

## 4. Thinking / Reasoning 控制

Qwen3.8 支持思考模式，**agent 场景通常应关闭**（见建议）。详见 `thinking-reasoning-guide.md`。

### 4.1 关闭思考（agent 推荐，快且干净）
```json
{
  "model": "qwen3.8-27b",
  "messages": [...],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

### 4.2 开启思考 + 指定深度（复杂推理任务）
```json
{
  "chat_template_kwargs": {"reasoning_effort": "xhigh"}
}
```
- 三档：`low` / `medium` / `xhigh`（传 `high` 自动→`xhigh`）
- 思考内容在 `message.reasoning`（**不是** `reasoning_content`）
- 思考 token 数在 `usage.completion_tokens_details.reasoning_tokens`

### 4.3 ⚠️ 关键坑
- effort / enable_thinking **必须放在 `chat_template_kwargs` 里**，放请求顶层会被忽略（不报错但无效）。
- 读思考用 `message.reasoning`，旧字段 `reasoning_content` 在本服务返回空。

---

## 5. ⚠️ 模型名必须精确匹配（最常见接入错误）

服务**只认 `model: "qwen3.8-27b"`**。发 `gpt-4`、`gpt-5`、`o1` 等会返回 **HTTP 404**。

Codex/codex-cli 默认可能发 OpenAI 模型名，两种解决：

**方案 A（改 agent 侧，推荐）**：在 Codex 配置里显式指定
```
model = qwen3.8-27b
```

**方案 B（改 vLLM 侧）**：给服务加别名（改 `cluster.env` 或启动参数）
```
--served-model-name qwen3.8-27b gpt-4 gpt-5-codex
```
这样 agent 发任意别名都能匹配。**需重启服务生效**。

---

## 6. 推荐的 agent 请求配置

### 代码/工具调用任务（大多数 agent 请求）
```json
{
  "model": "qwen3.8-27b",
  "messages": [...],
  "tools": [...],
  "tool_choice": "auto",
  "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false},
  "stream": true,
  "max_tokens": 2048
}
```
- `temperature: 0` — 代码要确定、可复现；且 temp=0 让投机解码接受率更高，**decode 快 ~24%**（实测 96 tok/s）。
- `enable_thinking: false` — 快、干净，不干扰工具解析。
- `stream: true` — agent 交互式体验。

### 复杂推理任务（疑难 bug、算法设计）
```json
{
  "chat_template_kwargs": {"reasoning_effort": "xhigh"},
  "temperature": 0.6,
  "max_tokens": 4096
}
```

---

## 7. 性能预期（实测参考）

| 指标 | 值 | 条件 |
|------|---:|------|
| 首字节延迟 TTFT | ~100ms | 内网、短 prompt |
| 单请求 decode | ~90-110 tok/s | temp=0 + DFlash |
| 并发 4 聚合 | ~230-310 tok/s | temp=0 |
| 32K 输入 prefill | ~0.65s | 稳态 |
| 128K 输入 prefill | ~1.5s | 稳态 |
| 最大上下文 | 131072 tokens | 超过返回 400 |

**超长请求处理**：prompt + max_tokens > 131072 时服务返回 **HTTP 400**（明确错误），agent 应捕获并截断上下文/重试，而非等待超时。

---

## 8. 错误处理（agent 需处理的响应码）

| HTTP | 含义 | agent 应对 |
|------|------|------|
| 200 | 成功 | 正常 |
| 400 | 上下文超 128K / 请求格式错 | 截断上下文重试 |
| 404 | 模型名不匹配 | 检查 model 字段（见 §5） |
| 503 | 服务未就绪（启动/编译中） | 退避重试 |

健康检查：`GET /health` 返回 200 表示可服务。

---

## 9. 当前限制与注意

- **无鉴权**：`API_KEY` 为空，仅限内网。对外必须设 API_KEY 并封 Ray 6379 端口（见部署文档 §5/§12）。
- **并发上限**：`max_num_seqs=4`，为少量 agent（1-4）优化。若 agent 数量增加需调大并重测。
- **模型能力**：27B 中等规模，适合中等复杂度任务；超高难度推理不及数百 B 大模型。
- **thinking 默认开**：不传 `enable_thinking` 时默认开启思考，agent 建议显式关闭。

---

## 10. 联调验证清单

对接 Codex 时逐项确认：
```
[ ] GET /health 返回 200
[ ] GET /v1/models 显示 qwen3.8-27b, max_model_len=131072
[ ] 基础对话：model=qwen3.8-27b 返回 200（非 404）
[ ] 工具调用：tools 请求返回 tool_calls
[ ] 多轮工具：tool role 消息能续接
[ ] 流式：stream=true 逐块返回
[ ] 关思考：enable_thinking=false 时 reasoning_tokens=0
[ ] 超长请求：>128K 返回 400 而非挂起
```

---

## 附：相关文档
- `thinking-reasoning-guide.md` — thinking/effort 详解
- `comprehensive-comparison.md` — 全配置性能对比
- `使用说明.md` / `四节点...手册.md` — 部署运维
