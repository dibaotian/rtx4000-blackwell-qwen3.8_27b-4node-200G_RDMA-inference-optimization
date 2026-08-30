# Qwen3.8-27B-FP8 262K Context 性能测试

测试时间：2026-08-29

## 配置

- GPU：4 x NVIDIA RTX PRO 4000 Blackwell 24GB，一节点一卡
- 网络：双 200Gb/s ARNIC RoCE v2，NCCL `NET/IB`
- 模型：`Qwen/Qwen3.8-27B-FP8`
- vLLM：`0.28.1rc1.dev43+g6f7df92a8`
- CUDA：13.0
- 并行：`TP=1, PP=4`
- `max_model_len=262144`
- FP8 KV cache、prefix cache、chunked prefill
- eager mode、text-only
- `max_num_seqs=8`、`max_num_batched_tokens=4096`

服务初始化结果：

- GPU KV cache：1,217,312 tokens
- 262,144-token 请求的静态最大并发估算：4.64x
- API：`http://<NODE0_IP>:8000/v1`

## 方法

使用 `vllm bench serve`、OpenAI completions endpoint、随机固定长度 token、`ignore_eos` 和固定输出长度。

长上下文冷测分别使用 seed 101、202、303。测试前后 `prefix_cache_hits_total` 均为 161,504，证明这三条测量没有 prefix-cache 命中。随机数据集会迭代执行 decode/encode，将服务端实际 prompt 长度校准到目标值；服务 metrics 也确认 262,112 prompt tokens 被完整处理。

输入 262,112 tokens 加输出 32 tokens，正好达到 262,144-token 上限。

## 短请求吞吐

固定每请求 1,024 输入、128 输出：

| 并发 | 成功/失败 | 输出吞吐 | 总 token 吞吐 | 平均 TTFT | 平均 TPOT | 平均 E2E |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4/0 | 18.04 tok/s | 162.40 tok/s | 343 ms | 53.15 ms | 7.09 s |
| 4 | 16/0 | 58.98 tok/s | 530.86 tok/s | 1.22 s | 58.68 ms | 8.67 s |
| 8 | 16/0 | **105.31 tok/s** | **947.83 tok/s** | 1.78 s | 62.31 ms | 9.69 s |
| 16 | 32/0 | 96.00 tok/s | 863.99 tok/s | 9.51 s | 72.55 ms | 18.73 s |

本次历史基线配置下，并发 8 是短请求吞吐甜点。并发 16 已饱和，吞吐下降且 TTFT 大幅增加。当前生产候选已按业务要求将调度并发收敛到 4，不能把这条历史扫描结论当作当前服务上限。

## 冷长上下文

单并发、32 输出 tokens：

| 输入长度 | 成功/失败 | TTFT | 近似 prefill 吞吐 | 平均 TPOT | 近似 decode 速度 | E2E |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 1/0 | 5.39 s | 6,082 tok/s | 54.07 ms | 18.50 tok/s | 7.06 s |
| 131,072 | 1/0 | 27.95 s | 4,690 tok/s | 57.96 ms | 17.25 tok/s | 29.74 s |
| 262,112 | 1/0 | 79.82 s | 3,284 tok/s | 60.52 ms | 16.52 tok/s | 81.70 s |

262K 边界请求完整成功，服务没有截断、OOM 或错误。随着 context 增长，prefill 吞吐下降，首 token 延迟增长明显。

## 满 Context 并发压力

每请求 262,112 输入、32 输出：

| 并发 | 成功/失败 | 平均 TTFT | P99 TTFT | 平均 TPOT | 平均 E2E | Preemption |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1/0 | 79.82 s | 79.82 s | 60.52 ms | 81.70 s | 0 |
| 2 | 2/0 | 156.06 s | 230.30 s | 1,652.93 ms | 207.30 s | 1 |
| 4 | 4/0 | 293.45 s | 477.09 s | 2,468.84 ms | 369.98 s | 3 |

四路满 context 的总 prompt 为 1,048,448 tokens，低于静态 KV pool，但运行中 KV 使用率多次接近 99%，触发抢占与重计算。因此 4.64x 是容量估算，不是可接受延迟下的并发能力。

## 建议

- 保留 `max_model_len=262144`，它已通过边界验证。
- 对接近 262K 的请求设置应用层并发上限 1；最多允许 2 时必须接受分钟级尾延迟。
- 本次历史扫描中普通 1K/128 请求的吞吐甜点约为并发 8；当前生产候选的业务并发上限为 4。
- 按输入长度分队列：例如 `<32K`、`32K-128K`、`>128K`，超长请求单独限流。
- 客户端必须为输出保留 token：`prompt_tokens + max_tokens <= 262144`。
- 本报告记录的是 eager 基线。后续 CUDA Graph A/B 已完成并保留；MTP/DFlash2 因当前 `PP=4` 不兼容而禁用，详见 [../spec-decode/README.md](../spec-decode/README.md)。

原始 JSON 和汇总 CSV 均保存在本目录。压力测试后服务仍为 ready，短请求正确返回 `323`，HTTP 200，耗时约 0.36 秒；Ray 保持 4 个 active node，未出现 OOM 或请求错误。
