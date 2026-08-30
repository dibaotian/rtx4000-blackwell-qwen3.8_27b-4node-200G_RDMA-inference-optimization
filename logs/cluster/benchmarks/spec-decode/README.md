# Qwen3.8-27B-FP8 推测解码与 CUDA Graph 优化验证

测试时间：2026-08-29

## 结论

- 本报告记录历史 `MAX_CUDAGRAPH_CAPTURE_SIZE=8` 的 eager/Graph 单变量 A/B；它证明应保留 CUDA Graph，但不代表当前 capture 上限。
- 当前生产候选保留 `TP=1, PP=4`、`ENFORCE_EAGER=false`，并按业务并发上限使用 `MAX_CUDAGRAPH_CAPTURE_SIZE=4`，自动捕获 batch `1,2,4`。
- 保持 `SPECULATIVE_CONFIG=''`。当前拓扑不能安全启用 MTP 或 DFlash2。
- CUDA Graph 在测试负载下将输出吞吐提高约 8% 至 11%，并完整通过 262,144-token 边界请求。
- Graph 模式日志无 `Traceback`、`ERROR` 或 OOM，算术探针正确返回 `323`。

## 历史 A/B 配置

```text
Model: Qwen/Qwen3.8-27B-FP8
vLLM: 0.28.1rc1.dev43+g6f7df92a8
GPU: 4 x NVIDIA RTX PRO 4000 Blackwell 24GB
Parallelism: TP=1, PP=4
MAX_MODEL_LEN=262144
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=4096
KV cache dtype: FP8
CUDA Graph capture sizes: 1,2,4,8
```

Graph 首次启动的 `torch.compile` 约需 16 至 17 秒，每个 stage 的 graph capture 约需 1 至 2 秒。Graph memory 日志为约 0.04 GiB；最小 KV stage 决定的总 KV pool 为 1,187,180 tokens，相比 eager 的 1,217,312 tokens 减少约 2.5%，262K 静态最大并发估算仍为 4.53x。

当前运行配置已经收敛为：

```text
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=2048
MAX_CUDAGRAPH_CAPTURE_SIZE=4
CUDA Graph capture sizes: 1,2,4
GPU KV cache: 1,208,272 tokens
262K static concurrency estimate: 4.61x
```

当前参数的完整搭建、运行和验收方法见 [四节点 Blackwell vLLM 环境搭建与运行手册](../../../../四节点%20Blackwell%20vLLM%20环境搭建与运行手册.md)，并发 1 至 4 的结果见 [../concurrency-1-4/](../concurrency-1-4/)。

## CUDA Graph A/B

使用 `vllm bench serve`、OpenAI completions endpoint、固定 1,024 输入 / 512 输出、`ignore_eos`、温度 0。每组在正式测量前执行一次 warmup。

| 模式 | 并发 | 成功/失败 | 输出吞吐 | 平均 TTFT | 平均 TPOT | 平均 E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eager | 1 | 2/0 | 18.65 tok/s | 353 ms | 53.04 ms | 27.46 s |
| CUDA Graph（capture 8） | 1 | 2/0 | **20.63 tok/s** | **318 ms** | **47.95 ms** | **24.82 s** |
| eager | 8 | 8/0 | 131.78 tok/s | **1.73 s** | 57.39 ms | 31.05 s |
| CUDA Graph（capture 8） | 8 | 8/0 | **142.60 tok/s** | 1.88 s | **52.50 ms** | **28.70 s** |

相对 eager，Graph 模式单并发输出吞吐提高 10.6%，并发 8 提高 8.2%；TPOT 分别降低 9.6% 和 8.5%。并发 8 的平均 TTFT 有约 0.15 秒波动，因此不能把 Graph 视为所有排队场景都降低首 token 延迟。

## 长上下文回归

| 输入 / 输出 | 成功/失败 | TTFT | 平均 TPOT | E2E |
| ---: | ---: | ---: | ---: | ---: |
| 32,768 / 32 | 1/0 | 4.78 s | 48.47 ms | 6.28 s |
| 262,112 / 32 | 1/0 | 74.63 s | 57.73 ms | 76.42 s |

第二条请求的输入与输出之和正好是 262,144，证明 Graph 模式没有缩短可用 context。与 eager 边界基线相比，TTFT 从 79.82 秒降到 74.63 秒。

## MTP 与 DFlash2

Qwen3.8 checkpoint 包含一层原生 MTP，但 vLLM 当前的 `Qwen3_5MTP` model runner 不支持 pipeline parallel。实际启动失败信息为：

```text
NotImplementedError: Pipeline parallelism is not supported for this model.
```

`DFlash2Qwen3ForCausalLM` 同样不支持 pipeline parallel。上游 PP 支持 PR [vllm-project/vllm#46994](https://github.com/vllm-project/vllm/pull/46994) 尚未成为当前镜像中的可用实现，并涉及多项正确性修复，不能只修改 `SupportsPP` 接口声明。

作为替代方案测试了 `TP=4, PP=1`，但当前四节点 ARNIC 环境停在 NCCL communicator 初始化，未完成模型加载。因此生产配置继续使用 `TP=1, PP=4`，并将 speculative config 保持为空。

## 原始结果

- `pp4-baseline-1k-512-c1.json`
- `pp4-baseline-1k-512-c8.json`
- `pp4-cudagraph8-1k-512-c1.json`
- `pp4-cudagraph8-1k-512-c8.json`
- `pp4-cudagraph8-32k-32-c1.json`
- `pp4-cudagraph8-262112-32-c1.json`