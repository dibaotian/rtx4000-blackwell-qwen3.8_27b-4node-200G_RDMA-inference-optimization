# 4 节点 RTX PRO 4000 Blackwell — Qwen3.8-27B 推理性能优化实战

在 4 台单卡服务器（每台 1× NVIDIA RTX PRO 4000 Blackwell 24GB，200G RDMA 互联）上，对 Qwen3.8-27B 做的一次**完整、可复现、单变量对照**的推理性能优化记录。用 vLLM + Ray 部署，覆盖并行策略、量化、投机解码、长文本、Agent 对接等维度，全部基于实测数据（`vllm bench serve` + 真实 ShareGPT 数据集），不用理论值。

> 硬件：4× RTX PRO 4000 Blackwell 24GB（sm_120）· 200G ARNIC RoCE v2 · 每节点单卡
> 软件：vLLM 0.28.1（nightly）· Ray · CUDA 13.0 · Qwen3.8-27B

---

## 核心发现速览

从最初的 20.6 tok/s 优化到 100+ tok/s（单请求 decode），**5 倍提升**，每一步都是实测推翻了先前的"不可能"结论：

| 优化步骤 | 单请求 decode | 关键发现 |
|---------|---:|---------|
| PP=4（初始） | 20.6 tok/s | 因 FP8 权重 28.75GB 放不进单卡 24GB，被迫跨卡 |
| **TP=4** | 42 tok/s | **推翻"TP 跑不起来"**：TP=4 在 ARNIC RDMA 上能跑，且单请求比 PP 快 2 倍 |
| **+ 投机解码** | 78-97 tok/s | **推翻"MTP 不可用"**：TP 下 MTP/DFlash 可启用，decode 大幅加速 |
| **+ NVFP4 量化** | 97 tok/s | NVFP4 比 FP8 快 15%，能力无损，KV +26% |
| **+ temp=0** | 96-100+ tok/s | 贪心解码让投机接受率↑，再 +24%（仅适合确定性任务） |

### 几个反直觉的实测结论
- **TP 完胜 PP**：同卡数下 TP 单请求快 70-105%；PP 加卡几乎不提速（串行流水线）。
- **投机解码必须用真实数据测**：随机 token 会系统性压低接受率、**颠倒 DFlash/MTP 排序**。
- **NVFP4 上 MTP 反超 DFlash**（FP8 上相反）——量化方式影响投机算法选择。
- **单请求 decode 是 compute-bound**（投机把 batch=1 变成小批量矩阵乘），不是常见的显存带宽瓶颈。
- **GPU 锁频反而更慢 17%**：145W 功耗墙下，动态 boost 比锁频更优。
- **长文本无退化**：205K token 大海捞针任意深度命中，NVFP4 量化不损害长上下文检索。

---

## 最终推荐配置

```
Model:        Qwen3.8-27B（NVFP4 量化，unsloth）
Runtime:      vLLM（OpenAI 兼容）
并行:          TP=4, PP=1（4 节点各 1/4 层）
投机解码:      DFlash n=7（并发优先）或 MTP n=3（单请求优先）
KV cache:     FP8
Context:      131072（128K，Agent 场景；可到 262K）
采样:          确定性任务用 temperature=0（额外 +24% 加速）
```

实测性能：单请求 ~90-100 tok/s，并发 4 聚合 ~300 tok/s，TTFT ~100ms（内网）。

---

## 文档导航

### 性能报告（`results/final/`）
| 报告 | 内容 |
|------|------|
| [**性能速览.md**](results/final/性能速览.md) | ⭐ **一页看全**（3分钟看清所有配置结果 + 选型速查，先看这份） |
| [comprehensive-comparison.md](results/final/comprehensive-comparison.md) | 全配置详表（量化×并行×投机，深挖细节看这份） |
| [tp4-vs-pp4-report.md](results/final/tp4-vs-pp4-report.md) | TP=4 vs PP=4，推翻"TP 跑不起来" |
| [matrix-fp8-vs-nvfp4.md](results/final/matrix-fp8-vs-nvfp4.md) | FP8 vs NVFP4 × TP2/TP4 矩阵 + 能力测试 |
| [dflash-vs-mtp-report.md](results/final/dflash-vs-mtp-report.md) | DFlash vs MTP 投机解码对比 |
| [real-data-correction.md](results/final/real-data-correction.md) | 真实数据 vs 随机数据（重要方法论） |
| [single-request-profiling.md](results/final/single-request-profiling.md) | 单请求瓶颈 profiling |
| [long-context-report.md](results/final/long-context-report.md) | 长文本输入输出 + 大海捞针 |
| [hardware-tuning-report.md](results/final/hardware-tuning-report.md) | GPU/CPU 调频尝试（负面结果） |
| [rdma-full-analysis.md](results/final/rdma-full-analysis.md) | RDMA 全面分析（带宽/延迟/质量/瓶颈判定） |
| [rdma-monitoring-guide.md](results/final/rdma-monitoring-guide.md) | RDMA 监控指南（ARNIC 计数器 + 脚本用法） |
| [nvfp4-search-report.md](results/final/nvfp4-search-report.md) | NVFP4 量化模型调研 |

### 操作文档（`docs/`）
| 文档 | 内容 |
|------|------|
| [环境搭建与运行手册.md](docs/环境搭建与运行手册.md) | 从零部署、启停、故障恢复 |
| [AGENT-接入指南.md](docs/AGENT-接入指南.md) | Codex 等 Agent 对接（工具调用、流式、模型名匹配） |
| [重启指导书.md](docs/重启指导书.md) | 三种重启场景 + 常见故障 |
| [thinking-reasoning-guide.md](docs/thinking-reasoning-guide.md) | thinking / reasoning effort 三档控制 |
| [性能优化目标与方法论.md](docs/性能优化目标与方法论.md) | 优化方法论（单变量、真实数据、可复现） |
| [使用说明.md](docs/使用说明.md) | 部署演进记录 |

原始 benchmark 数据在 `logs/cluster/benchmarks/`（82 个 JSON），汇总在 `results/final/benchmark.csv`。

---

## 快速开始

> ⚠️ 配置里的 IP、路径、主机名均已脱敏为 `<NODE0_IP>`、`<ARNIC_RDMA_ROOT>` 等占位符，使用前替换成你自己的值。

```bash
# 1. 从模板创建配置，填入你的 4 节点 IP、ARNIC 路径等
cp cluster.env.example cluster.env
vim cluster.env   # 替换所有 <...> 占位符

# 2. 头节点执行：预检 → 构建镜像 → 下载模型 → 分发
./scripts/vllm_cluster_ctl.sh preflight
./scripts/vllm_cluster_prepare.sh build
./scripts/vllm_cluster_ctl.sh sync-image
./scripts/vllm_cluster_prepare.sh download
./scripts/vllm_cluster_ctl.sh sync-model

# 3. 启动 Ray + vLLM
./scripts/vllm_cluster_ctl.sh start
./scripts/vllm_cluster_service.sh start

# 4. 验证
curl http://<NODE0_IP>:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","prompt":"17*19=","max_tokens":8,"temperature":0}'
# 应返回 323
```

详细步骤见 [docs/环境搭建与运行手册.md](docs/环境搭建与运行手册.md)。

---

## 说明与致谢

- 模型：[Qwen3.8-27B](https://huggingface.co/Qwen)（Qwen 团队）
- NVFP4 量化：[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)（社区第三方量化）
- DFlash2 草稿模型：[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
- 推理框架：[vLLM](https://github.com/vllm-project/vllm) · [Ray](https://github.com/ray-project/ray)

本仓库是特定硬件（RTX PRO 4000 Blackwell 24GB + ARNIC RDMA）上的实测记录，数字与硬件强相关，换硬件需重测。ARNIC 为定制 RDMA provider，非通用环境。
