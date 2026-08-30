# 单路（并发1）Profiling 与优化报告

> 日期：2026-08-29 | 起点：TP=4 + DFlash2 n=7（当前最优）
> 数据集：ShareGPT 真实对话，seed=42，输出 512
> 目标：profile 单请求瓶颈，找是否还能继续优化

---

## 0. 核心发现

| 发现 | 结论 |
|------|------|
| **单请求瓶颈是什么** | **compute-bound**（SM 97% 满载，显存带宽仅 45%）——不是常见的 memory-bandwidth bound |
| **最大优化点** | **temperature=0（贪心）**：单请求 78→**96.5 tok/s（+24%）**，并发4 242→**313（+29%）** |
| **DFlash n 上限** | n=7 是硬上限（草稿模型 block_size=8，n=8 启动失败） |
| **接受率是关键杠杆** | 默认采样 27%，temp=0 提到 35%，直接转化为吞吐 |
| **通信/CPU** | 非瓶颈（CPU 93% 空闲，跨节点 all-reduce 未饱和） |

---

## 1. GPU 遥测（单请求 decode 稳态，4 节点）

| 指标 | 实测值 | 判读 |
|------|--------|------|
| GPU 利用率 | **97%**（4 卡全满） | SM 打满 |
| 显存带宽利用率 | **41-46%** | **远未饱和** |
| 功耗 | 95-101W / 145W（~68%） | 未触功耗墙 |
| SM 时钟 | 2437-2497 MHz（boost 上限 3090 的 ~80%） | 接近满频 |
| CPU 利用率 | 3-7%（93% 空闲） | 非 CPU bound |

**判读（按 goal §16 判据）**：GPU util 97% + 显存带宽仅 45% = **compute-bound（Case A）**。

### 为什么 decode 是 compute-bound（反常识）
一般单请求 decode 是 memory-bandwidth bound（batch=1 反复读权重）。但这里 **DFlash 投机一次验证 7+1 个 token**，把 batch=1 的 decode 变成了「小批量矩阵乘」，吃满了 SM 算力，而权重只读一次被摊薄——所以瓶颈从显存带宽转移到了 compute。这解释了为何 GPU 97% 但显存带宽只有 45%。

---

## 2. DFlash n=7 接受率剖析（真实数据）

每轮投机 7 个草稿 token，接受率随位置衰减：

| position | 接受率 |
|---:|---:|
| pos0 | 100% |
| pos1 | 63% |
| pos2 | 40% |
| pos3 | 28% |
| pos4 | 19% |
| pos5 | 14% |
| pos6 | 9% |

- 总接受率（默认采样）：**27.2%**，平均每轮主模型 forward 产出 ~3.7 token（2.7 接受 + 1 bonus）。
- 尾部（pos5/6）接受率已 <15%，大量草稿浪费——但因 compute 有余量，浪费的代价小。
- **接受率是单请求吞吐的直接杠杆**：提高接受率 = 每次 forward 产出更多 token。

---

## 3. 优化实测结果

### ✅ 优化1：temperature=0（贪心解码）— 最大收益
| 指标 | 默认采样 | temp=0 | 提升 |
|------|---:|---:|---:|
| 单请求 tok/s | 78.2 | **96.5** | **+24%** |
| 并发4 tok/s | 241.8 | **312.8** | **+29%** |
| 接受率 | 27.2% | **35.2%** | — |
| 单请求 TPOT | 12.6ms | **9.9ms** | −21% |

**原理**：贪心解码时目标模型输出确定，草稿模型更容易命中 → 接受率从 27% 升到 35% → 直接转化为吞吐。

**⚠️ 重要 caveat（这不是纯免费加速）**：temp=0 **改变输出行为**——变成确定性贪心。
- **适合**：代码生成、数学、信息抽取、分类等确定性任务（贪心本就是首选）。
- **不适合**：创意写作、多样化对话（会降低多样性）。
- 这是**质量/速度权衡**，需按业务场景决定。投机解码本身对 temp>0 仍无损（输出分布不变），只是接受率较低、加速较小。

### ❌ 优化2：DFlash n=8 — 不可行
草稿模型 `block_size=8` 硬限制 num_speculative_tokens ≤ 7。n=8 启动报 `AssertionError: expected size 8==7`。**n=7 是 DFlash 上限。**

### 已在前期排除的优化
- `max_num_batched_tokens` 4096：MTP 实测更差（102→86），已 REJECT。
- MTP vs DFlash：真实数据 DFlash 全面胜出（见 real-data-correction.md）。

---

## 4. 还能继续优化的方向（未穷尽）

| 方向 | 预期 | 状态 |
|------|------|------|
| **temp=0 用于确定性任务** | 单请求 +24% | ✅ 已验证，按场景启用 |
| **微调 DFlash 草稿模型** | 提高接受率 → 更高吞吐 | 需训练，成本高 |
| **NVFP4 量化替代 FP8** | 权重更小、compute 更快（Blackwell FP4） | ❌ 无权重，需下载 |
| **提高 GPU 功耗上限/超频** | 当前 100W/145W，SM 已 97% | 收益有限（compute 已满） |
| **减少跨节点 all-reduce**（NVLink/单机多卡） | 降低 TP 通信开销 | 硬件限制，当前无 NVLink |
| **draft_tensor_parallel_size 调整** | 草稿模型并行度 | 可试，草稿模型小收益可能有限 |

**核心判断**：单请求已是 compute-bound 且 SM 97% 满载，**在当前硬件+权重下，temp=0 是唯一的大收益优化**（+24%）。进一步提升需要换权重（NVFP4）或换硬件（NVLink 降通信/更强算力）。

---

## 5. 优化后的单请求最优

| 配置 | 单请求 tok/s | 场景 |
|------|---:|------|
| DFlash n=7 + 默认采样 | 78.2 | 通用（保多样性） |
| **DFlash n=7 + temp=0** | **96.5** | 确定性任务（代码/数学/抽取） |

**从最初 PP=4 的 20.6 → 现在 96.5 tok/s（确定性任务），累计 4.7 倍。**

---

## 6. 数据与复现

- CSV：`results/final/benchmark.csv`（`profiling` 行）
- temp=0 复现：benchmark 加 `--temperature 0`；生产 API 请求设 `"temperature": 0`。
- 遥测：`nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm --format=csv` 单请求时采样。

```bash
# temp=0 单请求 benchmark
vllm bench serve --model /models/Qwen_Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b \
  --tokenizer /models/Qwen_Qwen3.8-27B-FP8 --base-url http://<NODE0_IP>:8000 \
  --endpoint /v1/completions --dataset-name sharegpt \
  --dataset-path /root/.cache/huggingface/sharegpt.json \
  --sharegpt-output-len 512 --num-prompts 10 --max-concurrency 1 --seed 42 --temperature 0
```
