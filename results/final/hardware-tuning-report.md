# 硬件/系统级调优报告 — 提升 prefill/TPOT 的尝试

> 日期：2026-08-29 | 配置：NVFP4 TP=4 + DFlash n=7
> 问题：能否通过提高 GPU 时钟/CPU 频率等硬件手段，在保准确度下提升 prefill 和 TPOT？
> 结论：**不能。GPU 锁频反而更慢（−17%），CPU governor 无帮助。当前默认已是最优。**

---

## 0. 核心结论

| 尝试 | 结果 | 原因 |
|------|:---:|------|
| GPU 锁定最高频 3090MHz | ❌ **更慢 −17%** | 145W 功耗墙下，锁频破坏动态功耗分配 |
| CPU 切 performance governor | ❌ 无帮助 | 推理不是 CPU-bound（CPU 93% 空闲） |
| 提高 GPU 功耗上限 | ❌ 不可行 | 145W 是硬顶（max=default=145，锁死） |

**已全部恢复默认。默认的动态 boost + schedutil 就是最优配置。**

---

## 1. 硬件 headroom 诊断

| 项目 | 状态 | 是否有余量 |
|------|------|:---:|
| GPU 功耗上限 | 145W（max=default=145，**锁死**） | ❌ 无法提高 |
| GPU 时钟 | 满载自动 boost 到 2437-2497MHz（上限 3090） | ⚠️ 表面有，实际受功耗限 |
| GPU 降速触发 | 无 power cap / 热 / 硬件降速（throttle reasons 全 0） | 满载时未被限制 |
| CPU governor | schedutil（动态，空闲 1510MHz，上限 3550） | ⚠️ 表面有 |
| 显存带宽 | 单请求 decode 仅用 45% | 有余量但非瓶颈 |

**关键**：GPU 满载时时钟只到 2497MHz（而非上限 3090），不是因为被"限制"，而是因为 **145W 功耗预算不够同时喂满所有单元**。这是这张卡的物理特性。

---

## 2. A/B 实测（单请求 decode，NVFP4+DFlash，temp=0，真实数据）

| 配置 | run1 | run2 | run3 | 中位 |
|------|---:|---:|---:|---:|
| **默认（动态 boost + schedutil）** | 105 | 115 | 111 | **~110 tok/s** |
| GPU 锁 3090 + CPU performance | 88 | 95 | 91 | **~91 tok/s** |

**调优反而慢 17%。** 已 REJECT，恢复默认。

### 为什么锁频有害
- GPU 强制锁在 3090MHz，但 145W 功耗预算不足以支撑所有单元同时全速。
- 锁死时钟后，GPU 无法在功耗墙内**智能地**把有限的瓦数分配给真正的瓶颈单元（compute 核心 / 显存控制器）。
- 默认的动态 boost 会根据实时负载和功耗余量优化分配，在**功耗受限的卡上反而更高效**。
- 结论：**功耗受限的 GPU（如这张 145W 的 RTX PRO 4000）不应锁频**，动态调度更优。

### 为什么 CPU governor 无帮助
- LLM 推理的 compute 全在 GPU，CPU 只做调度/tokenization/采样，实测 CPU 93% 空闲。
- CPU 不是瓶颈，提高其频率不改变 GPU-bound 的 decode/prefill。

---

## 3. 附带修正：prefill TTFT 之前数据是冷启动失真

调优测试中发现，之前 `long-context-report.md` 的 prefill TTFT 是**冷启动首次请求**（含 JIT/kernel 编译），严重偏大。稳态（warmup 后）实测：

| 输入长度 | 之前（冷启动，失真） | **稳态（正确）** | 修正幅度 |
|---:|---:|---:|---:|
| 32K | 5549 ms | **~650 ms** | 快 8.5× |
| 128K | 33039 ms | **1468 ms** | 快 22× |
| 262K | 97148 ms | **63051 ms** | 快 1.5× |

**真实的 prefill 能力比之前报告的好得多**：
- 32K 输入首字节仅 0.65s（不是 5.5s）
- 128K 输入 1.5s（不是 33s）
- 262K 边界 63s（超长 prefill 本身 compute 密集，这个量级合理）

> 教训：benchmark 必须在服务 warmup 后测稳态，冷启动首次请求含编译开销，不能作为性能数据。`long-context-report.md` 的 prefill 表应以本报告的稳态值为准。

---

## 4. 那到底怎么提升 prefill/TPOT？（有效手段回顾）

硬件层无余量，但**软件/算法层已经做了有效优化**（前几轮已验证）：

| 手段 | 对 TPOT | 对 prefill/TTFT | 状态 |
|------|:---:|:---:|:---:|
| **NVFP4 量化** | ✅ +13% | ✅ 略帮 | 已用 |
| **DFlash/MTP 投机** | ✅ 大幅（TPOT 23→6ms） | ❌ 不帮 | 已用 |
| **temp=0（确定性任务）** | ✅ +24% | ❌ 不帮 | 按场景用 |
| **CUDA Graph** | ✅ +10% | ⚠️ 轻微 | 已用 |
| **TP=4（vs TP=2）** | ✅ 并行加速 | ✅ 帮 | 已用 |
| GPU 锁频 | ❌ −17% | ❌ | 已弃 |
| CPU 提频 | ❌ 无 | ❌ 无 | 已弃 |

**提升 TPOT 的有效路径全在软件层**（量化+投机+采样+graph），已全部用上。
**提升 prefill/TTFT** 的主要手段是 TP 并行（已 TP=4）；进一步需要更强算力硬件或更激进的 attention kernel（如 FlashAttention 变体，vLLM 已默认用）。

### 还能试的（软件层，未做）
1. **prefix caching 命中**：长文档问答若有共享前缀，缓存命中可让 prefill TTFT 从秒级降到毫秒级（实测 32K 缓存命中 720ms→已见效）。适合 RAG/多轮同文档场景。
2. **chunked prefill 调参**：调 `max_num_batched_tokens` 平衡 prefill 分块——但前期实测对 decode 无益，对长 prefill 需单独评估。
3. **更快的 attention backend**：当前已用 FlashInfer/FLASH_ATTN，无明显更优选项。

---

## 5. 最终建议

- **硬件层：保持默认**（GPU 动态 boost、CPU schedutil）。不要锁频，不要改 governor——实测有害或无益。
- **这张卡的性能已被软件优化榨到接近硬件极限**（145W 功耗墙 + compute-bound）。
- 要再上一个台阶，只能换硬件：更高功耗预算的 GPU（如 RTX PRO 6000 300W+）、NVLink（降 TP 通信）、或更多卡。

---

## 6. 数据与复现

- CSV：`results/final/benchmark.csv`（`hwtune` / `prefill-fix` 行）
- 恢复默认命令：
```bash
# GPU 解锁频率
sudo nvidia-smi -rgc
# CPU 恢复 schedutil
echo schedutil | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```
- 测调优效果：切换后用相同 `vllm bench serve`（sharegpt, temp=0, c1）多次取中位对比。
