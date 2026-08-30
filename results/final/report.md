# 4-Node RTX PRO 4000 Blackwell — Qwen3.8-27B 推理性能优化报告

> 生成日期：2026-08-29
> 数据来源：`/data/vllm/logs/cluster/benchmarks/` 下的真实 benchmark（vLLM `bench serve`），非理论推算。
> 本报告严格区分「已实测」「受限未测」「失败」三类结论，不用理论值冒充实测。

---

## 0. 阅读导航（先看这里）

这份报告要回答 `/goal` 文档里的 10 个核心问题。下面先给一句话结论，细节在后文各章。

| # | 问题 | 结论 | 依据 |
|---|------|------|------|
| 1 | 最佳量化模型 | **Qwen3.8-27B-FP8**（官方 checkpoint） | §3 |
| 2 | 最佳 runtime | **vLLM 0.28.1rc1**（nightly，sm_120） | §3、§8 |
| 3 | 最佳 MTP | **无法启用**（当前 PP=4 拓扑不支持），改用 CUDA Graph 替代增益 | §6 |
| 4 | 单请求 decode（1并发） | **20.6 tok/s**（1K 输入/512 输出，Graph 开） | §4 |
| 5 | 4-node aggregate（1K/512, C4） | 当前配置 **76.1 tok/s** | §4、§5 |
| 6 | DP scaling efficiency | **本集群非 DP=4 架构**，是 PP=4 单副本；不适用该指标 | §2、§9 |
| 7 | 4-node TP | **✅ TP=4 实测可运行**：c1=42.2 / c2=76.4 / c4=142.3 tok/s（见 tp4-vs-pp4-report.md） | §7 |
| 8 | TP 通信开销 | **非瓶颈**（压测 4 卡 99% util，近线性扩展；精确 % 待 hw_counter） | §7 |
| 9 | 200G RDMA 是否瓶颈 | **否**：PP=4 与 TP=4 下均非瓶颈（TP all-reduce 走 IB，仍 compute bound） | §7、§9 |
| 10 | 最终推荐配置 | **decode 首选 TP=4**（单请求快 2 倍）；FP8 + vLLM + FP8 KV + CUDA Graph | §10、tp4-vs-pp4-report.md |

> ⚠️ **重要更新（2026-08-29 晚）**：本报告主体基于 PP=4 数据写成，其中「TP=4 启动失败」的结论已被**实测推翻**。TP=4 不仅能跑，单请求 decode（42 tok/s）是 PP=4（20.6）的 **2 倍**。完整 TP=4 实测见同目录 **`tp4-vs-pp4-report.md`**。下方 §7 保留原始记述以说明「为何曾误判」，但最终结论以 TP=4 报告为准。

**一句话总览**：本项目实际落地的架构与原始 `/goal` 设想的「NVFP4 + DP=4 单卡独立副本」**不同**。因为 27B 模型的可用量化权重（FP8，约 28.75GiB）**放不进单张 24GB 卡**，集群采用了 **1 副本横跨 4 卡的 Pipeline Parallel（TP=1, PP=4）**。这直接改变了「单 GPU baseline」「DP scaling」「聚合吞吐」这些指标的定义。下面详细解释为什么，以及实测到什么。

---

## 1. 硬件与软件环境（已确认）

| 项目 | 值 |
|------|-----|
| 节点 | 4 × SR655（node0 ~ node3） |
| GPU | 每节点 1 × NVIDIA RTX PRO 4000 Blackwell 24GB（共 4 张） |
| Compute capability | 12.0（sm_120，Blackwell） |
| 单卡显存 | 24467 MiB（约 23.9 GiB） |
| 4 卡合计显存 | 约 95.6 GiB |
| 驱动 | 580.159.04 |
| 宿主 CUDA Toolkit | 12.8；容器内 CUDA 13.0 |
| 内存 | 每节点约 124 GiB |
| 网络 | 双口 200Gb/s ARNIC RoCE v2（`arnic_0` / `arnic_1`），MTU 4096，GID index 1 |
| RDMA 状态 | 2 条 link ACTIVE；NCCL 走 `NET/IB`，GPUDirect（DMA-BUF）可用 |
| GPU↔GPU 互联 | **无 NVLink**，跨节点走 PCIe + 200G RDMA |
| Runtime | vLLM `0.28.1rc1.dev43+g6f7df92a8`（Ray 分布式后端） |
| 容器 | `vllm-blackwell-ray:20260828`，Docker 29.1.3 + NVIDIA runtime |

**网卡命名差异**（运维关键）：1/2 号节点用 `ens7`，3/4 号节点用 `ens8`；脚本用 `RAY_INTERFACE=auto` 自动处理，不要写死。

模型结构（`Qwen3.8-27B-FP8/config.json`）：`model_type=qwen3_5`，64 层，hidden 5120，24 个 attention head，4 个 KV head（GQA），head_dim 256，vocab 248320，max_position 262144，量化 **fp8**。64 层 ÷ 4 stage = 每节点正好 16 层，PP 切分非常干净。

---

## 2. 关键架构决策：为什么是 PP=4 而不是 DP=4

原始 `/goal` 假设「27B 量化后能塞进单张 24GB 卡，于是 4 卡各跑一个完整副本（DP=4）」。**实测证明这个前提不成立**：

- 目前唯一稳定可用的量化权重是官方 **FP8**，磁盘 30.89 GB / 约 28.75 GiB（66 个 shard）。
- 单张卡只有 23.9 GiB，光权重就放不下，还没算 KV cache / CUDA context / graph。
- 因此模型必须**跨卡切分**。在「一节点一卡」拓扑下，vLLM 的合适布局是 **TP=1, PP=4**：把 64 层按 stage 分到 4 张卡，每张卡放 16 层。

这与 DP=4 有本质区别：

| | DP=4（goal 设想） | PP=4（实际落地） |
|---|---|---|
| 每卡内容 | 一个完整模型副本 | 模型的 1/4（16 层） |
| 请求路由 | 4 个请求 → 4 卡并行，互不通信 | 1 个请求依次流过 4 卡（流水线） |
| 跨节点通信 | 几乎无（仅调度/控制） | 每层边界传 activation（点对点，量小） |
| 单请求延迟 | 单卡算力决定 | 4 卡串行 + 3 次跨节点 hop |
| 聚合吞吐来源 | 4 副本相加 | 流水线并发 + 批处理 |
| "单 GPU baseline" | 有意义 | **无意义**（模型跑不起来） |
| "DP scaling efficiency" | 核心指标 | **不适用** |

**结论**：由于显存约束，`/goal` 里「单 GPU baseline → DP=4 线性 scaling」这条主线在本硬件上无法执行。真实的性能问题变成了「PP=4 流水线在小并发下的吞吐与延迟」。本报告据此重新组织指标。

> 若坚持要做真正的 DP=4，需要一个**能塞进 24GB 的量化模型**（如可用的 NVFP4/W4A16 权重）。截至目前磁盘上**没有** NVFP4 版本的 Qwen3.8-27B（只在 vLLM 源码里有 `nvfp4_kv_cache_kernels.cu` 内核，没有对应权重）。这是继续优化的头号前置条件，见 §10。

---

## 3. 量化与 Runtime 选择（P0/P1/P2）

### 量化：FP8 胜出（唯一稳定可用）
- **FP8**：官方 checkpoint，Blackwell 原生支持，实测稳定跑通 262K 边界、无 OOM。**已采用**。
- **NVFP4**：`/goal` 首选，但磁盘上无权重，未测。Blackwell 理论上 FP4 吞吐更高，是最值得补的实验（见 §10）。
- **BF16**：27B BF16 约 54GiB，4 卡 PP 也吃力且无收益，不考虑。

### Runtime：vLLM
- vLLM nightly 已支持 sm_120 + FP8 + PP=4 + Ray 跨节点 + FP8 KV + CUDA Graph，且 200G RDMA 经 NCCL `NET/IB` 正常工作。**已采用**。
- SGLang / llama.cpp 未测。按 `/goal` 优先级，只有当 vLLM 的 NVFP4/MTP 出现明显问题时才切换；当前 vLLM 能满足除 MTP 外的全部需求。

### 一个 Blackwell 专属细节
启动日志显示 vLLM 在本架构上**自动禁用了 DeepGemm**：
> `Auto-disabled DeepGemm for model_type=qwen3_5_text on Blackwell. DeepGemm E8M0 scale format causes accuracy degradation ... Falling back to CUTLASS.`

即 FP8 GEMM 走 CUTLASS 而非 DeepGemm，这是精度正确性的保护，不用干预。

---

## 4. 核心实测：并发 1 / 2 / 4（1K 输入 / 512 输出，CUDA Graph 开）

这是最贴合 `/goal`「只测并发 1/2/4」要求的一组数据，配置固定、可比较。
来源：`logs/cluster/benchmarks/concurrency-1-4/`（`maxseq4-batched2048-c{1,2,4}.json`，PP 层分布 16/16/16/16）。

| 并发 | 完成请求 | 输出吞吐 (tok/s) | 总 token 吞吐 | 平均 TTFT | 中位 TTFT | 平均 TPOT | 相对 C1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | 4 | **20.62** | 61.9 | 318 ms | 319 ms | 47.97 ms | 1.00× |
| **2** | 8 | **40.91** | 122.7 | 440 ms | 380 ms | 48.10 ms | **1.98×** |
| **4** | 12 | **76.07** | 228.2 | 818 ms | 963 ms | 51.03 ms | **3.69×** |

**解读**：
- **单请求 decode = 20.6 tok/s**（≈48.5 ms/token）。这是「一条请求流过 4 卡流水线」的速度，受跨节点 hop 延迟影响，不是单卡算力上限。
- **并发扩展几乎线性**：C2 达 1.98×，C4 达 3.69×。TPOT 从 48.0ms 仅微增到 51.0ms（约 +6%），说明 PP=4 流水线在低并发下靠并发填充流水线气泡提升吞吐。
- **TTFT 随并发上升**（319ms → 963ms 中位）：因为 prefill 也在 4 卡串行，多请求排队时首 token 变慢。这是 PP 相对 DP 的固有代价。

**补充：中间并发验证**（`maxseq4-graph1234-c3.json`，C3）→ 59.5 tok/s，落在 C2(40.9) 与 C4(75.7) 之间，趋势一致，交叉验证了线性性。

**配置不敏感性**（重要的负面结论，避免无效调参）：
- 将 `max_num_seqs` 和 Graph capture 上限从 8 收敛到 4，对 c1/c2 吞吐无实质影响，因此按业务上限保留 4。
- 显式加入 batch-3 Graph 后，c3 吞吐仅提高约 0.05%，属于噪声，因此继续使用自动 `[1,2,4]`。
- `max_num_batched_tokens=2048` 相对 4096 时，c1/c2 吞吐变化小于 0.1%，c4 吞吐提高 0.54%，但 c4 平均 TTFT 降低 21.1%。32K TTFT 降低 10.6%，代价是该长上下文 TPOT 增加 5.6%。由于目标偏重并发 1 至 4 的首 token 延迟，保留 2048。
- `max_num_batched_tokens=8192` 让 32K TTFT 恶化约 41%，KV pool 减少约 6.2%，已拒绝。

这说明低并发 decode 吞吐主要受流水线结构影响，调度参数的价值主要体现在 prefill 延迟与 KV 容量取舍。

---

## 5. CUDA Graph A/B（唯一成功的「单变量优化」）

由于 MTP 被拓扑封死（§6），CUDA Graph 成为唯一跑通的加速项。严格单变量对照（1K 输入/512 输出，仅切换 eager↔graph）。
来源：`logs/cluster/benchmarks/spec-decode/`。

| 模式 | 并发 | 输出吞吐 | 平均 TTFT | 平均 TPOT | 平均 E2E |
|------|---:|---:|---:|---:|---:|
| eager | 1 | 18.65 tok/s | 353 ms | 53.04 ms | 27.46 s |
| **Graph** | 1 | **20.63 tok/s** | **318 ms** | **47.95 ms** | **24.82 s** |
| eager | 8 | 131.78 tok/s | **1.73 s** | 57.39 ms | 31.05 s |
| **Graph** | 8 | **142.60 tok/s** | 1.88 s | **52.50 ms** | **28.70 s** |

**判定（按 /goal 有效性标准）**：
- 单并发输出吞吐 **+10.6%**、TPOT **−9.6%** → 「明显性能变化」。**KEEP**。
- 并发 8 输出吞吐 **+8.2%**、TPOT **−8.5%** → 「有意义变化」。**KEEP**。
- 代价：graph 占显存约 0.04 GiB，KV pool 从 1,217,312 → 1,187,180 tokens（−2.5%），可接受。
- 注意：并发 8 的 TTFT 有约 0.15s 波动，不能宣称 Graph 在所有排队场景都降低首 token 延迟。

**结论**：这组历史 A/B 证明应保留 `ENFORCE_EAGER=false`。当前并发上限为 4，因此将 `MAX_CUDAGRAPH_CAPTURE_SIZE` 收敛为 4，自动捕获 `[1,2,4]`；当前配置已再次通过 262K 边界请求，无 Traceback/OOM。

---

## 6. MTP：受阻，无法启用（P3 卡点）

`/goal` 把 MTP 列为核心优化项（P3），要求测 MTP OFF/1/2/3。**实测结论：当前拓扑无法启用 MTP。**

- Qwen3.8 checkpoint 自带一层原生 MTP head，但 vLLM 当前的 `Qwen3_5MTP` model runner **不支持 pipeline parallel**，启动直接抛：
  ```
  NotImplementedError: Pipeline parallelism is not supported for this model.
  ```
- 备选 `DFlash2Qwen3ForCausalLM` 同样不支持 PP，且实现仍有待解决的正确性问题，不能作为生产绕过。
- 上游 PP 支持在 PR [vllm-project/vllm#46994](https://github.com/vllm-project/vllm/pull/46994)，含多处调度/正确性修复，**不能只改 `SupportsPP` 接口声明**蒙混。

**这是一个结构性矛盾**：
> MTP 需要 TP（或单卡），而本硬件因显存必须用 PP → 两者当前互斥。

要解锁 MTP，二选一：
1. 有一个能塞进 24GB 的量化模型（NVFP4/W4A16）→ 单卡或 TP，可开 MTP；或
2. 等上游把 MTP + PP 支持合并发布。

**因此 `/goal` 问题 3「MTP 把单 GPU 从 X 提升到 Y」当前无法回答**，只能记录受阻原因（符合 `/goal`「不要修改权重、记录原因、继续其他测试」的要求）。CUDA Graph 的 +10.6% 是当前拿到的替代性 decode 增益。

---

## 7. TP=4：启动失败，通信开销未能测量

`/goal` 第七/八阶段要求测 TP=2 / TP=4，并测 RDMA 通信开销。**实测：TP=4 无法在当前 ARNIC 环境启动。**

`TP=4, PP=1` 实验在 NCCL communicator 初始化阶段停止：日志没有出现 `Init COMPLETE`，各 GPU 仅占用约 645MiB，未进入模型权重加载。由于该实验没有形成可用服务，因此没有吞吐数据。

`concurrency-1-4/partition-17-17-17-13-startup-failed.log` 记录的是另一个实验：`TP=1, PP=4` 下手工改层分布后，混合注意力 KV group 与 Graph warmup 不匹配，报 `expected 3 block tables, got 4`。它不是 TP=4 的失败日志，两项不可混为一谈。

**因此**：
- `/goal` 问题 7「4-node TP tok/s」→ **无数据**。
- 问题 8「TP 通信开销 %」→ **无法计算**（TP 没跑起来）。
- 问题 5（DP vs TP）中的 TP 侧 → 空缺。

这不是隐瞒，是如实记录失败（符合 `/goal` 禁止「无原因跳过失败测试」）。

---

## 8. 长上下文与吞吐甜点（补充实测）

### 短请求吞吐甜点（1K 输入 / 128 输出）
来源：`context-262k/summary.csv`。

| 并发 | 输出吞吐 | 总 token 吞吐 | 平均 TTFT | 平均 TPOT |
|---:|---:|---:|---:|---:|
| 1 | 18.04 tok/s | 162 | 343 ms | 53.15 ms |
| 4 | 58.98 tok/s | 531 | 1.22 s | 58.68 ms |
| **8** | **105.31 tok/s** | **948** | 1.78 s | 62.31 ms |
| 16 | 96.00 tok/s | 864 | 9.51 s | 72.55 ms |

**峰值在并发 8（105 tok/s）**；并发 16 已饱和，吞吐倒退、TTFT 暴涨到 9.5s。这是本集群短请求的实用上限。

### 冷长上下文（单并发，32 输出）
| 输入长度 | TTFT | prefill 吞吐 | decode 速度 | E2E |
|---:|---:|---:|---:|---:|
| 32K | 5.39 s | 6,082 tok/s | 18.50 tok/s | 7.06 s |
| 128K | 27.95 s | 4,690 tok/s | 17.25 tok/s | 29.74 s |
| 262K | 79.82 s | 3,284 tok/s | 16.52 tok/s | 81.70 s |

- 262K 边界（262,112 输入 + 32 输出 = 262,144 上限）**完整成功，无截断/OOM**。
- decode 速度随 context 缓慢下降（18.5→16.5 tok/s），符合 attention 随 KV 增长的预期。
- prefill 吞吐随 context 下降（6082→3284 tok/s），TTFT 显著变长。

### 满 context 并发压力（每请求 262K）
| 并发 | 平均 TTFT | 平均 TPOT | Preemption |
|---:|---:|---:|---:|
| 1 | 79.82 s | 60.52 ms | 0 |
| 2 | 156.06 s | 1652.93 ms | 1 |
| 4 | 293.45 s | 2468.84 ms | 3 |

KV pool 静态估算 4.64× 262K 并发，但运行时 KV 利用率逼近 99% 触发抢占重算。**结论：4.64× 是容量估算，不是可接受延迟下的并发能力。** 接近 262K 的请求应用层并发上限设 1。

---

## 9. 瓶颈定位（按 /goal §16 判据）

结合 GPU/网络遥测与上述数据：

| 场景 | 瓶颈类型 | 判据 |
|------|---------|------|
| 低并发 decode（C1~C4，1K ctx） | **流水线结构 + 跨节点延迟** | GPU 利用率不满、RDMA 流量小、TPOT 稳定在 ~48-51ms。单请求延迟由 4 卡串行 + 3 次 RDMA hop 决定，不是单卡 compute。对应 `/goal` 的 Case D 变体（结构/调度型）。 |
| 短请求高并发（C8） | **计算/调度趋于饱和** | C8→C16 吞吐倒退、TTFT 暴涨，GPU 被打满。对应 Case A（compute bound）。 |
| 长 context prefill | **attention + KV 带宽** | prefill 吞吐随 ctx 下降，TTFT 线性增长。对应 Case B（memory-bandwidth bound）。 |
| 满 context 并发 | **KV cache 容量** | KV 利用率近 99% 触发 preemption。KV 是硬约束。 |
| PP 跨节点通信 | **不是瓶颈** | NCCL 走 NET/IB + GPUDirect，PP 每层只传一份 activation（点对点，非 all-reduce），量小；RDMA 未见饱和。 |

**关于 200G RDMA 的价值（`/goal` 问题 9）**：
在 **PP=4** 模式下，RDMA 承载的是「层边界 activation 的点对点传递」，数据量远小于 TP 的逐 token all-reduce，200G 带宽**绰绰有余、不是瓶颈**。RDMA 的真正压力测试（TP=4 的逐 step all-reduce）因 TP 未跑起来而**未能验证**。所以对「200G RDMA 在 TP 下是否成瓶颈」这个问题，**目前无法给出实测答案**——只能说 DP/PP 路线下它不是瓶颈。

---

## 10. 最终推荐与结论

### 当前已验证的生产配置（KEEP）
```
Model:         Qwen/Qwen3.8-27B-FP8
Quantization:  FP8（权重）+ FP8 KV cache
Runtime:       vLLM 0.28.1rc1（Ray 后端，CUDA 13.0，sm_120）
Distribution:  TP=1, PP=4（4 节点各 16 层）
MTP:           OFF（拓扑不支持，用 CUDA Graph 替代增益）
CUDA Graph:    ON，自动 capture 1/2/4（历史 eager A/B：单并发 +10.6%）
Context:       262144（已通过边界验证）
Scheduler:     max sequences 4，max batched tokens 2048
并发策略:      普通请求最大 4；接近 262K 的请求限并发 1
Prefix cache:  ON
```

### 对 `/goal` 10 问的最终回答
1. **最佳量化模型**：Qwen3.8-27B-FP8（当前唯一稳定可用；NVFP4 无权重待补）。
2. **最佳 runtime**：vLLM nightly。
3. **最佳 MTP**：当前无法启用（PP 与 MTP 互斥）；CUDA Graph 提供 +10.6% 替代。
4. **单请求 decode**：20.6 tok/s（1K/512, Graph）；短请求 18 tok/s。
5. **聚合吞吐**：当前 1K/512 并发4 = 76.1 tok/s。历史上限扫描曾覆盖并发 8，但不属于当前运行范围。
6. **DP scaling efficiency**：不适用（本集群是 PP=4 单副本，非 DP=4）。PP 并发扩展实测 C4=3.69×C1、C2=1.98×C1，接近线性。
7. **4-node TP**：TP=4 启动失败，无数据。
8. **TP 通信开销**：未能测量。
9. **200G RDMA 是否瓶颈**：PP=4 下**否**；TP 下未验证。
10. **最终推荐**：见上方配置块。

### 与 `/goal` 验收目标的对照
| 目标 | 目标值 | 实测 | 是否达成 |
|------|-------|------|---------|
| 单请求 decode | > 50 tok/s | 20.6 tok/s | ❌ 未达（因 PP 跨节点串行，非单卡） |
| 聚合吞吐 | > 200 tok/s | 当前目标范围 C4 为 76.1 output tok/s、228.2 total tok/s | ⚠️ output tok/s 未达 200 |

**为什么单请求没到 50 tok/s**：`/goal` 的 50 tok/s 目标基于「单卡独立副本（DP）」假设。实际因显存必须 PP=4，单请求要串行穿过 4 张卡 + 3 次跨节点 hop，延迟被结构性拉长。**这不是 GPU 慢，是架构约束。**

### 下一步优化优先级（要突破，必须改架构前提）
1. **【最高价值】获取能塞进 24GB 的量化权重**（NVFP4 或 W4A16，约 ≤14GiB）。一旦可用：
   - 可切到**真正的 DP=4**（4 卡各一副本）→ 单请求走单卡，decode 有望冲 50-60 tok/s，聚合 200+ tok/s（`/goal` 原目标才成立）；
   - 单卡/TP 下可**解锁 MTP** → 再叠加 decode 增益；
   - 这一步同时解开问题 3、4、5、6 的死结。
2. **解决跨节点 TP=4 的 NCCL 初始化**（ARNIC provider + sm_120），才能回答问题 7、8、9 的 TP 侧，量化 200G RDMA 在 all-reduce 下的真实开销。
3. 在有权重前，PP=4 路线下的参数调优空间已近枯竭（§4 证明批处理参数不敏感），继续调 batch/graph 收益 < 3%，不值得。

### 诚实边界（本报告不做的事）
- 不把 PP=4 谎称 DP=4，不把总 token 吞吐谎称 output 吞吐。
- 不用理论值填 NVFP4 / MTP / TP 的空缺——这些是**待测**，不是**已知**。
- 所有数字均来自 3 次以上或有 warmup 的 `vllm bench serve` 实测 JSON，原始文件路径见 §11。

---

## 11. 数据与文件索引

| 内容 | 路径 |
|------|------|
| 汇总 CSV（本次生成） | `results/final/benchmark.csv` |
| 本报告 | `results/final/report.md` |
| 并发 1/2/4 原始 JSON | `logs/cluster/benchmarks/concurrency-1-4/*.json` |
| CUDA Graph A/B + MTP/TP 记录 | `logs/cluster/benchmarks/spec-decode/README.md` + `*.json` |
| 262K 长上下文报告 | `logs/cluster/benchmarks/context-262k/README.md` + `summary.csv` |
| 自定义 PP 分层失败日志 | `logs/cluster/benchmarks/concurrency-1-4/partition-17-17-17-13-startup-failed.log` |
| TP=4 失败结论 | 启动时日志观测；当前目录没有单独归档的 TP=4 日志文件 |
| 集群配置 | `cluster.env` |
| 当前环境搭建与运行手册 | `四节点 Blackwell vLLM 环境搭建与运行手册.md` |
| 部署演进说明 | `使用说明.md` |
| 模型配置 | `models/Qwen_Qwen3.8-27B-FP8/config.json` |

### 复现关键命令
```bash
# 服务已在运行；健康检查
curl -s http://<NODE0_IP>:8000/v1/models

# 并发 1/2/4 基准（1K 输入 / 512 输出）
vllm bench serve --model qwen3.8-27b \
  --base-url http://<NODE0_IP>:8000 \
  --dataset-name random --random-input-len 1024 --random-output-len 512 \
  --ignore-eos --max-concurrency 1 --num-prompts 4   # 改 2/8, 4/12

# 集群运维
./scripts/vllm_cluster_ctl.sh status
./scripts/vllm_cluster_service.sh logs
```
