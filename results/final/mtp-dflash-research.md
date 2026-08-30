# TP=4 基础上启用 MTP / DFlash 的可行性研究

> 研究日期：2026-08-29
> 方法：联网调研官方 vLLM recipe / 模型卡 / 独立 benchmark + 本地环境核实
> 核心问题：在已跑通的 TP=4 上启用 MTP 或 DFlash，能否提升 TPOT？能否同时改善 TTFT？

---

## 0. 直接回答你的两个问题

| 问题 | 结论 |
|------|------|
| **能否提升 TPOT（decode 加速）？** | ✅ **能，且提升显著**。这正是投机解码的设计目的。同系列 Qwen3.8-27B 官方/第三方数据：原生 MTP 约 **2.59×**，DFlash2 约 **3.43×**（并发1、H200）。你当前 TP=4 的 TPOT=23ms，理论上 MTP 可降到 **~9-11ms**，DFlash 可到 **~7ms**。 |
| **能否同时改善 TTFT（prefill 加速）？** | ❌ **不能**。投机解码**只加速 decode 阶段，对 prefill/TTFT 没有帮助**——这是所有权威来源的一致结论。TTFT 由 prefill 计算量决定，MTP/DFlash 不改变 prefill。 |

**一句话**：MTP/DFlash 能把 TPOT 打下来一大截（decode 快 2-3 倍），但 **TTFT 不会因此变快**。想同时降 TTFT 要用别的手段（见 §5）。

---

## 1. 为什么 TPOT 能提升，TTFT 不能

投机解码（speculative decoding）的原理：
- **Decode 阶段**：用一个轻量"草稿"机制一次猜多个 token，再由主模型**一次 forward 批量验证**。若猜中 k 个，就用「1 次主模型 forward」产出了「k+1 个 token」→ 每 token 的 wall-clock（TPOT）被摊薄。decode 在低并发下是 **memory-bandwidth bound**（反复读权重），投机把多个 token 的验证合并成一次读权重，正好击中瓶颈。
- **Prefill 阶段**：处理输入 prompt 本来就是**一次性并行**处理所有输入 token（compute bound），不存在"逐 token"问题，投机解码无从加速。**TTFT = prefill 时间，不变。**

> 权威原文（Jetson AI Lab / DFlash 模型卡）：*"speculative decoding accelerates the decode phase, not prefill/TTFT."*

---

## 2. 本地环境核实：具备启用条件（关键优势）

| 检查项 | 结果 | 含义 |
|--------|------|------|
| vLLM 版本 | 0.28.1rc1.dev43 | — |
| 支持的 method | ✅ `mtp` `qwen3_5_mtp` `qwen3_next_mtp` `dflash` `deepseek_mtp` `eagle3` `ngram` | **MTP 和 DFlash 都已内置**，无需打 PR |
| `use_dflash` 标志 | ✅ 存在于 SpeculativeConfig | DFlash 代码已在你的 nightly |
| 模型自带 MTP 权重 | ✅ config 有 `mtp_num_hidden_layers:1` + `mtp.layers.0.*`/`mtp.fc`/`mtp.norm` | **原生 MTP head 就在 checkpoint 里，不用额外下载** |
| DFlash 草稿模型 | ❌ 本地无 `incoai/Qwen3.8-27B-DFlash2` | 用 DFlash 需联网下载草稿模型（环境目前离线） |

**结论**：**MTP 现在就能试**（权重齐全、method 内置、离线可用）。**DFlash 需要先解决草稿模型下载**（约几 GB，需临时联网或手动拷入）。

---

## 3. 之前为什么 MTP 被判"不可用"——现在前提已变

旧文档记录 MTP 启动报 `NotImplementedError: Pipeline parallelism is not supported for this model`。**根因是 PP=4**：`Qwen3_5MTP` runner 不支持 pipeline parallel。

但你现在已经切到 **TP=4, PP=1**——**PP 的限制不再存在**。而官方 vLLM recipe 明确展示 MTP 与 tensor parallel 一起用：

```bash
# 官方 Qwen3-Next recipe（TP=4 + MTP）
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}' \
  --tensor-parallel-size 4 --no-enable-chunked-prefill

# 官方 Qwen3.6-27B recipe（同系列，dense 27B）
vllm serve Qwen/Qwen3.6-27B-FP8 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --reasoning-parser qwen3
```

**所以"MTP 不可用"是 PP 时代的结论，在 TP=4 下大概率被解锁。** 需要实测确认。

---

## 4. MTP vs DFlash：选哪个

| 维度 | 原生 MTP | DFlash2 |
|------|---------|---------|
| decode 加速（并发1，Qwen3.8-27B） | ~2.59× | **~3.43×**（更快） |
| 接受长度（accepted tokens/step） | 较低 | **更高**（所有测试任务胜出） |
| 本地就绪 | ✅ 权重自带，立即可用 | ❌ 需下载草稿模型 |
| 显存开销 | 小（1 层 MTP head） | 更大（额外草稿模型 + block draft） |
| 输出质量 | 无损 | 无损（构造上 lossless） |
| dense 模型上的表现 | vLLM AMD 研究：**dense Qwen 上原生 MTP 有时反超 DFlash** | MoE 上 DFlash 更强 |
| 你的模型类型 | Qwen3.8-27B 是 **dense** → MTP 可能更划算 | — |

**关键权衡**：
- DFlash 的 3.43× 是 **H200 + SGLang + 并发1** 的数字，**并发升高优势缩水**：并发32 时只剩 1.0-1.45×。你的场景是**并发 1-4**，正好落在投机解码的甜区。
- 但 vLLM 官方 AMD 研究发现：**dense Qwen 模型（如 Qwen3.6-27B）上，原生 MTP 峰值有时高于 DFlash**。你的 27B 是 dense，不是 MoE，所以 **MTP 未必输给 DFlash**。
- `num_speculative_tokens` 要调：DFlash 甜点 7-8，到 15 会让 27B 接受率崩到 12%；MTP 从 1 起步。

**推荐路线**：
1. **先测原生 MTP**（本地就绪、dense 上有优势、官方 recipe 直接支持）——这是性价比最高的第一步。
2. MTP 数据出来后，若想追更高 decode，再考虑联网下 DFlash2 草稿模型做对照。

---

## 5. 如果你也想降 TTFT（投机解码帮不上，用这些）

TTFT 由 prefill 决定。可单独优化（与 MTP 正交，可叠加）：

| 手段 | 对 TTFT 的作用 | 备注 |
|------|--------------|------|
| **TP=4 本身** | 已经帮了：TTFT 中位从 PP4 的 320ms 降到 **179ms**（prefill 也 4 卡并行） | 已生效 |
| **提高 max_num_batched_tokens** | 长 prompt 的 prefill 分块更少 → TTFT↓ | 你现在 2048，之前测过 8192 会让 32K TTFT 恶化，需针对目标 context 重调 |
| **prefix caching** | 有共享前缀（system prompt）时命中缓存，跳过重复 prefill → TTFT 大降 | 已启用；仅对有共享前缀的流量有效 |
| **chunked prefill 开关** | 影响 prefill 与 decode 的调度权衡 | ⚠️ 注意：MTP 官方 recipe 常配 `--no-enable-chunked-prefill`，两者可能冲突，需实测 |

**重要冲突提示**：官方 Qwen3-Next MTP recipe 用了 `--no-enable-chunked-prefill`，而你当前配置开着 `--enable-chunked-prefill`。启用 MTP 时可能需要关掉 chunked prefill，这会**反过来影响长 prompt 的 TTFT**。这是一个需要实测权衡的点。

---

## 6. 建议的实测方案（单变量、可对比）

固定 TP=4 + 1K输入/512输出，与现有 TP=4 baseline（TPOT 23ms、42 tok/s）对比：

```bash
# 步骤1：在 cluster.env 设 SPECULATIVE_CONFIG，逐级测 num_speculative_tokens
# MTP n=1（官方 Qwen3.6-27B 默认，最稳）
SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}'
# 若脚本注入 chunked-prefill 与 MTP 冲突，临时设 ENABLE 相关开关

# 步骤2：依次测 n=1 → 2 → 3，每档跑并发 1/2/4
# 记录：TPOT、accepted tokens、acceptance rate、decode tok/s、GPU util、显存
# 计算 speedup = MTP_tok/s / 42.2（TP=4 baseline）

# 步骤3：选"最高稳定 throughput"，不是简单选 n 最大
#   若 n=3 接受率明显下降或显存/延迟波动大，优先 n=2（遵循 goal §12.2）
```

**预期结果**（基于调研，需实测验证）：
- TPOT：23ms → 约 **9-14ms**（取决于接受率，MTP 约 1.6-2.6×）
- decode tok/s：42 → 约 **70-110 tok/s**（单请求）
- TTFT：**基本不变**（179ms 附近），甚至因关 chunked-prefill 在长 prompt 上略升
- 显存：MTP head 很小，KV 略降，应无 OOM 风险

**验收目标关联**：TP=4 baseline 单请求已 42 tok/s，MTP 若达 1.5× 即 **63 tok/s，稳超 goal 的 50 tok/s"优秀"线**，还没算并发聚合。

---

## 7. 风险与注意

1. **实测才算数**：以上加速倍数来自 H200/SGLang/其他模型，你的 RTX PRO 4000 + vLLM + ARNIC 必须实测，遵循 goal 的"单变量 + 3 次有效测试 + 报告 median 不只报 peak"。
2. **chunked prefill 冲突**：MTP 可能要求关闭它，会改变长 prompt TTFT 行为，需单独评估。
3. **DFlash 需联网**：草稿模型不在本地，且当前 `HF_HUB_OFFLINE=1`；要试得先临时放开下载或手动拷入。
4. **接受率是关键变量**：`num_speculative_tokens` 过大反而降接受率（27B + DFlash n=15 → 12% 接受率）。从小值起步。
5. **并发升高收益递减**：你在并发 1-4（甜区）没问题；若未来上高并发，投机收益会缩水。

---

## 8. 参考来源

- [MTP (Multi-Token Prediction) — vLLM 官方文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/)
- [Qwen3.6-27B — vLLM Recipes（同系列 dense 27B，MTP 配置）](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
- [recipes/Qwen/Qwen3-Next.md — vLLM 官方（MTP + TP=4 命令）](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3-Next.md)
- [z-lab/Qwen3.8-27B-DFlash2 — Hugging Face 模型卡](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
- [DFlash 2: Keep Drafting Parallel — Inco AI 技术博客](https://inco.ai/blog/dflash2/)
- [Speculative Decoding on Jetson: MTP, DFlash — Jetson AI Lab（TTFT 说明）](https://www.jetson-ai-lab.com/tutorials/speculative-decoding/)
- [Exploring Speculative Decoding on AMD GPUs — vLLM Blog（dense 上 MTP 反超 DFlash）](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus)
- [When Speculative Decoding Helps Local LLMs — Allen Kuo / Medium（并发缩水）](https://allenkuo.medium.com/when-speculative-decoding-helps-local-llms-and-when-it-doesnt-5c41dd804e4b)
