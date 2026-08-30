# DFlash2 vs MTP 实测对比报告（真实数据）

> 更新日期：2026-08-30 | 硬件：4 × RTX PRO 4000 Blackwell + 200G ARNIC RDMA
> 数据集：**ShareGPT 真实对话**（94145 条，seed=42，固定输出 512）
> 配置：TP=4，FP8 KV，CUDA Graph
> 草稿模型：`z-lab/Qwen3.8-27B-DFlash2`（3.58GB，已同步 4 节点）
>
> 注：本报告只保留真实数据（ShareGPT）的正确结论。早期随机数据的失真数字已删除。

---

## 0. 结论

**真实数据下 DFlash 全面领先 MTP**（单请求和并发都赢），但差距不大；NVFP4 上情形反转（MTP 单请求更快）。选型看量化和场景。

| 场景 | 最优 | 说明 |
|------|------|------|
| FP8，单请求低延迟 | **DFlash n=7**（78 tok/s） | 比 MTP n=3（69）快 13% |
| FP8，并发高吞吐 | **DFlash n=6**（并发4=252） | 比 MTP n=3（230）快 9% |
| **NVFP4，单请求** | **MTP n=3**（97 tok/s） | NVFP4 上 MTP 反超 DFlash（87） |
| **NVFP4，并发高吞吐** ⭐ | **DFlash n=7**（并发4=311） | 全项目最高聚合吞吐 |

⭐ = 当前生产配置（NVFP4 TP=4 + DFlash n=7）。

---

## 1. FP8 完整对比（ShareGPT 真实数据，默认采样）

| 配置 | 并发1 | 并发2 | 并发4 | 单请求TPOT |
|------|---:|---:|---:|---:|
| 无投机 | 42.6 | — | — | 23.4ms |
| MTP n=3 | 69.4 | 134.5 | 230.2 | 14.2ms |
| DFlash n=6 | 73.2 | 136.2 | **251.9** | 13.2ms |
| **DFlash n=7** | **78.2** | 137.0 | 241.8 | **12.6ms** |

**粗体** = 该列最优。

- **DFlash n=7 单请求最优**（78.2 tok/s），比 MTP n=3（69.4）快 13%。
- **DFlash n=6 并发4 最优**（251.9 tok/s），比 MTP n=3（230.2）快 9%。
- DFlash 在单请求和并发段**都领先** MTP（真实数据下）。

---

## 2. NVFP4 对比（ShareGPT 真实数据，temp=0）

| 配置 | 并发1 | 并发4 | 单请求TPOT |
|------|---:|---:|---:|
| 无投机 | 48.4 | — | 20.0ms |
| **MTP n=3** | **97.4** | 301.3 | **9.9ms** |
| DFlash n=7 | 86.6 | **311.0** | 11.1ms |

**关键反转**：NVFP4 上 **MTP 单请求（97.4）反超 DFlash（86.6）**，与 FP8 上相反。
- 原因：unsloth 的 NVFP4 checkpoint 里 MTP 层随权重一起 NVFP4 量化优化，匹配更好、接受率更高。
- DFlash 在 NVFP4 上并发4 仍最高（311），是全项目最高聚合吞吐。

---

## 3. MTP vs DFlash 的本质区别

| | MTP（原生多token预测） | DFlash2（block-diffusion 草稿） |
|---|---|---|
| 草稿方式 | 逐 token 顺序草稿 | 一次预测整块（block=8，最多7个草稿） |
| 接受率 | 高（草稿少而精） | 较低（草稿多而糙，靠"量"取胜） |
| 额外成本 | 0（权重自带 MTP 层） | 需 3.58GB 草稿模型 |
| FP8 表现 | 稍逊 DFlash | 单请求/并发都略优 |
| NVFP4 表现 | **单请求反超**（随权重量化优化） | 并发仍最优 |
| num_speculative_tokens | 1/2/3（n=3 最快） | n=7（block 上限，n=8 会启动失败） |

---

## 4. 选型建议

| 你的配置+场景 | 推荐 | 单请求 | 并发4 |
|---------|------|---:|---:|
| NVFP4 + 单请求低延迟 | **MTP n=3** | 97 | 301 |
| NVFP4 + 并发高吞吐 ⭐ | **DFlash n=7** | 87 | 311 |
| FP8 + 单请求 | DFlash n=7 | 78 | 242 |
| FP8 + 并发 | DFlash n=6 | 73 | 252 |

### 成本/运维
- **MTP**：0 额外成本，权重自带，最省事；NVFP4 上单请求还最快。
- **DFlash**：需额外 3.58GB 草稿模型，多一个依赖；并发吞吐最强。

### 综合
- **NVFP4（当前生产量化）下**：单请求选 MTP n=3，并发选 DFlash n=7。当前生产用 DFlash n=7（并发吞吐优先）。
- 两者都**远超 goal 的 50/60 tok/s 目标**。

---

## 5. 重要方法论：为什么必须用真实数据

投机解码靠预测真实语言的下一个 token 工作。**随机 token 数据会系统性压低接受率、颠倒排序**，绝不能用于投机解码 benchmark。

- 早期用随机数据曾误得出「DFlash 并发4 被 MTP 反超」——**是假象**。真实数据下 DFlash 并发4 反而领先。
- 随机数据对 DFlash（大块草稿）失真幅度（并发4 低估 21-26%）远大于 MTP（低估 6%）。
- 详见 `real-data-correction.md`。

---

## 6. 优化历程（真实数据）

```
PP=4（原始）              20.6 tok/s   1.0×
  ↓ TP=4
TP=4 无投机              42.6 tok/s   2.1×
  ↓ 投机 + 真实数据 temp=0
FP8 + DFlash n=7        96.5 tok/s   （随机数据曾虚报 126）
  ↓ NVFP4 量化
NVFP4 + MTP n=3         97.4 tok/s   单请求最优
NVFP4 + DFlash n=7      并发4 = 311   聚合吞吐最优 ⭐ 当前生产
```

---

## 7. 数据与复现

- 真实数据 JSON：`logs/cluster/benchmarks/real-data/`、`matrix/`
- CSV：`results/final/benchmark.csv`（`real-sharegpt`、`nvfp4` 行）
- 权威汇总：`comprehensive-comparison.md §0c`、`real-data-correction.md`
- 草稿模型：`hf-cache/dflash2-draft/`（4 节点已同步）

```bash
# 真实数据 benchmark（关键：--dataset-name sharegpt，不能用 random）
vllm bench serve --model <模型路径> --served-model-name qwen3.8-27b --tokenizer <模型路径> \
  --base-url http://<NODE0_IP>:8000 --endpoint /v1/completions \
  --dataset-name sharegpt --dataset-path /root/.cache/huggingface/sharegpt.json \
  --sharegpt-output-len 512 --num-prompts 16 --max-concurrency 4 --seed 42 --temperature 0

# DFlash 配置
SPECULATIVE_CONFIG='{"method":"dflash","model":"/root/.cache/huggingface/dflash2-draft","num_speculative_tokens":7}'
# MTP 配置（权重自带，无需下载）
SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":3}'
```
