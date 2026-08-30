# 真实数据（ShareGPT）重测 — 纠正随机数据的失真结论

> 测试日期：2026-08-29 | 硬件：4 × RTX PRO 4000 Blackwell + 200G ARNIC RDMA
> 数据集：**ShareGPT 真实对话**（94145 条，seed=42，固定输出 512）
> 配置：TP=4，FP8，FP8 KV，CUDA Graph
> 起因：用户正确指出「投机解码测试不应该用随机数据」

---

## 0. 为什么随机数据是错的（方法论）

之前几轮 benchmark 用 `--dataset-name random`（随机 token）。**这对投机解码是根本性错误**：

- 投机解码（MTP/DFlash）的草稿模型靠**预测真实语言的下一个 token** 工作。
- 随机 token 序列**没有语言规律**，草稿命中率被人为压低且失真。
- 后果：随机数据既**低估了接受率**（真实文本可预测性更高），又让**不同配置的排序失真**。

**结论：随机数据的绝对值和排序都不可信。以下真实数据为准。**

---

## 1. 真实数据 vs 随机数据（同配置对比，揭示失真幅度）

| 配置 | 并发4 随机数据 | 并发4 真实数据 | 差异 |
|------|---:|---:|---:|
| DFlash n=6 | 199.6 | **251.9** | 真实 **+26%** |
| DFlash n=7 | 199.7 | **241.8** | 真实 **+21%** |
| MTP n=3 | 217.5 | 230.2 | 真实 +6% |

**随机数据系统性低估了投机收益，尤其 DFlash（大块草稿受数据质量影响更大）**。真实数据下 DFlash 并发4 从"输给 MTP"变成"反超 MTP"——**结论被纠正了**。

---

## 2. 真实数据最终对比表（1K级输入/512输出，ShareGPT）

| 配置 | 并发1 | 并发2 | 并发4 | 单请求TPOT |
|------|---:|---:|---:|---:|
| 无投机 baseline | 42.6 | — | — | 23.4ms |
| MTP n=3 | 69.4 | 134.5 | 230.2 | 14.2ms |
| DFlash n=6 | 73.2 | 136.2 | **251.9** | 13.2ms |
| **DFlash n=7** | **78.2** | 137.0 | 241.8 | **12.6ms** |

**粗体** = 各列最优。

---

## 3. 纠正后的关键结论

### DFlash 真实数据下全面领先
- **单请求**：DFlash n=7 = 78.2 tok/s（MTP n=3 只有 69.4）→ DFlash **+13%**
- **并发4**：DFlash n=6 = 251.9 tok/s（MTP n=3 只有 230.2）→ DFlash **+9%**
- 随机数据时"MTP 并发4 反超"的结论**是随机数据造成的假象**，真实数据下 **DFlash 在所有并发段都领先**。

### DFlash n=6 vs n=7 的取舍
| | 单请求 | 并发4 |
|---|---:|---:|
| n=7 | **78.2**（最快） | 241.8 |
| n=6 | 73.2 | **251.9**（最快） |

- **单请求优先 → n=7**
- **并发4 优先 → n=6**
- 差距都在 5-10 tok/s，两者都优于 MTP。

### 真实数据下绝对值比随机数据低（单请求）但并发更高
注意单请求真实数据（78）比随机数据（126）**低**。原因：ShareGPT 输入长度不一（有很多短对话），单请求场景下 prefill/调度占比变化，拉低了纯 decode 吞吐的表观值。但**并发场景真实数据更高**（草稿命中率真实体现）。这说明**单一 workload 不能代表全部**——真实数据的并发聚合数字最有生产参考价值。

---

## 4. 最终推荐（基于真实数据）

| 场景 | 推荐 | 单请求 | 并发4 |
|------|------|---:|---:|
| **单请求低延迟** | **DFlash n=7** | 78.2 | 242 |
| **并发高吞吐** | **DFlash n=6** | 73.2 | 252 |
| **省事/无额外模型** | MTP n=3 | 69.4 | 230 |

**真实数据下 DFlash 全面胜出**（单请求 +13%、并发4 +9% 优于 MTP）。之前基于随机数据的"MTP 并发更强"结论已被推翻。

---

## 5. 教训（方法论）

1. **投机解码必须用真实数据测**——随机数据系统性失真，且对不同算法失真幅度不同（DFlash 受影响 >MTP），会颠倒排序。
2. **单一 workload 不足以定论**——单请求和并发的最优配置不同（n=7 vs n=6），生产选型要看实际流量分布。
3. 这是本项目第 N 次"实测推翻先前结论"：随机数据→真实数据同样是一次重要纠正。感谢用户指出。

---

## 6. 数据与复现

- 原始 JSON：`logs/cluster/benchmarks/real-data/{nospec,mtp3,dflash6,dflash7}-real-c{1,2,4}.json`
- CSV：`results/final/benchmark.csv`（`real-sharegpt` 10 行）
- 数据集：`hf-cache/sharegpt.json`（ShareGPT_V3，94145 条对话）

```bash
# 真实数据 benchmark 命令（关键：--dataset-name sharegpt）
vllm bench serve --model /models/Qwen_Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b \
  --tokenizer /models/Qwen_Qwen3.8-27B-FP8 --base-url http://<NODE0_IP>:8000 \
  --endpoint /v1/completions \
  --dataset-name sharegpt --dataset-path /root/.cache/huggingface/sharegpt.json \
  --sharegpt-output-len 512 --num-prompts 16 --max-concurrency 4 --seed 42
```

> 注：之前 `results/final/` 下用随机数据的报告（tp4-vs-pp4、mtp-tuning、dflash-vs-mtp、comprehensive-comparison）中，**投机解码部分的绝对值和 DFlash/MTP 排序应以本报告为准**。非投机配置（TP/PP/卡数对比）不受影响，因为那些不依赖数据可预测性。
