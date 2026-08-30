# NVFP4 量化模型搜索报告（HuggingFace）

> 日期：2026-08-29 | 目的：为 RTX PRO 4000 Blackwell（sm_120，支持 FP4）寻找 Qwen3.8-27B 的 NVFP4 权重
> 背景：goal 首选 NVFP4，但磁盘原本无此权重。NVFP4 可能解锁「单卡运行」+「Blackwell FP4 加速」。

---

## 0. 结论

**HuggingFace 上确实有多个 Qwen3.8-27B 的 NVFP4 量化（均为第三方，官方未发布 NVFP4）**，且多数能塞进单卡 24GB。但「单卡放下权重」≠「单卡可用」——还要留 KV cache 空间。详见 §2 的权衡分析。

**推荐候选**（按用途）：
- **要 MTP + 单卡** → `unsloth/Qwen3.8-27B-NVFP4`（23.4GB，保留 MTP，下载量最高）
- **要最小/最高质** → `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`（20.6GB，QAT 训练）
- **要 vLLM 生产稳定 + 大用户基数** → `RadixArk/Qwen3.8-27B-NVFP4`（21.9GB，NVIDIA ModelOpt，124万下载）

---

## 1. 候选对比表（实测 HF API 元数据）

| 模型 | 权重大小 | 单卡24GB放权重 | 含MTP | 下载量 | 量化方法 |
|------|---:|:---:|:---:|---:|------|
| **QUASAR-QAT/...QUASAR-NVFP4** | **20.6 GB** | ✅ | ❌ | 3.8k | QAT 量化感知训练，全线性层 NVFP4，号称最高质 |
| **RadixArk/Qwen3.8-27B-NVFP4** | 21.9 GB | ✅ | ❌ | **124万** | NVIDIA ModelOpt，MLP W4A4 group16，attn 用 FP8，MTP/vision 保 BF16 |
| **unsloth/Qwen3.8-27B-NVFP4** | 23.4 GB | ⚠️ 勉强 | ✅ | **209万** | Dynamic V3.0，168/496 线性层 NVFP4，attn/deltanet 多保 FP8，**保留 MTP** |
| Inferact/Qwen3.8-27B-NVFP4 | 26.4 GB | ❌ 超24GB | ✅ | 72万 | 304/496 线性层 NVFP4，deltanet 多保 BF16 |
| esatapedico/...NVFP4-MTP-GGUF | GGUF | — | ✅ | — | llama.cpp 路线（非 vLLM） |
| akopytko/...NVFP4-GGUF | GGUF | — | — | — | llama.cpp，报 ~1.6x、152 t/s @0.75 接受率 |

---

## 2. 关键权衡：单卡「放下权重」≠「可用」

这是必须提醒的核心问题：

- 单卡 24GB（实际可用 ~23.9GiB，×0.9 利用率 ≈ 21.5GiB）。
- **unsloth 23.4GB 权重 > 21.5GiB 可用**——单卡几乎放不下，或放下后 **KV cache 所剩无几**（放不了几 K context）。
- QUASAR 20.6GB / RadixArk 21.9GB 更宽裕，但单卡留给 KV 的也只有 ~1-3GiB，**长 context 受限**。

### 对比当前 FP8 多卡方案的 KV 空间
| 方案 | 权重/卡 | KV 空间 | 最大 context |
|------|---:|---:|---:|
| 当前 TP=4 FP8 | 7.1GiB/卡 | 12GiB/卡 → 1.5M tokens | 262K ✅ |
| NVFP4 单卡 | ~21GiB | ~1-3GiB | 可能只有几万 token |
| **NVFP4 TP=2** | ~11GiB/卡 | 更充裕 | 更长 |
| **NVFP4 TP=4** | ~5.5GiB/卡 | 最充裕 | 262K+ |

**所以 NVFP4 的最佳用法不一定是"单卡"**：
- **单卡 NVFP4**：适合短 context、追求单卡独立副本（真正的 DP=4，4 卡 = 4 个独立服务）。
- **NVFP4 + TP=4**：权重更小 → 每卡 compute 更省 → 可能比 FP8 TP=4 更快，且 KV 空间更大。这可能是**比当前 FP8 更优的生产方案**。

---

## 3. NVFP4 相对 FP8 的潜在收益（理论，需实测）

1. **Blackwell FP4 算力**：RTX PRO 4000 是 sm_120，原生支持 FP4。NVFP4 W4A4 用 FP4 做矩阵乘，理论比 FP8 快（unsloth 称 ~1.5× vs BF16；vs FP8 增益需实测）。
2. **权重更小**：FP8 28.75GB → NVFP4 ~21GB，省 ~27% 显存，KV 空间更大。
3. **单卡可行性**：解锁 goal 原设想的「单卡 DP=4」——4 卡跑 4 个独立完整模型副本，聚合吞吐可能更高（无跨节点通信）。

**但有风险**：
- **精度**：NVFP4 W4A4 比 FP8 精度低，unsloth 称保留 92-97% top-1 准确率——需在你的任务上验证质量。
- **MTP 兼容**：只有 unsloth/Inferact 保留了 MTP 层；QUASAR/RadixArk 无 MTP（但可外挂 DFlash 草稿模型）。
- **vLLM 加载**：NVFP4 W4A4 在当前 vLLM nightly 的支持需实测（源码有 nvfp4 内核，但端到端加载要验证）。

---

## 4. 建议的下一步（若要试 NVFP4）

优先级排序：

1. **RadixArk/Qwen3.8-27B-NVFP4（21.9GB）** — 首选试验对象：
   - NVIDIA ModelOpt 官方量化工具，vLLM 兼容性最好；
   - 124万下载，社区验证充分；
   - 可配 DFlash 草稿模型补投机（无自带 MTP 不影响）。
2. 先测 **NVFP4 + TP=4**（和当前 FP8 TP=4 同拓扑，单变量对比 NVFP4 vs FP8 的速度/质量）。
3. 若 TP=4 有增益，再测 **NVFP4 单卡 DP=4**（4 独立副本，验证 goal 原始设想的聚合吞吐）。

### 下载命令（宿主临时联网，同当前 DFlash 流程）
```bash
docker exec -d vllm-ray-node bash -lc \
  "HF_HUB_OFFLINE=0 hf download RadixArk/Qwen3.8-27B-NVFP4 --local-dir /root/.cache/huggingface/nvfp4-radixark"
# 同步到 4 节点后，cluster.env 改 MODEL_ID 指向 NVFP4 路径，quantization 设 nvfp4/modelopt
```

---

## 5. 是否现在就做？

**取决于你的目标**：
- 若当前 DFlash n=7（单请求 78/temp0 96.5 tok/s，并发4 252）**已满足需求** → NVFP4 是锦上添花，可暂缓。
- 若想**冲更高单卡吞吐 / 验证 goal 的单卡 DP=4 / 追求更大聚合吞吐** → 值得下 RadixArk NVFP4 实测。
- NVFP4 W4A4 有**精度下降风险**，生产切换前必须在你的真实任务上做质量对比（不只是速度）。

---

## 6. 参考来源

- [unsloth/Qwen3.8-27B-NVFP4（23.4GB，保留MTP，209万下载）](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
- [RadixArk/Qwen3.8-27B-NVFP4（21.9GB，NVIDIA ModelOpt，124万下载）](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
- [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4（20.6GB，QAT，最小）](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)
- [Inferact/Qwen3.8-27B-NVFP4（26.4GB，超单卡）](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4)
- [esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF（llama.cpp路线）](https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF)
- [Qwen3.8 - Unsloth 官方文档](https://unsloth.ai/docs/models/qwen3.8)
