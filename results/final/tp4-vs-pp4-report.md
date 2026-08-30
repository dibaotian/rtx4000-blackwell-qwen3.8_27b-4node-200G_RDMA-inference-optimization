# TP=4 实测报告 — 推翻「TP=4 无法运行」的旧结论

> 测试日期：2026-08-29
> 结论摘要：**TP=4 在当前 4 节点 ARNIC RDMA 集群上可以稳定运行**，且单请求 decode 速度是 PP=4 的 **2 倍**。此前手册/报告中「TP=4 卡在 NCCL 初始化」的结论，经复测**不成立**。

---

## 0. 为什么重做这个实验

原 `使用说明.md` 和 `benchmark_summary` 记载：

> 「`TP=4, PP=1` 实验未能通过 NCCL communicator 初始化……GPU 仅约 645 MiB，未进入模型加载。」

但复查磁盘日志发现：**没有任何一份真正 `TP=4` 的完整启动日志留存**。当前 `server.log` 里唯一那句 `Tensor parallel size (4)` 其实是 PP=4 服务里 vLLM `ray_utils` 的一句**误导性 warning**（它把 world_size=4 说成 TP），那次启动实际是 `TP=1, PP=4` 且**成功**了。

从显存角度看，TP=4 本应比 PP=4 更容易：TP 把每层切 4 份，单卡权重只占约 7GiB（PP 每卡放整段 16 层约 8GiB）。因此「放不下」不可能是失败原因。唯一存疑的是**跨节点 all-reduce 能否在 ARNIC RDMA 上跑通**。于是重做实验，全程抓日志。

---

## 1. 实测结果：TP=4 完整启动并正确服务

单变量改动：仅把 `TENSOR_PARALLEL_SIZE=1, PIPELINE_PARALLEL_SIZE=4` 改为 `TENSOR_PARALLEL_SIZE=4, PIPELINE_PARALLEL_SIZE=1`，其余配置（FP8、FP8 KV、CUDA Graph、max_model_len 262144、max_num_seqs=4、max_num_batched_tokens=2048）全部不变。

启动全程无 error，关键里程碑（日志存档 `logs/cluster/benchmarks/tp4-test/tp4-startup-success.log`）：

| 阶段 | 结果 |
|------|------|
| NCCL communicator | ✅ 4 rank `ncclCommInitRank ... nranks 4 Init START`，`tp:0` group 建立 |
| all-reduce backend | ✅ `Using ['PYNCCL'] all-reduce backends for group 'tp:0'` |
| 网络路径 | ✅ `Using network IB` —— TP all-reduce 走 200G ARNIC RDMA |
| 模型加载 | ✅ `Model loading took 7.14 GiB`（每卡 1/4 权重，符合 TP 预期） |
| torch.compile | ✅ 约 42s |
| CUDA Graph capture | ✅ PIECEWISE + FULL 全部成功（含 all-reduce 融合） |
| KV cache | ✅ **1,505,068 tokens**（比 PP=4 的 1.19M 多 27%），262K 并发估算 **5.74x** |
| GPU 显存 | 4 节点各 21.6 GiB |
| API | ✅ ready，`17*19=323` 正确返回，0 error |

**这些阶段全部越过了旧文档声称「进不去」的那一步。** 事实是 TP=4 能完整跑起来。

---

## 2. 性能对比：TP=4 vs PP=4（同 workload，1K 输入 / 512 输出）

| 并发 | 指标 | PP=4（旧基线） | **TP=4（新实测）** | 变化 |
|---:|------|---:|---:|---:|
| **1** | 输出吞吐 tok/s | 20.6 | **42.2** | **+105%** |
| | TPOT ms | 47.97 | **23.38** | **−51%** |
| | TTFT(中位) ms | 320 | **179** | −44% |
| | E2E s | 24.83 | ~13（估算） | ~−48% |
| **2** | 输出吞吐 tok/s | 40.9 | **76.4** | **+87%** |
| | 每请求 tok/s | 20.5 | **38.2** | +86% |
| | TPOT ms | 48.10 | **25.5** | −47% |
| **4** | 输出吞吐 tok/s | 75.7 | **142.3** | **+88%** |
| | 每请求 tok/s | 18.9 | **35.6** | +88% |
| | TPOT ms | 51.0 | **27.4** | −46% |

原始 JSON：`logs/cluster/benchmarks/tp4-test/tp4-1k-512-c{1,2,4}.json`。

### 为什么 TP=4 单请求快一倍
- **TP=4**：4 张卡**并行**计算同一层（每卡算 1/4 的 attention/FFN），单 token 的 wall-clock 约等于「1/4 计算 + 1 次 all-reduce」→ TPOT 腰斩到 23ms。
- **PP=4**：4 张卡**串行**流水线，一个 token 要依次穿过 4 段 + 3 次跨节点 hop → TPOT 约 48ms。
- 单请求延迟敏感场景，**TP 明显优于 PP**。

---

## 3. 瓶颈定位：200G RDMA 是不是 TP 的瓶颈？

压测（并发4）期间遥测：

| 观测 | 值 | 含义 |
|------|-----|------|
| 4 节点 GPU 利用率 | **全部 98-99%** | 4 卡真正并行满载（PP=4 时同一刻仅 1 卡忙） |
| 4 节点功耗 | 98-123 W（上限 145W） | 计算密集，非空转等通信 |
| TP all-reduce 路径 | `Using network IB` + PYNCCL | 逐层 all-reduce 走 200G ARNIC RDMA |
| decode 吞吐随并发 | 42→76→142，接近线性 | 通信未拖垮扩展性 |

**判定（按 goal §23 Case B）：compute bound，不是 communication bound。**
- GPU 高（99%）、扩展接近线性 → 200G RDMA 带宽**足够**承载 TP 的逐层 all-reduce，**没有成为瓶颈**。
- 注：ARNIC 用自定义 userspace provider，标准 `/sys/class/infiniband` 字节计数器读不到精确 GB/s；但「GPU 99% + 近线性扩展 + all-reduce 走 IB」三条已足以判定通信非瓶颈。若要精确量化通信占比，需用 ARNIC provider 自带的 hw_counters 或 NCCL profiler，属后续工作。

**这正面回答了 goal 问题 9**：在本硬件上，200G RDMA 对 TP=4 decode **不是瓶颈**——这是之前因 TP 跑不起来而无法回答的问题。

---

## 4. TP=4 vs PP=4：如何选择

| 维度 | TP=4 | PP=4 |
|------|------|------|
| 单请求 decode | **42 tok/s（快 2x）** | 20.6 tok/s |
| 低延迟场景 | **✅ 首选** | 劣势 |
| KV cache 容量 | **1.50M tokens（+27%）** | 1.19M tokens |
| 4 卡利用率 | **并行满载 99%** | 流水线，单刻 1 卡忙 |
| 聚合吞吐(并发4) | **142 tok/s** | 75.7 tok/s |
| 通信模式 | 每层 all-reduce（走 RDMA，但未成瓶颈） | 每段 activation 点对点（量更小） |
| 长 prompt prefill | 待测 | 已验证 262K |
| 稳定性 | 本次单轮验证通过 | 长期验证充分 |

**结论**：在这个「4 节点 × 单卡 + 200G RDMA」拓扑上，**TP=4 是比 PP=4 更好的 decode 配置**——单请求快一倍、KV 更大、4 卡真正并行。之前认为「单卡一节点必须用 PP」的前提，被 ARNIC RDMA 足够好的 all-reduce 性能打破了。

---

## 5. 对 goal 核心问题的更新回答

| # | 问题 | 旧结论 | **新结论（实测）** |
|---|------|--------|------|
| 4 | 单请求 decode | 20.6 tok/s（PP=4） | **42.2 tok/s（TP=4）** |
| 5 | 聚合吞吐(并发4) | 75.7 tok/s | **142.3 tok/s（TP=4）** |
| 7 | 4-node TP tok/s | 「启动失败，无数据」 | **c1=42.2 / c2=76.4 / c4=142.3 tok/s** |
| 8 | TP 通信开销 | 「无法测量」 | **非瓶颈**（GPU 99%，近线性扩展；精确 % 待 hw_counter） |
| 9 | 200G RDMA 是否瓶颈 | 「TP 下未验证」 | **否**（compute bound） |
| 验收 | 单请求 > 50 tok/s | 未达（20.6） | **接近达成**：TP=4 单请求 42，并发2 每请求 38；若叠加优化有望破 50 |

---

## 6. 后续可继续优化（都在 TP=4 基础上）

1. **提高 max_num_seqs / 并发**：TP=4 并发4 才 142 tok/s，GPU 已 99%，但可测更高并发看聚合上限（goal 限定只测 1/2/4，此为超出项）。
2. **精确量化 RDMA 通信占比**：用 ARNIC hw_counters 或 `NCCL profiler` 抓 all-reduce 字节数/耗时，回答「通信占单 step 多少 %」。
3. **TP=4 长上下文回归**：补 32K/128K/262K，确认 TP 下 KV 更大（1.5M）是否让长 context 并发更宽松。
4. **TP=4 + MTP**：TP 不受 PP 的限制，`Qwen3_5MTP` 可能可在 TP 下启用——这是解锁 MTP 的现实路径，值得单独试。
5. **修脚本**：`vllm_cluster_ctl.sh` 第 19-20 行硬性要求 `节点数==PP`，导致 `sync-files` 拒绝 TP=4 配置。应放宽为 `节点数 == TP×PP` 才能用标准编排跑 TP=4。

---

## 7. 复现方法

```bash
cd /data/vllm
cp cluster.env cluster.env.bak
# 改两行：TENSOR_PARALLEL_SIZE=4  /  PIPELINE_PARALLEL_SIZE=1
# 把 cluster.env scp 到 3 个 worker（绕过 ctl.sh 的 PP==节点数 校验）
for ip in <NODE1_IP> <NODE2_IP> <NODE3_IP>; do scp cluster.env $ip:/data/vllm/cluster.env; done
./scripts/vllm_cluster_service.sh start   # Ray 已在跑，直接起 vLLM
# benchmark（注意用本地路径做 tokenizer，环境离线）
docker exec vllm-ray-node vllm bench serve \
  --model /models/Qwen_Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b \
  --tokenizer /models/Qwen_Qwen3.8-27B-FP8 \
  --base-url http://<NODE0_IP>:8000 --endpoint /v1/completions \
  --dataset-name random --random-input-len 1024 --random-output-len 512 \
  --ignore-eos --num-prompts 4 --max-concurrency 1
```

### 存档文件
- `logs/cluster/benchmarks/tp4-test/tp4-startup-success.log` — TP=4 完整成功启动日志
- `logs/cluster/benchmarks/tp4-test/tp4-1k-512-c{1,2,4}.json` — 三组 benchmark 原始 JSON
- `logs/cluster/benchmarks/tp4-test/pp4-baseline-server.log` — 对照的 PP=4 启动日志
- `results/final/benchmark.csv` — 含 `tp4-actual` 三行
