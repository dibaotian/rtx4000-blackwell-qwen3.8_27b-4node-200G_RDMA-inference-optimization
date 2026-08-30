# NCCL 双网卡策略详解 — 为什么两个 200G 口都有流量

> 日期：2026-08-30 | 硬件：每节点 1 GPU + 2× 200G ARNIC RoCE v2 网卡
> 配置：NVFP4 + TP=4，NCCL 2.30.7，vLLM + Ray
> 本文回答：**每台机器只有 1 张 GPU，为什么 RDMA 的两个 200G 口都有流量？这正常吗？**

---

## 0. 一句话结论

**正常，而且是刻意配置的最优状态。** 每节点有 2 张独立 200G 网卡（合计 400G），NCCL 被配置为**同时使用两块卡做负载均衡**——把跨节点 all-reduce 通信分摊到两条链路上。两个口流量近 1:1（实测 1734 vs 1730 MB/s）正是配置生效的证明。**GPU 数量和网卡数量不需要 1:1。**

---

## 1. 先厘清硬件：两个口是两张独立网卡

```
arnic_0 → PCIe 81:00.0 → netdev ens7       （独立网卡 A，200G）
arnic_1 → PCIe 82:00.0 → netdev enp130s0   （独立网卡 B，200G）
```

- 不是"一张双口卡"，而是**两块独立的 RDMA 网卡**，各占一个 PCIe 槽。
- 合计 **400G** 网络能力。
- 两块都在 **NUMA node 0**，和 GPU（NUMA 0）同一节点 → 数据流不跨 NUMA，延迟最优。

---

## 2. 为什么单 GPU 也需要跨节点通信

关键误区：**"1 GPU 就不用网络" 是错的。**

当前是 **TP=4（张量并行）**：Qwen3.8-27B 的每一层被**切成 4 份**，4 个节点的 GPU 各算 1/4。但每算完一层，4 张 GPU 必须**交换并汇总**中间结果（all-reduce），才能继续下一层。

```
每个 decode step，每一层：
  4 张 GPU 各算自己的 1/4
        ↓
  all-reduce（跨 4 节点交换 + 求和）  ← 走 RDMA 网络
        ↓
  下一层……
```

所以虽然每节点 1 GPU，但 TP=4 让这张 GPU **频繁和其他 3 节点通信**。通信量不小，用两块网卡分摊正合适。

---

## 3. NCCL 如何调度这两块网卡（核心机制）

### 3.1 配置：明确告诉 NCCL 用两块卡
`cluster.env` 里：
```bash
NCCL_IB_HCA=arnic_0,arnic_1     # 两块 HCA 都用
NCCL_CROSS_NIC=1                # 允许跨网卡通信路径
NCCL_IB_GID_INDEX=1             # RoCE v2 的 GID
NCCL_SOCKET_IFNAME=ens7,enp130s0  # 带外通信（建链）用的网卡
```

### 3.2 NCCL 实际做了什么（服务日志实测）

**① 每个 rank 识别到 2 块网卡：**
```
Rank 0: 2 Net devices
Local Net device counts across ranks: min 2 max 2   ← 4 节点都是 2 块
NET/IB : Using [0]arnic_0:1/RoCE [1]arnic_1:1/RoCE   ← 两块都用
```

**② 建立 4 个通信通道（Channel），分摊到两块网卡：**
```
Channel 00/04 : 0 1 2 3
Channel 01/04 : 0 1 2 3
Channel 02/04 : 0 1 2 3
Channel 03/04 : 0 1 2 3
```
NCCL 把 all-reduce 拆成 **4 个并行 channel**，这些 channel 被分配到 arnic_0 和 arnic_1 两块网卡上并行传输 → **两块卡同时有流量**。

**③ 用 Tree 算法组织 4 节点的通信拓扑：**
```
Trees [0] 2/-1/-1->0->-1  [1] 2/-1/-1->0->-1  [2] -1/-1/-1->0->1  [3] -1/-1/-1->0->1
```
这是 NCCL 为 4 节点构建的**通信树**（每个 channel 一棵），决定数据在节点间的流动路径。不同 channel 的树结构不同，配合双网卡进一步分散负载。

### 3.3 一句话概括机制
```
all-reduce  →  拆成 4 个 channel  →  分摊到 2 块网卡并行传输  →  两口都有流量
```

`NCCL_CROSS_NIC=1` 允许一个通信路径的收发用不同网卡，进一步提升双卡利用率。

---

## 4. 为什么两口流量近 1:1 是健康的

实测（并发4 decode，每节点）：
| 节点 | arnic_0 TX | arnic_1 TX | 均衡度 |
|------|---:|---:|---:|
| node0 | 1734 MB/s | 1730 MB/s | 99.8% |
| node1 | 1753 | 1750 | 99.8% |
| node2 | 1774 | 1771 | 99.8% |
| node3 | 1793 | 1790 | 99.8% |

**两块网卡流量几乎完全相等** = NCCL 把 channel 均匀分配到两块卡，负载完美均衡。

| 如果只有一个口有流量 | 你现在两口均衡 |
|------|------|
| ❌ 浪费一半网络（只用 200G） | ✅ 400G 都用上 |
| 单卡可能先饱和成瓶颈 | ✅ 负载分摊，余量翻倍 |
| 可能是配置丢了 arnic_1 或网卡 down | ✅ 配置正确生效 |

---

## 5. 数据路径：GPUDirect RDMA（额外优化）

除了双网卡，你的配置还启用了 **GPUDirect RDMA**（日志实测）：
```
DMA-BUF is available on GPU device 0     ← GPU 数据可直接进网卡
nvidia_peermem 模块已加载                 ← GPUDirect 内核依赖就绪
```

意味着 all-reduce 时：
```
GPU 显存 ──直接 DMA──> 网卡 ──> 对端网卡 ──直接 DMA──> 对端 GPU 显存
         （不经过 CPU 内存中转）
```
延迟更低、CPU 开销更小。这是 RDMA + 双网卡之外的第三层优化。

---

## 6. 完整健康检查清单

| 检查项 | 实测 | 状态 |
|--------|------|------|
| 两块网卡都 ACTIVE | arnic_0/1 both LINK_UP | ✅ |
| 每 rank 识别 2 块网卡 | min 2 max 2 | ✅ |
| NCCL 走 RDMA（非 TCP） | NET/IB : Using arnic_0 + arnic_1 | ✅ |
| 双网卡流量均衡 | 两口 99.8% 相等 | ✅ |
| GPUDirect RDMA | DMA-BUF available | ✅ |
| nvidia_peermem | 已加载 | ✅ |
| GPU/网卡同 NUMA | 都在 NUMA 0 | ✅ |
| 通信通道 | 4 channel，Tree 算法 | ✅ |

**全绿 = 双网卡配置教科书级正确。**

---

## 7. 故障排查：什么情况才算异常

以下才是需要处理的问题（当前都正常）：

### 7.1 只有一个口有流量（另一口 0）
- 检查网卡链路：`rdma link`（两个都应 ACTIVE）
- 检查 NCCL 配置：`NCCL_IB_HCA` 是否还是 `arnic_0,arnic_1`（丢了一个会只用一块）
- 检查网卡是否 down：`ip link show ens7 / enp130s0`

### 7.2 NCCL 退回 TCP（NET/Socket 而非 NET/IB）
日志出现 `via NET/Socket` = RDMA 没生效，退回慢速 TCP。检查：
- ARNIC provider 挂载（`libarnic-rdmav34.so`）
- `nvidia_peermem` 是否加载
- GID index 是否为 1

### 7.3 两口流量严重不均衡（如 90% vs 10%）
- 可能一块网卡 PCIe 降速、或 NUMA 不亲和
- 检查 `NCCL_CROSS_NIC=1` 是否设置
- 用 `scripts/rdma_monitor.sh all 2` 观察两口实时流量

### 7.4 监控命令
```bash
# 实时看两块网卡流量是否均衡
./scripts/rdma_monitor.sh all 2

# 看 NCCL 是否用双 HCA
docker exec vllm-ray-node grep 'NET/IB : Using' /var/log/vllm/server.log
```

---

## 8. 延伸：这个配置对更大规模的意义

- **当前 Qwen3.8-27B TP=4**：双网卡利用率 <13%（见 `rdma-full-analysis.md`），400G 其实过剩。
- **但双网卡不是浪费**：更大模型、更高并发、或 TP 规模扩大时，通信量增加，双网卡的余量就是扩展空间。
- **单网卡 vs 双网卡的价值**：不在于当前是否用满，而在于消除"单条链路成为瓶颈"的可能，以及负载均衡带来的更稳定延迟。

---

## 9. 相关文档
- [rdma-full-analysis.md](rdma-full-analysis.md) — RDMA 全面性能分析
- [rdma-monitoring-guide.md](rdma-monitoring-guide.md) — 监控脚本与计数器
- [scripts/rdma_monitor.sh](../../scripts/rdma_monitor.sh) — 实时监控（支持双网卡）
