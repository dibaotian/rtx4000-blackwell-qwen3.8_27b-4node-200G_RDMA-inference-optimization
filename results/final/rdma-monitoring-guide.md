# RDMA 监控与统计分析指南（ARNIC）

> 日期：2026-08-30 | 硬件：双 200G ARNIC RoCE v2（arnic_0 / arnic_1）
> 适用：TP=4 跨节点部署下监控 RDMA 通信，判断是否成为瓶颈

---

## 0. 为什么要监控 RDMA

当前生产是 **TP=4**，每个 decode step 都要跨 4 节点做 all-reduce。RDMA 通信直接影响性能。监控能回答：
- RDMA 是不是瓶颈？（利用率/延迟）
- 网络质量如何？（重传/丢包/拥塞）
- 通信模式是否正常？（包大小/速率）

**已有结论**：当前 200G RDMA 利用率极低（<10%），retx/drop 极少，**不是瓶颈**（compute-bound）。本指南教你持续监控确认这一点。

---

## 1. 快速使用

```bash
cd /data/vllm
./scripts/rdma_monitor.sh              # 监控 arnic_0，每2秒
./scripts/rdma_monitor.sh arnic_1 5    # arnic_1，每5秒
./scripts/rdma_monitor.sh all 2        # 两块网卡同时
```
输出表格：`TX_MB/s | RX_MB/s | Kpkt/s | util% | retx | drop | RTT_us`

---

## 2. 数据来源：ARNIC hw_counters

⚠️ ARNIC 是定制 provider，**标准 `port_xmit_data` 不可读**，但提供了自己的 `cs_*` 计数器（更丰富）：
```
/sys/class/infiniband/arnic_0/ports/1/hw_counters/
```

| 计数器 | 含义 | 用途 |
|--------|------|------|
| `cs_tx_bytes` / `cs_rx_bytes` | 累计收发字节 | 算带宽（MB/s） |
| `cs_goodput_bytes` | 有效吞吐（去协议开销） | 算有效带宽 |
| `cs_tx_packets` / `cs_rx_packets` | 累计包数 | 算包速率、平均包大小 |
| `cs_new_wr_rtt_min/max` | RTT 延迟（纳秒） | 通信延迟，实测 3.9-6.8μs |
| `cs_retx_wr` | 重传 work request | **网络质量关键**，应接近 0 |
| `rx_drop` / `rx_cmpl_drop` | 丢包 | **拥塞/错误关键**，应接近 0 |
| `cs_rx_ecn_resp` | ECN 拥塞标记响应 | 拥塞信号，>0 说明有拥塞 |
| `cs_retx_wr` 增长率 | 重传率 | 网络劣化预警 |
| `local_mr_errors` / `remote_mr_errors` | 内存注册错误 | 严重故障，应为 0 |
| `tx_undersized` | 过小包数 | 通信效率 |

**计数器是累计值**，需采两次算增量除以时间间隔（脚本已自动处理）。

---

## 3. 值得统计分析的核心指标

### 3.1 带宽利用率（是否瓶颈的第一判据）
```
利用率 = 实际带宽 / 25 GB/s（200G 单向理论上限）
```
- **< 30%**：RDMA 有大量余量，不是瓶颈（当前情况）
- **30-70%**：中等负载，关注延迟
- **> 80%**：接近饱和，可能是 communication bound

### 3.2 延迟 RTT（小消息延迟瓶颈）
实测 `cs_new_wr_rtt`：**3.9-6.8 μs**。
- TP 的 all-reduce 是**小消息高频**通信，延迟比带宽更关键。
- RTT 稳定在个位数 μs 是健康的；若 max 突增到几十/上百 μs，说明有排队/拥塞。

### 3.3 网络质量（重传 + 丢包 + ECN）
| 指标 | 健康值 | 异常含义 |
|------|--------|---------|
| `cs_retx_wr` 增量 | ≈ 0 | 持续增长 = 丢包重传，网络劣化 |
| `rx_drop` 增量 | ≈ 0 | 增长 = 拥塞/缓冲溢出 |
| `cs_rx_ecn_resp` | 0 | > 0 = 网络拥塞（ECN 标记） |
| `local/remote_mr_errors` | 0 | > 0 = RDMA 内存故障，严重 |

### 3.4 通信模式（辅助诊断）
- **平均包大小** = tx_bytes / tx_packets：实测约 2000 bytes（接近 MTU 4096 的一半，正常）
- **包速率**：反映通信频率，TP decode 时高频小包

---

## 4. ⚠️ 关键注意事项（避免误判）

1. **计数器是全节点共享的**：包含 Ray 心跳、NCCL 常驻通信、其他服务，**不只是推理**。实测空闲时也有间歇性后台流量（Ray/NCCL keepalive）。分析推理影响要用"负载前后增量"对比。

2. **两块网卡都要看**：NCCL 配置 `NCCL_IB_HCA=arnic_0,arnic_1`，流量分摊在两块卡，单看一块会低估总带宽。

3. **采样窗口影响读数**：decode 活跃期 vs prefill 期 vs 排队期，RDMA 流量差异大。要判断稳态，采样窗口应 ≥ 10s 或覆盖多个请求。

4. **TP=4 的通信特征**：每 token 每层都 all-reduce → 小消息、高频、延迟敏感。带宽利用率低不代表 RDMA 没用，而是"够用且不饱和"。

---

## 5. 判断 RDMA 是否瓶颈（决策树）

```
采样负载下的 RDMA + GPU 利用率：
├── GPU util 高(>90%) + RDMA util 低(<30%)  → compute-bound（当前情况，RDMA 不是瓶颈）
├── GPU util 低 + RDMA util 高(>80%)         → communication-bound（RDMA 是瓶颈）
├── RDMA 带宽不高但 RTT/retx 高               → 小消息延迟瓶颈（查 QP/拥塞/CPU affinity）
└── retx/drop/ecn 持续增长                    → 网络质量问题（查链路/拥塞控制）
```

**当前实测**：GPU 97-99% + RDMA util <10% + retx/drop≈0 → **compute-bound，RDMA 健康且非瓶颈**。

---

## 6. 建议的持续监控方式

### 实时观察（人工排查时）
```bash
./scripts/rdma_monitor.sh all 2
```

### 长期趋势（生产告警）
可将关键计数器接入 Prometheus/Grafana，重点告警：
- `cs_retx_wr` 增长率突增（网络劣化）
- `rx_drop` / `cs_rx_ecn_resp` > 0（拥塞）
- `local/remote_mr_errors` > 0（RDMA 故障，立即告警）
- RTT max 持续 > 50μs（延迟异常）

vLLM 自身的 `/metrics`（Prometheus）也应一起采，结合 GPU util 才能判断瓶颈归属。

### RDMA 基线测试（可选，需停服务）
标准 perftest 工具宿主已安装（`ib_write_bw` / `ib_read_bw` / `ib_send_lat`），可建链路带宽/延迟基线：
```bash
# 一端 server，另一端 client（需指定 ARNIC 设备）
ib_write_bw -d arnic_0 --report_gbits        # server 端
ib_write_bw -d arnic_0 --report_gbits <对端IP>  # client 端
```
⚠️ 会占用网卡，建议在维护窗口做，不要和推理同时跑。

---

## 7. 脚本

`scripts/rdma_monitor.sh` — 实时采样表格输出，支持单/双网卡、自定义间隔。
计数器路径、含义、利用率计算已内置。
