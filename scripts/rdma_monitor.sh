#!/usr/bin/env bash
# RDMA 监控脚本 — 基于 ARNIC hw_counters 采样带宽/延迟/重传/丢包
# 用法：
#   ./rdma_monitor.sh                 # 默认监控 arnic_0，每2秒刷新
#   ./rdma_monitor.sh arnic_1 5       # 监控 arnic_1，每5秒
#   ./rdma_monitor.sh all 2           # 同时监控两块网卡
set -u

HCA="${1:-arnic_0}"
INTERVAL="${2:-2}"

# 200G 链路理论带宽（单向）：200 Gbit/s ≈ 25 GB/s
LINK_BW_BYTES=$((25 * 1000 * 1000 * 1000))

read_counter() {  # $1=hca $2=counter名
  cat "/sys/class/infiniband/$1/ports/1/hw_counters/$2" 2>/dev/null || echo 0
}

snapshot() {  # 输出一行关键计数器
  local h=$1
  echo "$(read_counter $h cs_tx_bytes) $(read_counter $h cs_rx_bytes) \
$(read_counter $h cs_tx_packets) $(read_counter $h cs_rx_packets) \
$(read_counter $h cs_retx_wr) $(read_counter $h rx_drop) \
$(read_counter $h cs_rx_ecn_resp) $(read_counter $h cs_new_wr_rtt_min) \
$(read_counter $h cs_new_wr_rtt_max) $(read_counter $h local_mr_errors) \
$(read_counter $h remote_mr_errors)"
}

hcas=()
if [ "$HCA" = "all" ]; then hcas=(arnic_0 arnic_1); else hcas=("$HCA"); fi

echo "RDMA 监控（ARNIC hw_counters）| 间隔 ${INTERVAL}s | Ctrl+C 退出"
echo "注意：计数器为全节点共享（含 Ray/NCCL 后台流量），不只推理"
echo "200G 链路理论上限 ≈ 25 GB/s（单向）"
echo "----------------------------------------------------------------------"

declare -A prev
for h in "${hcas[@]}"; do prev[$h]=$(snapshot "$h"); done

printf "%-9s %10s %10s %9s %8s %7s %7s %8s\n" \
  "HCA" "TX_MB/s" "RX_MB/s" "Kpkt/s" "util%" "retx" "drop" "RTT_us"

while true; do
  sleep "$INTERVAL"
  for h in "${hcas[@]}"; do
    cur=$(snapshot "$h")
    read -r p_tx p_rx p_tp p_rp p_retx p_drop p_ecn p_rmin p_rmax p_lme p_rme <<< "${prev[$h]}"
    read -r c_tx c_rx c_tp c_rp c_retx c_drop c_ecn c_rmin c_rmax c_lme c_rme <<< "$cur"
    prev[$h]=$cur

    python3 - "$h" "$INTERVAL" "$LINK_BW_BYTES" \
      "$p_tx" "$c_tx" "$p_rx" "$c_rx" "$p_tp" "$c_tp" \
      "$p_retx" "$c_retx" "$p_drop" "$c_drop" "$c_rmin" "$c_rmax" <<'PY'
import sys
h,itv,link = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
p_tx,c_tx,p_rx,c_rx,p_tp,c_tp = map(int, sys.argv[4:10])
p_retx,c_retx,p_drop,c_drop,rmin,rmax = map(int, sys.argv[10:16])
tx=(c_tx-p_tx)/itv; rx=(c_rx-p_rx)/itv; pkt=(c_tp-p_tp)/itv
util=max(tx,rx)/link*100
rtt_us = f"{rmin/1000:.1f}-{rmax/1000:.1f}"
print(f"{h:<9} {tx/1e6:10.1f} {rx/1e6:10.1f} {pkt/1000:9.0f} "
      f"{util:8.2f} {c_retx-p_retx:7d} {c_drop-p_drop:7d} {rtt_us:>8}")
PY
  done
done
