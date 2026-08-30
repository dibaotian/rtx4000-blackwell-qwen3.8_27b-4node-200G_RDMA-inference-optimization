#!/usr/bin/env bash
# ARNIC 驱动重载脚本 — 切换拥塞控制（CC）开关
# ⚠️ 会中断 vLLM + Ray（依赖 RDMA），需在 4 节点执行，需要 sudo
#
# 用法（在头节点执行，会 ssh 到其他节点）：
#   ./arnic_reload_cc.sh enable    # 打开 CC（disable_cc=0）
#   ./arnic_reload_cc.sh disable   # 关闭 CC（disable_cc=1，恢复原状）
#   ./arnic_reload_cc.sh status    # 只查看当前 CC 状态，不改
#
# 前置：4 节点 SSH 免密；每节点能 sudo modprobe
set -uo pipefail

NODES=(<NODE0_IP> <NODE1_IP> <NODE2_IP> <NODE3_IP>)
HEAD=<NODE0_IP>
ACTION="${1:-status}"

# 单节点上执行的操作（函数体会被 ssh 传过去）
node_status() {
  echo "  disable_cc=$(cat /sys/module/arnic/parameters/disable_cc 2>/dev/null)"
  echo "  cc_state=$(cat /sys/class/infiniband/arnic_0/cc_state 2>/dev/null)"
  echo "  cc_algorithm=$(cat /sys/class/infiniband/arnic_0/cc_algorithm 2>/dev/null)"
  for h in arnic_0 arnic_1; do
    echo "  $h link=$(cat /sys/class/infiniband/$h/ports/1/state 2>/dev/null | awk '{print $2}')"
  done
}

node_reload() {  # $1 = disable_cc 值（0=开CC，1=关CC）
  local dcc=$1
  # 1. 停可能持有 RDMA 的容器（vLLM/Ray 在容器里）
  sudo docker stop vllm-ray-node 2>/dev/null || true
  sleep 3
  # 2. 卸载依赖 arnic 的上层模块，再卸 arnic
  #    注意：nvidia_peermem / ib_uverbs 可能被别的东西引用，谨慎
  sudo modprobe -r nvidia_peermem 2>/dev/null || true
  # arnic 被 ib_uverbs 引用；ib_uverbs 可能被多方引用，尽量只卸 arnic
  if ! sudo modprobe -r arnic 2>/dev/null; then
    echo "  [!] arnic 卸载失败（被占用）。可能需要先停更多服务或重启节点。"
    echo "      当前占用：$(lsmod | grep '^arnic')"
    return 1
  fi
  # 3. 带参数重载
  sudo modprobe arnic disable_cc=${dcc}
  sleep 2
  # 4. 恢复 GPUDirect 模块
  sudo modprobe nvidia_peermem 2>/dev/null || true
  # 5. 确认
  echo "  重载后 disable_cc=$(cat /sys/module/arnic/parameters/disable_cc 2>/dev/null)"
  echo "  链路：$(rdma link 2>/dev/null | grep -c ACTIVE) active"
}

# 把函数序列化传给远程节点执行
run_remote() {  # $1=ip $2=funcname $3...=args
  local ip=$1; shift
  local fn=$1; shift
  if [ "$ip" = "$HEAD" ]; then
    "$fn" "$@"
  else
    ssh -o ConnectTimeout=8 "$ip" "$(declare -f "$fn"); $fn $*"
  fi
}

case "$ACTION" in
  status)
    for ip in "${NODES[@]}"; do
      echo "=== node $ip ==="
      run_remote "$ip" node_status
    done
    ;;
  enable|disable)
    dcc=1; [ "$ACTION" = "enable" ] && dcc=0
    echo "############################################################"
    echo "# 将在 4 节点重载 arnic 驱动，disable_cc=${dcc}（${ACTION} CC）"
    echo "# ⚠️ 这会中断 vLLM + Ray，需要手动重启集群"
    echo "############################################################"
    read -rp "确认继续？输入 yes： " ok
    [ "$ok" = "yes" ] || { echo "已取消"; exit 1; }
    for ip in "${NODES[@]}"; do
      echo "=== 重载 node $ip ==="
      run_remote "$ip" node_reload "$dcc" || echo "  [!] node $ip 重载有问题，检查后再继续其他节点"
    done
    echo
    echo "############################################################"
    echo "# 驱动重载完成。现在需要重启集群："
    echo "#   ./scripts/vllm_cluster_ctl.sh start      （若 ctl PP 校验拦，见重启指导书）"
    echo "#   ./scripts/vllm_cluster_service.sh start"
    echo "# 然后验证：curl .../v1/completions 应返回 323"
    echo "############################################################"
    ;;
  *)
    echo "用法: $0 enable|disable|status"
    exit 2
    ;;
esac
