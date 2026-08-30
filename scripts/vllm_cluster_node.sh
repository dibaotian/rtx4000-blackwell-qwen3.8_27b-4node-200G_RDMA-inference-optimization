#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/cluster.env}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  vllm_cluster_node.sh start head
  vllm_cluster_node.sh start worker
  vllm_cluster_node.sh stop
  vllm_cluster_node.sh status
  vllm_cluster_node.sh logs
  vllm_cluster_node.sh shell

Set ENV_FILE=/path/to/cluster.env to use a different configuration file.
EOF
}

load_config() {
    [[ -f "${ENV_FILE}" ]] || die "Missing ${ENV_FILE}; copy cluster.env.example first."
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a

    local required_vars=(
        VLLM_IMAGE CONTAINER_NAME HEAD_IP RAY_INTERFACE RAY_PORT
        NCCL_SOCKET_IFNAME NCCL_IB_HCA HF_CACHE_HOST
        VLLM_CACHE_HOST LOG_DIR_HOST MODEL_ROOT_HOST MODEL_ID SERVED_MODEL_NAME API_PORT
        MAX_MODEL_LEN GPU_MEMORY_UTILIZATION MAX_NUM_SEQS ARNIC_RDMA_ROOT
        NCCL_IB_GID_INDEX
    )
    local name
    for name in "${required_vars[@]}"; do
        [[ -n "${!name:-}" ]] || die "${name} is not set in ${ENV_FILE}."
    done
}

active_rdma_netdevs() {
    command -v rdma >/dev/null || die "rdma-core is not installed."
    rdma link show | awk '
        $1 == "link" && $4 == "ACTIVE" {
            for (field = 1; field <= NF; field++) {
                if ($field == "netdev") {
                    print $(field + 1)
                }
            }
        }
    '
}

resolve_network() {
    mapfile -t RDMA_NETDEVS < <(active_rdma_netdevs)
    ((${#RDMA_NETDEVS[@]} > 0)) || die "No active RDMA netdev found."

    if [[ "${RAY_INTERFACE}" == "auto" ]]; then
        RAY_INTERFACE="${RDMA_NETDEVS[0]}"
    fi
    if [[ "${NCCL_SOCKET_IFNAME}" == "auto" ]]; then
        NCCL_SOCKET_IFNAME="$(IFS=,; printf '%s' "${RDMA_NETDEVS[*]}")"
    fi
}

detect_node_ip() {
    local detected
    detected="$(ip -4 -o address show dev "${RAY_INTERFACE}" | awk '{split($4, address, "/"); print address[1]; exit}')"
    [[ -n "${detected}" ]] || die "No IPv4 address found on ${RAY_INTERFACE}."
    printf '%s\n' "${NODE_IP:-${detected}}"
}

check_host() {
    command -v docker >/dev/null || die "docker is not installed."
    docker info >/dev/null 2>&1 || die "Docker is not running or is not accessible."
    command -v nvidia-smi >/dev/null || die "nvidia-smi is not installed."
    nvidia-smi --query-gpu=name --format=csv,noheader | grep -q . || die "No NVIDIA GPU detected."
    [[ -f "${ARNIC_RDMA_ROOT}/lib/libarnic-rdmav34.so" ]] ||
        die "Missing custom ARNIC provider under ${ARNIC_RDMA_ROOT}."
    [[ -f "${ARNIC_RDMA_ROOT}/etc/libibverbs.d/arnic.driver" ]] ||
        die "Missing ARNIC provider configuration under ${ARNIC_RDMA_ROOT}."

    if ! lsmod | grep -q '^nvidia_peermem'; then
        sudo -n modprobe nvidia_peermem ||
            die "nvidia_peermem is not loaded; run: sudo modprobe nvidia_peermem"
    fi

    resolve_network
    ip link show "${RAY_INTERFACE}" >/dev/null 2>&1 || die "Missing interface ${RAY_INTERFACE}."

    docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1 ||
        die "Image ${VLLM_IMAGE} is missing; build it with scripts/vllm_cluster_prepare.sh."
}

start_node() {
    local role="${1:-}"
    [[ "${role}" == "head" || "${role}" == "worker" ]] || die "Role must be head or worker."

    check_host
    local node_ip
    node_ip="$(detect_node_ip)"

    if [[ "${role}" == "head" ]]; then
        [[ "${node_ip}" == "${HEAD_IP}" ]] ||
            die "Head IP mismatch: ${RAY_INTERFACE} has ${node_ip}, but HEAD_IP=${HEAD_IP}."
    else
        ping -c 1 -W 2 "${HEAD_IP}" >/dev/null 2>&1 || die "Cannot reach Ray head at ${HEAD_IP}."
    fi

    mkdir -p "${HF_CACHE_HOST}" "${VLLM_CACHE_HOST}" "${LOG_DIR_HOST}" "${MODEL_ROOT_HOST}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

    local ray_command
    if [[ "${role}" == "head" ]]; then
        ray_command="exec ray start --block --head --node-ip-address=${node_ip} --port=${RAY_PORT} --include-dashboard=false"
    else
        ray_command="exec ray start --block --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${node_ip}"
    fi

    local docker_args=(
        run -d
        --name "${CONTAINER_NAME}"
        --restart unless-stopped
        --runtime nvidia
        --gpus all
        --network host
        --ipc host
        --cap-add IPC_LOCK
        --ulimit memlock=-1:-1
        --ulimit stack=67108864
        --entrypoint /bin/bash
        -v "${HF_CACHE_HOST}:/root/.cache/huggingface"
        -v "${VLLM_CACHE_HOST}:/root/.cache/vllm"
        -v "${LOG_DIR_HOST}:/var/log/vllm"
        -v "${MODEL_ROOT_HOST}:/models:ro"
        -v "${ARNIC_RDMA_ROOT}:${ARNIC_RDMA_ROOT}:ro"
        -e "VLLM_HOST_IP=${node_ip}"
        -e "GLOO_SOCKET_IFNAME=${RAY_INTERFACE}"
        -e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
        -e "NCCL_IB_HCA=${NCCL_IB_HCA}"
        -e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}"
        -e "NCCL_IB_DISABLE=0"
        -e "NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}"
        -e "NCCL_DEBUG=${NCCL_DEBUG:-INFO}"
        -e "RAY_USAGE_STATS_ENABLED=0"
        -e "ARNIC_RDMA_ROOT=${ARNIC_RDMA_ROOT}"
        -e "MODEL_ID=${MODEL_ID}"
        -e "MODEL_REVISION=${MODEL_REVISION:-main}"
        -e "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
        -e "API_HOST=${API_HOST:-0.0.0.0}"
        -e "API_PORT=${API_PORT}"
        -e "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
        -e "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
        -e "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
        -e "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}"
        -e "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}"
        -e "PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-4}"
        -e "ENFORCE_EAGER=${ENFORCE_EAGER:-true}"
        -e "LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-true}"
        -e "ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-false}"
        -e "API_KEY=${API_KEY:-}"
        -e "HF_HUB_OFFLINE=1"
        -e "TRANSFORMERS_OFFLINE=1"
    )

    if [[ "${DOCKER_PRIVILEGED:-true}" == "true" ]]; then
        docker_args+=(--privileged)
    else
        local device
        for device in /dev/infiniband/*; do
            [[ -e "${device}" ]] && docker_args+=(--device "${device}")
        done
    fi

    ray_command="export LD_LIBRARY_PATH=${ARNIC_RDMA_ROOT}/lib:\${LD_LIBRARY_PATH:-}; export PATH=${ARNIC_RDMA_ROOT}/bin:\${PATH}; ${ray_command}"
    docker "${docker_args[@]}" "${VLLM_IMAGE}" -lc "${ray_command}" >/dev/null
    printf 'Started Ray %s node %s at %s (Ray: %s; NCCL: %s).\n' \
        "${role}" "${CONTAINER_NAME}" "${node_ip}" "${RAY_INTERFACE}" "${NCCL_SOCKET_IFNAME}"
}

show_status() {
    if ! docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
        die "Container ${CONTAINER_NAME} is not running."
    fi
    docker exec "${CONTAINER_NAME}" ray status
}

main() {
    load_config
    case "${1:-}" in
        start)
            start_node "${2:-}"
            ;;
        stop)
            docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
            printf 'Stopped %s.\n' "${CONTAINER_NAME}"
            ;;
        status)
            show_status
            ;;
        logs)
            exec docker logs --tail 200 -f "${CONTAINER_NAME}"
            ;;
        shell)
            exec docker exec -it "${CONTAINER_NAME}" /bin/bash
            ;;
        *)
            usage
            exit 2
            ;;
    esac
}

main "$@"