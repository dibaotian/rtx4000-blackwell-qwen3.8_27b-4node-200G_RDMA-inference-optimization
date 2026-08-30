#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/cluster.env}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

load_config() {
    [[ -f "${ENV_FILE}" ]] || die "Missing ${ENV_FILE}; copy cluster.env.example first."
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
    IFS=',' read -r -a NODES <<< "${CLUSTER_NODES}"
    ((${#NODES[@]} == PIPELINE_PARALLEL_SIZE)) ||
        die "CLUSTER_NODES count must equal PIPELINE_PARALLEL_SIZE."
    [[ "${NODES[0]}" == "${HEAD_IP}" ]] || die "The first CLUSTER_NODES entry must be HEAD_IP."
}

is_local_node() {
    ip -4 -o address show | awk '{split($4, address, "/"); print address[1]}' | grep -Fxq "$1"
}

run_on_node() {
    local node="$1"
    shift
    if is_local_node "${node}"; then
        "$@"
    else
        local remote_command
        printf -v remote_command '%q ' "$@"
        ssh -o BatchMode=yes -o ConnectTimeout=5 "${node}" "${remote_command}"
    fi
}

sync_files() {
    local node
    for node in "${NODES[@]:1}"; do
        printf 'Syncing cluster files to %s...\n' "${node}"
        ssh -o BatchMode=yes "${node}" \
            "sudo -n install -d -o \$(id -u) -g \$(id -g) '${REMOTE_ROOT}' '${REMOTE_ROOT}/docker' '${REMOTE_ROOT}/scripts'"
        rsync -a --relative \
            "${ROOT_DIR}/./cluster.env" \
            "${ROOT_DIR}/./cluster.env.example" \
            "${ROOT_DIR}/./docker/Dockerfile.nvidia-ray" \
            "${ROOT_DIR}/./scripts/vllm_cluster_node.sh" \
            "${ROOT_DIR}/./scripts/vllm_cluster_prepare.sh" \
            "${ROOT_DIR}/./scripts/vllm_cluster_service.sh" \
            "${ROOT_DIR}/./scripts/vllm_cluster_ctl.sh" \
            "${node}:${REMOTE_ROOT}/"
    done
    chmod +x "${ROOT_DIR}"/scripts/vllm_cluster_*.sh
    for node in "${NODES[@]:1}"; do
        ssh -o BatchMode=yes "${node}" \
            "chmod 600 '${REMOTE_ROOT}/cluster.env'; chmod +x '${REMOTE_ROOT}'/scripts/vllm_cluster_*.sh"
    done
}

preflight_all() {
    local node
    for node in "${NODES[@]}"; do
        printf '\n=== %s ===\n' "${node}"
        run_on_node "${node}" env ENV_FILE="${REMOTE_ROOT}/cluster.env" \
            "${REMOTE_ROOT}/scripts/vllm_cluster_prepare.sh" preflight
    done
}

sync_image() {
    docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1 || die "Build ${VLLM_IMAGE} on the head first."
    local node
    for node in "${NODES[@]:1}"; do
        printf 'Sending image %s to %s...\n' "${VLLM_IMAGE}" "${node}"
        docker save "${VLLM_IMAGE}" | ssh -o BatchMode=yes "${node}" docker load
    done
}

sync_model() {
    [[ "${MODEL_ID}" == /models/* ]] || die "MODEL_ID must be under /models."
    local relative_model="${MODEL_ID#/models/}"
    local source_dir="${MODEL_ROOT_HOST}/${relative_model}"
    [[ -f "${source_dir}/model.safetensors.index.json" ]] ||
        die "Model is incomplete at ${source_dir}; run vllm_cluster_prepare.sh download."

    local node
    local pids=()
    for node in "${NODES[@]:1}"; do
        printf 'Syncing model to %s...\n' "${node}"
        ssh -o BatchMode=yes "${node}" \
            "sudo -n install -d -o \$(id -u) -g \$(id -g) '${MODEL_ROOT_HOST}' '${MODEL_ROOT_HOST}/${relative_model}'"
        rsync -a --info=progress2 "${source_dir}/" "${node}:${MODEL_ROOT_HOST}/${relative_model}/" &
        pids+=("$!")
    done

    local failed=0
    local pid
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    ((failed == 0)) || die "One or more model sync jobs failed."
}

start_cluster() {
    local node
    for node in "${NODES[@]:1}"; do
        run_on_node "${node}" env ENV_FILE="${REMOTE_ROOT}/cluster.env" \
            "${REMOTE_ROOT}/scripts/vllm_cluster_node.sh" stop
    done
    ENV_FILE="${ENV_FILE}" "${ROOT_DIR}/scripts/vllm_cluster_node.sh" stop

    ENV_FILE="${ENV_FILE}" "${ROOT_DIR}/scripts/vllm_cluster_node.sh" start head
    python3 - "${HEAD_IP}" "${RAY_PORT}" <<'PY'
import socket
import sys
import time

host, port = sys.argv[1], int(sys.argv[2])
deadline = time.time() + 60
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1):
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit(f"Ray head {host}:{port} did not become ready")
PY
    for node in "${NODES[@]:1}"; do
        run_on_node "${node}" env ENV_FILE="${REMOTE_ROOT}/cluster.env" \
            "${REMOTE_ROOT}/scripts/vllm_cluster_node.sh" start worker
    done
    docker exec -i -e "EXPECTED_GPUS=${PIPELINE_PARALLEL_SIZE}" \
        "${CONTAINER_NAME}" python3 - <<'PY'
import os
import time

import ray

ray.init(address="auto", logging_level="ERROR")
deadline = time.time() + 120
expected = int(os.environ["EXPECTED_GPUS"])
available = 0
stable_checks = 0
while time.time() < deadline:
    available = int(ray.cluster_resources().get("GPU", 0))
    if available >= expected:
        stable_checks += 1
    else:
        stable_checks = 0
    if stable_checks >= 3:
        break
    time.sleep(2)

print(f"Ray GPUs: {available}/{expected}")
if stable_checks < 3:
    raise SystemExit("Ray workers did not register in time")
PY
    ENV_FILE="${ENV_FILE}" "${ROOT_DIR}/scripts/vllm_cluster_node.sh" status
}

stop_cluster() {
    local node
    for node in "${NODES[@]}"; do
        run_on_node "${node}" env ENV_FILE="${REMOTE_ROOT}/cluster.env" \
            "${REMOTE_ROOT}/scripts/vllm_cluster_node.sh" stop
    done
}

status_all() {
    local node
    for node in "${NODES[@]}"; do
        printf '\n=== %s ===\n' "${node}"
        run_on_node "${node}" docker ps --filter "name=${CONTAINER_NAME}" \
            --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
        run_on_node "${node}" nvidia-smi \
            --query-gpu=name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader
    done
    printf '\n=== Ray ===\n'
    ENV_FILE="${ENV_FILE}" "${ROOT_DIR}/scripts/vllm_cluster_node.sh" status
    printf '\n=== vLLM ===\n'
    ENV_FILE="${ENV_FILE}" "${ROOT_DIR}/scripts/vllm_cluster_service.sh" status
}

usage() {
    cat <<'EOF'
Usage: vllm_cluster_ctl.sh COMMAND

Commands:
  sync-files       Copy cluster configuration and scripts to worker nodes
  preflight        Check GPU, Docker, RDMA, and GPUDirect on all nodes
  sync-image       Copy the head node's built Docker image to workers
  sync-model       Copy the downloaded model directory to workers
  start            Start the Ray head and all workers
  stop             Stop vLLM/Ray containers on all nodes
  status           Show containers, GPUs, Ray resources, and API health
EOF
}

main() {
    load_config
    case "${1:-}" in
        sync-files) sync_files ;;
        preflight) preflight_all ;;
        sync-image) sync_image ;;
        sync-model) sync_model ;;
        start) start_cluster ;;
        stop) stop_cluster ;;
        status) status_all ;;
        *) usage; exit 2 ;;
    esac
}

main "$@"