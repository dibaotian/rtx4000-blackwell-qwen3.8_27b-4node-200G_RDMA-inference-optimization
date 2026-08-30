#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/cluster.env}"
PID_FILE=/run/vllm-server.pid
LOG_FILE=/var/log/vllm/server.log

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

load_config() {
    [[ -f "${ENV_FILE}" ]] || die "Missing ${ENV_FILE}."
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
}

container_running() {
    docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

service_running() {
    container_running && docker exec "${CONTAINER_NAME}" bash -lc \
        'test -s /run/vllm-server.pid && kill -0 "$(cat /run/vllm-server.pid)" 2>/dev/null'
}

check_ray_capacity() {
    local required_gpus=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))
    docker exec "${CONTAINER_NAME}" python3 -c \
        "import ray; ray.init(address='auto', logging_level='ERROR'); resources=ray.cluster_resources(); available=int(resources.get('GPU', 0)); print(f'Ray resources: {resources}'); assert available >= ${required_gpus}, f'need ${required_gpus} GPUs, found {available}'"
}

start_service() {
    container_running || die "Ray head container ${CONTAINER_NAME} is not running."
    service_running && die "vLLM service is already running."
    check_ray_capacity

    local args=(
        vllm serve "${MODEL_ID}"
        --host "${API_HOST}"
        --port "${API_PORT}"
        --served-model-name "${SERVED_MODEL_NAME}"
        --dtype auto
        --kv-cache-dtype fp8
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
        --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}"
        --distributed-executor-backend ray
        --max-model-len "${MAX_MODEL_LEN}"
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
        --max-num-seqs "${MAX_NUM_SEQS}"
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
        --enable-chunked-prefill
        --reasoning-parser qwen3
        --enable-auto-tool-choice
        --tool-call-parser qwen3_coder
    )

    [[ "${MODEL_ID}" != /* ]] && args+=(--revision "${MODEL_REVISION}")

    [[ "${ENFORCE_EAGER}" == "true" ]] && args+=(--enforce-eager)
    if [[ -n "${CUDAGRAPH_CAPTURE_SIZES:-}" ]]; then
        local capture_sizes
        IFS=',' read -r -a capture_sizes <<< "${CUDAGRAPH_CAPTURE_SIZES}"
        args+=(--cudagraph-capture-sizes "${capture_sizes[@]}")
    fi
    [[ -n "${MAX_CUDAGRAPH_CAPTURE_SIZE:-}" ]] &&
        args+=(--max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}")
    [[ "${LANGUAGE_MODEL_ONLY}" == "true" ]] && args+=(--language-model-only)
    if [[ "${ENABLE_PREFIX_CACHING}" == "true" ]]; then
        args+=(--enable-prefix-caching)
    else
        args+=(--no-enable-prefix-caching)
    fi
    [[ -n "${SPECULATIVE_CONFIG:-}" ]] && args+=(--speculative-config "${SPECULATIVE_CONFIG}")
    [[ -n "${API_KEY:-}" ]] && args+=(--api-key "${API_KEY}")

    local command
    printf -v command '%q ' "${args[@]}"
    local docker_exec_args=(docker exec -d)
    [[ -n "${VLLM_PP_LAYER_PARTITION:-}" ]] &&
        docker_exec_args+=(-e "VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION}")
    docker exec "${CONTAINER_NAME}" bash -lc ": > ${LOG_FILE}"
    "${docker_exec_args[@]}" "${CONTAINER_NAME}" bash -lc \
        "export LD_LIBRARY_PATH=${ARNIC_RDMA_ROOT}/lib:\${LD_LIBRARY_PATH:-}; echo \$\$ > ${PID_FILE}; exec ${command} >> ${LOG_FILE} 2>&1"
    printf 'vLLM is starting on http://%s:%s (model: %s).\n' "${HEAD_IP}" "${API_PORT}" "${SERVED_MODEL_NAME}"
    printf 'Follow startup with: %s logs\n' "$0"
}

stop_service() {
    if service_running; then
        docker exec "${CONTAINER_NAME}" bash -lc \
            'kill "$(cat /run/vllm-server.pid)" && rm -f /run/vllm-server.pid'
        printf 'Stopped vLLM service.\n'
    else
        printf 'vLLM service is not running.\n'
    fi
}

show_status() {
    if service_running; then
        printf 'vLLM process: running\n'
    else
        printf 'vLLM process: stopped\n'
    fi

    if curl -fsS --max-time 3 "http://${HEAD_IP}:${API_PORT}/health" >/dev/null 2>&1; then
        printf 'API health: ready\n'
    else
        printf 'API health: not ready\n'
    fi
}

test_api() {
    local auth_args=()
    [[ -n "${API_KEY:-}" ]] && auth_args=(-H "Authorization: Bearer ${API_KEY}")
    curl -fsS "http://${HEAD_IP}:${API_PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        "${auth_args[@]}" \
        -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话说明 RDMA 的作用。\"}],\"max_tokens\":128,\"temperature\":0.7,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
    printf '\n'
}

usage() {
    printf 'Usage: %s start|stop|status|logs|test\n' "$0"
}

main() {
    load_config
    case "${1:-}" in
        start) start_service ;;
        stop) stop_service ;;
        status) show_status ;;
        logs) exec docker exec "${CONTAINER_NAME}" tail -n 200 -f "${LOG_FILE}" ;;
        test) test_api ;;
        *) usage; exit 2 ;;
    esac
}

main "$@"