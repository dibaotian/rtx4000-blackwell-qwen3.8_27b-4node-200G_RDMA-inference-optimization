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
}

preflight() {
    command -v docker >/dev/null || die "docker is not installed."
    docker info >/dev/null 2>&1 || die "Docker is not running or is not accessible."
    command -v nvidia-smi >/dev/null || die "nvidia-smi is not installed."
    command -v rdma >/dev/null || die "rdma-core is not installed."
    [[ -f "${ARNIC_RDMA_ROOT}/lib/libarnic-rdmav34.so" ]] || die "ARNIC provider is missing."

    local gpu_info
    gpu_info="$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader)"
    [[ -n "${gpu_info}" ]] || die "No NVIDIA GPU detected."
    printf 'GPU: %s\n' "${gpu_info}"

    local active_rdma
    active_rdma="$(rdma link show | awk '$4 == "ACTIVE" {count++} END {print count + 0}')"
    ((active_rdma >= 1)) || die "No active RDMA link detected."
    printf 'RDMA: %s active link(s)\n' "${active_rdma}"

    if ! lsmod | grep -q '^nvidia_peermem'; then
        sudo -n modprobe nvidia_peermem || die "Cannot load nvidia_peermem."
    fi
    printf 'GPUDirect module: loaded\n'
}

build_image() {
    preflight
    docker build \
        --file "${ROOT_DIR}/docker/Dockerfile.nvidia-ray" \
        --build-arg "VLLM_BASE_IMAGE=${VLLM_BASE_IMAGE}" \
        --tag "${VLLM_IMAGE}" \
        "${ROOT_DIR}"
    docker run --rm --runtime nvidia --gpus all --entrypoint /usr/bin/python3 "${VLLM_IMAGE}" \
        -c 'import cupy, importlib.metadata as md, ray, torch, transformers, vllm; assert not any(d.metadata["Name"] == "cupy-cuda12x" for d in md.distributions()); print(f"vLLM={vllm.__version__} Ray={ray.__version__} Transformers={transformers.__version__} CUDA={torch.version.cuda} CuPy={cupy.__version__} GPU={torch.cuda.get_device_name(0)}")'
}

download_model() {
    docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1 || die "Build ${VLLM_IMAGE} first."
    [[ "${MODEL_ID}" == /models/* ]] || die "MODEL_ID must be under /models for local deployment."
    local model_dir="${MODEL_ROOT_HOST}/${MODEL_ID#/models/}"
    mkdir -p "${HF_CACHE_HOST}" "${model_dir}"
    docker run --rm --network host \
        -v "${HF_CACHE_HOST}:/root/.cache/huggingface" \
        -v "${MODEL_ROOT_HOST}:/models" \
        -e HF_XET_HIGH_PERFORMANCE=1 \
        -e MODEL_SOURCE="${MODEL_SOURCE}" \
        -e MODEL_REVISION="${MODEL_REVISION}" \
        -e MODEL_ID="${MODEL_ID}" \
        --entrypoint /bin/bash \
        "${VLLM_IMAGE}" -lc \
        'hf download "${MODEL_SOURCE}" --revision "${MODEL_REVISION}" --local-dir "${MODEL_ID}"'
    sudo -n chown -R "$(id -u):$(id -g)" "${model_dir}" ||
        die "Model downloaded, but could not restore ownership of ${model_dir}."
}

verify_model() {
    docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1 || die "Build ${VLLM_IMAGE} first."
    docker run --rm --network none \
        -v "${MODEL_ROOT_HOST}:/models:ro" \
        -e HF_HUB_OFFLINE=1 \
        -e TRANSFORMERS_OFFLINE=1 \
        -e MODEL_ID="${MODEL_ID}" \
        --entrypoint /bin/bash \
        "${VLLM_IMAGE}" -lc \
        'python3 - <<'"'"'PY'"'"'
import json
import os
from pathlib import Path

path = Path(os.environ["MODEL_ID"])
config = json.loads((path / "config.json").read_text())
index = json.loads((path / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (path / name).is_file()]
if missing:
    raise SystemExit(f"Missing {len(missing)} weight shards, first: {missing[0]}")
architectures = config.get("architectures")
print(f"model={path} architecture={architectures} shards={len(shards)}")
PY'
}

usage() {
    printf 'Usage: %s preflight|build|download|verify-model\n' "$0"
}

main() {
    load_config
    case "${1:-}" in
        preflight) preflight ;;
        build) build_image ;;
        download) download_model ;;
        verify-model) verify_model ;;
        *) usage; exit 2 ;;
    esac
}

main "$@"