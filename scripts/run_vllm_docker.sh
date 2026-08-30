# run this in the docker env 
# in the VLLM container
# python benchmark_throughput.py --backend vllm --input-len 128 --output-len 512 --model Qwen/Qwen2.5-7B-Instruct --num-prompts 100 --seed 1100 --trust-remote-code --max-model-len 2048 --tensor-parallel-size 1

# python benchmark_throughput.py --backend vllm --input-len 128 --output-len 512 --model Qwen/Qwen3-8B --num-prompts 100 --seed 1100 --trust-remote-code --max-model-len 2048 --tensor-parallel-size 1

CONTAINER_NAME="vllm-qwen-server"
IMAGE="rocm/vllm-dev:rocm7.1.1_navi_ubuntu24.04_py3.12_pytorch_2.8_vllm_0.10.2rc1"
MODEL_PATH="$(pwd)/models"
LOGS_PATH="$(pwd)/logs"
SCRIPTS_PATH="$(pwd)/scripts"

docker run -it --rm --network=host \
        --device=/dev/kfd \
        --device=/dev/dri \
        --ipc=host \
        --shm-size 16G \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -p 30000:30000 \
        -v "$MODEL_PATH:/models" \
        -v "$LOGS_PATH:/logs" \
        -v "$SCRIPTS_PATH:/scripts" \
        --env HUGGINGFACE_HUB_CACHE=/models \ 
    $IMAGE

