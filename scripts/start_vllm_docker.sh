#!/bin/bash

# 单GPU vLLM启动脚本 - 直接使用Docker命令
# 适用于AMD RX7900系列GPU

echo "========================================="
echo "vLLM 单GPU启动脚本 (Docker命令版)"
echo "========================================="
echo ""

# 设置变量
CONTAINER_NAME="vllm-qwen-server"
IMAGE="rocm/vllm-dev:rocm7.1.1_navi_ubuntu22.04_py3.10_pytorch_2.8_vllm_0.10.2rc1"
MODEL_PATH="$(pwd)/models"
LOGS_PATH="$(pwd)/logs"
SCRIPTS_PATH="$(pwd)/scripts"

# 检查Docker状态
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker未运行或权限不足"
    echo "请确保Docker已启动并且用户在docker组中"
    echo "运行: sudo usermod -aG docker $USER && newgrp docker"
    exit 1
fi

# 检查模型是否存在
if [ ! -d "$MODEL_PATH/Qwen_Qwen3-8B" ]; then
    echo "⚠️ 模型不存在: $MODEL_PATH/Qwen_Qwen3-8B"
    echo ""
    echo "请先下载模型到 ./models/Qwen_Qwen3-8B 目录"
    exit 1
fi

# 创建必要的目录
mkdir -p "$LOGS_PATH"

# 停止旧容器
echo "🔄 停止旧容器..."
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# 启动容器
echo ""
echo "🚀 启动vLLM服务..."
docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --shm-size 16gb \
    -p 8080:8080 \
    -v "$MODEL_PATH:/models" \
    -v "$LOGS_PATH:/logs" \
    -v "$SCRIPTS_PATH:/scripts" \
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    -e HIP_VISIBLE_DEVICES=0 \
    -e ROCR_VISIBLE_DEVICES=0 \
    -e GPU_MAX_HW_QUEUES=8 \
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
    -e VLLM_USE_V1=0 \
    $IMAGE \
    python -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen_Qwen3-8B \
    --host 0.0.0.0 \
    --port 8080 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.80 \
    --enforce-eager \
    --trust-remote-code

# 检查容器是否启动
sleep 3
if docker ps | grep -q $CONTAINER_NAME; then
    echo "✅ 容器已启动"
    echo ""
    echo "📋 查看日志："
    echo "docker logs -f $CONTAINER_NAME"
    echo ""
    
    # 等待服务就绪
    echo "⏳ 等待服务就绪（可能需要1-2分钟）..."
    for i in {1..24}; do
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            echo ""
            echo "✅ vLLM服务已就绪！"
            echo ""
            echo "🌐 API地址: http://localhost:8080"
            echo ""
            echo "📝 测试命令："
            echo 'curl http://localhost:8080/v1/completions \'
            echo '  -H "Content-Type: application/json" \'
            echo '  -d "{\"model\": \"/models/Qwen_Qwen3-8B\", \"prompt\": \"你好\", \"max_tokens\": 50}"'
            echo ""
            echo "📊 查看容器状态："
            echo "docker stats $CONTAINER_NAME"
            exit 0
        fi
        echo -n "."
        sleep 5
    done
    
    echo ""
    echo "⚠️ 服务未能在2分钟内就绪"
    echo "请查看日志: docker logs $CONTAINER_NAME"
else
    echo "❌ 容器启动失败"
    echo "查看错误: docker logs $CONTAINER_NAME"
    exit 1
fi
