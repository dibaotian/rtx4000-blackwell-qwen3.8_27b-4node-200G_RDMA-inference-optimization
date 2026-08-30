#!/bin/bash

# 单GPU vLLM启动脚本
# 适用于AMD RX7900系列GPU

echo "========================================="
echo "vLLM 单GPU启动脚本"
echo "========================================="
echo ""

# 检查Docker状态
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker未运行或权限不足"
    echo "请确保Docker已启动并且用户在docker组中"
    echo "运行: sudo usermod -aG docker $USER && newgrp docker"
    exit 1
fi

# 检查模型是否存在
if [ ! -d "./models/Qwen_Qwen3-8B" ]; then
    echo "⚠️ 模型不存在: ./models/Qwen_Qwen3-8B"
    echo ""
    echo "请先下载模型："
    echo "bash scripts/download_model.sh Qwen/Qwen3-8B-Instruct"
    echo ""
    read -p "是否现在下载? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        bash scripts/download_model.sh Qwen/Qwen3-8B-Instruct ./models/Qwen_Qwen3-8B
    else
        exit 1
    fi
fi

# 停止旧容器
echo "🔄 停止旧容器..."
docker-compose -f docker-compose-single-gpu.yml down 2>/dev/null

# 检查GPU
echo ""
echo "📊 检查GPU状态..."
rocm-smi 2>/dev/null | head -15 || echo "⚠️ 无法检测GPU状态，继续启动..."

# 启动服务
echo ""
echo "🚀 启动vLLM服务..."
docker-compose -f docker-compose-single-gpu.yml up -d

# 等待服务启动
echo ""
echo "⏳ 等待服务启动（可能需要1-2分钟）..."
sleep 10

# 检查容器状态
if docker ps | grep -q vllm-qwen-server; then
    echo "✅ 容器已启动"
    echo ""
    echo "📋 查看日志："
    echo "docker-compose -f docker-compose-single-gpu.yml logs -f"
    echo ""
    echo "📊 监控状态："
    echo "watch -n 5 'rocm-smi --showmeminfo vram --csv | head -5'"
    echo ""
    
    # 等待更长时间让服务完全启动
    echo "⏳ 等待服务完全启动..."
    for i in {1..12}; do
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            echo ""
            echo "✅ vLLM服务已就绪！"
            echo ""
            echo "🌐 API地址: http://localhost:8080"
            echo ""
            echo "📝 测试命令："
            echo 'curl http://localhost:8080/v1/completions \'
            echo '  -H "Content-Type: application/json" \'
            echo '  -d "{"model": "/models/Qwen_Qwen3-8B", "prompt": "你好", "max_tokens": 50}"'
            exit 0
        fi
        echo -n "."
        sleep 5
    done
    
    echo ""
    echo "⚠️ 服务未能在60秒内就绪"
    echo "请查看日志: docker-compose -f docker-compose-single-gpu.yml logs"
else
    echo "❌ 容器启动失败"
    echo "查看错误: docker-compose -f docker-compose-single-gpu.yml logs"
    exit 1
fi
