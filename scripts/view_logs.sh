#!/bin/bash

# vLLM 日志查看脚本

CONTAINER_NAME="vllm-qwen-server"

echo "========================================="
echo "vLLM 日志查看工具"
echo "========================================="
echo ""

# 检查容器是否存在
if ! docker ps -a --format "{{.Names}}" | grep -q "^$CONTAINER_NAME$"; then
    echo "❌ 容器 $CONTAINER_NAME 不存在"
    echo ""
    echo "请先启动vLLM服务："
    echo "bash scripts/start_vllm_docker.sh"
    exit 1
fi

# 显示容器状态
echo "📊 容器状态："
docker ps -a --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# 提供选项
echo "选择操作："
echo "1) 查看最新日志（最后50行）"
echo "2) 实时跟踪日志"
echo "3) 查看全部日志"
echo "4) 查看容器资源使用"
echo "5) 重启容器"
echo "6) 停止容器"
echo ""

read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        echo ""
        echo "📋 最新日志（最后50行）："
        echo "----------------------------------------"
        docker logs --tail 50 $CONTAINER_NAME
        ;;
    2)
        echo ""
        echo "📋 实时日志（按Ctrl+C退出）："
        echo "----------------------------------------"
        docker logs -f $CONTAINER_NAME
        ;;
    3)
        echo ""
        echo "📋 全部日志："
        echo "----------------------------------------"
        docker logs $CONTAINER_NAME
        ;;
    4)
        echo ""
        echo "📊 容器资源使用（按Ctrl+C退出）："
        echo "----------------------------------------"
        docker stats $CONTAINER_NAME
        ;;
    5)
        echo ""
        echo "🔄 重启容器..."
        docker restart $CONTAINER_NAME
        echo "✅ 容器已重启"
        echo ""
        echo "等待服务就绪..."
        sleep 10
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            echo "✅ 服务已就绪"
        else
            echo "⚠️ 服务可能还在启动中，请稍后再试"
        fi
        ;;
    6)
        echo ""
        echo "🛑 停止容器..."
        docker stop $CONTAINER_NAME
        echo "✅ 容器已停止"
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac
