#!/bin/bash

echo "========================================="
echo "vLLM 服务状态检查"
echo "========================================="

# 检查容器状态
echo "📦 Docker容器状态："
sudo docker ps | grep vllm-qwen-server | awk '{print "容器ID: "$1" | 状态: "$7" "$8}'

echo ""
echo "📊 模型下载进度："
sudo docker exec vllm-qwen-server tail -3 /tmp/vllm.log 2>/dev/null | grep "Downloading" || echo "无下载活动"

echo ""
echo "🔌 API状态："
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ API服务已就绪！"
    echo ""
    echo "📋 可用模型："
    curl -s http://localhost:8080/v1/models 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "获取模型列表失败"
else
    echo "⏳ API服务尚未就绪（模型可能还在下载或加载中）"
fi

echo ""
echo "💻 GPU使用情况："
sudo docker exec vllm-qwen-server rocm-smi --showmeminfo vram --csv 2>/dev/null | head -5 || echo "无法获取GPU信息"

echo ""
echo "========================================="
echo "使用提示："
echo "1. 模型下载完成后，API会自动启动"
echo "2. 测试API: curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
echo "3. 查看完整日志: sudo docker exec vllm-qwen-server tail -f /tmp/vllm.log"
echo "========================================="
