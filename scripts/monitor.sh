#!/bin/bash

echo "========================================="
echo "vLLM 服务监控"
echo "========================================="

while true; do
    # 检查健康状态
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ vLLM服务已经就绪！"
        echo ""
        echo "可用的模型："
        curl -s http://localhost:8080/v1/models | python3 -m json.tool
        break
    else
        echo "⏳ 服务启动中... (首次启动需要下载模型)"
        
        # 检查磁盘使用情况
        echo "📦 模型下载目录大小："
        sudo docker exec vllm-qwen-server du -sh /models 2>/dev/null || echo "  等待下载开始..."
        
        # 检查GPU使用情况
        echo "🖥️ GPU状态："
        sudo docker exec vllm-qwen-server rocm-smi --showmeminfo vram --csv | head -5
        
        sleep 10
        echo "---"
    fi
done

echo ""
echo "========================================="
echo "🎉 vLLM服务已就绪！"
echo "========================================="
echo ""
echo "测试命令："
echo "  curl http://localhost:8080/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{"
echo '      "messages": [{"role": "user", "content": "你好"}],'
echo '      "temperature": 0.7,'
echo '      "max_tokens": 100'
echo "    }'"
