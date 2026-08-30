#!/bin/bash

# HuggingFace模型下载脚本
# 直接从HuggingFace下载模型

MODEL_NAME="${1:-Qwen/Qwen3-8B}"
LOCAL_DIR="${2:-/dc2/minx/models/$(echo $MODEL_NAME | tr '/' '_')}"

echo "========================================="
echo "HuggingFace 模型下载工具"
echo "========================================="
echo "模型: $MODEL_NAME"
echo "保存到: $LOCAL_DIR"
echo ""

# 检查是否安装了huggingface-hub
if ! command -v huggingface-cli &> /dev/null; then
    echo "📦 安装huggingface-hub..."
    # Ubuntu 24.04+ 需要特殊处理
    pip install --user huggingface-hub 2>/dev/null || \
    pip install --user --break-system-packages huggingface-hub 2>/dev/null || \
    (echo "使用系统包管理器安装..." && sudo apt-get update && sudo apt-get install -y python3-huggingface-hub) || \
    (echo "创建虚拟环境..." && python3 -m venv ~/.hf_venv && ~/.hf_venv/bin/pip install huggingface-hub)
fi

# 确保huggingface-cli在PATH中
if [ -f ~/.hf_venv/bin/huggingface-cli ]; then
    export PATH="$HOME/.hf_venv/bin:$PATH"
fi

# 创建目录
mkdir -p "$(dirname "$LOCAL_DIR")"

echo ""
echo "⬇️ 开始下载模型..."
echo "提示: 可以设置 HF_ENDPOINT=https://hf-mirror.com 使用镜像加速"
echo ""

# 下载模型
if [ -n "$HF_ENDPOINT" ]; then
    echo "使用镜像站: $HF_ENDPOINT"
fi

# 使用Python脚本下载（更灵活）
# 优先使用虚拟环境的Python
PYTHON_CMD="python3"
if [ -f ~/.hf_venv/bin/python ]; then
    PYTHON_CMD="$HOME/.hf_venv/bin/python"
fi

$PYTHON_CMD << EOF
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("尝试安装 huggingface-hub...")
    import subprocess
    try:
        # 尝试在当前Python环境中安装
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "huggingface-hub"])
        from huggingface_hub import snapshot_download
    except:
        print("❌ 无法安装 huggingface-hub")
        print("请手动运行: pip install --user --break-system-packages huggingface-hub")
        sys.exit(1)

model_id = '$MODEL_NAME'
local_dir = '$LOCAL_DIR'

print(f'正在下载 {model_id} 到 {local_dir}')
print('-' * 50)

try:
    # 下载模型，显示进度条
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"]  # 忽略不需要的格式
    )
    print('-' * 50)
    print('✅ 模型下载完成！')
    
    # 显示下载的文件
    print('\n📁 下载的文件：')
    total_size = 0
    for root, dirs, files in os.walk(local_dir):
        # 忽略.cache目录
        if '.cache' in root:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path) / (1024**3)  # 转换为GB
            total_size += size
            if size > 0.001:  # 只显示大于1MB的文件
                rel_path = os.path.relpath(file_path, local_dir)
                print(f'  - {rel_path}: {size:.3f} GB')
    
    print(f'\n📊 总大小: {total_size:.2f} GB')
                
except Exception as e:
    print(f'❌ 下载失败: {e}')
    print('\n💡 提示:')
    print('1. 如果遇到连接问题，可以设置环境变量:')
    print('   export HF_ENDPOINT=https://hf-mirror.com')
    print('2. 如果需要登录，请运行:')
    print('   huggingface-cli login')
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ 模型下载成功！"
    echo "模型路径: $LOCAL_DIR"
    echo ""
    echo "📝 常用模型列表："
    echo ""
    echo "【Qwen3系列】"
    echo "  - Qwen/Qwen3-0.5B-Instruct    (0.5B，测试用)"
    echo "  - Qwen/Qwen3-1B-Instruct      (1B)"
    echo "  - Qwen/Qwen3-3B-Instruct      (3B)"
    echo "  - Qwen/Qwen3-7B-Instruct      (7B)"
    echo "  - Qwen/Qwen3-8B-Instruct      (8B，推荐)"
    echo "  - Qwen/Qwen3-14B-Instruct     (14B)"
    echo "  - Qwen/Qwen3-32B-Instruct     (32B)"
    echo ""
    echo "【Qwen2.5系列】(最新)"
    echo "  - Qwen/Qwen2.5-0.5B-Instruct  (0.5B)"
    echo "  - Qwen/Qwen2.5-3B-Instruct    (3B)"
    echo "  - Qwen/Qwen2.5-7B-Instruct    (7B)"
    echo "  - Qwen/Qwen2.5-14B-Instruct   (14B)"
    echo "  - Qwen/Qwen2.5-32B-Instruct   (32B)"
    echo ""
    echo "🚀 使用vLLM启动服务（适用于NVIDIA GPU）："
    echo "python -m vllm.entrypoints.openai.api_server \\"
    echo "  --model $LOCAL_DIR \\"
    echo "  --port 8080 \\"
    echo "  --host 0.0.0.0"
    echo ""
    echo "🦙 使用llama.cpp（适用于AMD GPU）："
    echo "1. 转换模型: python convert.py $LOCAL_DIR"
    echo "2. 量化: ./quantize ./models/model.gguf ./models/model-q4_0.gguf q4_0"
    echo "3. 启动: ./server -m ./models/model-q4_0.gguf -c 2048 --host 0.0.0.0 --port 8080"
    echo "========================================="
else
    echo "❌ 下载失败，请检查网络连接或模型名称"
    echo ""
    echo "💡 故障排除："
    echo "1. 检查模型名称是否正确"
    echo "2. 尝试使用镜像: export HF_ENDPOINT=https://hf-mirror.com"
    echo "3. 检查磁盘空间是否足够"
    echo "4. 如果需要认证，运行: huggingface-cli login"
fi
