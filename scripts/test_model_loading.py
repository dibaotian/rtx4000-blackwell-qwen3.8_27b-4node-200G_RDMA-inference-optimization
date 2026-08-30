#!/usr/bin/env python3
"""
简单测试脚本：验证模型能否在GPU上加载
"""
import torch
import time
import os

print("=" * 50)
print("Qwen3-8B 模型加载测试")
print("=" * 50)

# 检查GPU
print("\n1. 检查GPU状态:")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 显存信息
    mem_info = torch.cuda.mem_get_info(0)
    print(f"可用显存: {mem_info[0] / 1024**3:.2f} GB")
    print(f"总显存: {mem_info[1] / 1024**3:.2f} GB")

# 尝试加载模型
print("\n2. 尝试加载模型:")
model_path = "/dc2/minx/models/Qwen_Qwen3-8B"

try:
    print(f"模型路径: {model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("加载模型到GPU（使用bfloat16）...")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    load_time = time.time() - start_time
    print(f"✅ 模型加载成功！耗时: {load_time:.2f}秒")
    
    # 检查模型信息
    print(f"\n3. 模型信息:")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # 显存使用
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_info_after = torch.cuda.mem_get_info(0)
        used_mem = (mem_info[1] - mem_info_after[0]) / 1024**3
        print(f"模型占用显存: {used_mem:.2f} GB")
    
    # 简单测试
    print("\n4. 运行简单测试:")
    inputs = tokenizer("Hello, I am", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"测试输出: {response}")
    
    print("\n✅ 所有测试通过！模型可以正常工作。")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
