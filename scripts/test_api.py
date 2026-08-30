#!/usr/bin/env python3
"""
vLLM API测试脚本
用于测试vLLM服务器是否正常运行
"""

import requests
import json
import time
import sys

API_BASE_URL = "http://localhost:8080"

def test_health():
    """测试服务器健康状态"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ 服务器健康检查通过")
            return True
        else:
            print(f"❌ 服务器健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
        return False

def test_models():
    """获取可用模型列表"""
    try:
        response = requests.get(f"{API_BASE_URL}/v1/models")
        if response.status_code == 200:
            models = response.json()
            print("✅ 获取模型列表成功:")
            print(json.dumps(models, indent=2, ensure_ascii=False))
            return True
        else:
            print(f"❌ 获取模型列表失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def test_chat_completion():
    """测试聊天补全API"""
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "你好，请简单介绍一下你自己。"}
    ]
    
    payload = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False
    }
    
    try:
        print("\n正在发送聊天请求...")
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 聊天补全成功!")
            print("\n回复内容:")
            print(result['choices'][0]['message']['content'])
            print("\n使用情况:")
            print(f"  输入tokens: {result['usage']['prompt_tokens']}")
            print(f"  输出tokens: {result['usage']['completion_tokens']}")
            print(f"  总tokens: {result['usage']['total_tokens']}")
            return True
        else:
            print(f"❌ 聊天补全失败: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def test_streaming():
    """测试流式输出"""
    messages = [
        {"role": "user", "content": "数到5"}
    ]
    
    payload = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": True
    }
    
    try:
        print("\n测试流式输出...")
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=True
        )
        
        if response.status_code == 200:
            print("✅ 流式响应:")
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data = line_str[6:]
                        if data != "[DONE]":
                            try:
                                chunk = json.loads(data)
                                content = chunk['choices'][0].get('delta', {}).get('content', '')
                                if content:
                                    print(content, end='', flush=True)
                            except:
                                pass
            print("\n")
            return True
        else:
            print(f"❌ 流式输出失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def main():
    print("=" * 50)
    print("vLLM API 测试工具")
    print("=" * 50)
    
    # 等待服务器启动
    print("\n正在检查服务器状态...")
    max_retries = 30
    for i in range(max_retries):
        if test_health():
            break
        if i < max_retries - 1:
            print(f"重试中... ({i+1}/{max_retries})")
            time.sleep(2)
    else:
        print("\n⚠️  服务器未响应，请确保vLLM服务已启动")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("开始测试...")
    print("=" * 50)
    
    # 运行测试
    tests = [
        ("获取模型列表", test_models),
        ("聊天补全", test_chat_completion),
        ("流式输出", test_streaming)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📝 测试: {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
        time.sleep(1)
    
    # 输出总结
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠️  部分测试失败，请检查日志")
        sys.exit(1)

if __name__ == "__main__":
    main()
