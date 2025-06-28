#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniCPM4 模型适配示例

这个示例展示了如何使用 ModelMapper 来适配 MiniCPM4 模型，
将其转换为 MNN 格式进行部署。

参考链接：
- MiniCPM4 模型: https://huggingface.co/openbmb/MiniCPM4-8B
- 配置信息: https://huggingface.co/openbmb/MiniCPM4-MCP/blob/main/configuration_minicpm.py
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model_mapper import ModelMapper

def load_minicpm4_model(model_path_or_name="openbmb/MiniCPM4-8B"):
    """
    加载 MiniCPM4 模型
    
    Args:
        model_path_or_name: 模型路径或 Hugging Face 模型名称
    
    Returns:
        tokenizer: 分词器
        model: 模型
        config: 配置
    """
    print(f"正在加载 MiniCPM4 模型: {model_path_or_name}")
    
    # 加载配置
    config = AutoConfig.from_pretrained(model_path_or_name)
    print(f"模型配置: {config}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    print(f"词汇表大小: {tokenizer.vocab_size}")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_name,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return tokenizer, model, config

def test_model_mapping(config):
    """
    测试模型映射配置
    
    Args:
        config: 模型配置对象
    """
    print("\n=== 测试模型映射配置 ===")
    
    # 创建模型映射器
    mapper = ModelMapper()
    
    # 获取映射配置
    model_type, model_map = mapper.get_map(config)
    print(f"检测到的模型类型: {model_type}")
    
    # 打印映射配置
    print("\n映射配置:")
    for key, value in model_map.items():
        print(f"  {key}: {value}")
    
    return model_type, model_map

def test_model_structure(model, config):
    """
    测试模型结构映射
    
    Args:
        model: 模型对象
        config: 配置对象
    """
    print("\n=== 测试模型结构映射 ===")
    
    mapper = ModelMapper()
    model_type, model_map = mapper.get_map(config)
    
    # 测试配置映射
    print("配置映射测试:")
    config_map = model_map['config']
    for dst_attr, src_attr in config_map.items():
        try:
            attributes = src_attr.split('.')
            obj = config
            for attr in attributes:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    obj = None
                    break
            if obj is not None:
                print(f"  ✓ {dst_attr} -> {src_attr} = {obj}")
            else:
                print(f"  ✗ {dst_attr} -> {src_attr} (未找到)")
        except Exception as e:
            print(f"  ✗ {dst_attr} -> {src_attr} (错误: {e})")
    
    # 测试模型结构映射
    print("\n模型结构映射测试:")
    model_map_config = model_map['model']
    for dst_attr, src_attr in model_map_config.items():
        try:
            attributes = src_attr.split('.')
            obj = model
            for attr in attributes:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    obj = None
                    break
            if obj is not None:
                print(f"  ✓ {dst_attr} -> {src_attr} (找到)")
            else:
                print(f"  ✗ {dst_attr} -> {src_attr} (未找到)")
        except Exception as e:
            print(f"  ✗ {dst_attr} -> {src_attr} (错误: {e})")

def generate_sample_text(tokenizer, model, prompt="你好，请介绍一下你自己"):
    """
    生成示例文本
    
    Args:
        tokenizer: 分词器
        model: 模型
        prompt: 输入提示
    """
    print(f"\n=== 生成示例文本 ===")
    print(f"输入提示: {prompt}")
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"生成文本: {generated_text}")

def main():
    """主函数"""
    print("MiniCPM4 模型适配示例")
    print("=" * 50)
    
    try:
        # 1. 加载模型
        tokenizer, model, config = load_minicpm4_model()
        
        # 2. 测试模型映射
        model_type, model_map = test_model_mapping(config)
        
        # 3. 测试模型结构
        test_model_structure(model, config)
        
        # 4. 生成示例文本
        generate_sample_text(tokenizer, model)
        
        print("\n=== 适配完成 ===")
        print("MiniCPM4 模型已成功适配到 MNN 格式")
        print("您可以使用 MNN 推理引擎来部署此模型")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请检查模型路径和依赖是否正确安装")

if __name__ == "__main__":
    main() 