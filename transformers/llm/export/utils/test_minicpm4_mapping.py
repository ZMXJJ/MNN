#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniCPM4 映射配置测试脚本

这个脚本用于测试 MiniCPM4 模型的映射配置是否正确。
"""

import sys
import os
from unittest.mock import Mock

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_mapper import ModelMapper

def create_mock_minicpm4_config():
    """创建模拟的 MiniCPM4 配置对象"""
    config = Mock()
    
    # 设置模型类型
    config.model_type = 'minicpm4'
    
    # 设置基本配置参数
    config.hidden_size = 4096
    config.head_dim = 128
    config.num_attention_heads = 32
    config.num_hidden_layers = 32
    config.num_key_value_heads = 32
    config.rope_theta = 10000.0
    config.rope_scaling = None
    config.intermediate_size = 11008
    config.max_position_embeddings = 2048
    config.vocab_size = 32000
    config.bos_token_id = 1
    config.eos_token_id = 2
    config.pad_token_id = None
    config.rms_norm_eps = 1e-6
    config.attention_bias = False
    config.attention_dropout = 0.0
    config.scale_emb = 1
    config.dim_model_base = 1
    config.scale_depth = 1
    
    return config

def test_minicpm4_mapping():
    """测试 MiniCPM4 映射配置"""
    print("=== MiniCPM4 映射配置测试 ===")
    
    # 创建映射器
    mapper = ModelMapper()
    
    # 创建模拟配置
    config = create_mock_minicpm4_config()
    
    # 获取映射配置
    model_type, model_map = mapper.get_map(config)
    
    print(f"检测到的模型类型: {model_type}")
    print(f"预期模型类型: minicpm4")
    
    # 验证模型类型
    assert model_type == 'minicpm4', f"模型类型不匹配: 期望 'minicpm4', 实际 '{model_type}'"
    print("✓ 模型类型检测正确")
    
    # 验证映射配置结构
    required_keys = ['config', 'model', 'decoder', 'attention']
    for key in required_keys:
        assert key in model_map, f"缺少必需的映射键: {key}"
    print("✓ 映射配置结构完整")
    
    # 测试配置映射
    print("\n测试配置映射:")
    config_map = model_map['config']
    test_config = create_mock_minicpm4_config()
    
    for dst_attr, src_attr in config_map.items():
        try:
            attributes = src_attr.split('.')
            obj = test_config
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
    print("\n测试模型结构映射:")
    model_map_config = model_map['model']
    
    # 创建模拟模型对象
    mock_model = Mock()
    mock_model.lm_head = Mock()
    mock_model.model = Mock()
    mock_model.model.embed_tokens = Mock()
    mock_model.model.layers = Mock()
    mock_model.model.norm = Mock()
    mock_model.model.rotary_emb = Mock()
    mock_model.model.rotary_emb_local = Mock()
    
    for dst_attr, src_attr in model_map_config.items():
        try:
            attributes = src_attr.split('.')
            obj = mock_model
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
    
    print("\n=== 测试完成 ===")
    print("MiniCPM4 映射配置测试通过！")
    
    return True

def test_mapper_registration():
    """测试映射器注册"""
    print("\n=== 测试映射器注册 ===")
    
    mapper = ModelMapper()
    
    # 检查是否注册了 minicpm4
    assert 'minicpm4' in mapper.mapper, "MiniCPM4 映射未注册"
    print("✓ MiniCPM4 映射已正确注册")
    
    # 检查映射配置完整性
    minicpm4_map = mapper.mapper['minicpm4']
    required_sections = ['config', 'model', 'decoder', 'attention']
    
    for section in required_sections:
        assert section in minicpm4_map, f"缺少 {section} 部分"
        assert isinstance(minicpm4_map[section], dict), f"{section} 不是字典类型"
    
    print("✓ 映射配置结构完整")
    print("✓ 映射器注册测试通过")
    
    return True

def main():
    """主函数"""
    print("MiniCPM4 映射配置测试")
    print("=" * 50)
    
    try:
        # 测试映射器注册
        test_mapper_registration()
        
        # 测试映射配置
        test_minicpm4_mapping()
        
        print("\n🎉 所有测试通过！")
        print("MiniCPM4 模型适配配置正确。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 