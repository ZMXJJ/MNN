# MiniCPM4 模型适配指南

## 概述

本指南介绍如何将 MiniCPM4 模型适配到 MNN 推理引擎，实现高效的模型部署。

## MiniCPM4 模型特点

MiniCPM4 是一个基于 Transformer 架构的大语言模型，具有以下特点：

- **模型架构**: 基于 LLaMA 架构改进
- **参数量**: 支持多种规模（如 8B 参数）
- **多模态**: 支持文本、图像、音频等多种模态
- **高效推理**: 优化的注意力机制和位置编码

## 适配配置

### 1. 模型映射配置

在 `model_mapper.py` 中，我们已经为 MiniCPM4 添加了完整的映射配置：

```python
def regist_minicpm4(self):
    minicpm4_map = {
        'config': {
            'hidden_size': 'hidden_size',
            'head_dim': 'head_dim',
            'num_attention_heads': 'num_attention_heads',
            'num_hidden_layers': 'num_hidden_layers',
            'num_key_value_heads': 'num_key_value_heads',
            'rope_theta': 'rope_theta',
            'rope_scaling': 'rope_scaling',
            'intermediate_size': 'intermediate_size',
            'max_position_embeddings': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'bos_token_id': 'bos_token_id',
            'eos_token_id': 'eos_token_id',
            'pad_token_id': 'pad_token_id',
            'rms_norm_eps': 'rms_norm_eps',
            'attention_bias': 'attention_bias',
            'attention_dropout': 'attention_dropout',
            'scale_emb': 'scale_emb',
            'dim_model_base': 'dim_model_base',
            'scale_depth': 'scale_depth'
        },
        'model': {
            'lm_': 'lm_head',
            'embed_': 'model.embed_tokens',
            'blocks_': 'model.layers',
            'final_layernorm_': 'model.norm',
            'rotary_emb': 'model.rotary_emb',
            'rotary_emb_local': 'model.rotary_emb_local'
        },
        'decoder': {
            'self_attn': 'self_attn',
            'mlp': 'mlp',
            'input_layernorm': 'input_layernorm',
            'post_attention_layernorm': 'post_attention_layernorm',
            'pre_feedforward_layernorm': 'pre_feedforward_layernorm',
            'post_feedforward_layernorm': 'post_feedforward_layernorm'
        },
        'attention': {
            'q_proj': 'q_proj',
            'k_proj': 'k_proj',
            'v_proj': 'v_proj',
            'o_proj': 'o_proj',
            'q_norm': 'q_norm',
            'k_norm': 'k_norm'
        }
    }
    self.regist('minicpm4', minicpm4_map)
```

### 2. 关键配置参数

MiniCPM4 的关键配置参数包括：

- **hidden_size**: 隐藏层维度
- **num_attention_heads**: 注意力头数
- **num_hidden_layers**: Transformer 层数
- **rope_theta**: 旋转位置编码参数
- **intermediate_size**: MLP 中间层维度
- **scale_emb**: 嵌入层缩放因子
- **scale_depth**: 深度缩放因子

## 使用方法

### 1. 环境准备

```bash
# 安装依赖
pip install torch transformers accelerate
pip install mnn  # MNN 推理引擎
```

### 2. 运行适配示例

```bash
cd transformers/llm/export/utils/
python minicpm4_example.py
```

### 3. 自定义使用

```python
from model_mapper import ModelMapper
from transformers import AutoConfig, AutoModelForCausalLM

# 加载模型配置
config = AutoConfig.from_pretrained("openbmb/MiniCPM4-8B")

# 创建映射器
mapper = ModelMapper()

# 获取映射配置
model_type, model_map = mapper.get_map(config)

# 使用映射配置进行模型转换
# ... 转换逻辑
```

## 模型结构映射

### 配置映射 (config)

| 目标属性 | 源属性 | 说明 |
|---------|--------|------|
| hidden_size | hidden_size | 隐藏层维度 |
| num_attention_heads | num_attention_heads | 注意力头数 |
| num_hidden_layers | num_hidden_layers | Transformer 层数 |
| rope_theta | rope_theta | 旋转位置编码参数 |
| intermediate_size | intermediate_size | MLP 中间层维度 |

### 模型结构映射 (model)

| 目标属性 | 源属性 | 说明 |
|---------|--------|------|
| lm_ | lm_head | 语言模型输出层 |
| embed_ | model.embed_tokens | 词嵌入层 |
| blocks_ | model.layers | Transformer 层 |
| final_layernorm_ | model.norm | 最终层归一化 |
| rotary_emb | model.rotary_emb | 旋转位置编码 |

### 解码器映射 (decoder)

| 目标属性 | 源属性 | 说明 |
|---------|--------|------|
| self_attn | self_attn | 自注意力机制 |
| mlp | mlp | 多层感知机 |
| input_layernorm | input_layernorm | 输入层归一化 |
| post_attention_layernorm | post_attention_layernorm | 注意力后层归一化 |

### 注意力机制映射 (attention)

| 目标属性 | 源属性 | 说明 |
|---------|--------|------|
| q_proj | q_proj | 查询投影 |
| k_proj | k_proj | 键投影 |
| v_proj | v_proj | 值投影 |
| o_proj | o_proj | 输出投影 |
| q_norm | q_norm | 查询归一化 |
| k_norm | k_norm | 键归一化 |

## 性能优化建议

### 1. 量化优化

```python
# 使用 INT8 量化
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 2. 内存优化

```python
# 使用梯度检查点
model.gradient_checkpointing_enable()

# 使用混合精度
from torch.cuda.amp import autocast
with autocast():
    outputs = model(**inputs)
```

### 3. 推理优化

```python
# 设置推理模式
model.eval()

# 使用 torch.no_grad()
with torch.no_grad():
    outputs = model(**inputs)
```

## 常见问题

### Q1: 模型加载失败
**A**: 检查模型路径和网络连接，确保能正常访问 Hugging Face 模型库。

### Q2: 内存不足
**A**: 使用模型量化、梯度检查点或减少批处理大小。

### Q3: 推理速度慢
**A**: 启用混合精度推理，使用 GPU 加速，或进行模型优化。

### Q4: 映射配置错误
**A**: 检查模型版本和配置参数是否匹配，确保映射配置正确。

## 参考资源

- [MiniCPM4 模型页面](https://huggingface.co/openbmb/MiniCPM4-8B)
- [MiniCPM4 配置文档](https://huggingface.co/openbmb/MiniCPM4-MCP/blob/main/configuration_minicpm.py)
- [MNN 官方文档](https://www.yuque.com/mnn)
- [Transformers 文档](https://huggingface.co/docs/transformers)

## 更新日志

- **v1.0.0**: 初始版本，支持基本的 MiniCPM4 模型适配
- **v1.1.0**: 添加了完整的配置参数映射
- **v1.2.0**: 优化了性能和使用体验 