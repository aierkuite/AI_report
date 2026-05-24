"""第 2 步小参数量 GPT 类模型架构验证程序

本程序用于完成课程任务中的模型搭建部分
它只验证模型结构、前向传播和随机权重生成流程
预训练循环、数据准备和微调流程将在后续步骤中实现
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """保存小型 GPT 模型的核心超参数配置

    参数含义:
        vocab_size: 词表大小
        context_length: 模型一次最多能处理的 token 数量
        emb_dim: token 向量和隐藏状态的维度
        n_heads: 多头注意力的头数
        n_layers: Transformer 解码器模块层数
        dropout: Dropout 随机失活比例

    返回值含义:
        GPTConfig 实例用于传递模型结构配置
    """

    vocab_size: int = 50_257
    context_length: int = 128
    emb_dim: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """实现 GPT 中使用的因果多头自注意力模块"""

    def __init__(self, config: GPTConfig) -> None:
        """初始化因果多头自注意力层

        参数含义:
            config: GPTConfig 配置对象，提供隐藏维度、头数、上下文长度和 dropout

        返回值含义:
            无返回值，初始化注意力相关权重和因果掩码
        """

        super().__init__()
        if config.emb_dim % config.n_heads != 0:
            raise ValueError("emb_dim 必须能被 n_heads 整除")

        self.n_heads = config.n_heads
        self.head_dim = config.emb_dim // config.n_heads
        self.qkv = nn.Linear(config.emb_dim, 3 * config.emb_dim)
        self.proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算带因果掩码的多头自注意力输出

        参数含义:
            x: 输入隐藏状态，形状为 batch_size、seq_len、emb_dim

        返回值含义:
            返回注意力后的隐藏状态，形状与输入 x 相同
        """

        batch_size, seq_len, emb_dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :seq_len, :seq_len] == 0
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ value
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        out = self.proj(out)
        out = self.resid_drop(out)
        return out


class FeedForward(nn.Module):
    """实现 GPT 解码器块中的前馈网络模块"""

    def __init__(self, config: GPTConfig) -> None:
        """初始化前馈网络层

        参数含义:
            config: GPTConfig 配置对象，提供隐藏维度和 dropout

        返回值含义:
            无返回值，初始化两层线性变换和激活函数
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.emb_dim, 4 * config.emb_dim),
            nn.GELU(),
            nn.Linear(4 * config.emb_dim, config.emb_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算前馈网络输出

        参数含义:
            x: 输入隐藏状态，形状为 batch_size、seq_len、emb_dim

        返回值含义:
            返回前馈网络变换后的隐藏状态，形状与输入 x 相同
        """

        return self.net(x)


class TransformerBlock(nn.Module):
    """实现一个 GPT Transformer 解码器块"""

    def __init__(self, config: GPTConfig) -> None:
        """初始化 Transformer 解码器块

        参数含义:
            config: GPTConfig 配置对象，提供注意力、前馈网络和归一化层所需参数

        返回值含义:
            无返回值，初始化一层因果注意力、一层前馈网络和两个 LayerNorm
        """

        super().__init__()
        self.ln_1 = nn.LayerNorm(config.emb_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.emb_dim)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行一个解码器块的前向计算

        参数含义:
            x: 输入隐藏状态，形状为 batch_size、seq_len、emb_dim

        返回值含义:
            返回经过注意力、前馈网络和残差连接后的隐藏状态
        """

        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """实现小参数量 GPT 类语言模型"""

    def __init__(self, config: GPTConfig) -> None:
        """初始化小型 GPT 模型

        参数含义:
            config: GPTConfig 配置对象，提供词表大小、上下文长度和网络结构参数

        返回值含义:
            无返回值，初始化嵌入层、Transformer 层和输出语言模型头
        """

        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.emb_dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """初始化模型权重

        参数含义:
            module: 当前正在初始化的 PyTorch 模块

        返回值含义:
            无返回值，直接修改模块内部参数
        """

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """根据输入 token 序列计算下一个 token 的预测 logits

        参数含义:
            idx: 输入 token 编号，形状为 batch_size、seq_len

        返回值含义:
            返回每个位置对词表中下一个 token 的预测分数，形状为 batch_size、seq_len、vocab_size
        """

        _, seq_len = idx.shape
        if seq_len > self.config.context_length:
            raise ValueError(f"输入序列长度不能超过 context_length={self.config.context_length}")

        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)
        x = self.drop(token_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """统计模型参数量

    参数含义:
        model: 需要统计参数量的 PyTorch 模型

    返回值含义:
        返回二元组，第一个值是总参数量，第二个值是可训练参数量
    """

    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


@torch.no_grad()
def generate(
    model: MiniGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = 20,
) -> torch.Tensor:
    """基于当前模型连续生成新 token

    参数含义:
        model: 用于推理生成的 MiniGPT 模型
        idx: 初始 token 序列，形状为 batch_size、seq_len
        max_new_tokens: 需要继续生成的新 token 数量
        temperature: 采样温度，数值越大随机性越强
        top_k: 每一步只从概率最高的前 k 个 token 中采样，传入 None 表示不限制

    返回值含义:
        返回拼接了新 token 后的完整 token 序列
    """

    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.context_length :]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_idx), dim=1)

    return idx


def main() -> None:
    """运行第 2 步模型架构验证流程

    参数含义:
        无参数

    返回值含义:
        无返回值，直接在控制台打印模型配置、参数量、前向传播结果和生成结果
    """

    torch.manual_seed(42)
    config = GPTConfig()
    model = MiniGPT(config)

    total_params, trainable_params = count_parameters(model)
    print("模型配置:")
    print(config)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    batch_size = 2
    seq_len = 16
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits = model(dummy_input)

    print("\n前向传播验证:")
    print(f"输入 token 形状: {tuple(dummy_input.shape)}")
    print(f"输出 logits 形状: {tuple(logits.shape)}")
    print(f"期望 logits 形状: ({batch_size}, {seq_len}, {config.vocab_size})")

    prompt_tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    generated_tokens = generate(model, prompt_tokens, max_new_tokens=20, temperature=1.0, top_k=20)

    print("\n随机权重生成验证:")
    print(f"初始 token: {prompt_tokens.tolist()[0]}")
    print(f"生成 token: {generated_tokens.tolist()[0]}")


if __name__ == "__main__":
    main()
