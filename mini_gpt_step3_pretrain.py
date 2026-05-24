"""第 3 步小参数量 GPT 类模型预训练程序

本程序用于完成课程任务中的预训练部分
它包含文本数据读取、字符级分词、训练样本构建、预训练循环、效果测试和权重保存
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mini_gpt_step2 import GPTConfig, MiniGPT, count_parameters, generate


DEFAULT_CORPUS = """
人工智能正在改变学习、工作和科研的方式。一个小型语言模型也可以展示大模型的核心机制。
GPT 类模型通过阅读大量文本，学习上下文中 token 之间的统计关系，并预测下一个 token。
预训练阶段不需要人工标注答案，只需要连续文本。模型看到前面的内容，目标是预测后面的内容。
Transformer 解码器使用因果自注意力，保证当前位置只能关注当前位置及之前的信息。
词嵌入把离散 token 转换为向量，位置嵌入帮助模型理解 token 的顺序。
训练过程中，交叉熵损失会衡量模型预测分布和真实下一个 token 之间的差距。
当训练损失逐渐下降时，模型通常能生成更接近训练语料风格的文本。
从零构建大模型的实践，可以帮助理解分词、嵌入、注意力、优化器和生成策略。

Large language models learn by predicting the next token from context.
A small GPT model can be trained on a normal computer to demonstrate the same core ideas.
Pretraining uses unlabeled text, tokenization, embeddings, causal attention, loss calculation, and optimization.
After training, the model should generate text that is more consistent with the training corpus.
"""


@dataclass
class PretrainConfig:
    """保存第 3 步预训练流程的核心配置

    参数含义:
        data_dir: 存放预训练 txt 文本语料的目录
        output_dir: 存放模型权重、分词器和训练日志的目录
        context_length: 每个训练样本包含的输入 token 数量
        emb_dim: 模型隐藏向量维度
        n_heads: 多头注意力头数
        n_layers: Transformer 解码器层数
        dropout: Dropout 随机失活比例
        batch_size: 每次训练使用的样本数量
        max_steps: 最大训练步数
        eval_interval: 每隔多少步评估一次训练集和验证集损失
        eval_batches: 每次评估最多使用的 batch 数量
        learning_rate: AdamW 优化器学习率
        weight_decay: AdamW 优化器权重衰减系数
        grad_clip: 梯度裁剪阈值
        train_split: 训练集 token 占全部 token 的比例
        seed: 随机种子
        prompt: 训练前后用于生成效果对比的提示文本
        generate_tokens: 每次效果测试生成的新 token 数量
        temperature: 生成采样温度
        top_k: 生成时保留概率最高的候选 token 数量
        device: 运行设备，auto 表示自动选择 cuda 或 cpu

    返回值含义:
        PretrainConfig 实例用于统一传递预训练参数
    """

    data_dir: str = "data_pretrain"
    output_dir: str = "outputs_pretrain"
    context_length: int = 64
    emb_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    batch_size: int = 16
    max_steps: int = 300
    eval_interval: int = 50
    eval_batches: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    train_split: float = 0.9
    seed: int = 42
    prompt: str = "人工智能"
    generate_tokens: int = 120
    temperature: float = 0.8
    top_k: int = 20
    device: str = "auto"


class CharTokenizer:
    """实现面向小规模实验的字符级分词器"""

    pad_token = "<PAD>"
    unk_token = "<UNK>"

    def __init__(self, vocab: list[str]) -> None:
        """初始化字符级分词器

        参数含义:
            vocab: 按编号顺序排列的 token 字符列表，必须包含特殊 token

        返回值含义:
            无返回值，初始化 token 到编号和编号到 token 的映射
        """

        self.itos = vocab
        self.stoi = {token: idx for idx, token in enumerate(vocab)}
        self.pad_id = self.stoi[self.pad_token]
        self.unk_id = self.stoi[self.unk_token]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """根据训练文本构建字符级词表

        参数含义:
            text: 用于提取字符集合的训练文本

        返回值含义:
            返回根据文本字符集合构建完成的 CharTokenizer 实例
        """

        chars = sorted(set(text))
        chars = [char for char in chars if char not in {cls.pad_token, cls.unk_token}]
        vocab = [cls.pad_token, cls.unk_token] + chars
        return cls(vocab)

    def encode(self, text: str) -> list[int]:
        """把文本转换为 token 编号序列

        参数含义:
            text: 待编码的原始文本

        返回值含义:
            返回 token 编号列表，未知字符会映射为 unk_id
        """

        return [self.stoi.get(char, self.unk_id) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        """把 token 编号序列还原为文本

        参数含义:
            token_ids: 待解码的 token 编号列表

        返回值含义:
            返回解码后的字符串，PAD 会被跳过，UNK 会显示为问号
        """

        chars: list[str] = []
        for token_id in token_ids:
            if token_id == self.pad_id:
                continue
            if token_id == self.unk_id:
                chars.append("?")
                continue
            chars.append(self.itos[token_id])
        return "".join(chars)

    def to_dict(self) -> dict[str, object]:
        """把分词器转换为可保存的字典结构

        参数含义:
            无参数

        返回值含义:
            返回包含词表和特殊 token 信息的字典
        """

        return {
            "type": "char",
            "itos": self.itos,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }


class LanguageModelDataset(Dataset):
    """把连续 token 序列切分为 GPT 预训练样本"""

    def __init__(self, token_ids: list[int], context_length: int) -> None:
        """初始化语言模型数据集

        参数含义:
            token_ids: 连续文本编码后的 token 编号序列
            context_length: 每个输入样本包含的 token 数量

        返回值含义:
            无返回值，保存 token 序列和上下文长度
        """

        if len(token_ids) <= context_length:
            raise ValueError("token 数量必须大于 context_length")

        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        """返回可构造的训练样本数量

        参数含义:
            无参数

        返回值含义:
            返回数据集中可取出的样本数量
        """

        return len(self.token_ids) - self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """根据索引返回一个自回归语言模型训练样本

        参数含义:
            index: 样本起始位置索引

        返回值含义:
            返回二元组 x 和 y，x 是输入 token，y 是右移一位后的预测目标
        """

        x = self.token_ids[index : index + self.context_length]
        y = self.token_ids[index + 1 : index + self.context_length + 1]
        return x, y


def parse_args() -> PretrainConfig:
    """解析命令行参数并生成预训练配置

    参数含义:
        无参数

    返回值含义:
        返回从命令行参数生成的 PretrainConfig 实例
    """

    parser = argparse.ArgumentParser(description="第 3 步小参数量 GPT 类模型预训练程序")
    parser.add_argument("--data-dir", default="data_pretrain", help="存放 txt 预训练语料的目录")
    parser.add_argument("--output-dir", default="outputs_pretrain", help="保存预训练产物的目录")
    parser.add_argument("--context-length", type=int, default=64, help="上下文 token 长度")
    parser.add_argument("--emb-dim", type=int, default=128, help="隐藏向量维度")
    parser.add_argument("--n-heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--n-layers", type=int, default=4, help="Transformer 层数")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比例")
    parser.add_argument("--batch-size", type=int, default=16, help="每个 batch 的样本数量")
    parser.add_argument("--max-steps", type=int, default=1300, help="最大训练步数")
    parser.add_argument("--eval-interval", type=int, default=50, help="评估间隔步数")
    parser.add_argument("--eval-batches", type=int, default=10, help="每次评估最多使用的 batch 数量")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="权重衰减")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--train-split", type=float, default=0.9, help="训练集 token 比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--prompt", default="人工智能", help="训练前后用于生成对比的提示文本")
    parser.add_argument("--generate-tokens", type=int, default=120, help="生成的新 token 数量")
    parser.add_argument("--temperature", type=float, default=0.8, help="生成采样温度")
    parser.add_argument("--top-k", type=int, default=20, help="生成时保留的候选 token 数量")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    args = parser.parse_args()
    return PretrainConfig(**vars(args))


def set_seed(seed: int) -> None:
    """设置 Python 和 PyTorch 随机种子

    参数含义:
        seed: 随机种子整数

    返回值含义:
        无返回值，直接影响后续随机数生成过程
    """

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    """选择模型训练设备

    参数含义:
        device_name: 设备名称，支持 auto、cpu、cuda

    返回值含义:
        返回 PyTorch 设备对象
    """

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 cuda，但当前环境不可用")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_text(text: str) -> str:
    """对原始文本做轻量清洗

    参数含义:
        text: 从语料文件读取到的原始文本

    返回值含义:
        返回统一换行并去除首尾空白后的文本
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def load_corpus(data_dir: Path) -> tuple[str, list[Path]]:
    """读取预训练语料目录中的所有 txt 文件

    参数含义:
        data_dir: 存放 txt 文本语料的目录路径

    返回值含义:
        返回二元组，第一个值是合并后的文本，第二个值是实际读取的文件路径列表
    """

    text_files = sorted(data_dir.rglob("*.txt")) if data_dir.exists() else []
    parts: list[str] = []
    for file_path in text_files:
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            content = normalize_text(file.read())
        if content:
            parts.append(content)

    if parts:
        return "\n\n".join(parts), text_files

    return normalize_text(DEFAULT_CORPUS), []


def ensure_minimum_corpus_size(text: str, min_chars: int) -> str:
    """在语料过小时重复文本以保证训练样本数量

    参数含义:
        text: 原始或内置预训练文本
        min_chars: 期望达到的最少字符数量

    返回值含义:
        返回长度不小于 min_chars 的训练文本
    """

    if len(text) >= min_chars:
        return text
    repeats = (min_chars // max(len(text), 1)) + 1
    return ("\n" + text) * repeats


def split_tokens(token_ids: list[int], train_split: float, context_length: int) -> tuple[list[int], list[int]]:
    """把连续 token 序列切分为训练集和验证集

    参数含义:
        token_ids: 全部语料编码后的 token 编号序列
        train_split: 训练集 token 占比
        context_length: 上下文长度，用于保证验证集至少能构造一个样本

    返回值含义:
        返回二元组，第一个值是训练 token，第二个值是验证 token
    """

    split_index = int(len(token_ids) * train_split)
    split_index = max(split_index, context_length + 1)
    split_index = min(split_index, len(token_ids) - context_length - 1)
    train_ids = token_ids[:split_index]
    valid_ids = token_ids[split_index:]
    return train_ids, valid_ids


def build_dataloaders(
    token_ids: list[int],
    config: PretrainConfig,
) -> tuple[DataLoader, DataLoader, list[int], list[int]]:
    """构建训练集和验证集 DataLoader

    参数含义:
        token_ids: 全部语料编码后的 token 编号序列
        config: 预训练流程配置对象

    返回值含义:
        返回训练 DataLoader、验证 DataLoader、训练 token 列表和验证 token 列表
    """

    train_ids, valid_ids = split_tokens(token_ids, config.train_split, config.context_length)
    train_dataset = LanguageModelDataset(train_ids, config.context_length)
    valid_dataset = LanguageModelDataset(valid_ids, config.context_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    return train_loader, valid_loader, train_ids, valid_ids


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    """评估模型在指定数据集上的平均交叉熵损失

    参数含义:
        model: 待评估的 MiniGPT 模型
        data_loader: 训练集或验证集 DataLoader
        device: 执行评估的设备
        max_batches: 最多评估的 batch 数量

    返回值含义:
        返回平均交叉熵损失，若数据为空则返回无穷大
    """

    model.eval()
    losses: list[float] = []
    for batch_index, (x, y) in enumerate(data_loader):
        if batch_index >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    if not losses:
        return float("inf")
    return sum(losses) / len(losses)


def generate_text(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    """使用模型根据提示文本生成可读文本

    参数含义:
        model: 用于生成的 MiniGPT 模型
        tokenizer: 字符级分词器
        prompt: 生成起始提示文本
        max_new_tokens: 需要继续生成的新 token 数量
        temperature: 采样温度
        top_k: 每步采样保留的候选 token 数量
        device: 执行生成的设备

    返回值含义:
        返回解码后的完整生成文本
    """

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.unk_id]
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    generated = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(generated[0].tolist())


def train_model(
    model: MiniGPT,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: PretrainConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    """执行 GPT 预训练循环

    参数含义:
        model: 待训练的 MiniGPT 模型
        train_loader: 训练集 DataLoader
        valid_loader: 验证集 DataLoader
        config: 预训练流程配置对象
        device: 执行训练的设备

    返回值含义:
        返回训练日志列表，每个元素记录 step、train_loss 和 valid_loss
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.train()
    loss_log: list[dict[str, float]] = []
    train_iter = iter(train_loader)

    for step in range(1, config.max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        should_eval = step == 1 or step % config.eval_interval == 0 or step == config.max_steps
        if should_eval:
            train_loss = estimate_loss(model, train_loader, device, config.eval_batches)
            valid_loss = estimate_loss(model, valid_loader, device, config.eval_batches)
            loss_log.append(
                {
                    "step": float(step),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                }
            )
            print(f"step {step:4d} | train_loss {train_loss:.4f} | valid_loss {valid_loss:.4f}")

    return loss_log


def save_json(data: dict[str, object], file_path: Path) -> None:
    """把字典数据保存为 UTF-8 JSON 文件

    参数含义:
        data: 需要保存的字典数据
        file_path: JSON 文件保存路径

    返回值含义:
        无返回值，直接写入文件
    """

    with open(file_path, "w", encoding="utf-8", newline="\r\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_loss_log(loss_log: list[dict[str, float]], file_path: Path) -> None:
    """把训练损失日志保存为 CSV 文件

    参数含义:
        loss_log: 训练过程中记录的损失列表
        file_path: CSV 文件保存路径

    返回值含义:
        无返回值，直接写入文件
    """

    with open(file_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "train_loss", "valid_loss"], lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(loss_log)


def save_checkpoint(
    model: MiniGPT,
    model_config: GPTConfig,
    train_config: PretrainConfig,
    tokenizer: CharTokenizer,
    loss_log: list[dict[str, float]],
    output_dir: Path,
) -> None:
    """保存预训练模型权重、分词器和训练配置

    参数含义:
        model: 训练完成的 MiniGPT 模型
        model_config: 模型结构配置对象
        train_config: 预训练流程配置对象
        tokenizer: 字符级分词器
        loss_log: 训练损失日志
        output_dir: 输出目录路径

    返回值含义:
        无返回值，直接在输出目录写入文件
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "tokenizer": tokenizer.to_dict(),
        "loss_log": loss_log,
    }
    torch.save(checkpoint, output_dir / "mini_gpt_pretrained.pt")
    save_json(tokenizer.to_dict(), output_dir / "char_tokenizer.json")
    save_json({"model_config": asdict(model_config), "train_config": asdict(train_config)}, output_dir / "pretrain_config.json")
    save_loss_log(loss_log, output_dir / "loss_log.csv")


def main() -> None:
    """运行第 3 步预训练完整流程

    参数含义:
        无参数

    返回值含义:
        无返回值，直接执行数据准备、模型训练、效果展示和产物保存
    """

    config = parse_args()
    set_seed(config.seed)
    device = select_device(config.device)

    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    corpus, text_files = load_corpus(data_dir)
    min_chars = max(config.context_length * config.batch_size * 10, 8_000)
    corpus = ensure_minimum_corpus_size(corpus, min_chars)

    tokenizer = CharTokenizer.from_text(corpus)
    token_ids = tokenizer.encode(corpus)
    train_loader, valid_loader, train_ids, valid_ids = build_dataloaders(token_ids, config)

    model_config = GPTConfig(
        vocab_size=len(tokenizer.itos),
        context_length=config.context_length,
        emb_dim=config.emb_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    )
    model = MiniGPT(model_config).to(device)
    total_params, trainable_params = count_parameters(model)

    print("第 3 步预训练配置:")
    print(config)
    print(f"运行设备: {device}")
    print(f"读取语料文件数: {len(text_files)}")
    print(f"训练字符数: {len(corpus):,}")
    print(f"词表大小: {len(tokenizer.itos):,}")
    print(f"训练 token 数: {len(train_ids):,}")
    print(f"验证 token 数: {len(valid_ids):,}")
    print(f"模型总参数量: {total_params:,}")
    print(f"模型可训练参数量: {trainable_params:,}")

    print("\n训练前生成效果:")
    print(generate_text(model, tokenizer, config.prompt, config.generate_tokens, config.temperature, config.top_k, device))

    print("\n开始预训练:")
    loss_log = train_model(model, train_loader, valid_loader, config, device)

    print("\n训练后生成效果:")
    print(generate_text(model, tokenizer, config.prompt, config.generate_tokens, config.temperature, config.top_k, device))

    save_checkpoint(model, model_config, config, tokenizer, loss_log, output_dir)
    print(f"\n预训练产物已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
