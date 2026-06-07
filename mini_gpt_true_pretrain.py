"""真实中文语料 MiniGPT 预训练程序

本程序默认读取 data_pretrain/true_pretrain 下的 TinyStories-Zh 数据
它以字符级分词方式训练项目已有 MiniGPT 模型，并保存可继续微调的 checkpoint
中文维基只在显式选择 source 或 ratio 混合比例时参与训练
默认只抽取部分字符进行笔记本友好的预训练，并把低频字符映射为 UNK 以降低长尾噪声
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


TEXT_FIELD_CANDIDATES = (
    "text",
    "content",
    "story",
    "article",
    "正文",
    "内容",
)
TITLE_FIELD_CANDIDATES = ("title", "标题", "name", "名称")
RATIO_SOURCE_TYPES = ("tinystories", "wiki")
SUPPORTED_SOURCE_TYPES = ("ratio", "wiki", "tinystories", "json", "parquet", "text", "all")


@dataclass
class TruePretrainConfig:
    """保存真实中文语料预训练流程配置

    参数含义:
        data_dir: 存放真实预训练数据的目录，默认读取 data_pretrain/true_pretrain
        output_dir: 存放模型权重、分词器、训练日志和样例输出的目录
        init_checkpoint: 训练前可选加载的 MiniGPT checkpoint 路径，空字符串表示从随机初始化开始
        source: 数据来源类型，默认 tinystories 表示只使用 TinyStories-Zh 语料
        mix_ratio: ratio 模式下的语料比例，默认仅使用 TinyStories-Zh，可显式加入中文维基
        allow_missing_ratio_sources: ratio 模式下是否允许缺少某类语料并自动按已有语料重新归一化
        max_chars: 最多抽取多少个字符参与训练，0 表示不限制，默认保留删去 CLUE 后的 140 万文本字符份额
        max_docs: 最多抽取多少条文本记录，0 表示不限制
        min_text_chars: 短于该长度的文本记录会被跳过
        min_char_frequency: 构建字符词表时保留字符所需的最低出现次数，低频字符会映射到 UNK
        json_chunk_size: 流式读取 JSON 文件时每次读取的字符数
        parquet_batch_size: 流式读取 Parquet 文件时每个批次的记录数
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
        warmup_steps: 学习率线性预热步数
        min_learning_rate: 余弦衰减后的最低学习率
        weight_decay: AdamW 优化器权重衰减系数
        grad_clip: 梯度裁剪阈值
        early_stopping_patience: 验证集损失连续多少次评估无明显改善后提前停止，0 表示关闭
        early_stopping_min_delta: 判定验证集损失改善所需的最小下降幅度
        train_split: 训练集 token 占全部 token 的比例
        seed: 随机种子
        prompt: 训练前后用于生成效果对比的提示文本
        generate_tokens: 每次效果测试生成的新 token 数量
        temperature: 生成采样温度，0 表示贪心解码
        top_k: 生成时保留概率最高的候选 token 数量，0 表示不限制
        repetition_penalty: 生成时对已经出现过的 token 施加的重复惩罚，1 表示不惩罚
        no_repeat_ngram_size: 生成时禁止重复的 ngram 长度，0 表示不限制
        num_workers: DataLoader 工作进程数，Windows 上默认 0 更稳定
        device: 运行设备，auto 表示自动选择 cuda 或 cpu
        dry_run: 只检查数据和模型配置，不执行训练和保存

    返回值含义:
        TruePretrainConfig 实例用于统一传递真实语料预训练参数
    """

    data_dir: str = "data_pretrain/true_pretrain"
    output_dir: str = "outputs_true_pretrain_tinystories"
    init_checkpoint: str = ""
    source: str = "tinystories"
    mix_ratio: str = "tinystories=1.0"
    allow_missing_ratio_sources: bool = False
    max_chars: int = 1_400_000
    max_docs: int = 0
    min_text_chars: int = 20
    min_char_frequency: int = 2
    json_chunk_size: int = 1_048_576
    parquet_batch_size: int = 2048
    context_length: int = 192
    emb_dim: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.05
    batch_size: int = 12
    max_steps: int = 12_000
    eval_interval: int = 100
    eval_batches: int = 100
    learning_rate: float = 3e-4
    warmup_steps: int = 300
    min_learning_rate: float = 5e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    early_stopping_patience: int = 18
    early_stopping_min_delta: float = 0.005
    train_split: float = 0.9
    seed: int = 42
    prompt: str = "从前有一个小女孩"
    generate_tokens: int = 120
    temperature: float = 0.6
    top_k: int = 10
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 4
    num_workers: int = 0
    device: str = "auto"
    dry_run: bool = False


@dataclass
class CorpusBuildResult:
    """保存语料抽取结果和统计信息

    参数含义:
        corpus: 合并后的训练文本
        selected_files: 本次选择的数据文件路径列表
        ignored_files: 因格式不支持而跳过的文件路径列表
        records_seen: 已读取的原始文本记录数量
        records_used: 参与训练的文本记录数量
        chars_used: 参与训练的字符数量
        file_record_counts: 每个文件贡献的文本记录数量
        source_record_counts: 每类语料贡献的文本记录数量
        source_char_counts: 每类语料贡献的字符数量
        source_targets: ratio 模式下每类语料的目标抽样字符数
        source_ratio: ratio 模式下每类语料的目标比例
    返回值含义:
        CorpusBuildResult 实例用于向训练流程和日志传递数据统计
    """

    corpus: str
    selected_files: list[Path]
    ignored_files: list[Path]
    records_seen: int
    records_used: int
    chars_used: int
    file_record_counts: dict[str, int] = field(default_factory=dict)
    source_record_counts: dict[str, int] = field(default_factory=dict)
    source_char_counts: dict[str, int] = field(default_factory=dict)
    source_targets: dict[str, int] = field(default_factory=dict)
    source_ratio: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """把语料统计转换为可序列化字典

        参数含义:
            无参数

        返回值含义:
            返回可写入 JSON 的语料统计字典
        """

        return {
            "selected_files": [str(path) for path in self.selected_files],
            "ignored_files": [str(path) for path in self.ignored_files],
            "records_seen": self.records_seen,
            "records_used": self.records_used,
            "chars_used": self.chars_used,
            "file_record_counts": self.file_record_counts,
            "source_record_counts": self.source_record_counts,
            "source_char_counts": self.source_char_counts,
            "source_targets": self.source_targets,
            "source_ratio": self.source_ratio,
        }


@dataclass
class TrainingResult:
    """保存训练日志和验证集最优结果

    参数含义:
        loss_log: 训练过程中按评估间隔记录的损失和学习率
        best_step: 验证集损失最低时对应的训练步数
        best_train_loss: 最优验证点对应的训练集损失
        best_valid_loss: 训练过程中观察到的最低验证集损失
        final_step: 实际执行到的最后训练步数
        early_stopped: 是否因为验证集损失长期无改善而提前停止

    返回值含义:
        TrainingResult 实例用于保存训练过程摘要和最佳模型信息
    """

    loss_log: list[dict[str, float]]
    best_step: int
    best_train_loss: float
    best_valid_loss: float
    final_step: int
    early_stopped: bool

    def to_dict(self) -> dict[str, object]:
        """把训练结果转换为可序列化字典

        参数含义:
            无参数

        返回值含义:
            返回包含最佳 step、训练损失和验证损失的字典
        """

        return {
            "best_step": self.best_step,
            "best_train_loss": self.best_train_loss if math.isfinite(self.best_train_loss) else None,
            "best_valid_loss": self.best_valid_loss if math.isfinite(self.best_valid_loss) else None,
            "final_step": self.final_step,
            "early_stopped": self.early_stopped,
        }


@dataclass
class SourceBuildResult:
    """保存单类语料的抽取结果

    参数含义:
        source_name: 语料来源名称
        text_parts: 从文本语料抽取到的正文片段
        selected_files: 当前语料来源使用的数据文件
        records_seen: 当前语料来源读取过的记录数量
        records_used: 当前语料来源实际使用的记录数量
        chars_used: 当前语料来源使用的文本字符数
        file_record_counts: 当前语料来源中每个文件贡献的记录数量

    返回值含义:
        SourceBuildResult 实例用于汇总到 CorpusBuildResult
    """

    source_name: str
    text_parts: list[str]
    selected_files: list[Path]
    records_seen: int
    records_used: int
    chars_used: int
    file_record_counts: dict[str, int]


@dataclass
class TokenizerBuildResult:
    """保存字符级分词器构建结果和低频字符统计

    参数含义:
        tokenizer: 按字符频次过滤后构建出的字符级分词器
        min_char_frequency: 保留字符所需的最低出现次数
        total_unique_chars: 原始语料中出现过的唯一字符数量
        kept_unique_chars: 实际进入词表的唯一字符数量
        dropped_unique_chars: 因低频被映射到 UNK 的唯一字符数量
        dropped_char_occurrences: 因低频被映射到 UNK 的字符总出现次数

    返回值含义:
        TokenizerBuildResult 实例用于传递分词器和可保存的词表统计
    """

    tokenizer: CharTokenizer
    min_char_frequency: int
    total_unique_chars: int
    kept_unique_chars: int
    dropped_unique_chars: int
    dropped_char_occurrences: int

    def to_dict(self) -> dict[str, int]:
        """把分词器构建统计转换为可序列化字典

        参数含义:
            无参数

        返回值含义:
            返回可写入 JSON 的分词器构建统计字典
        """

        return {
            "min_char_frequency": self.min_char_frequency,
            "total_unique_chars": self.total_unique_chars,
            "kept_unique_chars": self.kept_unique_chars,
            "dropped_unique_chars": self.dropped_unique_chars,
            "dropped_char_occurrences": self.dropped_char_occurrences,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    """构建真实语料预训练命令行参数解析器

    参数含义:
        无参数

    返回值含义:
        返回配置完成的 argparse.ArgumentParser 实例
    """

    parser = argparse.ArgumentParser(description="真实中文语料 MiniGPT 预训练程序")
    parser.add_argument("--data-dir", default="data_pretrain/true_pretrain", help="真实预训练数据目录")
    parser.add_argument("--output-dir", default="outputs_true_pretrain_tinystories", help="保存预训练产物的目录")
    parser.add_argument("--init-checkpoint", default="", help="训练前加载的已有 MiniGPT checkpoint 路径，留空表示随机初始化")
    parser.add_argument("--source", default="tinystories", choices=SUPPORTED_SOURCE_TYPES, help="数据来源类型，默认只使用 TinyStories-Zh")
    parser.add_argument(
        "--mix-ratio",
        default="tinystories=1.0",
        help="ratio 模式下的数据比例，例如 tinystories=1.0 或 tinystories=0.3,wiki=0.7",
    )
    parser.add_argument(
        "--allow-missing-ratio-sources",
        action="store_true",
        help="ratio 模式下允许缺少某类语料，并把比例重新分配给已有语料",
    )
    parser.add_argument("--max-chars", type=int, default=1_400_000, help="最多抽取的训练字符数，0 表示不限制")
    parser.add_argument("--max-docs", type=int, default=0, help="最多抽取的文本记录数，0 表示不限制")
    parser.add_argument("--min-text-chars", type=int, default=20, help="短文本过滤阈值")
    parser.add_argument("--min-char-frequency", type=int, default=2, help="字符词表最低出现次数，低频字符映射为 UNK")
    parser.add_argument("--json-chunk-size", type=int, default=1_048_576, help="JSON 流式读取块大小")
    parser.add_argument("--parquet-batch-size", type=int, default=2048, help="Parquet 流式读取 batch 大小")
    parser.add_argument("--context-length", type=int, default=192, help="上下文 token 长度")
    parser.add_argument("--emb-dim", type=int, default=256, help="隐藏向量维度")
    parser.add_argument("--n-heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--n-layers", type=int, default=6, help="Transformer 层数")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout 比例")
    parser.add_argument("--batch-size", type=int, default=12, help="每个 batch 的样本数量")
    parser.add_argument("--max-steps", type=int, default=12_000, help="最大训练步数")
    parser.add_argument("--eval-interval", type=int, default=100, help="评估间隔步数")
    parser.add_argument("--eval-batches", type=int, default=100, help="每次评估最多使用的 batch 数量")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--warmup-steps", type=int, default=300, help="学习率线性预热步数")
    parser.add_argument("--min-learning-rate", type=float, default=5e-5, help="余弦衰减后的最低学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--early-stopping-patience", type=int, default=18, help="验证集损失连续无改善的最大评估次数，0 表示关闭")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.005, help="验证集损失视为改善所需的最小下降幅度")
    parser.add_argument("--train-split", type=float, default=0.9, help="训练集 token 比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--prompt", default="从前有一个小女孩", help="训练前后用于生成对比的提示文本")
    parser.add_argument("--generate-tokens", type=int, default=120, help="生成的新 token 数量")
    parser.add_argument("--temperature", type=float, default=0.6, help="生成采样温度，0 表示贪心解码")
    parser.add_argument("--top-k", type=int, default=10, help="生成时保留的候选 token 数量，0 表示不限制")
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="生成时对已经出现过的 token 施加的重复惩罚，1 表示不惩罚")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4, help="生成时禁止重复的 ngram 长度，0 表示不限制")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 工作进程数")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    parser.add_argument("--dry-run", action="store_true", help="只检查数据和模型配置，不执行训练")
    return parser


def parse_args() -> TruePretrainConfig:
    """解析命令行参数并生成预训练配置

    参数含义:
        无参数

    返回值含义:
        返回从命令行参数生成的 TruePretrainConfig 实例
    """

    parser = build_arg_parser()
    args = parser.parse_args()
    return TruePretrainConfig(**vars(args))


if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
    build_arg_parser().parse_args()
    raise SystemExit


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from mini_gpt_step2 import GPTConfig, MiniGPT, count_parameters
from mini_gpt_step3_pretrain import CharTokenizer


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


def validate_config(config: TruePretrainConfig) -> None:
    """校验预训练配置是否合理

    参数含义:
        config: 真实语料预训练配置对象

    返回值含义:
        无返回值，配置不合法时直接抛出异常
    """

    if config.max_chars < 0:
        raise ValueError("max_chars 不能小于 0")
    if config.max_docs < 0:
        raise ValueError("max_docs 不能小于 0")
    if config.min_text_chars < 0:
        raise ValueError("min_text_chars 不能小于 0")
    if config.min_char_frequency <= 0:
        raise ValueError("min_char_frequency 必须大于 0")
    if config.json_chunk_size <= 0:
        raise ValueError("json_chunk_size 必须大于 0")
    if config.parquet_batch_size <= 0:
        raise ValueError("parquet_batch_size 必须大于 0")
    if config.context_length <= 0:
        raise ValueError("context_length 必须大于 0")
    if config.emb_dim <= 0:
        raise ValueError("emb_dim 必须大于 0")
    if config.n_heads <= 0:
        raise ValueError("n_heads 必须大于 0")
    if config.n_layers <= 0:
        raise ValueError("n_layers 必须大于 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if config.max_steps < 0:
        raise ValueError("max_steps 不能小于 0")
    if config.eval_interval <= 0:
        raise ValueError("eval_interval 必须大于 0")
    if config.eval_batches <= 0:
        raise ValueError("eval_batches 必须大于 0")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate 必须大于 0")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps 不能小于 0")
    if config.min_learning_rate < 0:
        raise ValueError("min_learning_rate 不能小于 0")
    if config.min_learning_rate > config.learning_rate:
        raise ValueError("min_learning_rate 不能大于 learning_rate")
    if config.weight_decay < 0:
        raise ValueError("weight_decay 不能小于 0")
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience 不能小于 0")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta 不能小于 0")
    if not 0.0 < config.train_split < 1.0:
        raise ValueError("train_split 必须位于 0 到 1 之间")
    if config.temperature < 0:
        raise ValueError("temperature 不能小于 0")
    if config.top_k < 0:
        raise ValueError("top_k 不能小于 0")
    if config.repetition_penalty < 1.0:
        raise ValueError("repetition_penalty 必须大于等于 1")
    if config.no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size 不能小于 0")
    if config.num_workers < 0:
        raise ValueError("num_workers 不能小于 0")
    if config.init_checkpoint and not Path(config.init_checkpoint).is_file():
        raise FileNotFoundError(f"init_checkpoint 不存在: {config.init_checkpoint}")
    if config.source == "ratio":
        if config.max_chars <= 0:
            raise ValueError("ratio 模式必须设置大于 0 的 max_chars，才能按比例分配抽样字符数")
        parse_mix_ratio(config.mix_ratio)


def parse_mix_ratio(raw_ratio: str) -> dict[str, float]:
    """解析并归一化混合语料比例

    参数含义:
        raw_ratio: 命令行传入的比例字符串，例如 tinystories=0.3,wiki=0.7

    返回值含义:
        返回每类语料到归一化比例的映射
    """

    ratio: dict[str, float] = {}
    for item in raw_ratio.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"mix_ratio 条目必须使用 key=value 格式: {item}")
        key, value = item.split("=", 1)
        key = key.strip().lower()
        if key not in RATIO_SOURCE_TYPES:
            raise ValueError(f"mix_ratio 不支持语料类型 {key}，可选值为 {RATIO_SOURCE_TYPES}")
        try:
            ratio_value = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"mix_ratio 中 {key} 的比例不是有效数字: {value}") from exc
        if ratio_value < 0:
            raise ValueError(f"mix_ratio 中 {key} 的比例不能小于 0")
        if ratio_value == 0:
            continue
        ratio[key] = ratio.get(key, 0.0) + ratio_value

    total = sum(ratio.values())
    if total <= 0:
        raise ValueError("mix_ratio 至少需要一个大于 0 的语料比例")
    return {key: value / total for key, value in ratio.items()}


def build_source_targets(total_chars: int, source_ratio: dict[str, float]) -> dict[str, int]:
    """根据总字符数和比例计算每类语料目标字符数

    参数含义:
        total_chars: 本次计划抽取的总字符数
        source_ratio: 每类语料的归一化比例

    返回值含义:
        返回每类语料到目标字符数的映射，所有目标之和等于 total_chars
    """

    targets: dict[str, int] = {}
    remaining_chars = total_chars
    source_items = list(source_ratio.items())
    for index, (source_name, ratio_value) in enumerate(source_items):
        if index == len(source_items) - 1:
            target_chars = remaining_chars
        else:
            target_chars = int(total_chars * ratio_value)
            remaining_chars -= target_chars
        targets[source_name] = max(target_chars, 0)
    return targets


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
        返回统一换行、压缩空白并去除首尾空白后的文本
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.strip().split()) for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def extract_text(record: Any) -> str:
    """从一条数据记录中提取可用于预训练的正文文本

    参数含义:
        record: JSON 或 Parquet 中读取到的单条记录

    返回值含义:
        返回提取出的文本，无法提取时返回空字符串
    """

    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return ""

    title = ""
    for field_name in TITLE_FIELD_CANDIDATES:
        value = record.get(field_name)
        if value is not None and str(value).strip():
            title = str(value).strip()
            break

    body = ""
    for field_name in TEXT_FIELD_CANDIDATES:
        value = record.get(field_name)
        if value is not None and str(value).strip():
            body = str(value).strip()
            break

    if body:
        return f"{title}\n{body}" if title else body

    string_values = [str(value).strip() for value in record.values() if isinstance(value, str) and value.strip()]
    if not string_values:
        return ""
    return "\n".join(string_values)


def is_supported_text_data_file(file_path: Path) -> bool:
    """判断文件是否属于脚本可读取的文本语料格式

    参数含义:
        file_path: 待判断的数据文件路径

    返回值含义:
        返回布尔值，True 表示文件扩展名可被当前脚本读取
    """

    return file_path.suffix.lower() in {".json", ".jsonl", ".parquet", ".txt"}


def is_clue_file(file_path: Path) -> bool:
    """判断文件是否属于需要排除的 CLUECorpusSmall 相关文件

    参数含义:
        file_path: 待判断的数据文件路径

    返回值含义:
        返回布尔值，True 表示路径名属于 CLUECorpusSmall 或 ziya mmap 数据
    """

    lower_path = str(file_path).lower()
    file_name = file_path.name.lower()
    return "clue" in lower_path or "cluecorpus" in lower_path or file_name.startswith("ziya_mmap")


def collect_source_files(data_dir: Path) -> tuple[dict[str, list[Path]], list[Path]]:
    """收集数据目录中各类预训练语料文件

    参数含义:
        data_dir: 数据目录路径

    返回值含义:
        返回二元组，第一个值是语料类别到文件列表的映射，第二个值是明确排除或无法直接读取的文件列表
    """

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    wiki_files = sorted(data_dir.rglob("wiki_pretrain_part*.json"))
    tinystories_files = sorted(data_dir.rglob("train-*.parquet"))
    excluded_clue_files = sorted(path for path in data_dir.rglob("*") if path.is_file() and is_clue_file(path))
    json_files = sorted(
        path
        for path in data_dir.rglob("*.json")
        if path not in wiki_files and path not in excluded_clue_files
    )
    jsonl_files = sorted(path for path in data_dir.rglob("*.jsonl") if path not in excluded_clue_files)
    parquet_files = sorted(
        path
        for path in data_dir.rglob("*.parquet")
        if path not in tinystories_files and path not in excluded_clue_files
    )
    text_files = sorted(path for path in data_dir.rglob("*.txt") if path not in excluded_clue_files)
    bin_files = sorted(data_dir.rglob("*.bin"))

    source_files = {
        "wiki": wiki_files,
        "tinystories": tinystories_files,
        "json": wiki_files + json_files + jsonl_files,
        "parquet": tinystories_files + parquet_files,
        "text": text_files,
        "all": tinystories_files + wiki_files + json_files + jsonl_files + parquet_files + text_files,
    }
    ignored_files = sorted(set(bin_files + excluded_clue_files))
    return source_files, ignored_files


def select_source_files(data_dir: Path, source: str) -> tuple[list[Path], list[Path]]:
    """根据来源类型选择真实语料文件

    参数含义:
        data_dir: 数据目录路径
        source: 数据来源类型

    返回值含义:
        返回二元组，第一个值是会读取的文件，第二个值是不支持而跳过的文件
    """

    source_files, bin_files = collect_source_files(data_dir)
    if source not in source_files:
        raise ValueError(f"不支持的数据来源类型: {source}")

    selected_files = source_files[source]
    if not selected_files:
        raise FileNotFoundError(f"数据目录 {data_dir} 中没有找到 source={source} 对应的数据文件")

    ignored_files = [file_path for file_path in bin_files if file_path not in selected_files]
    return selected_files, ignored_files


def iter_json_array_records(file_path: Path, chunk_size: int) -> Iterator[Any]:
    """流式读取 JSON 数组文件中的记录

    参数含义:
        file_path: JSON 文件路径
        chunk_size: 每次从文件中读取的字符数

    返回值含义:
        返回逐条记录的迭代器，避免一次性加载大 JSON 文件
    """

    decoder = json.JSONDecoder()
    buffer = ""
    array_started = False
    eof = False

    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        while True:
            if not eof:
                chunk = file.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break

                if not array_started:
                    if buffer.startswith("["):
                        array_started = True
                        buffer = buffer[1:]
                        continue
                    try:
                        record, end_index = decoder.raw_decode(buffer)
                    except json.JSONDecodeError as exc:
                        if eof:
                            raise ValueError(f"无法解析 JSON 文件: {file_path}") from exc
                        break
                    yield record
                    buffer = buffer[end_index:]
                    continue

                if buffer.startswith("]"):
                    return
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue

                try:
                    record, end_index = decoder.raw_decode(buffer)
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(f"无法解析 JSON 数组记录: {file_path}") from exc
                    break

                yield record
                buffer = buffer[end_index:]

            if eof:
                if buffer.strip() and buffer.strip() != "]":
                    raise ValueError(f"JSON 文件末尾存在无法解析的内容: {file_path}")
                return


def iter_jsonl_records(file_path: Path) -> Iterator[Any]:
    """逐行读取 JSONL 文件中的记录

    参数含义:
        file_path: JSONL 文件路径

    返回值含义:
        返回逐条 JSON 记录的迭代器
    """

    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 文件 {file_path} 第 {line_number} 行无法解析") from exc


def iter_parquet_records(file_path: Path, batch_size: int) -> Iterator[Any]:
    """流式读取 Parquet 文件中的记录

    参数含义:
        file_path: Parquet 文件路径
        batch_size: 每次读取的记录数量

    返回值含义:
        返回逐条记录的迭代器
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("读取 Parquet 数据需要安装 pyarrow，请在虚拟环境中安装 pyarrow 或改用 --source wiki") from exc

    parquet_file = pq.ParquetFile(file_path)
    schema_names = parquet_file.schema.names
    columns = [field_name for field_name in (*TITLE_FIELD_CANDIDATES, *TEXT_FIELD_CANDIDATES) if field_name in schema_names]
    if not columns:
        columns = schema_names

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        batch_dict = batch.to_pydict()
        for row_index in range(batch.num_rows):
            yield {column: batch_dict[column][row_index] for column in columns}


def iter_text_file_records(file_path: Path) -> Iterator[str]:
    """读取普通文本文件内容

    参数含义:
        file_path: 文本文件路径

    返回值含义:
        返回包含整个文本文件内容的单条记录迭代器
    """

    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        yield file.read()


def iter_records_from_file(file_path: Path, config: TruePretrainConfig) -> Iterator[Any]:
    """根据文件扩展名选择合适的数据读取器

    参数含义:
        file_path: 需要读取的数据文件路径
        config: 真实语料预训练配置对象

    返回值含义:
        返回该文件中的逐条记录迭代器
    """

    suffix = file_path.suffix.lower()
    if suffix == ".json":
        yield from iter_json_array_records(file_path, config.json_chunk_size)
    elif suffix == ".jsonl":
        yield from iter_jsonl_records(file_path)
    elif suffix == ".parquet":
        yield from iter_parquet_records(file_path, config.parquet_batch_size)
    elif suffix == ".txt":
        yield from iter_text_file_records(file_path)
    else:
        raise ValueError(f"不支持的数据文件格式: {file_path}")


def build_text_source(
    source_name: str,
    selected_files: list[Path],
    target_chars: int,
    config: TruePretrainConfig,
) -> SourceBuildResult:
    """按目标字符数抽取某一类文本语料

    参数含义:
        source_name: 语料来源名称
        selected_files: 当前来源对应的数据文件列表
        target_chars: 该来源需要抽取的目标字符数，0 表示不限制
        config: 真实语料预训练配置对象

    返回值含义:
        返回当前文本来源的抽取结果
    """

    iterators: list[tuple[Path, Iterator[Any]]] = [
        (file_path, iter_records_from_file(file_path, config)) for file_path in selected_files
    ]
    active_iterators = iterators
    text_parts: list[str] = []
    file_record_counts: dict[str, int] = {}
    records_seen = 0
    records_used = 0
    chars_used = 0

    while active_iterators:
        next_active_iterators: list[tuple[Path, Iterator[Any]]] = []

        for file_path, record_iterator in active_iterators:
            try:
                record = next(record_iterator)
            except StopIteration:
                continue

            records_seen += 1
            text = normalize_text(extract_text(record))
            if len(text) >= config.min_text_chars:
                if target_chars > 0:
                    remaining_chars = target_chars - chars_used
                    if remaining_chars <= 0:
                        break
                    text = text[:remaining_chars]

                text_parts.append(text)
                records_used += 1
                chars_used += len(text)
                file_key = str(file_path)
                file_record_counts[file_key] = file_record_counts.get(file_key, 0) + 1

            if config.max_docs > 0 and records_used >= config.max_docs:
                break
            if target_chars > 0 and chars_used >= target_chars:
                break

            next_active_iterators.append((file_path, record_iterator))

        if config.max_docs > 0 and records_used >= config.max_docs:
            break
        if target_chars > 0 and chars_used >= target_chars:
            break

        active_iterators = next_active_iterators

    return SourceBuildResult(
        source_name=source_name,
        text_parts=text_parts,
        selected_files=selected_files,
        records_seen=records_seen,
        records_used=records_used,
        chars_used=chars_used,
        file_record_counts=file_record_counts,
    )


def build_source_result(
    source_name: str,
    selected_files: list[Path],
    target_units: int,
    config: TruePretrainConfig,
) -> SourceBuildResult:
    """根据语料来源类型抽取文本

    参数含义:
        source_name: 语料来源名称
        selected_files: 当前来源对应的数据文件列表
        target_units: 该来源的目标抽样字符数
        config: 真实语料预训练配置对象

    返回值含义:
        返回当前来源的抽取结果
    """

    return build_text_source(source_name, selected_files, target_units, config)


def renormalize_ratio_for_available_sources(
    source_ratio: dict[str, float],
    source_files: dict[str, list[Path]],
    allow_missing_sources: bool,
) -> dict[str, float]:
    """根据实际存在的数据文件校验或重算混合比例

    参数含义:
        source_ratio: 用户配置的语料比例
        source_files: 数据目录中各类语料文件映射
        allow_missing_sources: 是否允许缺失某类比例语料

    返回值含义:
        返回只包含可用语料来源的归一化比例
    """

    missing_sources = [source_name for source_name in source_ratio if not source_files.get(source_name)]
    if missing_sources and not allow_missing_sources:
        raise FileNotFoundError(
            "ratio 模式缺少以下语料来源: "
            + ", ".join(missing_sources)
            + "；请补齐数据或添加 --allow-missing-ratio-sources"
        )

    available_ratio = {
        source_name: ratio_value
        for source_name, ratio_value in source_ratio.items()
        if source_files.get(source_name)
    }
    total_ratio = sum(available_ratio.values())
    if total_ratio <= 0:
        raise FileNotFoundError("ratio 模式没有可用语料文件")
    return {source_name: ratio_value / total_ratio for source_name, ratio_value in available_ratio.items()}


def build_corpus(config: TruePretrainConfig) -> CorpusBuildResult:
    """从真实语料文件中抽取训练文本

    参数含义:
        config: 真实语料预训练配置对象

    返回值含义:
        返回包含合并文本和数据统计的 CorpusBuildResult 实例
    """

    data_dir = Path(config.data_dir)
    source_files, bin_files = collect_source_files(data_dir)
    if config.source == "ratio":
        source_ratio = renormalize_ratio_for_available_sources(
            parse_mix_ratio(config.mix_ratio),
            source_files,
            config.allow_missing_ratio_sources,
        )
        source_targets = build_source_targets(config.max_chars, source_ratio)
        source_names = list(source_ratio)
    else:
        if config.source not in source_files:
            raise ValueError(f"不支持的数据来源类型: {config.source}")
        if not source_files[config.source]:
            raise FileNotFoundError(f"数据目录 {data_dir} 中没有找到 source={config.source} 对应的数据文件")
        source_ratio = {config.source: 1.0}
        source_targets = {config.source: config.max_chars}
        source_names = [config.source]

    corpus_parts: list[str] = []
    selected_files: list[Path] = []
    file_record_counts: dict[str, int] = {}
    source_record_counts: dict[str, int] = {}
    source_char_counts: dict[str, int] = {}
    records_seen = 0
    records_used = 0

    for source_name in source_names:
        source_result = build_source_result(
            source_name,
            source_files[source_name],
            source_targets.get(source_name, 0),
            config,
        )
        corpus_parts.extend(source_result.text_parts)
        selected_files.extend(source_result.selected_files)
        records_seen += source_result.records_seen
        records_used += source_result.records_used
        source_record_counts[source_name] = source_result.records_used
        source_char_counts[source_name] = source_result.chars_used
        for file_path, record_count in source_result.file_record_counts.items():
            file_record_counts[file_path] = file_record_counts.get(file_path, 0) + record_count

    if not corpus_parts:
        raise ValueError("没有抽取到可用于预训练的文本，请检查数据格式或降低 min_text_chars")

    corpus = "\n\n".join(corpus_parts)
    ignored_files = [file_path for file_path in bin_files if file_path not in selected_files]
    return CorpusBuildResult(
        corpus=corpus,
        selected_files=selected_files,
        ignored_files=ignored_files,
        records_seen=records_seen,
        records_used=records_used,
        chars_used=len(corpus),
        file_record_counts=file_record_counts,
        source_record_counts=source_record_counts,
        source_char_counts=source_char_counts,
        source_targets=source_targets,
        source_ratio=source_ratio,
    )


def build_tokenizer(corpus: str, min_char_frequency: int) -> TokenizerBuildResult:
    """根据字符频次构建字符级分词器

    参数含义:
        corpus: 已抽取并合并后的训练语料文本
        min_char_frequency: 保留字符所需的最低出现次数，低于该次数的字符映射到 UNK

    返回值含义:
        返回分词器构建结果，包含 CharTokenizer 和低频字符过滤统计
    """

    char_counts = Counter(corpus)
    kept_chars = sorted(char for char, count in char_counts.items() if count >= min_char_frequency)
    kept_chars = [
        char
        for char in kept_chars
        if char not in {CharTokenizer.pad_token, CharTokenizer.unk_token}
    ]
    dropped_unique_chars = len(char_counts) - len(kept_chars)
    dropped_char_occurrences = sum(
        count
        for char, count in char_counts.items()
        if count < min_char_frequency or char in {CharTokenizer.pad_token, CharTokenizer.unk_token}
    )
    vocab = [CharTokenizer.pad_token, CharTokenizer.unk_token] + kept_chars
    tokenizer = CharTokenizer(vocab)
    return TokenizerBuildResult(
        tokenizer=tokenizer,
        min_char_frequency=min_char_frequency,
        total_unique_chars=len(char_counts),
        kept_unique_chars=len(kept_chars),
        dropped_unique_chars=dropped_unique_chars,
        dropped_char_occurrences=dropped_char_occurrences,
    )


def encode_text_token_ids(
    tokenizer: CharTokenizer,
    corpus_result: CorpusBuildResult,
    seed: int,
) -> list[int]:
    """把文本语料编码为字符级训练 token 序列

    参数含义:
        tokenizer: 根据文本语料构建出的字符级分词器
        corpus_result: 语料抽取结果和统计信息
        seed: 随机种子，用于打散文本片段顺序

    返回值含义:
        返回 MiniGPT 训练 token 编号列表
    """

    chunks: list[list[int]] = []
    separator_id = tokenizer.stoi.get("\n", tokenizer.unk_id)
    if corpus_result.corpus:
        for text_part in corpus_result.corpus.split("\n\n"):
            token_ids = tokenizer.encode(text_part)
            if token_ids:
                chunks.append(token_ids + [separator_id])

    if not chunks:
        raise ValueError("没有可编码的训练 token")

    random.Random(seed).shuffle(chunks)
    token_ids: list[int] = []
    for chunk in chunks:
        token_ids.extend(chunk)
        token_ids.append(separator_id)
    return token_ids


def split_tokens(token_ids: list[int], train_split: float, context_length: int) -> tuple[list[int], list[int]]:
    """把连续 token 序列切分为训练集和验证集

    参数含义:
        token_ids: 全部语料编码后的 token 编号序列
        train_split: 训练集 token 占比
        context_length: 上下文长度，用于保证验证集至少能构造一个样本

    返回值含义:
        返回二元组，第一个值是训练 token，第二个值是验证 token
    """

    if len(token_ids) <= context_length * 2 + 2:
        raise ValueError("token 数量过少，请增大 max_chars 或减小 context_length")

    split_index = int(len(token_ids) * train_split)
    split_index = max(split_index, context_length + 1)
    split_index = min(split_index, len(token_ids) - context_length - 1)
    train_ids = token_ids[:split_index]
    valid_ids = token_ids[split_index:]
    return train_ids, valid_ids


def build_dataloaders(
    token_ids: list[int],
    config: TruePretrainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, list[int], list[int]]:
    """构建训练集和验证集 DataLoader

    参数含义:
        token_ids: 全部语料编码后的 token 编号序列
        config: 真实语料预训练配置对象

    返回值含义:
        返回训练 DataLoader、训练评估 DataLoader、验证评估 DataLoader、训练 token 列表和验证 token 列表
    """

    train_ids, valid_ids = split_tokens(token_ids, config.train_split, config.context_length)
    train_dataset = LanguageModelDataset(train_ids, config.context_length)
    valid_dataset = LanguageModelDataset(valid_ids, config.context_length)
    eval_sample_count = config.eval_batches * config.batch_size
    train_eval_dataset = build_evenly_spaced_eval_dataset(train_dataset, eval_sample_count)
    valid_eval_dataset = build_evenly_spaced_eval_dataset(valid_dataset, eval_sample_count)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
    )
    valid_eval_loader = DataLoader(
        valid_eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
    )
    return train_loader, train_eval_loader, valid_eval_loader, train_ids, valid_ids


def build_evenly_spaced_eval_dataset(dataset: LanguageModelDataset, sample_count: int) -> Dataset:
    """构建固定均匀抽样的评估数据集

    参数含义:
        dataset: 原始语言模型数据集
        sample_count: 希望用于评估的样本数量

    返回值含义:
        返回原始数据集或按位置均匀抽样得到的 Subset
    """

    dataset_length = len(dataset)
    if sample_count <= 0 or sample_count >= dataset_length:
        return dataset
    if sample_count == 1:
        return Subset(dataset, [0])

    indices = sorted({round(index * (dataset_length - 1) / (sample_count - 1)) for index in range(sample_count)})
    return Subset(dataset, indices)


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
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    if not losses:
        return float("inf")
    return sum(losses) / len(losses)


def get_banned_ngram_next_tokens(token_ids: list[int], no_repeat_ngram_size: int) -> set[int]:
    """查找当前上下文下需要禁止生成的重复 ngram 后续 token

    参数含义:
        token_ids: 已有的完整生成 token 编号序列
        no_repeat_ngram_size: 禁止重复的 ngram 长度，0 表示不限制

    返回值含义:
        返回当前上下文下会造成重复 ngram 的候选 token 编号集合
    """

    if no_repeat_ngram_size <= 0 or len(token_ids) < no_repeat_ngram_size - 1:
        return set()

    prefix_length = no_repeat_ngram_size - 1
    current_prefix = tuple(token_ids[-prefix_length:]) if prefix_length > 0 else tuple()
    banned_tokens: set[int] = set()
    for start_index in range(0, len(token_ids) - no_repeat_ngram_size + 1):
        ngram = token_ids[start_index : start_index + no_repeat_ngram_size]
        prefix = tuple(ngram[:-1])
        if prefix == current_prefix:
            banned_tokens.add(ngram[-1])
    return banned_tokens


def generate_text(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    device: torch.device,
) -> str:
    """使用模型根据提示文本生成可读文本

    参数含义:
        model: 用于生成的 MiniGPT 模型
        tokenizer: 字符级分词器
        prompt: 生成起始提示文本
        max_new_tokens: 需要继续生成的新 token 数量
        temperature: 采样温度，0 表示贪心解码
        top_k: 每步采样保留的候选 token 数量，0 表示不限制
        repetition_penalty: 对已经出现过的 token 施加的重复惩罚，1 表示不惩罚
        no_repeat_ngram_size: 禁止重复的 ngram 长度，0 表示不限制
        device: 执行生成的设备

    返回值含义:
        返回解码后的完整生成文本
    """

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.unk_id]
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.context_length :]
        logits = model(idx_cond)
        logits = logits[:, -1, :].clone()

        if repetition_penalty > 1.0:
            for token_id in set(idx[0].tolist()):
                if 0 <= token_id < logits.size(-1):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

        banned_tokens = get_banned_ngram_next_tokens(idx[0].tolist(), no_repeat_ngram_size)
        for token_id in banned_tokens:
            if 0 <= token_id < logits.size(-1):
                logits[0, token_id] = float("-inf")

        if temperature == 0:
            next_idx = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            if top_k > 0:
                values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_idx), dim=1)

    return tokenizer.decode(idx[0].tolist())


def get_scheduled_learning_rate(step: int, config: TruePretrainConfig) -> float:
    """按 warmup 和余弦衰减计算当前学习率

    参数含义:
        step: 当前训练步数，从 1 开始
        config: 真实语料预训练配置对象

    返回值含义:
        返回当前训练步应使用的学习率
    """

    if config.max_steps <= 0:
        return config.learning_rate
    if config.warmup_steps > 0 and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if config.max_steps <= config.warmup_steps:
        return config.learning_rate

    decay_steps = max(1, config.max_steps - config.warmup_steps)
    decay_progress = min(1.0, max(0.0, (step - config.warmup_steps) / decay_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine_decay


def train_model(
    model: MiniGPT,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    valid_eval_loader: DataLoader,
    config: TruePretrainConfig,
    device: torch.device,
) -> TrainingResult:
    """执行真实中文语料 GPT 预训练循环

    参数含义:
        model: 待训练的 MiniGPT 模型
        train_loader: 训练集 DataLoader
        train_eval_loader: 固定抽样训练评估 DataLoader
        valid_eval_loader: 固定抽样验证评估 DataLoader
        config: 真实语料预训练配置对象
        device: 执行训练的设备

    返回值含义:
        返回训练结果对象，包含损失日志、最佳验证集 step 和是否提前终止
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.train()
    loss_log: list[dict[str, float]] = []
    train_iter = iter(train_loader)
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_step = 0
    best_train_loss = float("inf")
    best_valid_loss = float("inf")
    no_improve_count = 0
    final_step = 0
    early_stopped = False

    for step in range(1, config.max_steps + 1):
        current_lr = get_scheduled_learning_rate(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        final_step = step

        should_eval = step == 1 or step % config.eval_interval == 0 or step == config.max_steps
        if should_eval:
            train_loss = estimate_loss(model, train_eval_loader, device, config.eval_batches)
            valid_loss = estimate_loss(model, valid_eval_loader, device, config.eval_batches)
            is_improved = valid_loss < best_valid_loss - config.early_stopping_min_delta
            if is_improved:
                best_step = step
                best_train_loss = train_loss
                best_valid_loss = valid_loss
                no_improve_count = 0
                best_state_dict = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            else:
                no_improve_count += 1

            early_stopped = config.early_stopping_patience > 0 and no_improve_count >= config.early_stopping_patience
            loss_log.append(
                {
                    "step": float(step),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "learning_rate": current_lr,
                    "best_valid_loss": best_valid_loss,
                    "no_improve_count": float(no_improve_count),
                    "early_stopped": 1.0 if early_stopped else 0.0,
                }
            )
            print(
                f"step {step:4d} | train_loss {train_loss:.4f} | valid_loss {valid_loss:.4f} "
                f"| lr {current_lr:.2e} | best_valid_loss {best_valid_loss:.4f} "
                f"| no_improve {no_improve_count}"
            )
            model.train()

            if early_stopped:
                print(f"提前终止: 验证集损失连续 {no_improve_count} 次评估未明显改善，停止于 step {step}")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"已恢复验证集最优模型权重: step {best_step} | valid_loss {best_valid_loss:.4f}")

    return TrainingResult(
        loss_log=loss_log,
        best_step=best_step,
        best_train_loss=best_train_loss,
        best_valid_loss=best_valid_loss,
        final_step=final_step,
        early_stopped=early_stopped,
    )


def load_initial_checkpoint(
    model: MiniGPT,
    model_config: GPTConfig,
    tokenizer: CharTokenizer,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """加载已有 MiniGPT checkpoint 作为训练初始权重

    参数含义:
        model: 当前已经构建好的 MiniGPT 模型
        model_config: 当前模型结构配置对象
        tokenizer: 当前语料构建出的字符级分词器
        checkpoint_path: 待加载 checkpoint 文件路径
        device: checkpoint 张量加载到的目标设备

    返回值含义:
        无返回值，校验通过后直接把 checkpoint 权重加载到模型
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"init_checkpoint 格式不正确，期望字典结构: {checkpoint_path}")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"init_checkpoint 缺少 model_state_dict: {checkpoint_path}")

    checkpoint_model_config = checkpoint.get("model_config")
    if checkpoint_model_config is not None and checkpoint_model_config != asdict(model_config):
        raise ValueError(
            "init_checkpoint 的 model_config 与当前参数不一致，"
            "请保持 context_length、emb_dim、n_heads、n_layers、dropout 和词表大小一致"
        )

    checkpoint_tokenizer = checkpoint.get("tokenizer")
    if checkpoint_tokenizer is not None:
        if not isinstance(checkpoint_tokenizer, dict):
            raise ValueError(f"init_checkpoint 的 tokenizer 格式不正确: {checkpoint_path}")
        checkpoint_itos = checkpoint_tokenizer.get("itos")
        if checkpoint_itos is not None and checkpoint_itos != tokenizer.itos:
            raise ValueError("init_checkpoint 的字符词表与当前语料词表不一致，不能继续加载")

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"已加载训练初始 checkpoint: {checkpoint_path.resolve()}")


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
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "train_loss",
                "valid_loss",
                "learning_rate",
                "best_valid_loss",
                "no_improve_count",
                "early_stopped",
            ],
            lineterminator="\r\n",
        )
        writer.writeheader()
        writer.writerows(loss_log)


def save_sample_outputs(before_text: str, after_text: str, file_path: Path) -> None:
    """保存训练前后生成样例

    参数含义:
        before_text: 训练前生成文本
        after_text: 训练后生成文本
        file_path: 样例输出文件路径

    返回值含义:
        无返回值，直接写入文本文件
    """

    with open(file_path, "w", encoding="utf-8", newline="\r\n") as file:
        file.write("训练前生成效果\r\n")
        file.write(before_text)
        file.write("\r\n\r\n训练后生成效果\r\n")
        file.write(after_text)
        file.write("\r\n")


def save_checkpoint(
    model: MiniGPT,
    model_config: GPTConfig,
    train_config: TruePretrainConfig,
    tokenizer: CharTokenizer,
    tokenizer_result: TokenizerBuildResult,
    training_result: TrainingResult,
    corpus_result: CorpusBuildResult,
    sample_outputs: dict[str, str],
    output_dir: Path,
) -> None:
    """保存预训练模型权重、分词器、配置和数据统计

    参数含义:
        model: 训练完成的 MiniGPT 模型
        model_config: 模型结构配置对象
        train_config: 真实语料预训练流程配置对象
        tokenizer: 字符级分词器
        tokenizer_result: 分词器构建结果，包含低频字符过滤统计
        training_result: 训练结果对象，包含损失日志和最佳验证集信息
        corpus_result: 语料抽取结果和统计信息
        sample_outputs: 训练前后生成文本
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
        "tokenizer_stats": tokenizer_result.to_dict(),
        "loss_log": training_result.loss_log,
        "training_result": training_result.to_dict(),
        "data_stats": corpus_result.to_dict(),
        "sample_outputs": sample_outputs,
    }
    torch.save(checkpoint, output_dir / "mini_gpt_pretrained.pt")
    save_json(tokenizer.to_dict(), output_dir / "char_tokenizer.json")
    save_json(
        {
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "training_result": training_result.to_dict(),
            "data_stats": corpus_result.to_dict(),
            "tokenizer_stats": tokenizer_result.to_dict(),
        },
        output_dir / "true_pretrain_config.json",
    )
    save_loss_log(training_result.loss_log, output_dir / "loss_log.csv")
    save_sample_outputs(sample_outputs["before"], sample_outputs["after"], output_dir / "sample_outputs.txt")


def print_file_summary(corpus_result: CorpusBuildResult) -> None:
    """打印本次语料文件选择和抽取摘要

    参数含义:
        corpus_result: 语料抽取结果和统计信息

    返回值含义:
        无返回值，直接向控制台打印摘要
    """

    print(f"选择数据文件数: {len(corpus_result.selected_files)}")
    for file_path in corpus_result.selected_files:
        print(f"  - {file_path}")
    if corpus_result.ignored_files:
        print("未纳入本次来源选择的被排除数据文件:")
        for file_path in corpus_result.ignored_files:
            print(f"  - {file_path}")


def main() -> None:
    """运行真实中文语料预训练完整流程

    参数含义:
        无参数

    返回值含义:
        无返回值，直接执行数据抽取、模型训练、效果展示和产物保存
    """

    config = parse_args()
    validate_config(config)
    set_seed(config.seed)
    device = select_device(config.device)

    corpus_result = build_corpus(config)
    tokenizer_result = build_tokenizer(corpus_result.corpus, config.min_char_frequency)
    tokenizer = tokenizer_result.tokenizer
    token_ids = encode_text_token_ids(tokenizer, corpus_result, config.seed)
    train_loader, train_eval_loader, valid_eval_loader, train_ids, valid_ids = build_dataloaders(token_ids, config)

    model_config = GPTConfig(
        vocab_size=len(tokenizer.itos),
        context_length=config.context_length,
        emb_dim=config.emb_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    )
    model = MiniGPT(model_config).to(device)
    if config.init_checkpoint:
        load_initial_checkpoint(model, model_config, tokenizer, Path(config.init_checkpoint), device)

    total_params, trainable_params = count_parameters(model)

    print("真实中文语料预训练配置:")
    print(config)
    print(f"运行设备: {device}")
    print_file_summary(corpus_result)
    print(f"读取原始记录数: {corpus_result.records_seen:,}")
    print(f"参与训练记录数: {corpus_result.records_used:,}")
    print(f"参与训练文本字符数: {len(corpus_result.corpus):,}")
    print(f"字符词表大小: {len(tokenizer.itos):,}")
    print(f"原始唯一字符数: {tokenizer_result.total_unique_chars:,}")
    print(f"保留唯一字符数: {tokenizer_result.kept_unique_chars:,}")
    print(f"低频映射 UNK 唯一字符数: {tokenizer_result.dropped_unique_chars:,}")
    print(f"低频映射 UNK 字符出现次数: {tokenizer_result.dropped_char_occurrences:,}")
    print(f"模型词表大小: {model_config.vocab_size:,}")
    print(f"训练 token 数: {len(train_ids):,}")
    print(f"验证 token 数: {len(valid_ids):,}")
    print(f"模型总参数量: {total_params:,}")
    print(f"模型可训练参数量: {trainable_params:,}")

    if config.dry_run:
        print("\ndry-run 已完成，未执行训练和保存")
        return

    print("\n训练前生成效果:")
    before_text = generate_text(
        model,
        tokenizer,
        config.prompt,
        config.generate_tokens,
        config.temperature,
        config.top_k,
        config.repetition_penalty,
        config.no_repeat_ngram_size,
        device,
    )
    print(before_text)

    print("\n开始预训练:")
    training_result = train_model(model, train_loader, train_eval_loader, valid_eval_loader, config, device)

    print("\n训练后生成效果:")
    after_text = generate_text(
        model,
        tokenizer,
        config.prompt,
        config.generate_tokens,
        config.temperature,
        config.top_k,
        config.repetition_penalty,
        config.no_repeat_ngram_size,
        device,
    )
    print(after_text)

    output_dir = Path(config.output_dir)
    save_checkpoint(
        model=model,
        model_config=model_config,
        train_config=config,
        tokenizer=tokenizer,
        tokenizer_result=tokenizer_result,
        training_result=training_result,
        corpus_result=corpus_result,
        sample_outputs={"before": before_text, "after": after_text},
        output_dir=output_dir,
    )
    print(f"\n真实语料预训练产物已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
