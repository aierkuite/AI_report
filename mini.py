"""第 4 步小参数量 GPT 类模型指令微调程序

本程序用于完成课程任务中的监督式指令微调部分
它会加载第 3 步预训练权重，构建指令数据集，执行全参数微调，并展示微调前后效果
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
from mini_gpt_step3_pretrain import CharTokenizer


DEFAULT_INSTRUCTION_EXAMPLES = [
    {
        "instruction": "请用一句话解释什么是人工智能",
        "input": "",
        "output": "人工智能是让计算机模拟人类学习、理解和决策能力的技术。",
    },
    {
        "instruction": "请总结下面这段话",
        "input": "小型 GPT 模型通过预测下一个 token 来学习文本规律，虽然参数量较小，但能展示大语言模型的核心机制。",
        "output": "小型 GPT 模型可以通过下一个 token 预测展示大模型核心机制。",
    },
    {
        "instruction": "请判断这句话的情感倾向",
        "input": "这次模型训练终于跑通了，我很满意。",
        "output": "正向。",
    },
    {
        "instruction": "请把下面的话改写得更正式",
        "input": "这个模型能跑起来，说明流程基本没问题。",
        "output": "该模型能够正常运行，说明整体流程基本正确。",
    },
    {
        "instruction": "请回答问题",
        "input": "GPT 类模型为什么要使用因果注意力？",
        "output": "因为因果注意力可以阻止模型看到未来 token，使模型只能根据已有上下文预测下一个 token。",
    },
    {
        "instruction": "请列出预训练的三个关键步骤",
        "input": "",
        "output": "预训练通常包括准备文本数据、进行分词编码、使用下一个 token 预测目标训练模型。",
    },
    {
        "instruction": "请解释词嵌入的作用",
        "input": "",
        "output": "词嵌入用于把离散 token 编号转换为连续向量，方便神经网络进行计算。",
    },
    {
        "instruction": "请给下面内容起一个标题",
        "input": "模型先在无标注文本上预训练，再用指令数据微调，使它更适合回答用户问题。",
        "output": "从预训练到指令微调。",
    },
    {
        "instruction": "请判断下面说法是否正确",
        "input": "微调通常发生在预训练之后。",
        "output": "正确。",
    },
    {
        "instruction": "请用简短语言说明残差连接的作用",
        "input": "",
        "output": "残差连接有助于信息和梯度在深层网络中稳定传递。",
    },
]


@dataclass
class InstructionExample:
    """保存一条监督式指令微调样本

    参数含义:
        instruction: 用户希望模型执行的任务指令
        input: 任务需要参考的输入内容，可以为空
        output: 期望模型生成的标准回答

    返回值含义:
        InstructionExample 实例用于统一表示一条指令微调数据
    """

    instruction: str
    input: str
    output: str


@dataclass
class FinetuneConfig:
    """保存第 4 步指令微调流程的核心配置

    参数含义:
        data_dir: 存放指令微调 json 或 jsonl 数据的目录
        pretrained: 第 3 步保存的预训练 checkpoint 路径
        output_dir: 存放微调权重、分词器、日志和样例输出的目录
        batch_size: 每次训练使用的样本数量
        max_steps: 最大微调步数
        eval_interval: 每隔多少步评估一次训练集和验证集损失
        eval_batches: 每次评估最多使用的 batch 数量
        learning_rate: AdamW 优化器学习率
        weight_decay: AdamW 优化器权重衰减系数
        grad_clip: 梯度裁剪阈值
        train_split: 训练样本占全部样本的比例
        seed: 随机种子
        sample_instruction: 训练前后用于生成效果对比的指令
        sample_input: 训练前后用于生成效果对比的输入内容
        generate_tokens: 每次效果测试生成的新 token 数量
        temperature: 生成采样温度
        top_k: 生成时保留概率最高的候选 token 数量
        device: 运行设备，auto 表示自动选择 cuda 或 cpu

    返回值含义:
        FinetuneConfig 实例用于统一传递指令微调参数
    """

    data_dir: str = "data_instruction"
    pretrained: str = "outputs_pretrain/mini_gpt_pretrained.pt"
    output_dir: str = "outputs_instruction_finetune"
    batch_size: int = 4
    max_steps: int = 300
    eval_interval: int = 50
    eval_batches: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    train_split: float = 0.9
    seed: int = 42
    sample_instruction: str = "请用一句话解释什么是人工智能"
    sample_input: str = ""
    generate_tokens: int = 120
    temperature: float = 0.8
    top_k: int = 20
    device: str = "auto"


class InstructionDataset(Dataset):
    """把指令样本转换为 GPT 监督微调训练样本"""

    def __init__(
        self,
        examples: list[InstructionExample],
        tokenizer: CharTokenizer,
        context_length: int,
    ) -> None:
        """初始化指令微调数据集

        参数含义:
            examples: 指令样本列表
            tokenizer: 字符级分词器
            context_length: 模型支持的最大输入 token 数量

        返回值含义:
            无返回值，预先编码所有样本以便训练时读取
        """

        self.samples = [
            build_supervised_sample(example, tokenizer, context_length)
            for example in examples
        ]

    def __len__(self) -> int:
        """返回数据集样本数量

        参数含义:
            无参数

        返回值含义:
            返回数据集中指令样本数量
        """

        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """根据索引返回一个指令微调训练样本

        参数含义:
            index: 样本索引

        返回值含义:
            返回二元组 input_ids 和 labels，labels 中非回答区域为 -100
        """

        return self.samples[index]


def parse_args() -> FinetuneConfig:
    """解析命令行参数并生成指令微调配置

    参数含义:
        无参数

    返回值含义:
        返回从命令行参数生成的 FinetuneConfig 实例
    """

    parser = argparse.ArgumentParser(description="第 4 步小参数量 GPT 类模型指令微调程序")
    parser.add_argument("--data-dir", default="data_instruction", help="存放 json 或 jsonl 指令微调数据的目录")
    parser.add_argument("--pretrained", default="outputs_pretrain/mini_gpt_pretrained.pt", help="第 3 步预训练 checkpoint 路径")
    parser.add_argument("--output-dir", default="outputs_instruction_finetune", help="保存微调产物的目录")
    parser.add_argument("--batch-size", type=int, default=4, help="每个 batch 的样本数量")
    parser.add_argument("--max-steps", type=int, default=1000, help="最大微调步数")
    parser.add_argument("--eval-interval", type=int, default=50, help="评估间隔步数")
    parser.add_argument("--eval-batches", type=int, default=10, help="每次评估最多使用的 batch 数量")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="权重衰减")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--train-split", type=float, default=0.9, help="训练集样本比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--sample-instruction", default="请用一句话解释什么是人工智能", help="训练前后用于生成对比的指令")
    parser.add_argument("--sample-input", default="", help="训练前后用于生成对比的输入内容")
    parser.add_argument("--generate-tokens", type=int, default=120, help="生成的新 token 数量")
    parser.add_argument("--temperature", type=float, default=0.8, help="生成采样温度")
    parser.add_argument("--top-k", type=int, default=20, help="生成时保留的候选 token 数量")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    args = parser.parse_args()
    return FinetuneConfig(**vars(args))


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
    """对指令数据中的文本做轻量清洗

    参数含义:
        text: 原始文本

    返回值含义:
        返回统一换行并去除首尾空白后的文本
    """

    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def format_prompt(instruction: str, input_text: str, output: str | None = None) -> str:
    """把指令样本转换为统一提示词模板

    参数含义:
        instruction: 任务指令文本
        input_text: 任务输入文本，可以为空
        output: 标准回答文本，传入 None 时只生成回答前缀

    返回值含义:
        返回格式化后的完整提示词或回答前缀
    """

    prompt = (
        f"### 指令:\n{normalize_text(instruction)}\n\n"
        f"### 输入:\n{normalize_text(input_text)}\n\n"
        "### 回答:\n"
    )
    if output is None:
        return prompt
    return prompt + normalize_text(output)


def parse_instruction_record(record: dict[str, object], source: Path, record_number: int) -> InstructionExample:
    """把 JSON 字典转换为指令样本对象

    参数含义:
        record: 从 JSON 或 JSONL 中读取的一条字典
        source: 当前指令数据文件路径
        record_number: 当前记录序号，读取 JSONL 时等同于行号

    返回值含义:
        返回 InstructionExample 实例，字段缺失时抛出 ValueError
    """

    instruction = normalize_text(record.get("instruction", ""))
    input_text = normalize_text(record.get("input", ""))
    output = normalize_text(record.get("output", ""))
    if not instruction or not output:
        raise ValueError(f"{source} 第 {record_number} 条必须包含非空 instruction 和 output")
    return InstructionExample(instruction=instruction, input=input_text, output=output)


def collect_instruction_files(data_path: Path) -> list[Path]:
    """收集指令微调数据文件

    参数含义:
        data_path: 指令数据目录路径，也可以直接传入单个 json 或 jsonl 文件路径

    返回值含义:
        返回按路径排序后的 json 和 jsonl 文件列表，路径不存在或格式不支持时返回空列表
    """

    supported_suffixes = {".json", ".jsonl"}
    if not data_path.exists():
        return []
    if data_path.is_file():
        return [data_path] if data_path.suffix.lower() in supported_suffixes else []
    files = [
        file_path
        for file_path in data_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in supported_suffixes
    ]
    return sorted(files)


def load_jsonl_instruction_examples(file_path: Path) -> list[InstructionExample]:
    """读取 JSONL 指令微调样本

    参数含义:
        file_path: JSONL 指令数据文件路径，每行应包含 instruction、input 和 output 字段

    返回值含义:
        返回从文件中解析出的 InstructionExample 列表
    """

    examples: list[InstructionExample] = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{file_path} 第 {line_number} 行必须是 JSON 对象")
            examples.append(parse_instruction_record(record, file_path, line_number))
    return examples


def load_json_instruction_examples(file_path: Path) -> list[InstructionExample]:
    """读取 JSON 指令微调样本

    参数含义:
        file_path: JSON 指令数据文件路径，文件内容应为样本数组或包含样本数组的字典

    返回值含义:
        返回从文件中解析出的 InstructionExample 列表
    """

    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        data = json.load(file)

    if isinstance(data, dict):
        if {"instruction", "output"}.issubset(data):
            records = [data]
        else:
            records = None
            for key in ("data", "train", "examples", "records"):
                value = data.get(key)
                if isinstance(value, list):
                    records = value
                    break
            if records is None:
                raise ValueError(f"{file_path} 必须是样本数组，或在 data/train/examples/records 字段中包含样本数组")
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"{file_path} 顶层结构必须是 JSON 数组或 JSON 对象")

    examples: list[InstructionExample] = []
    for record_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{file_path} 第 {record_number} 条必须是 JSON 对象")
        examples.append(parse_instruction_record(record, file_path, record_number))
    return examples


def load_instruction_examples(data_dir: Path) -> tuple[list[InstructionExample], list[Path]]:
    """读取指令微调目录中的所有 json 和 jsonl 文件

    参数含义:
        data_dir: 存放 json 或 jsonl 指令微调数据的目录路径，也可以是单个数据文件路径

    返回值含义:
        返回二元组，第一个值是指令样本列表，第二个值是实际读取的文件路径列表
    """

    data_files = collect_instruction_files(data_dir)
    examples: list[InstructionExample] = []
    for file_path in data_files:
        if file_path.suffix.lower() == ".jsonl":
            examples.extend(load_jsonl_instruction_examples(file_path))
        else:
            examples.extend(load_json_instruction_examples(file_path))

    if examples:
        return examples, data_files

    default_examples = [
        InstructionExample(
            instruction=item["instruction"],
            input=item["input"],
            output=item["output"],
        )
        for item in DEFAULT_INSTRUCTION_EXAMPLES
    ]
    return default_examples, []


def load_pretrained_checkpoint(pretrained_path: Path) -> dict[str, object]:
    """加载第 3 步保存的预训练 checkpoint

    参数含义:
        pretrained_path: 预训练 checkpoint 文件路径

    返回值含义:
        返回 torch.load 读取到的 checkpoint 字典
    """

    if not pretrained_path.exists():
        raise FileNotFoundError(f"找不到预训练 checkpoint: {pretrained_path}")
    return torch.load(pretrained_path, map_location="cpu")


def tokenizer_from_checkpoint(checkpoint: dict[str, object]) -> CharTokenizer:
    """从 checkpoint 中还原字符级分词器

    参数含义:
        checkpoint: 第 3 步保存的 checkpoint 字典

    返回值含义:
        返回 CharTokenizer 实例
    """

    tokenizer_data = checkpoint.get("tokenizer")
    if not isinstance(tokenizer_data, dict) or "itos" not in tokenizer_data:
        raise ValueError("checkpoint 中缺少 tokenizer.itos")
    return CharTokenizer(list(tokenizer_data["itos"]))


def collect_text_for_vocab(examples: list[InstructionExample], config: FinetuneConfig) -> str:
    """收集需要纳入字符词表的所有指令文本

    参数含义:
        examples: 指令样本列表
        config: 指令微调配置对象

    返回值含义:
        返回拼接后的文本，用于检查并扩展字符词表
    """

    texts = [
        format_prompt(example.instruction, example.input, example.output)
        for example in examples
    ]
    texts.append(format_prompt(config.sample_instruction, config.sample_input))
    return "\n".join(texts)


def expand_tokenizer(tokenizer: CharTokenizer, text: str) -> tuple[CharTokenizer, int]:
    """根据指令文本扩展字符级词表

    参数含义:
        tokenizer: 从预训练 checkpoint 还原的字符级分词器
        text: 指令样本和测试提示词拼接后的文本

    返回值含义:
        返回二元组，第一个值是扩展后的分词器，第二个值是新增 token 数量
    """

    existing_tokens = set(tokenizer.itos)
    new_tokens = sorted(char for char in set(text) if char not in existing_tokens)
    if not new_tokens:
        return tokenizer, 0
    return CharTokenizer(tokenizer.itos + new_tokens), len(new_tokens)


def build_model_config(checkpoint: dict[str, object], vocab_size: int) -> GPTConfig:
    """根据 checkpoint 和当前词表大小构建模型配置

    参数含义:
        checkpoint: 第 3 步保存的 checkpoint 字典
        vocab_size: 指令微调阶段实际使用的词表大小

    返回值含义:
        返回适用于当前词表的 GPTConfig 实例
    """

    model_config_data = checkpoint.get("model_config")
    if not isinstance(model_config_data, dict):
        raise ValueError("checkpoint 中缺少 model_config")
    model_config = GPTConfig(**model_config_data)
    model_config.vocab_size = vocab_size
    return model_config


def load_state_dict_with_resize(model: MiniGPT, checkpoint: dict[str, object]) -> tuple[list[str], list[str], list[str]]:
    """加载预训练权重并适配扩展后的词表大小

    参数含义:
        model: 已按新词表大小创建的 MiniGPT 模型
        checkpoint: 第 3 步保存的 checkpoint 字典

    返回值含义:
        返回三元组，分别是直接加载的参数名、经过尺寸适配的参数名、跳过的参数名
    """

    pretrained_state = checkpoint.get("model_state_dict")
    if not isinstance(pretrained_state, dict):
        raise ValueError("checkpoint 中缺少 model_state_dict")

    target_state = model.state_dict()
    loaded_keys: list[str] = []
    resized_keys: list[str] = []
    skipped_keys: list[str] = []

    for key, value in pretrained_state.items():
        if key not in target_state:
            skipped_keys.append(key)
            continue

        target_value = target_state[key]
        if target_value.shape == value.shape:
            target_state[key] = value
            loaded_keys.append(key)
            continue

        can_resize_vocab_weight = (
            key in {"token_embedding.weight", "lm_head.weight"}
            and target_value.ndim == 2
            and value.ndim == 2
            and target_value.shape[1] == value.shape[1]
        )
        if can_resize_vocab_weight:
            resized_value = target_value.clone()
            rows = min(target_value.shape[0], value.shape[0])
            resized_value[:rows, :] = value[:rows, :]
            target_state[key] = resized_value
            resized_keys.append(key)
            continue

        skipped_keys.append(key)

    model.load_state_dict(target_state)
    return loaded_keys, resized_keys, skipped_keys


def split_examples(
    examples: list[InstructionExample],
    train_split: float,
    seed: int,
) -> tuple[list[InstructionExample], list[InstructionExample]]:
    """把指令样本切分为训练集和验证集

    参数含义:
        examples: 全部指令样本列表
        train_split: 训练集样本比例
        seed: 控制样本打乱的随机种子

    返回值含义:
        返回训练样本列表和验证样本列表
    """

    if not examples:
        raise ValueError("指令样本不能为空")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, shuffled

    split_index = int(len(shuffled) * train_split)
    split_index = max(1, min(split_index, len(shuffled) - 1))
    return shuffled[:split_index], shuffled[split_index:]


def build_supervised_sample(
    example: InstructionExample,
    tokenizer: CharTokenizer,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把一条指令样本编码为带监督掩码的训练样本

    参数含义:
        example: 一条指令微调样本
        tokenizer: 字符级分词器
        context_length: 模型最大输入 token 数量

    返回值含义:
        返回二元组 input_ids 和 labels，长度均为 context_length
    """

    prefix = format_prompt(example.instruction, example.input)
    output = normalize_text(example.output)
    prefix_ids = tokenizer.encode(prefix)
    output_ids = tokenizer.encode(output)
    if not output_ids:
        output_ids = [tokenizer.unk_id]

    max_total_length = context_length + 1
    if len(prefix_ids) >= max_total_length:
        prefix_ids = prefix_ids[-(max_total_length - 1) :]

    available_output_length = max_total_length - len(prefix_ids)
    output_ids = output_ids[:available_output_length]
    ids = prefix_ids + output_ids
    answer_start = len(prefix_ids)

    input_ids = ids[:-1]
    labels = ids[1:]
    ignore_count = max(answer_start - 1, 0)
    labels[:ignore_count] = [-100] * min(ignore_count, len(labels))

    pad_length = context_length - len(input_ids)
    if pad_length > 0:
        input_ids = input_ids + [tokenizer.pad_id] * pad_length
        labels = labels + [-100] * pad_length

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def build_dataloaders(
    examples: list[InstructionExample],
    tokenizer: CharTokenizer,
    model_config: GPTConfig,
    config: FinetuneConfig,
) -> tuple[DataLoader, DataLoader, list[InstructionExample], list[InstructionExample]]:
    """构建训练集和验证集 DataLoader

    参数含义:
        examples: 全部指令样本列表
        tokenizer: 字符级分词器
        model_config: 模型结构配置对象
        config: 指令微调配置对象

    返回值含义:
        返回训练 DataLoader、验证 DataLoader、训练样本列表和验证样本列表
    """

    train_examples, valid_examples = split_examples(examples, config.train_split, config.seed)
    train_dataset = InstructionDataset(train_examples, tokenizer, model_config.context_length)
    valid_dataset = InstructionDataset(valid_examples, tokenizer, model_config.context_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    return train_loader, valid_loader, train_examples, valid_examples


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
    for batch_index, (input_ids, labels) in enumerate(data_loader):
        if batch_index >= max_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        losses.append(loss.item())
    model.train()
    if not losses:
        return float("inf")
    return sum(losses) / len(losses)


def train_model(
    model: MiniGPT,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: FinetuneConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    """执行 GPT 指令微调训练循环

    参数含义:
        model: 待微调的 MiniGPT 模型
        train_loader: 训练集 DataLoader
        valid_loader: 验证集 DataLoader
        config: 指令微调流程配置对象
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
            input_ids, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, labels = next(train_iter)

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

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


@torch.no_grad()
def generate_instruction_answer(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    """使用模型根据指令生成回答文本

    参数含义:
        model: 用于生成的 MiniGPT 模型
        tokenizer: 字符级分词器
        instruction: 任务指令文本
        input_text: 任务输入文本
        max_new_tokens: 需要继续生成的新 token 数量
        temperature: 采样温度
        top_k: 每步采样保留的候选 token 数量
        device: 执行生成的设备

    返回值含义:
        返回解码后的完整提示词和生成内容
    """

    model.eval()
    prompt = format_prompt(instruction, input_text)
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.unk_id]
    token_ids = token_ids[-model.config.context_length :]
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    generated = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(generated[0].tolist())


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


def save_sample_outputs(
    before_text: str,
    after_text: str,
    file_path: Path,
) -> None:
    """保存微调前后的样例生成结果

    参数含义:
        before_text: 微调前生成结果
        after_text: 微调后生成结果
        file_path: 样例输出文件保存路径

    返回值含义:
        无返回值，直接写入文本文件
    """

    content = (
        "训练前生成效果:\n"
        f"{before_text}\n\n"
        "训练后生成效果:\n"
        f"{after_text}\n"
    )
    with open(file_path, "w", encoding="utf-8", newline="\r\n") as file:
        file.write(content)


def save_checkpoint(
    model: MiniGPT,
    model_config: GPTConfig,
    finetune_config: FinetuneConfig,
    tokenizer: CharTokenizer,
    loss_log: list[dict[str, float]],
    before_text: str,
    after_text: str,
    output_dir: Path,
) -> None:
    """保存指令微调模型权重、分词器、配置、日志和样例输出

    参数含义:
        model: 微调完成的 MiniGPT 模型
        model_config: 模型结构配置对象
        finetune_config: 指令微调流程配置对象
        tokenizer: 字符级分词器
        loss_log: 训练损失日志
        before_text: 微调前生成结果
        after_text: 微调后生成结果
        output_dir: 输出目录路径

    返回值含义:
        无返回值，直接在输出目录写入文件
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "finetune_config": asdict(finetune_config),
        "tokenizer": tokenizer.to_dict(),
        "loss_log": loss_log,
        "sample_outputs": {
            "before": before_text,
            "after": after_text,
        },
    }
    torch.save(checkpoint, output_dir / "mini_gpt_instruction_finetuned.pt")
    save_json(tokenizer.to_dict(), output_dir / "char_tokenizer.json")
    save_json(
        {
            "model_config": asdict(model_config),
            "finetune_config": asdict(finetune_config),
        },
        output_dir / "finetune_config.json",
    )
    save_loss_log(loss_log, output_dir / "loss_log.csv")
    save_sample_outputs(before_text, after_text, output_dir / "sample_outputs.txt")


def main() -> None:
    """运行第 4 步指令微调完整流程

    参数含义:
        无参数

    返回值含义:
        无返回值，直接执行数据读取、权重加载、模型微调、效果展示和产物保存
    """

    config = parse_args()
    set_seed(config.seed)
    device = select_device(config.device)

    checkpoint = load_pretrained_checkpoint(Path(config.pretrained))
    examples, data_files = load_instruction_examples(Path(config.data_dir))
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    tokenizer, added_tokens = expand_tokenizer(tokenizer, collect_text_for_vocab(examples, config))
    model_config = build_model_config(checkpoint, len(tokenizer.itos))
    model = MiniGPT(model_config)
    loaded_keys, resized_keys, skipped_keys = load_state_dict_with_resize(model, checkpoint)
    model = model.to(device)
    train_loader, valid_loader, train_examples, valid_examples = build_dataloaders(examples, tokenizer, model_config, config)
    total_params, trainable_params = count_parameters(model)

    print("第 4 步指令微调配置:")
    print(config)
    print(f"运行设备: {device}")
    print(f"读取指令文件数: {len(data_files)}")
    print(f"指令样本数: {len(examples):,}")
    print(f"训练样本数: {len(train_examples):,}")
    print(f"验证样本数: {len(valid_examples):,}")
    print(f"词表大小: {len(tokenizer.itos):,}")
    print(f"新增 token 数: {added_tokens:,}")
    print(f"模型总参数量: {total_params:,}")
    print(f"模型可训练参数量: {trainable_params:,}")
    print(f"直接加载参数数: {len(loaded_keys):,}")
    print(f"尺寸适配参数数: {len(resized_keys):,}")
    print(f"跳过参数数: {len(skipped_keys):,}")

    print("\n微调前生成效果:")
    before_text = generate_instruction_answer(
        model,
        tokenizer,
        config.sample_instruction,
        config.sample_input,
        config.generate_tokens,
        config.temperature,
        config.top_k,
        device,
    )
    print(before_text)

    print("\n开始指令微调:")
    loss_log = train_model(model, train_loader, valid_loader, config, device)

    print("\n微调后生成效果:")
    after_text = generate_instruction_answer(
        model,
        tokenizer,
        config.sample_instruction,
        config.sample_input,
        config.generate_tokens,
        config.temperature,
        config.top_k,
        device,
    )
    print(after_text)

    output_dir = Path(config.output_dir)
    save_checkpoint(model, model_config, config, tokenizer, loss_log, before_text, after_text, output_dir)
    print(f"\n指令微调产物已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
