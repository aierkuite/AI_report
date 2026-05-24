"""第 4 步 Qwen2.5 LoRA 指令微调程序

本程序用于完成课程任务中的监督式中文指令微调部分
它会加载 Qwen2.5 因果语言模型和 tokenizer，读取中文指令数据，执行 LoRA 指令微调，并保存本地适配器产物
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None


DEFAULT_INSTRUCTION_EXAMPLES = [
    {
        "instruction": "请用一句话解释什么是人工智能",
        "input": "",
        "output": "人工智能是让计算机模拟人类学习、理解和决策能力的技术。",
    },
    {
        "instruction": "请总结下面这段话",
        "input": "小型 GPT 模型通过预测下一个 token 来学习文本规律，并展示大语言模型的核心机制。",
        "output": "小型 GPT 模型通过下一个 token 预测展示大模型机制。",
    },
    {
        "instruction": "请判断下面句子的情感倾向",
        "input": "这次模型训练终于跑通了，我很满意。",
        "output": "这句话的情感倾向是正向。",
    },
    {
        "instruction": "请把下面的话改写得更正式",
        "input": "这个模型能跑起来，说明流程基本没问题。",
        "output": "该模型能够正常运行，说明整体流程基本正确。",
    },
    {
        "instruction": "请回答问题",
        "input": "GPT 类模型为什么要使用因果注意力？",
        "output": "因为因果注意力可以防止模型看到未来 token，使模型只能根据已有上下文预测下一个 token。",
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
class SampleEncodingStats:
    """保存单条样本编码过程中的长度和截断信息

    参数含义:
        original_prefix_length: 截断前提示词 token 数量
        original_output_length: 截断前回答 token 数量，包含结尾 token
        final_prefix_length: 截断后提示词 token 数量
        final_output_length: 截断后回答 token 数量
        effective_length: padding 前实际参与模型输入的 token 数量
        prompt_truncated: 提示词是否发生截断
        output_truncated: 回答是否发生截断

    返回值含义:
        SampleEncodingStats 实例用于统计数据集编码质量
    """

    original_prefix_length: int
    original_output_length: int
    final_prefix_length: int
    final_output_length: int
    effective_length: int
    prompt_truncated: bool
    output_truncated: bool


@dataclass
class DatasetEncodingStats:
    """保存一个数据集编码后的聚合统计

    参数含义:
        sample_count: 样本数量
        prompt_truncated_count: 提示词被截断的样本数量
        output_truncated_count: 回答被截断的样本数量
        average_effective_length: padding 前平均有效长度
        max_effective_length: padding 前最大有效长度

    返回值含义:
        DatasetEncodingStats 实例用于展示和保存数据截断统计
    """

    sample_count: int
    prompt_truncated_count: int
    output_truncated_count: int
    average_effective_length: float
    max_effective_length: int


DEFAULT_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass
class FinetuneConfig:
    """保存 Qwen2.5 LoRA 指令微调流程的核心配置

    参数含义:
        model_name: Hugging Face 模型名称，默认使用本地 Qwen2.5 模型
        model_dir: Qwen2.5 模型本地保存目录，存在时优先从本地加载
        data_file: 优先读取的指令微调数据文件，支持 json 数组和 jsonl
        data_dir: 回退读取的指令微调 jsonl 数据目录
        output_dir: 存放微调权重、配置、日志和样例输出的目录
        template_style: 指令模板风格，qwen_chat 表示 Qwen 对话模板
        system_prompt: Qwen 对话模板中的 system 消息
        max_length: tokenizer 编码后的最大序列长度
        max_samples: 最多使用的指令样本数量，0 或负数表示使用全量数据
        batch_size: 每次训练使用的样本数量
        gradient_accumulation_steps: 累积多少个 batch 后执行一次参数更新
        max_steps: 最大参数更新步数，不是 micro batch 数
        eval_interval: 每隔多少次参数更新评估一次训练集和验证集损失
        eval_batches: 每次评估最多使用的 batch 数量
        early_stopping_patience: 验证损失连续多少次评估未改进后提前终止，0 或负数表示关闭
        early_stopping_min_delta: 验证损失下降超过该阈值才视为有效改进
        learning_rate: AdamW 优化器学习率
        warmup_steps: 学习率预热的 optimizer step 数量
        lr_scheduler_type: 学习率调度策略，支持 none、linear、cosine
        min_lr_ratio: 学习率衰减结束时保留的最低比例
        lora_r: LoRA 低秩矩阵秩
        lora_alpha: LoRA 缩放系数
        lora_dropout: LoRA dropout 比例
        lora_target_modules: 逗号分隔的 LoRA 目标模块名称
        trust_remote_code: 加载模型和 tokenizer 时是否信任远程代码
        torch_dtype: 模型加载 dtype，支持 auto、float32、float16、bfloat16
        weight_decay: AdamW 优化器权重衰减系数
        grad_clip: 梯度裁剪阈值
        train_split: 训练样本占全部样本的比例
        seed: 随机种子
        sample_instruction: 训练前后用于生成效果对比的指令
        sample_input: 训练前后用于生成效果对比的输入内容
        generate_tokens: 每次效果测试生成的新 token 数量
        do_sample: 生成样例时是否使用随机采样
        temperature: 生成采样温度
        top_k: 生成时保留概率最高的候选 token 数量
        repetition_penalty: 生成时对已出现 token 的重复惩罚系数
        no_repeat_ngram_size: 生成时禁止重复的 ngram 长度，0 表示关闭
        device: 运行设备，auto 表示自动选择 cuda 或 cpu
        local_files_only: 只从本地缓存或本地目录加载模型
        force_download: 忽略本地 model_dir 并重新从模型名称加载

    返回值含义:
        FinetuneConfig 实例用于统一传递 Qwen2.5 LoRA 指令微调参数
    """

    model_name: str = "models/qwen2.5-0.5b"
    model_dir: str = "models/qwen2.5-0.5b"
    data_file: str = "data_instruction/alpaca_gpt4_data_zh.json"
    data_dir: str = "data_instruction"
    output_dir: str = "outputs_qwen_lora_finetune"
    template_style: str = "qwen_chat"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_length: int = 1024
    max_samples: int = 0
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 6000
    eval_interval: int = 50
    eval_batches: int = 50
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    lr_scheduler_type: str = "linear"
    min_lr_ratio: float = 0.1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = DEFAULT_LORA_TARGET_MODULES
    trust_remote_code: bool = False
    torch_dtype: str = "auto"
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    train_split: float = 0.9
    seed: int = 42
    sample_instruction: str = "请用一句话解释什么是人工智能"
    sample_input: str = ""
    generate_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.8
    top_k: int = 50
    repetition_penalty: float = 1.05
    no_repeat_ngram_size: int = 0
    device: str = "auto"
    local_files_only: bool = False
    force_download: bool = False


class InstructionDataset(Dataset):
    """把指令样本转换为因果语言模型监督微调训练样本"""

    def __init__(
        self,
        examples: list[InstructionExample],
        tokenizer: AutoTokenizer,
        max_length: int,
        template_style: str,
        system_prompt: str,
    ) -> None:
        """初始化指令微调数据集

        参数含义:
            examples: 指令样本列表
            tokenizer: 因果语言模型 tokenizer
            max_length: 编码后的最大输入 token 数量
            template_style: 指令模板风格，控制样本编码格式
            system_prompt: Qwen 对话模板中的 system 消息

        返回值含义:
        无返回值，预先编码所有样本以便训练时读取
    """

        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        encoding_stats: list[SampleEncodingStats] = []
        for example in examples:
            input_ids, attention_mask, labels, sample_stats = build_supervised_sample(
                example,
                tokenizer,
                max_length,
                template_style,
                system_prompt,
            )
            self.samples.append((input_ids, attention_mask, labels))
            encoding_stats.append(sample_stats)
        self.stats = summarize_encoding_stats(encoding_stats)

    def __len__(self) -> int:
        """返回数据集样本数量

        参数含义:
            无参数

        返回值含义:
            返回数据集中指令样本数量
        """

        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据索引返回一个指令微调训练样本

        参数含义:
            index: 样本索引

        返回值含义:
            返回 input_ids、attention_mask 和 labels，labels 中非回答区域为 -100
        """

        return self.samples[index]


def parse_args() -> FinetuneConfig:
    """解析命令行参数并生成指令微调配置

    参数含义:
        无参数

    返回值含义:
        返回从命令行参数生成的 FinetuneConfig 实例
    """

    parser = argparse.ArgumentParser(description="第 4 步 Qwen2.5 LoRA 中文指令微调程序")
    parser.add_argument("--model-name", default="models/qwen2.5-0.5b", help="Hugging Face 模型名称或本地模型目录")
    parser.add_argument("--model-dir", default="models/qwen2.5-0.5b", help="Qwen2.5 模型本地保存目录")
    parser.add_argument("--data-file", default="data_instruction/alpaca_gpt4_data_zh.json", help="优先读取的指令数据文件，支持 json 数组和 jsonl")
    parser.add_argument("--data-dir", default="data_instruction", help="存放 jsonl 指令微调数据的目录")
    parser.add_argument("--output-dir", default="outputs_qwen_lora_finetune", help="保存微调产物的目录")
    parser.add_argument("--template-style", default="qwen_chat", choices=["alpaca", "zh", "qwen_chat"], help="指令模板风格")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="Qwen 对话模板中的 system 消息")
    parser.add_argument("--max-length", type=int, default=1024, help="tokenizer 编码后的最大序列长度")
    parser.add_argument("--max-samples", type=int, default=0, help="最多使用的指令样本数量，0 或负数表示使用全量数据")
    parser.add_argument("--batch-size", type=int, default=1, help="每个 batch 的样本数量")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="累积多少个 batch 后执行一次参数更新")
    parser.add_argument("--max-steps", type=int, default=6000, help="最大参数更新步数，不是 micro batch 数")
    parser.add_argument("--eval-interval", type=int, default=50, help="每隔多少次参数更新评估一次")
    parser.add_argument("--eval-batches", type=int, default=50, help="每次评估最多使用的 batch 数量")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="验证损失连续未改进多少次后提前终止，0 或负数表示关闭")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0, help="验证损失下降超过该阈值才视为有效改进")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--warmup-steps", type=int, default=100, help="学习率预热的 optimizer step 数量")
    parser.add_argument("--lr-scheduler-type", default="linear", choices=["none", "linear", "cosine"], help="学习率调度策略")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1, help="学习率衰减结束时保留的最低比例")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA 低秩矩阵秩")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA 缩放系数")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout 比例")
    parser.add_argument("--lora-target-modules", default=DEFAULT_LORA_TARGET_MODULES, help="逗号分隔的 LoRA 目标模块名称")
    parser.add_argument("--trust-remote-code", action="store_true", help="加载模型和 tokenizer 时信任远程代码")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="模型加载 dtype")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="权重衰减")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--train-split", type=float, default=0.9, help="训练集样本比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--sample-instruction", default="请用一句话解释什么是人工智能", help="训练前后用于生成对比的指令")
    parser.add_argument("--sample-input", default="", help="训练前后用于生成对比的输入内容")
    parser.add_argument("--generate-tokens", type=int, default=128, help="生成的新 token 数量")
    parser.add_argument("--do-sample", action="store_true", help="生成样例时启用随机采样，默认使用确定性生成")
    parser.add_argument("--temperature", type=float, default=0.8, help="生成采样温度")
    parser.add_argument("--top-k", type=int, default=50, help="生成时保留的候选 token 数量")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="生成时对已出现 token 的重复惩罚系数")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0, help="生成时禁止重复的 ngram 长度，0 表示关闭")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    parser.add_argument("--local-files-only", action="store_true", help="只从本地缓存或本地目录加载模型")
    parser.add_argument("--force-download", action="store_true", help="忽略本地 model_dir 并重新从 model_name 加载")
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


def format_prompt(
    instruction: str,
    input_text: str,
    output: str | None = None,
    template_style: str = "alpaca",
    tokenizer: AutoTokenizer | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """把指令样本转换为统一提示词模板

    参数含义:
        instruction: 任务指令文本
        input_text: 任务输入文本，可以为空
        output: 标准回答文本，传入 None 时只生成回答前缀
        template_style: 指令模板风格，qwen_chat 表示 Qwen 对话模板
        tokenizer: Qwen 对话模板需要使用的 tokenizer，非 qwen_chat 模板可为空
        system_prompt: Qwen 对话模板中的 system 消息

    返回值含义:
        返回格式化后的完整提示词或回答前缀
    """

    instruction = normalize_text(instruction)
    input_text = normalize_text(input_text)
    if template_style == "alpaca":
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
        )
        if input_text:
            prompt += f"### Input:\n{input_text}\n\n"
        prompt += "### Response:\n"
    elif template_style == "zh":
        prompt = (
            f"### 指令:\n{instruction}\n\n"
            f"### 输入:\n{input_text}\n\n"
            "### 回答:\n"
        )
    elif template_style == "qwen_chat":
        if tokenizer is None:
            raise ValueError("qwen_chat 模板必须传入 tokenizer")
        user_content = build_user_content(instruction, input_text)
        messages = [
            {"role": "system", "content": normalize_text(system_prompt)},
            {"role": "user", "content": user_content},
        ]
        if output is None:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        messages.append({"role": "assistant", "content": normalize_text(output)})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        raise ValueError(f"不支持的模板风格: {template_style}")

    if output is None:
        return prompt
    return prompt + normalize_text(output)


def build_user_content(instruction: str, input_text: str) -> str:
    """构建 Qwen 对话模板中的用户消息内容

    参数含义:
        instruction: 任务指令文本
        input_text: 任务输入文本，可以为空

    返回值含义:
        返回合并指令和输入后的用户消息文本
    """

    instruction = normalize_text(instruction)
    input_text = normalize_text(input_text)
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def parse_instruction_record(record: dict[str, object], source: Path, item_number: int) -> InstructionExample:
    """把 JSON 字典转换为指令样本对象

    参数含义:
        record: 从 JSON 或 JSONL 中读取的一条字典
        source: 当前 JSON 或 JSONL 文件路径
        item_number: 当前记录序号

    返回值含义:
        返回 InstructionExample 实例，字段缺失时抛出 ValueError
    """

    instruction = normalize_text(record.get("instruction", ""))
    input_text = normalize_text(record.get("input", ""))
    output = normalize_text(record.get("output", ""))
    if not instruction or not output:
        raise ValueError(f"{source} 第 {item_number} 条记录必须包含非空 instruction 和 output")
    return InstructionExample(instruction=instruction, input=input_text, output=output)


def load_json_instruction_file(file_path: Path) -> list[InstructionExample]:
    """读取 json 数组格式的指令微调数据文件

    参数含义:
        file_path: 指令微调 json 文件路径

    返回值含义:
        返回从 json 数组中解析出的指令样本列表
    """

    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"{file_path} 必须是 JSON 数组格式")

    examples: list[InstructionExample] = []
    for item_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{file_path} 第 {item_number} 条记录必须是 JSON 对象")
        examples.append(parse_instruction_record(record, file_path, item_number))
    return examples


def load_jsonl_instruction_file(file_path: Path) -> list[InstructionExample]:
    """读取 jsonl 格式的指令微调数据文件

    参数含义:
        file_path: 指令微调 jsonl 文件路径

    返回值含义:
        返回从 jsonl 文件中解析出的指令样本列表
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


def resolve_data_file(data_file: str) -> Path | None:
    """把命令行中的数据文件参数解析为路径

    参数含义:
        data_file: 命令行传入的数据文件字符串，空字符串表示不使用单文件

    返回值含义:
        返回 Path 对象，若传入空字符串则返回 None
    """

    if not data_file.strip():
        return None
    return Path(data_file)


def load_instruction_examples(data_file: Path | None, data_dir: Path) -> tuple[list[InstructionExample], list[Path]]:
    """读取指令微调数据文件或目录

    参数含义:
        data_file: 优先读取的指令数据文件路径，传入 None 表示跳过
        data_dir: 回退读取的 jsonl 指令微调数据目录路径

    返回值含义:
        返回二元组，第一个值是指令样本列表，第二个值是实际读取的文件路径列表
    """

    if data_file is not None and str(data_file) and data_file.exists():
        suffix = data_file.suffix.lower()
        if suffix == ".json":
            return load_json_instruction_file(data_file), [data_file]
        if suffix == ".jsonl":
            return load_jsonl_instruction_file(data_file), [data_file]
        raise ValueError(f"不支持的指令数据文件类型: {data_file}")

    jsonl_files = sorted(data_dir.rglob("*.jsonl")) if data_dir.exists() else []
    examples: list[InstructionExample] = []
    for file_path in jsonl_files:
        examples.extend(load_jsonl_instruction_file(file_path))

    if examples:
        return examples, jsonl_files

    default_examples = [
        InstructionExample(
            instruction=item["instruction"],
            input=item["input"],
            output=item["output"],
        )
        for item in DEFAULT_INSTRUCTION_EXAMPLES
    ]
    return default_examples, []


def get_end_token_id(tokenizer: AutoTokenizer) -> int:
    """获取可用于兜底停止的 token 编号

    参数含义:
        tokenizer: 因果语言模型 tokenizer

    返回值含义:
        返回 eos、sep、pad、unk 中第一个可用的 token 编号
    """

    for token_id in (
        tokenizer.eos_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    ):
        if token_id is not None:
            return int(token_id)
    raise ValueError("tokenizer 缺少 eos、sep、pad、unk 等可用特殊 token")


def get_qwen_im_end_token_id(tokenizer: AutoTokenizer) -> int | None:
    """获取 Qwen 对话结束 token 编号

    参数含义:
        tokenizer: Qwen tokenizer

    返回值含义:
        返回 <|im_end|> 的 token 编号，若 tokenizer 不包含该 token 则返回 None
    """

    token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if token_id is None or token_id == tokenizer.unk_token_id:
        return None
    return int(token_id)


def get_generation_eos_token_ids(tokenizer: AutoTokenizer) -> int | list[int]:
    """获取生成时使用的停止 token 编号

    参数含义:
        tokenizer: 因果语言模型 tokenizer

    返回值含义:
        返回 eos token 编号，Qwen tokenizer 会同时包含 <|endoftext|> 和 <|im_end|>
    """

    token_ids: list[int] = []
    end_token_id = get_end_token_id(tokenizer)
    token_ids.append(end_token_id)
    im_end_token_id = get_qwen_im_end_token_id(tokenizer)
    if im_end_token_id is not None and im_end_token_id not in token_ids:
        token_ids.append(im_end_token_id)
    return token_ids[0] if len(token_ids) == 1 else token_ids


def ensure_tokenizer_special_tokens(tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> None:
    """补齐 tokenizer 和模型配置中的特殊 token

    参数含义:
        tokenizer: 因果语言模型 tokenizer
        model: 因果语言模型

    返回值含义:
        无返回值，直接更新 tokenizer 和 model.config 中的特殊 token 设置
    """

    if tokenizer.eos_token is None and tokenizer.sep_token is not None:
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.eos_token is None and tokenizer.unk_token is not None:
        tokenizer.eos_token = tokenizer.unk_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

    end_token_id = get_end_token_id(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = end_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = end_token_id
    model.config.pad_token_id = tokenizer.pad_token_id


def parse_lora_target_modules(target_modules: str) -> list[str]:
    """解析 LoRA 目标模块参数

    参数含义:
        target_modules: 逗号分隔的 LoRA 目标模块名称

    返回值含义:
        返回去除空白后的模块名称列表
    """

    modules = [item.strip() for item in target_modules.split(",") if item.strip()]
    if not modules:
        raise ValueError("lora_target_modules 不能为空")
    return modules


def resolve_torch_dtype(dtype_name: str) -> torch.dtype | str:
    """解析模型加载 dtype 参数

    参数含义:
        dtype_name: dtype 字符串，支持 auto、float32、float16、bfloat16

    返回值含义:
        返回 transformers.from_pretrained 可接收的 torch_dtype 参数
    """

    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"不支持的 torch_dtype: {dtype_name}")


def ensure_peft_available() -> None:
    """确认 PEFT 依赖可用

    参数含义:
        无参数

    返回值含义:
        无返回值，PEFT 不可用时抛出 RuntimeError
    """

    if LoraConfig is None or TaskType is None or get_peft_model is None:
        raise RuntimeError("缺少 peft 依赖，请在虚拟环境中安装 peft 后再运行 LoRA 微调")


def apply_lora_to_model(model: AutoModelForCausalLM, config: FinetuneConfig) -> AutoModelForCausalLM:
    """给基础因果语言模型挂载 LoRA 适配器

    参数含义:
        model: 已加载的基础因果语言模型
        config: 指令微调流程配置对象，包含 LoRA 参数

    返回值含义:
        返回挂载 LoRA 后的 PEFT 模型
    """

    ensure_peft_available()
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=parse_lora_target_modules(config.lora_target_modules),
    )
    return get_peft_model(model, lora_config)


def load_causal_lm_and_tokenizer(config: FinetuneConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """加载因果语言模型和 tokenizer，并挂载 LoRA 适配器

    参数含义:
        config: 指令微调配置对象，包含模型名称、本地目录和加载策略

    返回值含义:
        返回 LoRA 模型、tokenizer 和实际加载来源说明
    """

    model_dir = Path(config.model_dir)
    has_local_model = (
        model_dir.exists()
        and (model_dir / "config.json").exists()
        and any(model_dir.glob("*.json"))
    )
    use_local_dir = has_local_model and not config.force_download
    load_source = str(model_dir) if use_local_dir else config.model_name

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            load_source,
            local_files_only=config.local_files_only if not use_local_dir else True,
            trust_remote_code=config.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            load_source,
            local_files_only=config.local_files_only if not use_local_dir else True,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=resolve_torch_dtype(config.torch_dtype),
        )
    except OSError as exc:
        message = (
            f"无法加载因果语言模型来源 {load_source}\n"
            "如果是第一次运行，请确保网络可用，或提前把模型保存到本地模型目录\n"
            "如果需要离线运行，请确认本地目录已包含 config.json、模型权重和 tokenizer 文件"
        )
        raise RuntimeError(message) from exc

    ensure_tokenizer_special_tokens(tokenizer, model)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = apply_lora_to_model(model, config)

    if not use_local_dir and not config.local_files_only:
        model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(model_dir)

    source_label = str(model_dir) if use_local_dir else config.model_name
    return model, tokenizer, source_label


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


def sample_examples(
    examples: list[InstructionExample],
    max_samples: int,
    seed: int,
) -> list[InstructionExample]:
    """按固定随机种子抽样指令样本

    参数含义:
        examples: 全部指令样本列表
        max_samples: 最多使用的样本数量，0 或负数表示使用全量数据
        seed: 控制抽样顺序的随机种子

    返回值含义:
        返回抽样后的指令样本列表
    """

    if max_samples <= 0 or max_samples >= len(examples):
        return list(examples)
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:max_samples]


def build_supervised_sample(
    example: InstructionExample,
    tokenizer: AutoTokenizer,
    max_length: int,
    template_style: str,
    system_prompt: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SampleEncodingStats]:
    """把一条指令样本编码为带监督掩码的因果语言模型训练样本

    参数含义:
        example: 一条指令微调样本
        tokenizer: 因果语言模型 tokenizer
        max_length: 编码后的最大输入 token 数量
        template_style: 指令模板风格，控制样本编码格式
        system_prompt: Qwen 对话模板中的 system 消息

    返回值含义:
        返回 input_ids、attention_mask、labels 和样本编码统计，前三者长度均为 max_length
    """

    prefix = format_prompt(
        example.instruction,
        example.input,
        template_style=template_style,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    full_text = format_prompt(
        example.instruction,
        example.input,
        output=example.output,
        template_style=template_style,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    output_ids = full_ids[len(prefix_ids) :]
    if not output_ids:
        output_ids = [get_end_token_id(tokenizer)]

    original_prefix_length = len(prefix_ids)
    original_output_length = len(output_ids)
    max_total_length = max_length + 1
    if len(prefix_ids) >= max_total_length:
        prefix_ids = prefix_ids[-(max_total_length - 1) :]

    available_output_length = max_total_length - len(prefix_ids)
    output_ids = output_ids[:available_output_length]
    ids = prefix_ids + output_ids
    answer_start = len(prefix_ids)
    effective_length = max(len(ids) - 1, 0)

    input_ids = ids[:-1]
    labels = ids[1:]
    ignore_count = max(answer_start - 1, 0)
    labels[:ignore_count] = [-100] * min(ignore_count, len(labels))

    attention_mask = [1] * len(input_ids)
    pad_length = max_length - len(input_ids)
    if pad_length > 0:
        input_ids = input_ids + [int(tokenizer.pad_token_id)] * pad_length
        attention_mask = attention_mask + [0] * pad_length
        labels = labels + [-100] * pad_length

    sample_stats = SampleEncodingStats(
        original_prefix_length=original_prefix_length,
        original_output_length=original_output_length,
        final_prefix_length=len(prefix_ids),
        final_output_length=len(output_ids),
        effective_length=effective_length,
        prompt_truncated=len(prefix_ids) < original_prefix_length,
        output_truncated=len(output_ids) < original_output_length,
    )
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        sample_stats,
    )


def summarize_encoding_stats(stats: list[SampleEncodingStats]) -> DatasetEncodingStats:
    """汇总样本编码统计

    参数含义:
        stats: 每条样本的编码统计列表

    返回值含义:
        返回数据集级别的样本数量、截断数量和有效长度统计
    """

    if not stats:
        return DatasetEncodingStats(
            sample_count=0,
            prompt_truncated_count=0,
            output_truncated_count=0,
            average_effective_length=0.0,
            max_effective_length=0,
        )

    sample_count = len(stats)
    effective_lengths = [item.effective_length for item in stats]
    return DatasetEncodingStats(
        sample_count=sample_count,
        prompt_truncated_count=sum(1 for item in stats if item.prompt_truncated),
        output_truncated_count=sum(1 for item in stats if item.output_truncated),
        average_effective_length=sum(effective_lengths) / sample_count,
        max_effective_length=max(effective_lengths),
    )


def merge_encoding_stats(stats_items: list[DatasetEncodingStats]) -> DatasetEncodingStats:
    """合并多个数据集编码统计

    参数含义:
        stats_items: 多个数据集的聚合编码统计

    返回值含义:
        返回合并后的整体编码统计
    """

    total_samples = sum(item.sample_count for item in stats_items)
    if total_samples <= 0:
        return DatasetEncodingStats(
            sample_count=0,
            prompt_truncated_count=0,
            output_truncated_count=0,
            average_effective_length=0.0,
            max_effective_length=0,
        )

    weighted_effective_length = sum(
        item.average_effective_length * item.sample_count
        for item in stats_items
    )
    return DatasetEncodingStats(
        sample_count=total_samples,
        prompt_truncated_count=sum(item.prompt_truncated_count for item in stats_items),
        output_truncated_count=sum(item.output_truncated_count for item in stats_items),
        average_effective_length=weighted_effective_length / total_samples,
        max_effective_length=max(item.max_effective_length for item in stats_items),
    )


def build_dataloaders(
    examples: list[InstructionExample],
    tokenizer: AutoTokenizer,
    config: FinetuneConfig,
) -> tuple[
    DataLoader,
    DataLoader,
    list[InstructionExample],
    list[InstructionExample],
    DatasetEncodingStats,
]:
    """构建训练集和验证集 DataLoader

    参数含义:
        examples: 全部指令样本列表
        tokenizer: 因果语言模型 tokenizer
        config: 指令微调配置对象

    返回值含义:
        返回训练 DataLoader、验证 DataLoader、训练样本列表、验证样本列表和整体编码统计
    """

    train_examples, valid_examples = split_examples(examples, config.train_split, config.seed)
    train_dataset = InstructionDataset(
        train_examples,
        tokenizer,
        config.max_length,
        config.template_style,
        config.system_prompt,
    )
    valid_dataset = InstructionDataset(
        valid_examples,
        tokenizer,
        config.max_length,
        config.template_style,
        config.system_prompt,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    encoding_stats = merge_encoding_stats([train_dataset.stats, valid_dataset.stats])
    return train_loader, valid_loader, train_examples, valid_examples, encoding_stats


@torch.no_grad()
def estimate_loss(
    model: AutoModelForCausalLM,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    """评估模型在指定数据集上的平均交叉熵损失

    参数含义:
        model: 待评估的因果语言模型
        data_loader: 训练集或验证集 DataLoader
        device: 执行评估的设备
        max_batches: 最多评估的 batch 数量

    返回值含义:
        返回平均交叉熵损失，若数据为空则返回无穷大
    """

    model.eval()
    losses: list[float] = []
    for batch_index, (input_ids, attention_mask, labels) in enumerate(data_loader):
        if batch_index >= max_batches:
            break
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        losses.append(float(outputs.loss.item()))
    model.train()
    if not losses:
        return float("inf")
    return sum(losses) / len(losses)


def validate_config(config: FinetuneConfig) -> None:
    """检查指令微调配置是否可执行

    参数含义:
        config: 指令微调流程配置对象

    返回值含义:
        无返回值，配置非法时抛出 ValueError
    """

    if config.max_length <= 0:
        raise ValueError("max_length 必须大于 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps 必须大于 0")
    if config.max_steps <= 0:
        raise ValueError("max_steps 必须大于 0")
    if config.eval_interval <= 0:
        raise ValueError("eval_interval 必须大于 0")
    if config.eval_batches <= 0:
        raise ValueError("eval_batches 必须大于 0")
    if config.train_split <= 0 or config.train_split >= 1:
        raise ValueError("train_split 必须在 0 和 1 之间")
    if config.lr_scheduler_type not in {"none", "linear", "cosine"}:
        raise ValueError("lr_scheduler_type 必须是 none、linear 或 cosine")
    if config.min_lr_ratio < 0 or config.min_lr_ratio > 1:
        raise ValueError("min_lr_ratio 必须在 0 和 1 之间")
    if config.generate_tokens <= 0:
        raise ValueError("generate_tokens 必须大于 0")
    if config.repetition_penalty < 1.0:
        raise ValueError("repetition_penalty 必须大于等于 1.0")
    if config.no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size 必须大于等于 0")
    if config.lora_r <= 0:
        raise ValueError("lora_r 必须大于 0")
    if config.lora_alpha <= 0:
        raise ValueError("lora_alpha 必须大于 0")
    if config.lora_dropout < 0 or config.lora_dropout >= 1:
        raise ValueError("lora_dropout 必须在 0 和 1 之间")
    parse_lora_target_modules(config.lora_target_modules)


def compute_learning_rate_scale(
    optimizer_step: int,
    total_optimizer_steps: int,
    warmup_steps: int,
    scheduler_type: str,
    min_lr_ratio: float,
) -> float:
    """计算当前 optimizer step 的学习率缩放系数

    参数含义:
        optimizer_step: 当前已经执行的参数更新步数，从 1 开始
        total_optimizer_steps: 计划执行的参数更新总步数
        warmup_steps: 学习率预热步数
        scheduler_type: 学习率调度策略，支持 none、linear、cosine
        min_lr_ratio: 学习率缩放系数的最低值

    返回值含义:
        返回学习率相对初始 learning_rate 的缩放系数
    """

    if scheduler_type == "none":
        return 1.0
    if warmup_steps > 0 and optimizer_step <= warmup_steps:
        return max(optimizer_step / warmup_steps, 1e-8)

    decay_steps = max(total_optimizer_steps - warmup_steps, 1)
    progress = min(max((optimizer_step - warmup_steps) / decay_steps, 0.0), 1.0)
    if scheduler_type == "linear":
        return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)
    if scheduler_type == "cosine":
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_scale
    return 1.0


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_learning_rate: float,
    scale: float,
) -> float:
    """更新 optimizer 中所有参数组的学习率

    参数含义:
        optimizer: 需要更新学习率的 PyTorch optimizer
        base_learning_rate: 初始学习率
        scale: 当前学习率缩放系数

    返回值含义:
        返回实际设置后的学习率
    """

    current_learning_rate = base_learning_rate * scale
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_learning_rate
    return current_learning_rate


def train_model(
    model: AutoModelForCausalLM,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: FinetuneConfig,
    device: torch.device,
) -> list[dict[str, object]]:
    """执行因果语言模型指令微调训练循环

    参数含义:
        model: 待微调的因果语言模型
        train_loader: 训练集 DataLoader
        valid_loader: 验证集 DataLoader
        config: 指令微调流程配置对象
        device: 执行训练的设备

    返回值含义:
        返回训练日志列表，每个元素记录损失、最佳验证损失和提前终止状态
    """

    accumulation_steps = config.gradient_accumulation_steps
    total_optimizer_steps = config.max_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    loss_log: list[dict[str, object]] = []
    train_iter = iter(train_loader)
    early_stopping_enabled = config.early_stopping_patience > 0
    best_valid_loss = float("inf")
    best_model_state: dict[str, torch.Tensor] | None = None
    best_step = 0
    best_optimizer_step = 0
    bad_eval_count = 0
    optimizer_step = 0
    micro_step = 0
    current_learning_rate = config.learning_rate

    while optimizer_step < config.max_steps:
        try:
            input_ids, attention_mask, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, attention_mask, labels = next(train_iter)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        raw_loss = outputs.loss
        loss = raw_loss / accumulation_steps

        loss.backward()
        micro_step += 1
        should_update = micro_step % accumulation_steps == 0
        if should_update:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer_step += 1
            lr_scale = compute_learning_rate_scale(
                optimizer_step,
                total_optimizer_steps,
                config.warmup_steps,
                config.lr_scheduler_type,
                config.min_lr_ratio,
            )
            current_learning_rate = set_optimizer_learning_rate(
                optimizer,
                config.learning_rate,
                lr_scale,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        should_eval = should_update and (
            optimizer_step == 1
            or optimizer_step % config.eval_interval == 0
            or optimizer_step == config.max_steps
        )
        if should_eval:
            train_loss = estimate_loss(model, train_loader, device, config.eval_batches)
            valid_loss = estimate_loss(model, valid_loader, device, config.eval_batches)
            is_best = best_model_state is None or valid_loss < best_valid_loss - config.early_stopping_min_delta
            if is_best:
                best_valid_loss = valid_loss
                best_step = micro_step
                best_optimizer_step = optimizer_step
                bad_eval_count = 0
                best_model_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            elif early_stopping_enabled:
                bad_eval_count += 1

            should_stop = early_stopping_enabled and not is_best and bad_eval_count >= config.early_stopping_patience
            loss_log.append(
                {
                    "micro_step": float(micro_step),
                    "optimizer_step": float(optimizer_step),
                    "learning_rate": current_learning_rate,
                    "last_micro_batch_loss": float(raw_loss.item()),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "best_valid_loss": best_valid_loss,
                    "bad_eval_count": bad_eval_count,
                    "is_best": is_best,
                    "early_stopped": should_stop,
                }
            )
            print(
                f"micro_step {micro_step:5d} | optimizer_step {optimizer_step:4d} | lr {current_learning_rate:.2e} | "
                f"train_loss {train_loss:.4f} | valid_loss {valid_loss:.4f} | "
                f"best_valid_loss {best_valid_loss:.4f} | bad_eval_count {bad_eval_count}"
            )
            if should_stop:
                print(
                    f"提前终止触发: 验证损失连续 {bad_eval_count} 次评估未改进，"
                    f"将恢复第 {best_step} 个 micro batch、第 {best_optimizer_step} 次参数更新的最佳模型"
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(
            f"已恢复第 {best_step} 个 micro batch、第 {best_optimizer_step} 次参数更新的最佳验证模型，"
            f"best_valid_loss {best_valid_loss:.4f}"
        )
    return loss_log


@torch.no_grad()
def generate_instruction_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instruction: str,
    input_text: str,
    template_style: str,
    system_prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    device: torch.device,
) -> str:
    """使用因果语言模型根据指令生成回答文本

    参数含义:
        model: 用于生成的因果语言模型
        tokenizer: 因果语言模型 tokenizer
        instruction: 任务指令文本
        input_text: 任务输入文本
        template_style: 指令模板风格，控制提示词格式
        system_prompt: Qwen 对话模板中的 system 消息
        max_new_tokens: 需要继续生成的新 token 数量
        do_sample: 是否使用随机采样生成
        temperature: 采样温度
        top_k: 每步采样保留的候选 token 数量
        repetition_penalty: 对已出现 token 的重复惩罚系数
        no_repeat_ngram_size: 禁止重复的 ngram 长度，0 表示关闭
        device: 执行生成的设备

    返回值含义:
        返回解码后的完整提示词和生成内容
    """

    model.eval()
    prompt = format_prompt(
        instruction,
        input_text,
        template_style=template_style,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max(8, min(tokenizer.model_max_length, 1024)),
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": get_generation_eos_token_ids(tokenizer),
        "repetition_penalty": repetition_penalty,
    }
    if no_repeat_ngram_size > 0:
        generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_k"] = top_k
    generated = model.generate(**generation_kwargs)
    new_token_ids = generated[0, input_ids.shape[-1] :]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


def count_parameters(model: AutoModelForCausalLM) -> tuple[int, int]:
    """统计模型参数量

    参数含义:
        model: 需要统计参数量的因果语言模型

    返回值含义:
        返回二元组，第一个值是总参数量，第二个值是可训练参数量
    """

    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


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


def save_loss_log(loss_log: list[dict[str, object]], file_path: Path) -> None:
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
                "micro_step",
                "optimizer_step",
                "learning_rate",
                "last_micro_batch_loss",
                "train_loss",
                "valid_loss",
                "best_valid_loss",
                "bad_eval_count",
                "is_best",
                "early_stopped",
            ],
            lineterminator="\r\n",
        )
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


def build_data_stats(
    config: FinetuneConfig,
    raw_sample_count: int,
    used_sample_count: int,
    train_sample_count: int,
    valid_sample_count: int,
    encoding_stats: DatasetEncodingStats,
    loss_log: list[dict[str, object]],
) -> dict[str, object]:
    """构建数据和训练统计保存内容

    参数含义:
        config: 指令微调流程配置对象
        raw_sample_count: 抽样前原始样本数量
        used_sample_count: 实际参与训练和验证的样本数量
        train_sample_count: 训练集样本数量
        valid_sample_count: 验证集样本数量
        encoding_stats: 样本编码和截断统计
        loss_log: 训练损失日志

    返回值含义:
        返回可序列化为 JSON 的统计字典
    """

    best_valid_loss = None
    final_best_valid_loss = None
    if loss_log:
        best_valid_loss = min(float(item["valid_loss"]) for item in loss_log)
        final_best_valid_loss = float(loss_log[-1]["best_valid_loss"])

    return {
        "raw_sample_count": raw_sample_count,
        "used_sample_count": used_sample_count,
        "train_sample_count": train_sample_count,
        "valid_sample_count": valid_sample_count,
        "train_split": config.train_split,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "max_length": config.max_length,
        "max_optimizer_steps": config.max_steps,
        "estimated_max_micro_steps": config.max_steps * config.gradient_accumulation_steps,
        "min_lr_ratio": config.min_lr_ratio,
        "generate_tokens": config.generate_tokens,
        "repetition_penalty": config.repetition_penalty,
        "no_repeat_ngram_size": config.no_repeat_ngram_size,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_target_modules": parse_lora_target_modules(config.lora_target_modules),
        "torch_dtype": config.torch_dtype,
        "eval_batches": config.eval_batches,
        "encoding_stats": asdict(encoding_stats),
        "best_valid_loss": best_valid_loss,
        "final_best_valid_loss": final_best_valid_loss,
    }


def save_outputs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: FinetuneConfig,
    loss_log: list[dict[str, object]],
    before_text: str,
    after_text: str,
    data_stats: dict[str, object],
    output_dir: Path,
) -> None:
    """保存微调后的 LoRA 适配器、tokenizer、配置、日志和样例输出

    参数含义:
        model: 微调完成的 LoRA 模型
        tokenizer: 因果语言模型 tokenizer
        config: 指令微调流程配置对象
        loss_log: 训练损失日志
        before_text: 微调前生成结果
        after_text: 微调后生成结果
        data_stats: 数据集、截断和训练结果统计
        output_dir: 输出目录路径

    返回值含义:
        无返回值，直接在输出目录写入文件
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    finetuned_dir = output_dir / "qwen2_5_0_5b_lora_finetuned"
    finetuned_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(finetuned_dir)
    tokenizer.save_pretrained(finetuned_dir)
    save_json({"finetune_config": asdict(config)}, output_dir / "finetune_config.json")
    save_loss_log(loss_log, output_dir / "loss_log.csv")
    save_sample_outputs(before_text, after_text, output_dir / "sample_outputs.txt")
    save_json(data_stats, output_dir / "data_stats.json")


def main() -> None:
    """运行 Qwen2.5 LoRA 指令微调完整流程

    参数含义:
        无参数

    返回值含义:
        无返回值，直接执行数据读取、模型加载、模型微调、效果展示和产物保存
    """

    config = parse_args()
    validate_config(config)
    set_seed(config.seed)
    device = select_device(config.device)

    data_file = resolve_data_file(config.data_file)
    examples, data_files = load_instruction_examples(data_file, Path(config.data_dir))
    raw_sample_count = len(examples)
    examples = sample_examples(examples, config.max_samples, config.seed)
    model, tokenizer, model_source = load_causal_lm_and_tokenizer(config)
    model = model.to(device)
    train_loader, valid_loader, train_examples, valid_examples, encoding_stats = build_dataloaders(
        examples,
        tokenizer,
        config,
    )
    total_params, trainable_params = count_parameters(model)
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps

    print("第 4 步 Qwen2.5 LoRA 指令微调配置:")
    print(config)
    print(f"运行设备: {device}")
    print(f"模型来源: {model_source}")
    print(f"本地模型目录: {Path(config.model_dir).resolve()}")
    print(f"模板风格: {config.template_style}")
    print(f"LoRA 目标模块: {parse_lora_target_modules(config.lora_target_modules)}")
    print(f"LoRA 配置: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"最大序列长度: {config.max_length}")
    print(f"梯度累积步数: {config.gradient_accumulation_steps}")
    print(f"有效 batch size: {effective_batch_size}")
    print(f"最大参数更新步数: {config.max_steps}")
    print(f"预计最多 micro batch 步数: {config.max_steps * config.gradient_accumulation_steps}")
    print(f"评估间隔参数更新步数: {config.eval_interval}")
    print(f"早停耐心值: {config.early_stopping_patience}，0 或负数表示关闭")
    print(f"学习率调度: {config.lr_scheduler_type}, warmup_steps={config.warmup_steps}, min_lr_ratio={config.min_lr_ratio}")
    print(f"生成是否采样: {config.do_sample}")
    print(
        f"生成长度和反重复: generate_tokens={config.generate_tokens}, "
        f"repetition_penalty={config.repetition_penalty}, "
        f"no_repeat_ngram_size={config.no_repeat_ngram_size}"
    )
    print(f"读取指令文件数: {len(data_files)}")
    if data_files:
        print(f"指令数据文件: {data_files[0]}")
    print(f"原始指令样本数: {raw_sample_count:,}")
    print(f"实际使用样本数: {len(examples):,}")
    print(f"训练样本数: {len(train_examples):,}")
    print(f"验证样本数: {len(valid_examples):,}")
    print(f"模型总参数量: {total_params:,}")
    print(f"模型可训练参数量: {trainable_params:,}")
    print(
        "编码截断统计: "
        f"prompt_truncated={encoding_stats.prompt_truncated_count:,}, "
        f"output_truncated={encoding_stats.output_truncated_count:,}, "
        f"avg_effective_length={encoding_stats.average_effective_length:.2f}, "
        f"max_effective_length={encoding_stats.max_effective_length}"
    )

    print("\n微调前生成效果:")
    before_text = generate_instruction_answer(
        model,
        tokenizer,
        config.sample_instruction,
        config.sample_input,
        config.template_style,
        config.system_prompt,
        config.generate_tokens,
        config.do_sample,
        config.temperature,
        config.top_k,
        config.repetition_penalty,
        config.no_repeat_ngram_size,
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
        config.template_style,
        config.system_prompt,
        config.generate_tokens,
        config.do_sample,
        config.temperature,
        config.top_k,
        config.repetition_penalty,
        config.no_repeat_ngram_size,
        device,
    )
    print(after_text)

    output_dir = Path(config.output_dir)
    data_stats = build_data_stats(
        config,
        raw_sample_count,
        len(examples),
        len(train_examples),
        len(valid_examples),
        encoding_stats,
        loss_log,
    )
    save_outputs(model, tokenizer, config, loss_log, before_text, after_text, data_stats, output_dir)
    print(f"\n指令微调产物已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
