"""Qwen2.5 文本分类 LoRA 微调程序

本程序用于在本地 Qwen2.5-0.5B 模型基础上进行外卖评论二分类微调
默认读取 data_instruction/waimai_10k.csv，并保存 LoRA 适配器、标签映射、指标日志和样例预测
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import AutoTokenizer


DEFAULT_MODEL_DIR = "models/qwen2.5-0.5b"
DEFAULT_DATA_FILE = "data_instruction/waimai_10k.csv"
DEFAULT_OUTPUT_DIR = "outputs_qwen_classification_finetune"
DEFAULT_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
DEFAULT_SAMPLE_TEXT = "下次还来我是狗"
DEFAULT_HARD_EXAMPLE_REPEAT = 20
TEXT_FIELD_CANDIDATES = ("text", "review", "content", "input", "sentence", "句子", "文本", "评论")
LABEL_FIELD_CANDIDATES = ("label", "category", "class", "target", "标签", "类别")
DEFAULT_LABEL_NAMES = {"0": "差评", "1": "好评"}
NOISY_POSITIVE_TO_NEGATIVE_PATTERNS = (
    "我是狗",
    "谁再点谁是狗",
    "再点我是狗",
    "再买我是狗",
    "再吃我是狗",
    "再来我是狗",
    "就是狗",
    "再也不会",
    "再也不买",
    "再也不点",
    "再也不订",
    "再也不定",
    "再也不吃",
    "再也不叫",
    "再也不来",
    "不会再",
    "不再光顾",
    "不再来了",
    "绝不再",
    "永远不会",
    "永不再",
    "下次不会",
    "下次不再",
    "不会再爱",
    "拉黑",
    "差评",
    "垃圾",
    "难吃",
    "不好吃",
    "没法吃",
    "不能吃",
    "吃不下",
    "恶心",
    "失望",
    "投诉",
    "退款",
    "退单",
    "拉肚子",
    "馊",
    "坏的",
)
CURATED_HARD_TRAINING_EXAMPLES = (
    ("下次还来我是狗", "差评"),
    ("以后再点我是狗", "差评"),
    ("谁再点这家谁是狗", "差评"),
    ("我再买这家就是狗", "差评"),
    ("这家我再吃一次就是狗", "差评"),
    ("味道很好，下次还来我是狗", "差评"),
    ("配送挺快，下次还点我是狗", "差评"),
    ("服务不错，但下次再来我是狗", "差评"),
    ("不是说不好，反正以后再也不会点了", "差评"),
    ("好吃是好吃，但再也不买了", "差评"),
    ("味道很好，下次还来", "好评"),
    ("配送挺快，下次还点", "好评"),
    ("服务不错，还会再来", "好评"),
    ("很好吃，以后还会买", "好评"),
    ("下次一定继续点", "好评"),
)


@dataclass
class ClassificationExample:
    """保存一条文本分类样本

    参数含义:
        text: 需要分类的输入文本
        label: 文本对应的类别标签

    返回值含义:
        ClassificationExample 实例用于统一表示一条分类微调数据
    """

    text: str
    label: str


@dataclass
class ClassificationConfig:
    """保存 Qwen 分类微调流程的核心配置

    参数含义:
        model_dir: 本地 Qwen2.5 base 模型目录
        data_file: 分类数据文件路径，支持 csv、json、jsonl
        output_dir: 保存 LoRA 适配器和训练产物的目录
        max_length: tokenizer 最大序列长度
        batch_size: 每次训练使用的样本数量
        max_steps: 最大训练步数
        eval_interval: 每隔多少步评估一次训练集和验证集指标
        eval_batches: 每次评估最多使用的 batch 数量
        learning_rate: AdamW 优化器学习率
        weight_decay: AdamW 优化器权重衰减系数
        grad_clip: 梯度裁剪阈值
        dtype: 模型加载和训练使用的浮点精度
        train_split: 训练样本占全部样本的比例
        max_train_samples: 最多使用的训练前原始样本数量，0 表示不限制
        seed: 随机种子
        fix_noisy_labels: 是否修正 waimai_10k 中明显与文本矛盾的标签
        hard_example_repeat: 困难样本重复加入训练集的次数，0 表示不加入
        use_lora: 是否使用 LoRA 微调，关闭后执行全参数分类头/模型微调
        lora_rank: LoRA 低秩矩阵秩
        lora_alpha: LoRA 缩放系数
        lora_dropout: LoRA dropout 比例
        lora_target_modules: 逗号分隔的 LoRA 目标模块名称
        sample_text: 训练前后用于预测效果对比的样例文本
        device: 运行设备，auto 表示自动选择 cuda 或 cpu

    返回值含义:
        ClassificationConfig 实例用于统一传递分类微调参数
    """

    model_dir: str = DEFAULT_MODEL_DIR
    data_file: str = DEFAULT_DATA_FILE
    output_dir: str = DEFAULT_OUTPUT_DIR
    max_length: int = 128
    batch_size: int = 4
    max_steps: int = 300
    eval_interval: int = 25
    eval_batches: int = 20
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dtype: str = "auto"
    train_split: float = 0.9
    max_train_samples: int = 0
    seed: int = 42
    fix_noisy_labels: bool = True
    hard_example_repeat: int = DEFAULT_HARD_EXAMPLE_REPEAT
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = DEFAULT_LORA_TARGET_MODULES
    sample_text: str = DEFAULT_SAMPLE_TEXT
    device: str = "auto"


class ClassificationDataset:
    """把文本分类样本转换为 Qwen 分类训练样本"""

    def __init__(
        self,
        examples: list[ClassificationExample],
        tokenizer: "AutoTokenizer",
        label_to_id: dict[str, int],
        max_length: int,
    ) -> None:
        """初始化文本分类数据集

        参数含义:
            examples: 文本分类样本列表
            tokenizer: Qwen tokenizer
            label_to_id: 类别标签到类别编号的映射
            max_length: tokenizer 最大序列长度

        返回值含义:
            无返回值，保存样本和编码配置供训练时读取
        """

        self.examples = examples
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
        if not self.examples:
            raise ValueError("没有可用的分类样本")

    def __len__(self) -> int:
        """返回数据集样本数量

        参数含义:
            无参数

        返回值含义:
            返回数据集中分类样本数量
        """

        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, "torch.Tensor"]:
        """根据索引返回一个分类训练样本

        参数含义:
            index: 样本索引

        返回值含义:
            返回包含 input_ids、attention_mask 和 labels 的字典
        """

        torch = import_torch()
        example = self.examples[index]
        encoded = self.tokenizer(
            example.text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"][0].to(torch.long),
            "attention_mask": encoded["attention_mask"][0].to(torch.long),
            "labels": torch.tensor(self.label_to_id[example.label], dtype=torch.long),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    """构建分类微调命令行参数解析器

    参数含义:
        无参数

    返回值含义:
        返回配置完成的 argparse.ArgumentParser 实例
    """

    parser = argparse.ArgumentParser(description="Qwen2.5 文本分类 LoRA 微调程序")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="本地 Qwen2.5 base 模型目录")
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE, help="分类数据文件路径，支持 csv、json、jsonl")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="保存分类微调产物的目录")
    parser.add_argument("--max-length", type=int, default=128, help="tokenizer 最大序列长度")
    parser.add_argument("--batch-size", type=int, default=4, help="每个 batch 的样本数量")
    parser.add_argument("--max-steps", type=int, default=300, help="最大训练步数")
    parser.add_argument("--eval-interval", type=int, default=25, help="评估间隔步数")
    parser.add_argument("--eval-batches", type=int, default=20, help="每次评估最多使用的 batch 数量")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="模型浮点精度，auto 在 CUDA 支持时使用 bfloat16，否则使用 float32",
    )
    parser.add_argument("--train-split", type=float, default=0.9, help="训练集样本比例")
    parser.add_argument("--max-train-samples", type=int, default=0, help="最多使用的原始样本数量，0 表示不限制")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.set_defaults(fix_noisy_labels=True)
    parser.add_argument("--fix-noisy-labels", dest="fix_noisy_labels", action="store_true", help="修正明显与文本矛盾的 waimai_10k 标签")
    parser.add_argument("--no-fix-noisy-labels", dest="fix_noisy_labels", action="store_false", help="不修正原始数据标签")
    parser.add_argument("--hard-example-repeat", type=int, default=DEFAULT_HARD_EXAMPLE_REPEAT, help="困难样本重复加入训练集的次数，0 表示不加入")
    parser.set_defaults(use_lora=True)
    parser.add_argument("--use-lora", dest="use_lora", action="store_true", help="开启 LoRA 微调")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false", help="关闭 LoRA，执行普通微调")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora-target-modules", default=DEFAULT_LORA_TARGET_MODULES, help="逗号分隔的 LoRA 目标模块名称")
    parser.add_argument("--sample-text", default=DEFAULT_SAMPLE_TEXT, help="训练前后用于预测对比的样例文本")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    return parser


def parse_args() -> ClassificationConfig:
    """解析命令行参数并生成分类微调配置

    参数含义:
        无参数

    返回值含义:
        返回从命令行参数生成的 ClassificationConfig 实例
    """

    args = build_arg_parser().parse_args()
    return ClassificationConfig(**vars(args))


def import_torch():
    """延迟导入 torch 依赖

    参数含义:
        无参数

    返回值含义:
        返回已导入的 torch 模块，缺少依赖时抛出 RuntimeError
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("缺少 torch 依赖，请在虚拟环境中安装 torch 后再运行分类微调") from exc
    return torch


def import_transformers():
    """延迟导入 transformers 依赖

    参数含义:
        无参数

    返回值含义:
        返回 AutoModelForSequenceClassification 和 AutoTokenizer，缺少依赖时抛出 RuntimeError
    """

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("缺少 transformers 依赖，请在虚拟环境中安装 transformers 后再运行分类微调") from exc
    return AutoModelForSequenceClassification, AutoTokenizer


def import_peft_objects():
    """延迟导入 peft 依赖

    参数含义:
        无参数

    返回值含义:
        返回 LoraConfig、TaskType 和 get_peft_model，缺少依赖时抛出 RuntimeError
    """

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("缺少 peft 依赖，请在虚拟环境中安装 peft 后再运行 LoRA 分类微调") from exc
    return LoraConfig, TaskType, get_peft_model


def validate_config(config: ClassificationConfig) -> None:
    """校验分类微调配置是否满足训练基本约束

    参数含义:
        config: 分类微调配置对象

    返回值含义:
        无返回值，配置非法时抛出 ValueError
    """

    if config.max_length <= 0:
        raise ValueError("max_length 必须大于 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if config.max_steps <= 0:
        raise ValueError("max_steps 必须大于 0")
    if config.eval_interval <= 0:
        raise ValueError("eval_interval 必须大于 0")
    if config.eval_batches <= 0:
        raise ValueError("eval_batches 必须大于 0")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate 必须大于 0")
    if config.weight_decay < 0:
        raise ValueError("weight_decay 不能小于 0")
    if config.grad_clip <= 0:
        raise ValueError("grad_clip 必须大于 0")
    if config.dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("dtype 必须是 auto、float32、float16 或 bfloat16")
    if not 0 < config.train_split < 1:
        raise ValueError("train_split 必须位于 0 和 1 之间")
    if config.max_train_samples < 0:
        raise ValueError("max_train_samples 不能小于 0")
    if config.hard_example_repeat < 0:
        raise ValueError("hard_example_repeat 不能小于 0")
    if config.lora_rank <= 0:
        raise ValueError("lora_rank 必须大于 0")
    if config.lora_alpha <= 0:
        raise ValueError("lora_alpha 必须大于 0")
    if not 0 <= config.lora_dropout < 1:
        raise ValueError("lora_dropout 必须位于 0 和 1 之间")
    if not normalize_text(config.sample_text):
        raise ValueError("sample_text 不能为空")


def set_seed(seed: int) -> None:
    """设置 Python 和 PyTorch 随机种子

    参数含义:
        seed: 随机种子整数

    返回值含义:
        无返回值，直接影响后续随机数生成过程
    """

    torch = import_torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str):
    """选择模型训练设备

    参数含义:
        device_name: 设备名称，支持 auto、cpu、cuda

    返回值含义:
        返回 PyTorch 设备对象
    """

    torch = import_torch()
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 cuda，但当前环境不可用")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cuda_supports_bfloat16() -> bool:
    """检查当前 CUDA 设备是否支持 bfloat16

    参数含义:
        无参数

    返回值含义:
        返回 True 表示当前 CUDA 设备支持 bfloat16，否则返回 False
    """

    torch = import_torch()
    if not torch.cuda.is_available():
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(checker()) if checker is not None else False


def resolve_torch_dtype(config: ClassificationConfig, device):
    """根据配置和设备选择模型浮点精度

    参数含义:
        config: 分类微调配置对象
        device: 执行训练的 PyTorch 设备

    返回值含义:
        返回 PyTorch dtype 对象，用于加载分类模型
    """

    torch = import_torch()
    if config.dtype == "float32":
        return torch.float32
    if config.dtype == "float16":
        return torch.float16
    if config.dtype == "bfloat16":
        if device.type == "cuda" and not cuda_supports_bfloat16():
            raise RuntimeError("指定了 bfloat16，但当前 CUDA 设备不支持，请改用 --dtype float32")
        return torch.bfloat16
    if config.dtype == "auto":
        if device.type == "cuda" and cuda_supports_bfloat16():
            return torch.bfloat16
        return torch.float32
    raise ValueError("dtype 必须是 auto、float32、float16 或 bfloat16")


def assert_finite_tensor(tensor, tensor_name: str, context: str) -> None:
    """检查张量是否全部为有限数值

    参数含义:
        tensor: 需要检查的 PyTorch 张量
        tensor_name: 张量名称，用于错误提示
        context: 当前训练或评估上下文，用于定位问题

    返回值含义:
        无返回值，发现 NaN 或 Inf 时抛出 RuntimeError
    """

    torch = import_torch()
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(
            f"{context} 的 {tensor_name} 出现 NaN 或 Inf，"
            "请优先使用 --dtype float32，并把 --learning-rate 降到 1e-5"
        )


def normalize_text(text: object) -> str:
    """对分类数据中的文本做轻量清洗

    参数含义:
        text: 原始文本对象

    返回值含义:
        返回统一换行并去除首尾空白后的文本
    """

    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_waimai_label(label: object) -> str:
    """把外卖评论标签转换为中文类别名

    参数含义:
        label: 原始标签对象，常见取值为 0 或 1

    返回值含义:
        返回规范化后的类别标签文本
    """

    label_text = normalize_text(label)
    return DEFAULT_LABEL_NAMES.get(label_text, label_text)


def has_noisy_positive_pattern(text: str) -> bool:
    """判断文本是否包含明显负向或反讽模式

    参数含义:
        text: 待检查的评论文本

    返回值含义:
        返回 True 表示文本包含应优先判为差评的关键词模式
    """

    compact_text = normalize_text(text).replace(" ", "")
    for pattern in NOISY_POSITIVE_TO_NEGATIVE_PATTERNS:
        if pattern == "难吃" and any(phrase in compact_text for phrase in ("不难吃", "不是难吃", "没有难吃")):
            continue
        if pattern in compact_text:
            return True
    return False


def fix_obvious_noisy_labels(examples: list[ClassificationExample]) -> tuple[list[ClassificationExample], int]:
    """修正 waimai_10k 中明显负向但标为好评的样本

    参数含义:
        examples: 原始分类样本列表

    返回值含义:
        返回修正后的样本列表和被修正的样本数量
    """

    fixed_examples: list[ClassificationExample] = []
    fixed_count = 0
    positive_label = DEFAULT_LABEL_NAMES["1"]
    negative_label = DEFAULT_LABEL_NAMES["0"]
    for example in examples:
        if example.label == positive_label and has_noisy_positive_pattern(example.text):
            fixed_examples.append(ClassificationExample(text=example.text, label=negative_label))
            fixed_count += 1
        else:
            fixed_examples.append(example)
    return fixed_examples, fixed_count


def build_hard_training_examples(repeat: int) -> list[ClassificationExample]:
    """构建反讽和否定表达的困难训练样本

    参数含义:
        repeat: 困难样本重复加入训练集的次数，0 表示不加入

    返回值含义:
        返回可追加到训练集的分类样本列表
    """

    hard_examples: list[ClassificationExample] = []
    for _ in range(repeat):
        hard_examples.extend(
            ClassificationExample(text=text, label=label)
            for text, label in CURATED_HARD_TRAINING_EXAMPLES
        )
    return hard_examples


def first_non_empty_field(record: dict[str, object], candidates: tuple[str, ...]) -> str:
    """从记录中读取第一个非空候选字段

    参数含义:
        record: 分类数据记录
        candidates: 候选字段名元组

    返回值含义:
        返回第一个非空字段值，找不到时返回空字符串
    """

    for key in candidates:
        if key in record:
            value = normalize_text(record[key])
            if value:
                return value
    return ""


def parse_classification_record(
    record: dict[str, object],
    source: Path,
    record_number: int,
) -> ClassificationExample:
    """把字典记录转换为文本分类样本

    参数含义:
        record: 从 CSV、JSON 或 JSONL 中读取的一条字典
        source: 当前分类数据文件路径
        record_number: 当前记录序号，读取 CSV 时等同于行号

    返回值含义:
        返回 ClassificationExample 实例，字段缺失时抛出 ValueError
    """

    text = first_non_empty_field(record, TEXT_FIELD_CANDIDATES)
    label = first_non_empty_field(record, LABEL_FIELD_CANDIDATES)
    if not text:
        raise ValueError(f"{source} 第 {record_number} 条缺少非空文本字段，支持字段: {', '.join(TEXT_FIELD_CANDIDATES)}")
    if not label:
        raise ValueError(f"{source} 第 {record_number} 条缺少非空标签字段，支持字段: {', '.join(LABEL_FIELD_CANDIDATES)}")
    return ClassificationExample(text=text, label=normalize_waimai_label(label))


def load_csv_classification_examples(file_path: Path) -> list[ClassificationExample]:
    """读取 CSV 分类微调样本

    参数含义:
        file_path: CSV 分类数据文件路径，表头应包含文本字段和标签字段

    返回值含义:
        返回从文件中解析出的 ClassificationExample 列表
    """

    examples: list[ClassificationExample] = []
    with open(file_path, "r", encoding="utf-8-sig", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{file_path} 缺少 CSV 表头")
        for row_number, row in enumerate(reader, start=2):
            examples.append(parse_classification_record(dict(row), file_path, row_number))
    return examples


def load_json_classification_examples(file_path: Path) -> list[ClassificationExample]:
    """读取 JSON 分类微调样本

    参数含义:
        file_path: JSON 分类数据文件路径，文件内容应为样本数组或包含样本数组的字典

    返回值含义:
        返回从文件中解析出的 ClassificationExample 列表
    """

    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        data = json.load(file)
    if isinstance(data, dict):
        records = None
        if any(key in data for key in TEXT_FIELD_CANDIDATES) and any(key in data for key in LABEL_FIELD_CANDIDATES):
            records = [data]
        else:
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

    examples: list[ClassificationExample] = []
    for record_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{file_path} 第 {record_number} 条必须是 JSON 对象")
        examples.append(parse_classification_record(record, file_path, record_number))
    return examples


def load_jsonl_classification_examples(file_path: Path) -> list[ClassificationExample]:
    """读取 JSONL 分类微调样本

    参数含义:
        file_path: JSONL 分类数据文件路径，每行应包含文本字段和标签字段

    返回值含义:
        返回从文件中解析出的 ClassificationExample 列表
    """

    examples: list[ClassificationExample] = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{file_path} 第 {line_number} 行必须是 JSON 对象")
            examples.append(parse_classification_record(record, file_path, line_number))
    return examples


def load_classification_examples(data_file: Path, max_samples: int, seed: int) -> tuple[list[ClassificationExample], Path]:
    """读取分类微调样本并可选抽样

    参数含义:
        data_file: 分类数据文件路径
        max_samples: 最多保留的原始样本数量，0 表示不限制
        seed: 抽样随机种子

    返回值含义:
        返回分类样本列表和实际读取的数据文件路径
    """

    if not data_file.exists():
        raise FileNotFoundError(f"找不到分类数据文件: {data_file}")
    suffix = data_file.suffix.lower()
    if suffix == ".csv":
        examples = load_csv_classification_examples(data_file)
    elif suffix == ".json":
        examples = load_json_classification_examples(data_file)
    elif suffix == ".jsonl":
        examples = load_jsonl_classification_examples(data_file)
    else:
        raise ValueError(f"不支持的分类数据文件格式: {data_file.suffix}")

    if not examples:
        raise ValueError("分类数据文件中没有可用样本")
    if max_samples > 0 and len(examples) > max_samples:
        examples = random.Random(seed).sample(examples, max_samples)
    return examples, data_file


def prepare_training_examples(
    examples: list[ClassificationExample],
    config: ClassificationConfig,
) -> tuple[list[ClassificationExample], int]:
    """根据配置修正训练前样本标签

    参数含义:
        examples: 原始分类样本列表
        config: 分类微调配置对象

    返回值含义:
        返回二元组，第一个值是处理后的样本列表，第二个值是修正的标签数量
    """

    if not config.fix_noisy_labels:
        return examples, 0
    return fix_obvious_noisy_labels(examples)


def build_label_mapping(examples: list[ClassificationExample]) -> dict[str, int]:
    """根据分类样本构建标签到编号的映射

    参数含义:
        examples: 分类样本列表

    返回值含义:
        返回标签文本到整数编号的映射
    """

    labels = sorted({example.label for example in examples})
    ordered_default_labels = [DEFAULT_LABEL_NAMES[key] for key in sorted(DEFAULT_LABEL_NAMES)]
    if set(labels).issubset(set(ordered_default_labels)):
        labels = [label for label in ordered_default_labels if label in labels]
    if len(labels) < 2:
        raise ValueError("分类微调至少需要 2 个不同类别")
    return {label: index for index, label in enumerate(labels)}


def split_examples(
    examples: list[ClassificationExample],
    train_split: float,
    seed: int,
) -> tuple[list[ClassificationExample], list[ClassificationExample]]:
    """把分类样本划分为训练集和验证集

    参数含义:
        examples: 全部分类样本
        train_split: 训练集样本比例
        seed: 随机种子

    返回值含义:
        返回训练样本列表和验证样本列表
    """

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    split_index = int(len(shuffled) * train_split)
    split_index = max(1, min(split_index, len(shuffled) - 1))
    return shuffled[:split_index], shuffled[split_index:]


def parse_lora_target_modules(target_modules: str) -> list[str]:
    """解析 LoRA 目标模块名称

    参数含义:
        target_modules: 逗号分隔的 LoRA 目标模块名称

    返回值含义:
        返回清洗后的模块名称列表
    """

    modules = [item.strip() for item in target_modules.split(",") if item.strip()]
    if not modules:
        raise ValueError("lora_target_modules 不能为空")
    return modules


def load_tokenizer(model_dir: Path):
    """加载本地 Qwen tokenizer

    参数含义:
        model_dir: 本地 Qwen2.5 base 模型目录

    返回值含义:
        返回加载完成的 tokenizer
    """

    if not model_dir.exists():
        raise FileNotFoundError(f"找不到 base 模型目录: {model_dir}")
    _, AutoTokenizer = import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_sequence_classification_model(
    model_dir: Path,
    num_labels: int,
    label_to_id: dict[str, int],
    config: ClassificationConfig,
    device,
):
    """加载本地 Qwen 序列分类模型

    参数含义:
        model_dir: 本地 Qwen2.5 base 模型目录
        num_labels: 分类任务类别数量
        label_to_id: 类别标签到类别编号的映射
        config: 分类微调配置对象
        device: 模型加载后移动到的设备

    返回值含义:
        返回已移动到目标设备的序列分类模型
    """

    if not model_dir.exists():
        raise FileNotFoundError(f"找不到 base 模型目录: {model_dir}")
    AutoModelForSequenceClassification, _ = import_transformers()
    id_to_label = {index: label for label, index in label_to_id.items()}
    torch_dtype = resolve_torch_dtype(config, device)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    return model.to(device)


def apply_lora_to_model(model, config: ClassificationConfig):
    """给 Qwen 分类模型挂载 LoRA 适配器

    参数含义:
        model: Qwen 序列分类模型
        config: 分类微调配置对象

    返回值含义:
        返回已挂载 LoRA 的模型
    """

    LoraConfig, TaskType, get_peft_model = import_peft_objects()
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=parse_lora_target_modules(config.lora_target_modules),
        modules_to_save=["score"],
    )
    return get_peft_model(model, lora_config)


def build_dataloaders(
    examples: list[ClassificationExample],
    tokenizer: "AutoTokenizer",
    label_to_id: dict[str, int],
    config: ClassificationConfig,
) -> tuple[object, object, list[ClassificationExample], list[ClassificationExample], int]:
    """构建训练集和验证集 DataLoader

    参数含义:
        examples: 全部分类样本列表
        tokenizer: Qwen tokenizer
        label_to_id: 类别标签到类别编号的映射
        config: 分类微调配置对象

    返回值含义:
        返回训练 DataLoader、验证 DataLoader、训练样本列表、验证样本列表和追加的困难样本数量
    """

    torch = import_torch()
    train_examples, valid_examples = split_examples(examples, config.train_split, config.seed)
    hard_examples = build_hard_training_examples(config.hard_example_repeat)
    hard_example_count = len(hard_examples)
    if hard_examples:
        train_examples = train_examples + hard_examples
    train_dataset = ClassificationDataset(train_examples, tokenizer, label_to_id, config.max_length)
    valid_dataset = ClassificationDataset(valid_examples, tokenizer, label_to_id, config.max_length)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    return train_loader, valid_loader, train_examples, valid_examples, hard_example_count


def move_batch_to_device(batch: dict[str, "torch.Tensor"], device) -> dict[str, "torch.Tensor"]:
    """把 batch 中的张量移动到目标设备

    参数含义:
        batch: DataLoader 返回的张量字典
        device: 目标 PyTorch 设备

    返回值含义:
        返回移动到目标设备后的 batch 字典
    """

    return {key: value.to(device) for key, value in batch.items()}


def evaluate_model(model, data_loader, device, max_batches: int) -> tuple[float, float]:
    """评估模型在指定数据集上的平均损失和准确率

    参数含义:
        model: 待评估的 Qwen 分类模型
        data_loader: 训练集或验证集 DataLoader
        device: 执行评估的设备
        max_batches: 最多评估的 batch 数量

    返回值含义:
        返回二元组，第一个值是平均交叉熵损失，第二个值是分类准确率
    """

    torch = import_torch()
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            if batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            assert_finite_tensor(outputs.loss, "loss", f"评估 batch {batch_index + 1}")
            assert_finite_tensor(outputs.logits, "logits", f"评估 batch {batch_index + 1}")
            losses.append(float(outputs.loss.item()))
            predictions = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    model.train()
    if not losses or total == 0:
        return float("inf"), 0.0
    return sum(losses) / len(losses), correct / total


def train_model(model, train_loader, valid_loader, config: ClassificationConfig, device) -> list[dict[str, float]]:
    """执行 Qwen 分类微调训练循环

    参数含义:
        model: 待微调的 Qwen 分类模型
        train_loader: 训练集 DataLoader
        valid_loader: 验证集 DataLoader
        config: 分类微调流程配置对象
        device: 执行训练的设备

    返回值含义:
        返回训练指标日志列表，每个元素记录 step、loss 和 accuracy
    """

    torch = import_torch()
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise ValueError("没有可训练参数，请检查 LoRA 或模型冻结配置")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_log: list[dict[str, float]] = []
    train_iter = iter(train_loader)
    model.train()

    for step in range(1, config.max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = move_batch_to_device(batch, device)
        outputs = model(**batch)
        loss = outputs.loss
        assert_finite_tensor(loss, "loss", f"训练 step {step}")
        assert_finite_tensor(outputs.logits, "logits", f"训练 step {step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, config.grad_clip)
        assert_finite_tensor(grad_norm, "grad_norm", f"训练 step {step}")
        optimizer.step()

        should_eval = step == 1 or step % config.eval_interval == 0 or step == config.max_steps
        if should_eval:
            train_loss, train_accuracy = evaluate_model(model, train_loader, device, config.eval_batches)
            valid_loss, valid_accuracy = evaluate_model(model, valid_loader, device, config.eval_batches)
            loss_log.append(
                {
                    "step": float(step),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "train_accuracy": train_accuracy,
                    "valid_accuracy": valid_accuracy,
                }
            )
            print(
                f"step {step:4d} | "
                f"train_loss {train_loss:.4f} | valid_loss {valid_loss:.4f} | "
                f"train_acc {train_accuracy:.4f} | valid_acc {valid_accuracy:.4f}"
            )
    return loss_log


def predict_text(model, tokenizer: "AutoTokenizer", text: str, id_to_label: dict[int, str], max_length: int, device) -> dict[str, object]:
    """使用分类模型预测单条文本类别

    参数含义:
        model: 用于预测的 Qwen 分类模型
        tokenizer: Qwen tokenizer
        text: 待分类文本
        id_to_label: 类别编号到标签文本的映射
        max_length: tokenizer 最大序列长度
        device: 执行预测的设备

    返回值含义:
        返回包含输入文本、预测标签、置信度和类别概率的字典
    """

    torch = import_torch()
    model.eval()
    encoded = tokenizer(
        normalize_text(text),
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        assert_finite_tensor(logits, "logits", "样例预测")
        probabilities = torch.softmax(logits, dim=-1)[0]
        assert_finite_tensor(probabilities, "probabilities", "样例预测")
    predicted_id = int(torch.argmax(probabilities).item())
    probability_items = {
        id_to_label[index]: float(probabilities[index].item())
        for index in range(len(id_to_label))
    }
    return {
        "text": normalize_text(text),
        "predicted_label": id_to_label[predicted_id],
        "confidence": float(probabilities[predicted_id].item()),
        "probabilities": probability_items,
    }


def count_parameters(model) -> tuple[int, int]:
    """统计模型参数量

    参数含义:
        model: 需要统计参数量的 PyTorch 模型

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


def save_metric_log(metric_log: list[dict[str, float]], file_path: Path) -> None:
    """把训练指标日志保存为 CSV 文件

    参数含义:
        metric_log: 训练过程中记录的指标列表
        file_path: CSV 文件保存路径

    返回值含义:
        无返回值，直接写入文件
    """

    fieldnames = ["step", "train_loss", "valid_loss", "train_accuracy", "valid_accuracy"]
    with open(file_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(metric_log)


def build_data_stats(
    examples: list[ClassificationExample],
    train_examples: list[ClassificationExample],
    valid_examples: list[ClassificationExample],
    label_to_id: dict[str, int],
    data_file: Path,
    config: ClassificationConfig,
    loss_log: list[dict[str, float]],
    fixed_label_count: int,
    hard_example_count: int,
) -> dict[str, object]:
    """构建分类微调数据和训练统计

    参数含义:
        examples: 全部分类样本
        train_examples: 训练样本列表
        valid_examples: 验证样本列表
        label_to_id: 类别标签到类别编号的映射
        data_file: 实际读取的数据文件
        config: 分类微调配置对象
        loss_log: 训练指标日志
        fixed_label_count: 自动修正的明显错标样本数量
        hard_example_count: 追加到训练集的困难样本数量

    返回值含义:
        返回可保存到 JSON 的统计字典
    """

    label_counts: dict[str, int] = {}
    for example in examples:
        label_counts[example.label] = label_counts.get(example.label, 0) + 1
    best_valid_accuracy = max((float(item["valid_accuracy"]) for item in loss_log), default=0.0)
    best_valid_loss = min((float(item["valid_loss"]) for item in loss_log), default=float("inf"))
    return {
        "data_file": str(data_file),
        "model_dir": config.model_dir,
        "total_examples": len(examples),
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "base_train_examples": len(train_examples) - hard_example_count,
        "fixed_label_count": fixed_label_count,
        "hard_example_count": hard_example_count,
        "label_to_id": label_to_id,
        "label_counts": label_counts,
        "max_train_samples": config.max_train_samples,
        "max_length": config.max_length,
        "fix_noisy_labels": config.fix_noisy_labels,
        "hard_example_repeat": config.hard_example_repeat,
        "use_lora": config.use_lora,
        "lora_target_modules": parse_lora_target_modules(config.lora_target_modules),
        "best_valid_loss": best_valid_loss,
        "best_valid_accuracy": best_valid_accuracy,
    }


def save_outputs(
    model,
    tokenizer: "AutoTokenizer",
    config: ClassificationConfig,
    label_to_id: dict[str, int],
    loss_log: list[dict[str, float]],
    before_prediction: dict[str, object],
    after_prediction: dict[str, object],
    data_stats: dict[str, object],
    output_dir: Path,
) -> None:
    """保存分类微调模型、配置、日志和样例预测

    参数含义:
        model: 微调完成的 Qwen 分类模型
        tokenizer: Qwen tokenizer
        config: 分类微调流程配置对象
        label_to_id: 类别标签到类别编号的映射
        loss_log: 训练指标日志
        before_prediction: 微调前样例预测结果
        after_prediction: 微调后样例预测结果
        data_stats: 数据和训练统计
        output_dir: 输出目录路径

    返回值含义:
        无返回值，直接在输出目录写入文件
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "qwen2_5_0_5b_classification_lora"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    save_json(asdict(config), output_dir / "classification_config.json")
    id_to_label = {str(index): label for label, index in label_to_id.items()}
    save_json({"label_to_id": label_to_id, "id_to_label": id_to_label}, output_dir / "label_mapping.json")
    save_metric_log(loss_log, output_dir / "metrics_log.csv")
    save_json({"before": before_prediction, "after": after_prediction}, output_dir / "sample_predictions.json")
    save_json(data_stats, output_dir / "data_stats.json")


def main() -> None:
    """运行 Qwen2.5 文本分类微调完整流程

    参数含义:
        无参数

    返回值含义:
        无返回值，直接执行数据读取、模型加载、分类微调、样例预测和产物保存
    """

    config = parse_args()
    validate_config(config)
    set_seed(config.seed)
    device = select_device(config.device)

    raw_examples, data_file = load_classification_examples(Path(config.data_file), config.max_train_samples, config.seed)
    examples, fixed_label_count = prepare_training_examples(raw_examples, config)
    label_to_id = build_label_mapping(examples)
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = load_tokenizer(Path(config.model_dir))
    torch_dtype = resolve_torch_dtype(config, device)
    model = load_sequence_classification_model(Path(config.model_dir), len(label_to_id), label_to_id, config, device)
    if config.use_lora:
        model = apply_lora_to_model(model, config)
    model = model.to(device)

    train_loader, valid_loader, train_examples, valid_examples, hard_example_count = build_dataloaders(
        examples,
        tokenizer,
        label_to_id,
        config,
    )
    total_params, trainable_params = count_parameters(model)

    print("Qwen2.5 分类微调配置:")
    print(config)
    print(f"运行设备: {device}")
    print(f"模型浮点精度: {torch_dtype}")
    print(f"base 模型目录: {Path(config.model_dir).resolve()}")
    print(f"分类数据文件: {data_file.resolve()}")
    print(f"原始分类样本数: {len(raw_examples):,}")
    print(f"处理后分类样本数: {len(examples):,}")
    print(f"自动修正明显错标样本数: {fixed_label_count:,}")
    print(f"追加困难训练样本数: {hard_example_count:,}")
    print(f"训练样本数: {len(train_examples):,}")
    print(f"验证样本数: {len(valid_examples):,}")
    print(f"类别映射: {label_to_id}")
    print(f"LoRA 微调: {config.use_lora}")
    print(f"LoRA 目标模块: {parse_lora_target_modules(config.lora_target_modules)}")
    print(f"模型总参数量: {total_params:,}")
    print(f"模型可训练参数量: {trainable_params:,}")

    print("\n微调前样例预测:")
    before_prediction = predict_text(model, tokenizer, config.sample_text, id_to_label, config.max_length, device)
    print(before_prediction)

    print("\n开始分类微调:")
    loss_log = train_model(model, train_loader, valid_loader, config, device)

    print("\n微调后样例预测:")
    after_prediction = predict_text(model, tokenizer, config.sample_text, id_to_label, config.max_length, device)
    print(after_prediction)

    data_stats = build_data_stats(
        examples,
        train_examples,
        valid_examples,
        label_to_id,
        data_file,
        config,
        loss_log,
        fixed_label_count,
        hard_example_count,
    )
    output_dir = Path(config.output_dir)
    save_outputs(model, tokenizer, config, label_to_id, loss_log, before_prediction, after_prediction, data_stats, output_dir)
    print(f"\n分类微调产物已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
