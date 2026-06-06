"""Qwen2.5 外卖评论分类推理程序

本程序用于加载分类微调得到的 LoRA adapter，对单条或多条外卖评论判断差评或好评
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import AutoTokenizer


DEFAULT_MODEL_DIR = "models/qwen2.5-0.5b"
DEFAULT_ADAPTER_DIR = "outputs_qwen_classification_finetune/qwen2_5_0_5b_classification_lora"
DEFAULT_LABEL_MAPPING_FILE = "outputs_qwen_classification_finetune/label_mapping.json"


@dataclass
class InferenceConfig:
    """保存分类推理流程配置

    参数含义:
        model_dir: 本地 Qwen2.5 base 模型目录
        adapter_dir: 分类 LoRA adapter 目录
        label_mapping_file: 训练阶段保存的标签映射 JSON 文件
        text: 需要判断的单条评论文本
        input_file: 批量预测输入文本文件路径，每行一条评论
        output_file: 批量预测输出 JSONL 文件路径，空值表示输出到控制台
        max_length: tokenizer 最大序列长度
        device: 运行设备，auto 表示自动选择 cuda 或 cpu
        dtype: 模型加载使用的浮点精度

    返回值含义:
        InferenceConfig 实例用于统一传递分类推理参数
    """

    model_dir: str = DEFAULT_MODEL_DIR
    adapter_dir: str = DEFAULT_ADAPTER_DIR
    label_mapping_file: str = DEFAULT_LABEL_MAPPING_FILE
    text: str = ""
    input_file: str = ""
    output_file: str = ""
    max_length: int = 128
    device: str = "auto"
    dtype: str = "auto"


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器

    参数含义:
        无参数

    返回值含义:
        返回配置完成的 argparse.ArgumentParser 实例
    """

    parser = argparse.ArgumentParser(
        description="使用 Qwen2.5 分类 LoRA 模型判断外卖评论情感",
        epilog=(
            "示例:\n"
            "  python qwen_classification_infer.py --text \"下次还来我是狗\"\n"
            "  python qwen_classification_infer.py --input-file comments.txt --output-file predictions.jsonl"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="本地 Qwen2.5 base 模型目录")
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR, help="分类 LoRA adapter 目录")
    parser.add_argument("--label-mapping-file", default=DEFAULT_LABEL_MAPPING_FILE, help="标签映射 JSON 文件路径")
    parser.add_argument("--text", default="", help="需要判断的单条评论文本")
    parser.add_argument("--input-file", default="", help="批量预测输入文本文件路径，每行一条评论")
    parser.add_argument("--output-file", default="", help="批量预测输出 JSONL 文件路径，空值表示输出到控制台")
    parser.add_argument("--max-length", type=int, default=128, help="tokenizer 最大序列长度")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="模型浮点精度，auto 在 CUDA 支持时使用 bfloat16，否则使用 float32",
    )
    return parser


def parse_args() -> InferenceConfig:
    """解析命令行参数并生成推理配置

    参数含义:
        无参数

    返回值含义:
        返回从命令行参数生成的 InferenceConfig 实例
    """

    args = build_arg_parser().parse_args()
    return InferenceConfig(**vars(args))


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
        raise RuntimeError("缺少 torch 依赖，请在虚拟环境中安装 torch 后再运行分类推理") from exc
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
        raise RuntimeError("缺少 transformers 依赖，请在虚拟环境中安装 transformers 后再运行分类推理") from exc
    return AutoModelForSequenceClassification, AutoTokenizer


def import_peft_model():
    """延迟导入 peft 依赖

    参数含义:
        无参数

    返回值含义:
        返回 PeftModel，缺少依赖时抛出 RuntimeError
    """

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("缺少 peft 依赖，请在虚拟环境中安装 peft 后再加载分类 LoRA adapter") from exc
    return PeftModel


def validate_config(config: InferenceConfig) -> None:
    """校验分类推理配置

    参数含义:
        config: 分类推理配置对象

    返回值含义:
        无返回值，配置非法时抛出 ValueError
    """

    if config.max_length <= 0:
        raise ValueError("max_length 必须大于 0")
    if config.dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("dtype 必须是 auto、float32、float16 或 bfloat16")
    if normalize_text(config.text) and normalize_text(config.input_file):
        raise ValueError("text 和 input_file 只能二选一")
    if normalize_text(config.output_file) and not normalize_text(config.input_file):
        raise ValueError("output_file 只能和 input_file 一起使用")


def select_device(device_name: str):
    """选择模型推理设备

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


def resolve_torch_dtype(dtype_name: str, device):
    """根据配置和设备选择模型浮点精度

    参数含义:
        dtype_name: 浮点精度名称，支持 auto、float32、float16、bfloat16
        device: 执行推理的 PyTorch 设备

    返回值含义:
        返回 PyTorch dtype 对象，用于加载分类模型
    """

    torch = import_torch()
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        if device.type == "cuda" and not cuda_supports_bfloat16():
            raise RuntimeError("指定了 bfloat16，但当前 CUDA 设备不支持，请改用 --dtype float32")
        return torch.bfloat16
    if dtype_name == "auto":
        if device.type == "cuda" and cuda_supports_bfloat16():
            return torch.bfloat16
        return torch.float32
    raise ValueError("dtype 必须是 auto、float32、float16 或 bfloat16")


def normalize_text(text: object) -> str:
    """对输入文本做轻量清洗

    参数含义:
        text: 原始文本对象

    返回值含义:
        返回统一换行并去除首尾空白后的文本
    """

    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def load_label_mapping(label_mapping_file: Path) -> tuple[dict[str, int], dict[int, str]]:
    """读取训练阶段保存的标签映射

    参数含义:
        label_mapping_file: 标签映射 JSON 文件路径

    返回值含义:
        返回标签到编号映射和编号到标签映射
    """

    if not label_mapping_file.exists():
        raise FileNotFoundError(f"找不到标签映射文件: {label_mapping_file}")
    with open(label_mapping_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    label_to_id = data.get("label_to_id")
    id_to_label = data.get("id_to_label")
    if not isinstance(label_to_id, dict) or not isinstance(id_to_label, dict):
        raise ValueError(f"{label_mapping_file} 必须包含 label_to_id 和 id_to_label 字典")
    parsed_label_to_id = {str(label): int(index) for label, index in label_to_id.items()}
    parsed_id_to_label = {int(index): str(label) for index, label in id_to_label.items()}
    if set(parsed_label_to_id.values()) != set(parsed_id_to_label.keys()):
        raise ValueError("label_to_id 和 id_to_label 的类别编号不一致")
    for label, index in parsed_label_to_id.items():
        if parsed_id_to_label.get(index) != label:
            raise ValueError("label_to_id 和 id_to_label 的类别名称不一致")
    expected_waimai_mapping = {"差评": 0, "好评": 1}
    if set(parsed_label_to_id) == set(expected_waimai_mapping) and parsed_label_to_id != expected_waimai_mapping:
        raise ValueError("外卖评论标签映射必须保持 差评 -> 0、好评 -> 1")
    return parsed_label_to_id, parsed_id_to_label


def load_tokenizer(model_dir: Path, adapter_dir: Path):
    """加载 Qwen tokenizer

    参数含义:
        model_dir: 本地 Qwen2.5 base 模型目录
        adapter_dir: 分类 LoRA adapter 目录，优先从此目录读取 tokenizer

    返回值含义:
        返回加载完成的 tokenizer
    """

    _, AutoTokenizer = import_transformers()
    tokenizer_dir = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else model_dir
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"找不到 tokenizer 目录: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_classification_model(
    model_dir: Path,
    adapter_dir: Path,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    device,
    dtype_name: str,
):
    """加载 base 分类模型并挂载 LoRA adapter

    参数含义:
        model_dir: 本地 Qwen2.5 base 模型目录
        adapter_dir: 分类 LoRA adapter 目录
        label_to_id: 标签文本到类别编号的映射
        id_to_label: 类别编号到标签文本的映射
        device: 模型加载后移动到的设备
        dtype_name: 模型加载使用的浮点精度名称

    返回值含义:
        返回已挂载 LoRA adapter 且处于 eval 模式的分类模型
    """

    if not model_dir.exists():
        raise FileNotFoundError(f"找不到 base 模型目录: {model_dir}")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"找不到分类 LoRA adapter 目录: {adapter_dir}")
    AutoModelForSequenceClassification, _ = import_transformers()
    PeftModel = import_peft_model()
    torch_dtype = resolve_torch_dtype(dtype_name, device)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = base_model.config.eos_token_id
    model = PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=True)
    model.to(device)
    model.eval()
    return model


def assert_finite_tensor(tensor, tensor_name: str) -> None:
    """检查张量是否全部为有限数值

    参数含义:
        tensor: 需要检查的 PyTorch 张量
        tensor_name: 张量名称，用于错误提示

    返回值含义:
        无返回值，发现 NaN 或 Inf 时抛出 RuntimeError
    """

    torch = import_torch()
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{tensor_name} 出现 NaN 或 Inf，请尝试使用 --dtype float32")


def predict_text(
    model,
    tokenizer: "AutoTokenizer",
    text: str,
    id_to_label: dict[int, str],
    max_length: int,
    device,
) -> dict[str, object]:
    """使用分类模型预测单条评论

    参数含义:
        model: 已加载的 Qwen 分类模型
        tokenizer: 已加载的 Qwen tokenizer
        text: 待判断的评论文本
        id_to_label: 类别编号到标签文本的映射
        max_length: tokenizer 最大序列长度
        device: 执行推理的设备

    返回值含义:
        返回包含输入文本、预测标签、置信度和类别概率的字典
    """

    torch = import_torch()
    normalized_text = normalize_text(text)
    if not normalized_text:
        raise ValueError("待判断文本不能为空")
    encoded = tokenizer(
        normalized_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        assert_finite_tensor(logits, "logits")
        probabilities = torch.softmax(logits, dim=-1)[0]
        assert_finite_tensor(probabilities, "probabilities")
    predicted_id = int(torch.argmax(probabilities).item())
    probability_items = {
        id_to_label[index]: float(probabilities[index].item())
        for index in sorted(id_to_label)
    }
    return {
        "text": normalized_text,
        "predicted_label": id_to_label[predicted_id],
        "confidence": float(probabilities[predicted_id].item()),
        "probabilities": probability_items,
    }


def load_input_texts(config: InferenceConfig) -> list[str]:
    """读取需要预测的评论文本列表

    参数含义:
        config: 分类推理配置对象

    返回值含义:
        返回需要预测的评论文本列表
    """

    text = normalize_text(config.text)
    if text:
        return [text]
    input_file = normalize_text(config.input_file)
    if input_file:
        file_path = Path(input_file)
        if not file_path.exists():
            raise FileNotFoundError(f"找不到批量输入文件: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            return [line.strip() for line in file if line.strip()]
    interactive_text = input("请输入要判断的评论: ")
    return [normalize_text(interactive_text)]


def write_predictions(predictions: list[dict[str, object]], output_file: str) -> None:
    """输出预测结果

    参数含义:
        predictions: 预测结果字典列表
        output_file: 输出 JSONL 文件路径，空值表示输出到控制台

    返回值含义:
        无返回值，直接写入文件或打印到控制台
    """

    output_path_text = normalize_text(output_file)
    if output_path_text:
        output_path = Path(output_path_text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as file:
            for prediction in predictions:
                file.write(json.dumps(prediction, ensure_ascii=False) + "\r\n")
        print(f"预测结果已保存到: {output_path.resolve()}")
        return
    if len(predictions) == 1:
        print(json.dumps(predictions[0], ensure_ascii=False, indent=2))
        return
    for prediction in predictions:
        print(json.dumps(prediction, ensure_ascii=False))


def main() -> None:
    """运行分类推理完整流程

    参数含义:
        无参数

    返回值含义:
        无返回值，直接加载模型、执行预测并输出结果
    """

    config = parse_args()
    validate_config(config)
    device = select_device(config.device)
    label_to_id, id_to_label = load_label_mapping(Path(config.label_mapping_file))
    tokenizer = load_tokenizer(Path(config.model_dir), Path(config.adapter_dir))
    model = load_classification_model(
        Path(config.model_dir),
        Path(config.adapter_dir),
        label_to_id,
        id_to_label,
        device,
        config.dtype,
    )
    texts = load_input_texts(config)
    predictions = [
        predict_text(model, tokenizer, text, id_to_label, config.max_length, device)
        for text in texts
    ]
    write_predictions(predictions, config.output_file)


if __name__ == "__main__":
    main()
