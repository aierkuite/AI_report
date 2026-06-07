"""本地模型文本续写程序

本程序用于读取 txt 文件内容，并根据文件内容继续生成文本
它从 outputs_true_pretrain MiniGPT checkpoint 加载模型
它只执行推理生成，不执行训练或微调
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MINIGPT_DIR = "outputs_true_pretrain"
MINIGPT_CHECKPOINT_NAME = "mini_gpt_pretrained.pt"


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
        raise RuntimeError("缺少 torch 依赖，请在虚拟环境中安装 torch 后再运行推理") from exc
    return torch


def import_minigpt_components():
    """延迟导入 MiniGPT 本地模型组件

    参数含义:
        无参数

    返回值含义:
        返回 GPTConfig、MiniGPT 和 CharTokenizer，缺少依赖时抛出 RuntimeError
    """

    try:
        from mini_gpt_step2 import GPTConfig, MiniGPT
        from mini_gpt_step3_pretrain import CharTokenizer
    except ImportError as exc:
        raise RuntimeError("缺少 MiniGPT 本地组件或 torch 依赖，请在项目根目录和虚拟环境中运行推理") from exc
    return GPTConfig, MiniGPT, CharTokenizer


def parse_args() -> argparse.Namespace:
    """解析命令行参数

    参数含义:
        无参数

    返回值含义:
        返回 argparse.Namespace，其中包含 MiniGPT 模型路径、输入和生成参数
    """

    parser = argparse.ArgumentParser(
        description="加载 outputs_true_pretrain MiniGPT 模型并读取 txt 文件进行续写",
        epilog=(
            "示例:\n"
            "  python qwen_instruction_infer.py\n"
            "  python qwen_instruction_infer.py --model-dir outputs_true_pretrain\n"
            "  python qwen_instruction_infer.py --model-dir outputs_true_pretrain/mini_gpt_pretrained.pt"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model-dir", default=DEFAULT_MINIGPT_DIR, help="MiniGPT 输出目录或 mini_gpt_pretrained.pt 文件路径")
    parser.add_argument("--input-file", default="data_pretrain/111.txt", help="作为续写开头的 txt 文件路径")
    parser.add_argument("--output-file", default="artical/神女赋_续写", help="保存完整续写结果的 txt 文件路径，留空则不保存")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="最多生成的新 token 数量")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度，MiniGPT 中 0 表示贪心解码")
    parser.add_argument("--top-p", type=float, default=0.9, help="核采样概率阈值，仅启用随机采样时生效")
    parser.add_argument("--top-k", type=int, default=10, help="MiniGPT 每步保留的候选 token 数量，0 表示不限制")
    parser.add_argument("--max-input-tokens", type=int, default=768, help="最多保留的输入 token 数量，保留文件末尾内容")
    parser.add_argument("--repetition-penalty", type=float, default=1.08, help="生成时对已出现 token 的重复惩罚系数")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0, help="MiniGPT 禁止重复的 ngram 长度，0 表示不限制")
    parser.add_argument("--no-sample", action="store_true", help="关闭随机采样，改用贪心生成")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """校验命令行生成参数

    参数含义:
        args: argparse 解析后的命令行参数对象

    返回值含义:
        无返回值，参数不合法时直接抛出异常
    """

    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须大于 0")
    if args.max_input_tokens <= 0:
        raise ValueError("max_input_tokens 必须大于 0")
    if args.temperature < 0:
        raise ValueError("temperature 不能小于 0")
    if not 0 < args.top_p <= 1:
        raise ValueError("top_p 必须位于 0 到 1 之间")
    if args.top_k < 0:
        raise ValueError("top_k 不能小于 0")
    if args.repetition_penalty < 1.0:
        raise ValueError("repetition_penalty 必须大于等于 1")
    if args.no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size 不能小于 0")


def select_device(device_name: str) -> Any:
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


def read_text_file(file_path: Path) -> str:
    """按 UTF-8 读取 txt 文件内容

    参数含义:
        file_path: 需要读取的 txt 文件路径

    返回值含义:
        返回文件中的文本内容
    """

    if not file_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {file_path}")
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"输入文件内容为空: {file_path}")
    return text


def resolve_minigpt_checkpoint_path(model_path: Path) -> Path:
    """解析 MiniGPT checkpoint 文件路径

    参数含义:
        model_path: MiniGPT 输出目录或 mini_gpt_pretrained.pt 文件路径

    返回值含义:
        返回实际存在的 mini_gpt_pretrained.pt 文件路径
    """

    checkpoint_path = model_path if model_path.is_file() else model_path / MINIGPT_CHECKPOINT_NAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 MiniGPT checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_torch_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """读取 PyTorch checkpoint 字典

    参数含义:
        checkpoint_path: 需要读取的 checkpoint 文件路径

    返回值含义:
        返回 checkpoint 字典内容
    """

    torch = import_torch()
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"MiniGPT checkpoint 格式不正确，期望字典: {checkpoint_path}")
    return checkpoint


def load_tokenizer_data(checkpoint: dict[str, Any], model_path: Path) -> dict[str, Any]:
    """从 checkpoint 或 char_tokenizer.json 读取字符级分词器数据

    参数含义:
        checkpoint: 已读取的 MiniGPT checkpoint 字典
        model_path: MiniGPT 输出目录或 checkpoint 路径

    返回值含义:
        返回字符级分词器配置字典
    """

    tokenizer_data = checkpoint.get("tokenizer")
    if isinstance(tokenizer_data, dict):
        return tokenizer_data

    tokenizer_path = model_path.parent / "char_tokenizer.json" if model_path.is_file() else model_path / "char_tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"checkpoint 中缺少 tokenizer，且找不到分词器文件: {tokenizer_path}")
    with open(tokenizer_path, "r", encoding="utf-8") as file:
        tokenizer_data = json.load(file)
    if not isinstance(tokenizer_data, dict):
        raise ValueError(f"字符级分词器文件格式不正确: {tokenizer_path}")
    return tokenizer_data


def build_char_tokenizer(tokenizer_data: dict[str, Any]) -> Any:
    """根据保存的 tokenizer 数据恢复 CharTokenizer

    参数含义:
        tokenizer_data: checkpoint 或 char_tokenizer.json 中保存的分词器字典

    返回值含义:
        返回恢复完成的 CharTokenizer 实例
    """

    _, _, CharTokenizer = import_minigpt_components()
    vocab = tokenizer_data.get("itos")
    if not isinstance(vocab, list) or not vocab:
        raise ValueError("字符级分词器数据中缺少非空 itos 列表")
    return CharTokenizer([str(token) for token in vocab])


def load_minigpt_model_and_tokenizer(model_path: Path, device: Any) -> tuple[Any, Any]:
    """加载 outputs_true_pretrain 中保存的 MiniGPT 模型和字符级分词器

    参数含义:
        model_path: MiniGPT 输出目录或 mini_gpt_pretrained.pt 文件路径
        device: 模型加载后移动到的设备

    返回值含义:
        返回加载完成的 MiniGPT 模型和 CharTokenizer
    """

    GPTConfig, MiniGPT, _ = import_minigpt_components()
    checkpoint_path = resolve_minigpt_checkpoint_path(model_path)
    checkpoint = load_torch_checkpoint(checkpoint_path)

    model_config_data = checkpoint.get("model_config")
    if not isinstance(model_config_data, dict):
        raise ValueError(f"MiniGPT checkpoint 中缺少 model_config: {checkpoint_path}")
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise ValueError(f"MiniGPT checkpoint 中缺少 model_state_dict: {checkpoint_path}")

    tokenizer = build_char_tokenizer(load_tokenizer_data(checkpoint, model_path))
    model_config = GPTConfig(**model_config_data)
    model = MiniGPT(model_config)
    incompatible_keys = model.load_state_dict(model_state_dict)
    if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
        raise RuntimeError(
            "MiniGPT checkpoint 权重与模型结构不匹配，"
            f"missing={incompatible_keys.missing_keys}, unexpected={incompatible_keys.unexpected_keys}"
        )
    model = model.to(device)
    model.eval()
    return model, tokenizer


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


def apply_top_p_filter(logits: Any, top_p: float) -> Any:
    """对 logits 应用核采样过滤

    参数含义:
        logits: 当前步生成 logits，形状为 batch_size、vocab_size
        top_p: 核采样概率阈值

    返回值含义:
        返回已经把低概率候选置为负无穷的 logits
    """

    if top_p >= 1.0:
        return logits

    torch = import_torch()
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    for batch_index in range(logits.shape[0]):
        remove_indices = sorted_indices[batch_index, sorted_indices_to_remove[batch_index]]
        logits[batch_index, remove_indices] = float("-inf")
    return logits


def apply_minigpt_repetition_controls(
    logits: Any,
    token_ids: list[int],
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> Any:
    """对 MiniGPT logits 应用重复惩罚和 ngram 禁止规则

    参数含义:
        logits: 当前步生成 logits，形状为 1、vocab_size
        token_ids: 已有上下文 token 编号序列
        repetition_penalty: 对已经出现过的 token 施加的重复惩罚，1 表示不惩罚
        no_repeat_ngram_size: 禁止重复的 ngram 长度，0 表示不限制

    返回值含义:
        返回处理重复候选后的 logits
    """

    if repetition_penalty > 1.0:
        for token_id in set(token_ids):
            if 0 <= token_id < logits.size(-1):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

    banned_tokens = get_banned_ngram_next_tokens(token_ids, no_repeat_ngram_size)
    for token_id in banned_tokens:
        if 0 <= token_id < logits.size(-1):
            logits[0, token_id] = float("-inf")
    return logits


def sample_minigpt_next_token(
    logits: Any,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
) -> Any:
    """根据 MiniGPT 当前 logits 选择下一个 token

    参数含义:
        logits: 当前步生成 logits，形状为 1、vocab_size
        temperature: 采样温度，0 表示贪心解码
        top_p: 核采样概率阈值，仅采样时生效
        top_k: 每步保留的候选 token 数量，0 表示不限制
        do_sample: 是否启用随机采样

    返回值含义:
        返回形状为 1、1 的下一个 token 编号张量
    """

    torch = import_torch()
    if not do_sample or temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = float("-inf")
    logits = apply_top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_minigpt_continuation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    args: argparse.Namespace,
) -> tuple[str, str]:
    """根据输入文本使用 MiniGPT 生成续写内容

    参数含义:
        model: 已加载的 MiniGPT 模型
        tokenizer: 已加载的 CharTokenizer
        prompt: 从 txt 文件读取的原始文本
        device: 执行推理的设备
        args: 命令行生成参数

    返回值含义:
        返回二元组，第一个值是续写文本，第二个值是原文加续写的完整文本
    """

    torch = import_torch()
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.unk_id]
    if len(token_ids) > args.max_input_tokens:
        token_ids = token_ids[-args.max_input_tokens :]

    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    generated_token_ids: list[int] = []
    model.eval()
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            idx_cond = idx[:, -model.config.context_length :]
            logits = model(idx_cond)
            logits = logits[:, -1, :].clone()
            logits = apply_minigpt_repetition_controls(
                logits=logits,
                token_ids=idx[0].tolist(),
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            next_idx = sample_minigpt_next_token(
                logits=logits,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=not args.no_sample,
            )
            generated_token_ids.append(int(next_idx.item()))
            idx = torch.cat((idx, next_idx), dim=1)

    continuation = tokenizer.decode(generated_token_ids).strip()
    if not continuation:
        continuation = f"[未生成可见文本，原始新 token 数: {len(generated_token_ids)}]"
    return continuation, prompt + continuation


def write_output_file(output_file: str, full_text: str) -> None:
    """把完整续写文本写入输出文件

    参数含义:
        output_file: 输出文件路径字符串，留空表示不写入
        full_text: 原文加续写后的完整文本

    返回值含义:
        无返回值，直接写入文件并打印保存路径
    """

    if not output_file:
        return
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\r\n") as file:
        file.write(full_text)
    print(f"\n完整续写结果已保存到: {output_path.resolve()}")


def main() -> None:
    """运行本地模型文本续写流程

    参数含义:
        无参数

    返回值含义:
        无返回值，直接在控制台打印续写内容，并可选保存完整文本
    """

    args = parse_args()
    validate_args(args)
    input_file = Path(args.input_file)
    model_path = Path(args.model_dir)
    device = select_device(args.device)
    prompt = read_text_file(input_file)

    model, tokenizer = load_minigpt_model_and_tokenizer(model_path, device)
    continuation, full_text = generate_minigpt_continuation(model, tokenizer, prompt, device, args)

    print("模型类型: MiniGPT")
    print(f"模型路径: {model_path.resolve()}")
    print(f"运行设备: {device}")
    print("\n原文:")
    print(prompt)
    print("\n续写:")
    print(continuation)

    write_output_file(args.output_file, full_text)


if __name__ == "__main__":
    main()
