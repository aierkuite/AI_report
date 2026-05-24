"""Qwen 文本续写程序

本程序用于从本地目录加载 Qwen 模型，读取 txt 文件内容，并根据文件内容继续生成文本
它只执行推理生成，不执行训练或微调
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    """解析命令行参数

    参数含义:
        无参数

    返回值含义:
        返回 argparse.Namespace，其中包含模型路径、指令、输入和生成参数
    """

    parser = argparse.ArgumentParser(description="加载本地 Qwen 模型并读取 txt 文件进行续写")
    parser.add_argument("--model-dir", default="models/qwen2.5-0.5b", help="本地 Qwen 模型目录")
    parser.add_argument("--input-file", default="data_pretrain/神女赋.txt", help="作为续写开头的 txt 文件路径")
    parser.add_argument("--output-file", default="artical/神女赋_续写", help="保存完整续写结果的 txt 文件路径，留空则不保存")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="最多生成的新 token 数量")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="核采样概率阈值")
    parser.add_argument("--max-input-tokens", type=int, default=768, help="最多保留的输入 token 数量，保留文件末尾内容")
    parser.add_argument("--no-sample", action="store_true", help="关闭随机采样，改用贪心生成")
    parser.add_argument("--stop-at-eos", action="store_true", help="生成遇到 eos token 时停止")
    return parser.parse_args()


def select_device(device_name: str) -> torch.device:
    """选择模型推理设备

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


def read_text_file(file_path: Path) -> str:
    """按 UTF-8 读取 txt 文件内容

    参数含义:
        file_path: 需要读取的 txt 文件路径

    返回值含义:
        返回文件中的文本内容
    """

    if not file_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()


def load_model_and_tokenizer(model_dir: Path, device: torch.device) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """从本地目录加载 Qwen 模型和 tokenizer

    参数含义:
        model_dir: 本地模型目录路径
        device: 模型加载后移动到的设备

    返回值含义:
        返回加载完成的模型和 tokenizer
    """

    if not model_dir.exists():
        raise FileNotFoundError(f"找不到模型目录: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    model.eval()
    return model, tokenizer


def get_end_token_id(tokenizer: AutoTokenizer) -> int:
    """获取可用于生成结束和 padding 的 token 编号

    参数含义:
        tokenizer: 已加载的 Qwen tokenizer

    返回值含义:
        返回 eos 或 pad 中第一个可用的 token 编号
    """

    for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        if token_id is not None:
            return int(token_id)
    raise ValueError("tokenizer 缺少 eos_token_id 和 pad_token_id")


@torch.no_grad()
def generate_continuation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    max_input_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    stop_at_eos: bool,
) -> tuple[str, str]:
    """根据输入文本生成续写内容

    参数含义:
        model: 已加载的 Qwen 因果语言模型
        tokenizer: 已加载的 Qwen tokenizer
        prompt: 从 txt 文件读取的原始文本
        device: 执行推理的设备
        max_new_tokens: 最多生成的新 token 数量
        max_input_tokens: 最多保留的输入 token 数量
        temperature: 采样温度
        top_p: 核采样概率阈值
        do_sample: 是否启用随机采样
        stop_at_eos: 是否遇到 eos token 就停止

    返回值含义:
        返回二元组，第一个值是续写文本，第二个值是原文加续写的完整文本
    """

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if input_ids.shape[1] > max_input_tokens:
        input_ids = input_ids[:, -max_input_tokens:]
        attention_mask = attention_mask[:, -max_input_tokens:]

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    end_token_id = get_end_token_id(tokenizer)
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": 1.08,
        "pad_token_id": end_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    if stop_at_eos:
        generation_kwargs["eos_token_id"] = end_token_id

    output_ids = model.generate(**generation_kwargs)
    new_tokens = output_ids[0, input_ids.shape[1] :]
    continuation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not continuation:
        token_text = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        continuation = f"[未生成可见文本，原始新 token 数: {new_tokens.numel()}，原始解码: {token_text}]"
    return continuation, prompt + continuation


def main() -> None:
    """运行 Qwen 文本续写流程

    参数含义:
        无参数

    返回值含义:
        无返回值，直接在控制台打印续写内容，并可选保存完整文本
    """

    args = parse_args()
    input_file = Path(args.input_file)
    device = select_device(args.device)
    model, tokenizer = load_model_and_tokenizer(Path(args.model_dir), device)
    prompt = read_text_file(input_file)
    continuation, full_text = generate_continuation(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
        stop_at_eos=args.stop_at_eos,
    )

    print("\n原文:")
    print(prompt)
    print("\n续写:")
    print(continuation)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="\r\n") as file:
            file.write(full_text)
        print(f"\n完整续写结果已保存到: {output_path.resolve()}")


if __name__ == "__main__":
    main()
