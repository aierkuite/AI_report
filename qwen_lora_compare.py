"""Qwen LoRA 微调前后回答对比程序

本程序用于输入一条指令，分别加载本地 Qwen2.5 base 模型和 LoRA 微调适配器，比较训练前后回答差异
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_DIR = "models/qwen2.5-0.5b"
DEFAULT_ADAPTER_DIR = "outputs_qwen_lora_finetune/qwen2_5_0_5b_lora_finetuned"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


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


def import_transformers():
    """延迟导入 transformers 依赖

    参数含义:
        无参数

    返回值含义:
        返回 AutoModelForCausalLM 和 AutoTokenizer，缺少依赖时抛出 RuntimeError
    """

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("缺少 transformers 依赖，请在虚拟环境中安装 transformers 后再运行推理") from exc
    return AutoModelForCausalLM, AutoTokenizer


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
        raise RuntimeError("缺少 peft 依赖，请在虚拟环境中安装 peft 后再加载 LoRA adapter") from exc
    return PeftModel


def parse_args() -> argparse.Namespace:
    """解析命令行参数

    参数含义:
        无参数

    返回值含义:
        返回 argparse.Namespace，其中包含模型路径、输入指令和生成参数
    """

    parser = argparse.ArgumentParser(
        description="比较 Qwen2.5 base 模型和 LoRA 微调模型的指令回答",
        epilog=(
            "示例:\n"
            "  python qwen_lora_compare.py \"请用一句话解释什么是人工智能\"\n"
            "  python qwen_lora_compare.py --instruction \"请概括下面内容\" --input \"待概括文本\""
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("positional_instruction", nargs="?", default="", metavar="instruction", help="指令文本，可替代 --instruction")
    parser.add_argument("positional_input", nargs="?", default="", metavar="input", help="输入文本，可替代 --input")
    parser.add_argument("--instruction", default="", help="指令文本")
    parser.add_argument("--input", default="", help="输入文本，作为指令需要参考的补充内容")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="本地 Qwen2.5 base 模型目录")
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR, help="LoRA 微调适配器目录")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    parser.add_argument("--max-new-tokens", type=int, default=60, help="最多生成的新 token 数量")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度，仅在启用采样时生效")
    parser.add_argument("--top-p", type=float, default=0.9, help="核采样概率阈值，仅在启用采样时生效")
    parser.add_argument("--do-sample", action="store_true", help="启用随机采样，默认使用确定性生成")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="生成时对已出现 token 的重复惩罚系数")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="Qwen 对话模板中的 system 消息")
    args = parser.parse_args()
    use_interactive_optional_fields = not any(
        [
            args.instruction.strip(),
            args.positional_instruction.strip(),
            args.input.strip(),
            args.positional_input.strip(),
        ]
    )
    args.instruction = resolve_instruction(args.instruction, args.positional_instruction)
    args.input = resolve_optional_text(args.input, args.positional_input, "请输入输入，可直接回车跳过: ", use_interactive_optional_fields)
    return args


def resolve_instruction(instruction: str, positional_instruction: str) -> str:
    """解析最终使用的指令文本

    参数含义:
        instruction: 通过 --instruction 传入的指令文本
        positional_instruction: 通过位置参数传入的指令文本

    返回值含义:
        返回最终用于模型推理的非空指令文本
    """

    instruction = instruction.strip() or positional_instruction.strip()
    if instruction:
        return instruction
    instruction = input("请输入指令: ").strip()
    if not instruction:
        raise ValueError("指令不能为空")
    return instruction


def resolve_optional_text(value: str, positional_value: str, prompt: str, should_prompt: bool) -> str:
    """解析可选文本字段

    参数含义:
        value: 命令行选项传入的文本
        positional_value: 命令行位置参数传入的文本
        prompt: 交互式输入时展示的提示语
        should_prompt: 是否在文本缺失时进入交互式输入

    返回值含义:
        返回命令行文本、位置参数文本或交互式输入文本，允许为空
    """

    if value.strip():
        return value.strip()
    if positional_value.strip():
        return positional_value.strip()
    if should_prompt:
        return input(prompt).strip()
    return ""


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


def build_user_content(instruction: str, input_text: str) -> str:
    """构造 Qwen 对话模板中的用户消息

    参数含义:
        instruction: 用户输入的任务指令
        input_text: 任务需要参考的补充输入内容，可以为空

    返回值含义:
        返回合并后的用户消息文本
    """

    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def build_chat_prompt(
    tokenizer,
    instruction: str,
    input_text: str,
    system_prompt: str,
) -> str:
    """使用 Qwen chat template 构造生成提示词

    参数含义:
        tokenizer: 已加载的 Qwen tokenizer
        instruction: 用户输入的任务指令
        input_text: 任务需要参考的补充输入内容，可以为空
        system_prompt: Qwen 对话模板中的 system 消息

    返回值含义:
        返回包含 system、user 和 assistant 起始标记的提示词文本
    """

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": build_user_content(instruction, input_text)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def get_generation_eos_token_ids(tokenizer) -> int | list[int]:
    """获取 Qwen 生成时使用的停止 token 编号

    参数含义:
        tokenizer: 已加载的 Qwen tokenizer

    返回值含义:
        返回 eos token 编号列表，包含 <|endoftext|> 和 <|im_end|> 中可用的 token
    """

    token_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))

    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_token_id is not None and im_end_token_id != tokenizer.unk_token_id:
        im_end_token_id = int(im_end_token_id)
        if im_end_token_id not in token_ids:
            token_ids.append(im_end_token_id)

    if not token_ids:
        raise ValueError("tokenizer 缺少可用的 eos token")
    return token_ids[0] if len(token_ids) == 1 else token_ids


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
    return tokenizer


def load_base_model(model_dir: Path, device):
    """加载本地 Qwen2.5 base 模型

    参数含义:
        model_dir: 本地 Qwen2.5 base 模型目录
        device: 模型加载后移动到的设备

    返回值含义:
        返回已移动到目标设备并设为 eval 模式的 base 模型
    """

    if not model_dir.exists():
        raise FileNotFoundError(f"找不到 base 模型目录: {model_dir}")

    torch = import_torch()
    AutoModelForCausalLM, _ = import_transformers()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.config.pad_token_id = model.config.eos_token_id
    model = model.to(device)
    model.eval()
    return model


def load_lora_model(
    model_dir: Path,
    adapter_dir: Path,
    device,
):
    """加载 Qwen2.5 base 模型并挂载 LoRA 适配器

    参数含义:
        model_dir: 本地 Qwen2.5 base 模型目录
        adapter_dir: LoRA 微调适配器目录
        device: 模型加载后移动到的设备

    返回值含义:
        返回已挂载 LoRA 适配器并设为 eval 模式的模型
    """

    if not adapter_dir.exists():
        raise FileNotFoundError(f"找不到 LoRA adapter 目录: {adapter_dir}")

    base_model = load_base_model(model_dir, device)
    PeftModel = import_peft_model()
    lora_model = PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=True)
    lora_model.eval()
    return lora_model


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    """根据 Qwen chat prompt 生成回答

    参数含义:
        model: 用于生成的 Qwen 因果语言模型
        tokenizer: 已加载的 Qwen tokenizer
        prompt: 已由 Qwen chat template 构造好的提示词
        device: 执行推理的设备
        max_new_tokens: 最多生成的新 token 数量
        do_sample: 是否启用随机采样
        temperature: 采样温度，仅在启用采样时生效
        top_p: 核采样概率阈值，仅在启用采样时生效
        repetition_penalty: 对已出现 token 的重复惩罚系数

    返回值含义:
        返回只包含新生成内容的回答文本
    """

    torch = import_torch()
    with torch.no_grad():
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
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
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        output_ids = model.generate(**generation_kwargs)
    new_token_ids = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


def release_model(model, device) -> None:
    """释放模型引用并在 CUDA 上清理显存缓存

    参数含义:
        model: 需要释放的模型对象
        device: 当前执行推理的设备

    返回值含义:
        无返回值，直接触发 Python 垃圾回收和可选 CUDA 缓存清理
    """

    del model
    gc.collect()
    if device.type == "cuda":
        torch = import_torch()
        torch.cuda.empty_cache()


def print_comparison(
    instruction: str,
    input_text: str,
    model_dir: Path,
    adapter_dir: Path,
    before_answer: str,
    after_answer: str,
) -> None:
    """打印训练前后回答对比结果

    参数含义:
        instruction: 用户输入的任务指令
        input_text: 任务需要参考的补充输入内容
        model_dir: base 模型目录
        adapter_dir: LoRA adapter 目录
        before_answer: base 模型生成的训练前回答
        after_answer: LoRA 模型生成的训练后回答

    返回值含义:
        无返回值，直接向控制台打印对比内容
    """

    print("Qwen 训练前后回答对比")
    print(f"base 模型目录: {model_dir.resolve()}")
    print(f"LoRA adapter 目录: {adapter_dir.resolve()}")
    print("\n指令:")
    print(instruction)
    print("\n输入:")
    print(input_text if input_text.strip() else "[空]")
    print("\n训练前回答:")
    print(before_answer or "[未生成可见文本]")
    print("\n训练后回答:")
    print(after_answer or "[未生成可见文本]")


def main() -> None:
    """运行 Qwen 训练前后回答对比流程

    参数含义:
        无参数

    返回值含义:
        无返回值，直接加载 base 模型和 LoRA 模型并打印回答对比
    """

    args = parse_args()
    model_dir = Path(args.model_dir)
    adapter_dir = Path(args.adapter_dir)
    device = select_device(args.device)
    tokenizer = load_tokenizer(model_dir)
    prompt = build_chat_prompt(tokenizer, args.instruction, args.input, args.system_prompt)

    base_model = load_base_model(model_dir, device)
    before_answer = generate_answer(
        model=base_model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    release_model(base_model, device)
    base_model = None

    lora_model = load_lora_model(model_dir, adapter_dir, device)
    after_answer = generate_answer(
        model=lora_model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    release_model(lora_model, device)
    lora_model = None

    print_comparison(
        instruction=args.instruction,
        input_text=args.input,
        model_dir=model_dir,
        adapter_dir=adapter_dir,
        before_answer=before_answer,
        after_answer=after_answer,
    )


if __name__ == "__main__":
    main()
