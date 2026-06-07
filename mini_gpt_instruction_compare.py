"""MiniGPT 指令微调前后回答对比程序

本程序用于输入一条指令和可选输入内容，分别加载第 3 步预训练模型与第 4 步指令微调模型，输出微调前后回答作为参照
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PRETRAINED_CHECKPOINT = "outputs_true_pretrain_tinystories/mini_gpt_pretrained.pt"
DEFAULT_FINETUNED_CHECKPOINT = "outputs_true_pretrain_alpaca_instruction_finetune/mini_gpt_instruction_finetuned.pt"
DEFAULT_EOS_TOKEN = "<EOS>"
DEFAULT_PROMPT_STYLE = "compact"


@dataclass
class InferenceConfig:
    """保存 MiniGPT 指令推理对比配置

    参数含义:
        instruction: 用户希望模型执行的任务指令
        input: 任务需要参考的输入内容，可以为空
        pretrained: 第 3 步预训练 checkpoint 路径
        finetuned: 第 4 步指令微调 checkpoint 路径
        device: 推理设备，auto 表示自动选择 cuda 或 cpu
        max_new_tokens: 每个模型最多生成的新 token 数量
        temperature: 生成采样温度，0 表示贪心解码
        top_k: 生成时保留概率最高的候选 token 数量，0 表示不限制
        prompt_style: 提示词模板风格，auto 表示读取微调 checkpoint 中的配置
        eos_token: 回答结束特殊 token，auto 表示读取微调 checkpoint 中的配置
        seed: 随机种子，None 表示读取微调 checkpoint 中的训练种子

    返回值含义:
        InferenceConfig 实例用于统一传递 MiniGPT 指令推理对比参数
    """

    instruction: str
    input: str
    pretrained: str = DEFAULT_PRETRAINED_CHECKPOINT
    finetuned: str = DEFAULT_FINETUNED_CHECKPOINT
    device: str = "auto"
    max_new_tokens: int = 60
    temperature: float = 0.0
    top_k: int = 0
    prompt_style: str = "auto"
    eos_token: str = "auto"
    seed: int | None = None


def parse_args() -> InferenceConfig:
    """解析命令行参数并生成 MiniGPT 指令推理对比配置

    参数含义:
        无参数

    返回值含义:
        返回从命令行参数和可选交互输入生成的 InferenceConfig 实例
    """

    parser = argparse.ArgumentParser(
        description="比较 MiniGPT 预训练模型和指令微调模型的回答",
        epilog=(
            "示例:\n"
            "  python mini_gpt_instruction_compare.py \"请用一句话解释什么是人工智能\"\n"
            "  python mini_gpt_instruction_compare.py --instruction \"请总结下面这段话\" --input \"待总结文本\""
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("positional_instruction", nargs="?", default="", metavar="instruction", help="指令文本，可替代 --instruction")
    parser.add_argument("positional_input", nargs="?", default="", metavar="input", help="输入文本，可替代 --input")
    parser.add_argument("--instruction", default="", help="指令文本")
    parser.add_argument("--input", default="", help="输入文本，作为指令需要参考的补充内容")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED_CHECKPOINT, help="第 3 步预训练 checkpoint 路径")
    parser.add_argument("--finetuned", default=DEFAULT_FINETUNED_CHECKPOINT, help="第 4 步指令微调 checkpoint 路径")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    parser.add_argument("--max-new-tokens", type=int, default=60, help="每个模型最多生成的新 token 数量")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成采样温度，传入 0 表示贪心解码")
    parser.add_argument("--top-k", type=int, default=0, help="生成时保留的候选 token 数量，传入 0 表示不限制")
    parser.add_argument("--prompt-style", default="auto", choices=["auto", "compact", "alpaca"], help="提示词模板风格")
    parser.add_argument("--eos-token", default="auto", help="回答结束特殊 token，默认读取微调 checkpoint 配置")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，默认读取微调 checkpoint 中的训练种子")
    args = parser.parse_args()

    use_interactive_optional_fields = not any(
        [
            args.instruction.strip(),
            args.positional_instruction.strip(),
            args.input.strip(),
            args.positional_input.strip(),
        ]
    )
    instruction = resolve_instruction(args.instruction, args.positional_instruction)
    input_text = resolve_optional_text(
        args.input,
        args.positional_input,
        "请输入输入，可直接回车跳过: ",
        use_interactive_optional_fields,
    )
    return InferenceConfig(
        instruction=instruction,
        input=input_text,
        pretrained=args.pretrained,
        finetuned=args.finetuned,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        prompt_style=args.prompt_style,
        eos_token=args.eos_token,
        seed=args.seed,
    )


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
    """解析可选输入文本

    参数含义:
        value: 通过 --input 传入的文本
        positional_value: 通过位置参数传入的文本
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


def validate_config(config: InferenceConfig) -> None:
    """校验 MiniGPT 指令推理对比配置

    参数含义:
        config: MiniGPT 指令推理对比配置对象

    返回值含义:
        无返回值，配置非法时抛出 ValueError
    """

    if config.max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须大于 0")
    if config.temperature < 0:
        raise ValueError("temperature 不能小于 0")
    if config.top_k < 0:
        raise ValueError("top_k 不能小于 0")
    if not config.pretrained.strip():
        raise ValueError("pretrained 不能为空")
    if not config.finetuned.strip():
        raise ValueError("finetuned 不能为空")


def import_runtime_components():
    """延迟导入 MiniGPT 推理所需运行组件

    参数含义:
        无参数

    返回值含义:
        返回二元组，第一个值是 torch 模块，第二个值是 mini.py 模块
    """

    try:
        import torch
        import mini as mini_runtime
    except ImportError as exc:
        raise RuntimeError("缺少 MiniGPT 推理依赖，请在包含 torch 的虚拟环境中运行本程序") from exc
    return torch, mini_runtime


def select_device(torch_module, device_name: str):
    """选择 MiniGPT 推理设备

    参数含义:
        torch_module: 已导入的 torch 模块
        device_name: 设备名称，支持 auto、cpu、cuda

    返回值含义:
        返回 PyTorch 设备对象
    """

    if device_name == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("指定了 cuda，但当前环境不可用")
        return torch_module.device("cuda")
    if device_name == "cpu":
        return torch_module.device("cpu")
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def load_checkpoint(torch_module, checkpoint_path: Path, label: str) -> dict[str, object]:
    """读取 MiniGPT checkpoint 文件

    参数含义:
        torch_module: 已导入的 torch 模块
        checkpoint_path: checkpoint 文件路径
        label: 用于错误信息的模型阶段名称

    返回值含义:
        返回 torch.load 读取到的 checkpoint 字典
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到{label} checkpoint: {checkpoint_path}")
    checkpoint = torch_module.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{label} checkpoint 顶层结构必须是字典")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"{label} checkpoint 中缺少 model_state_dict")
    return checkpoint


def get_finetune_option(
    finetuned_checkpoint: dict[str, object],
    option_name: str,
    fallback: str,
) -> str:
    """从微调 checkpoint 中读取字符串配置

    参数含义:
        finetuned_checkpoint: 第 4 步指令微调 checkpoint 字典
        option_name: 需要读取的 finetune_config 字段名称
        fallback: checkpoint 中缺少该字段时使用的默认值

    返回值含义:
        返回读取到的字符串配置或默认值
    """

    finetune_config = finetuned_checkpoint.get("finetune_config")
    if isinstance(finetune_config, dict):
        value = finetune_config.get(option_name)
        if isinstance(value, str) and value:
            return value
    return fallback


def get_finetune_int_option(
    finetuned_checkpoint: dict[str, object],
    option_name: str,
    fallback: int,
) -> int:
    """从微调 checkpoint 中读取整数配置

    参数含义:
        finetuned_checkpoint: 第 4 步指令微调 checkpoint 字典
        option_name: 需要读取的 finetune_config 字段名称
        fallback: checkpoint 中缺少该字段时使用的默认值

    返回值含义:
        返回读取到的整数配置或默认值
    """

    finetune_config = finetuned_checkpoint.get("finetune_config")
    if isinstance(finetune_config, dict):
        value = finetune_config.get(option_name)
        if isinstance(value, int):
            return value
    return fallback


def resolve_generation_template(
    config: InferenceConfig,
    finetuned_checkpoint: dict[str, object],
) -> tuple[str, str]:
    """确定推理时使用的提示词模板和 EOS token

    参数含义:
        config: MiniGPT 指令推理对比配置对象
        finetuned_checkpoint: 第 4 步指令微调 checkpoint 字典

    返回值含义:
        返回二元组，第一个值是提示词模板风格，第二个值是 EOS token 文本
    """

    prompt_style = config.prompt_style
    if prompt_style == "auto":
        prompt_style = get_finetune_option(finetuned_checkpoint, "prompt_style", DEFAULT_PROMPT_STYLE)

    eos_token = config.eos_token
    if eos_token == "auto":
        eos_token = get_finetune_option(finetuned_checkpoint, "eos_token", DEFAULT_EOS_TOKEN)

    return prompt_style, eos_token


def resolve_seed(config: InferenceConfig, finetuned_checkpoint: dict[str, object]) -> int:
    """确定推理时使用的随机种子

    参数含义:
        config: MiniGPT 指令推理对比配置对象
        finetuned_checkpoint: 第 4 步指令微调 checkpoint 字典

    返回值含义:
        返回用于模型初始化和可选采样生成的随机种子
    """

    if config.seed is not None:
        return config.seed
    return get_finetune_int_option(finetuned_checkpoint, "seed", 42)


def set_seed(torch_module, seed: int) -> None:
    """设置 MiniGPT 推理对比流程的随机种子

    参数含义:
        torch_module: 已导入的 torch 模块
        seed: 随机种子整数

    返回值含义:
        无返回值，直接影响扩展词表初始化和可选采样生成
    """

    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def build_tokenizer(mini_runtime, finetuned_checkpoint: dict[str, object]):
    """从微调 checkpoint 中还原字符级 tokenizer

    参数含义:
        mini_runtime: 已导入的 mini.py 模块
        finetuned_checkpoint: 第 4 步指令微调 checkpoint 字典

    返回值含义:
        返回 CharTokenizer 实例
    """

    return mini_runtime.tokenizer_from_checkpoint(finetuned_checkpoint)


def build_finetuned_model_config(mini_runtime, finetuned_checkpoint: dict[str, object], vocab_size: int):
    """从微调 checkpoint 中构建 MiniGPT 模型配置

    参数含义:
        mini_runtime: 已导入的 mini.py 模块
        finetuned_checkpoint: 第 4 步指令微调 checkpoint 字典
        vocab_size: tokenizer 实际词表大小

    返回值含义:
        返回 GPTConfig 实例，配置非法时抛出 ValueError
    """

    model_config_data = finetuned_checkpoint.get("model_config")
    if not isinstance(model_config_data, dict):
        raise ValueError("微调 checkpoint 中缺少 model_config")
    model_config = mini_runtime.GPTConfig(**model_config_data)
    if model_config.vocab_size != vocab_size:
        raise ValueError(
            f"微调 checkpoint 的 vocab_size={model_config.vocab_size} 与 tokenizer 大小={vocab_size} 不一致"
        )
    return model_config


def get_eos_id(tokenizer, eos_token: str) -> int:
    """获取 EOS token 在字符级 tokenizer 中的编号

    参数含义:
        tokenizer: 从微调 checkpoint 中还原的 CharTokenizer
        eos_token: 回答结束特殊 token 文本

    返回值含义:
        返回 EOS token 编号，token 缺失时抛出 ValueError
    """

    eos_id = tokenizer.stoi.get(eos_token)
    if eos_id is None:
        raise ValueError(f"微调 tokenizer 中缺少 EOS token: {eos_token}")
    return int(eos_id)


def load_model(mini_runtime, checkpoint: dict[str, object], model_config, device):
    """按统一模型配置加载 MiniGPT 权重

    参数含义:
        mini_runtime: 已导入的 mini.py 模块
        checkpoint: 预训练或微调 checkpoint 字典
        model_config: 当前推理使用的 GPTConfig 实例
        device: 模型加载后移动到的设备

    返回值含义:
        返回四元组，依次为模型、直接加载参数名列表、尺寸适配参数名列表、跳过参数名列表
    """

    model = mini_runtime.MiniGPT(model_config)
    loaded_keys, resized_keys, skipped_keys = mini_runtime.load_state_dict_with_resize(model, checkpoint)
    model = model.to(device)
    model.eval()
    return model, loaded_keys, resized_keys, skipped_keys


def generate_answer(
    mini_runtime,
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    eos_id: int,
    prompt_style: str,
    device,
) -> str:
    """生成并提取 MiniGPT 指令回答正文

    参数含义:
        mini_runtime: 已导入的 mini.py 模块
        model: 用于推理的 MiniGPT 模型
        tokenizer: 字符级 tokenizer
        instruction: 用户输入的任务指令
        input_text: 任务需要参考的补充输入内容，可以为空
        max_new_tokens: 最多生成的新 token 数量
        temperature: 采样温度，0 表示贪心解码
        top_k: 每步采样保留的候选 token 数量，0 表示不限制
        eos_id: 回答结束特殊 token 的编号
        prompt_style: 提示词模板风格
        device: 执行推理的设备

    返回值含义:
        返回只包含模型新生成内容的回答文本
    """

    prompt = mini_runtime.format_prompt(instruction, input_text, prompt_style=prompt_style)
    full_text = mini_runtime.generate_instruction_answer(
        model=model,
        tokenizer=tokenizer,
        instruction=instruction,
        input_text=input_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
        prompt_style=prompt_style,
        device=device,
    )
    if full_text.startswith(prompt):
        return full_text[len(prompt) :].strip()
    return mini_runtime.strip_generated_answer(full_text)


def release_model(torch_module, model, device) -> None:
    """释放模型引用并按需清理 CUDA 缓存

    参数含义:
        torch_module: 已导入的 torch 模块
        model: 需要释放的模型对象
        device: 当前推理设备

    返回值含义:
        无返回值，直接触发 Python 垃圾回收和可选 CUDA 缓存清理
    """

    del model
    gc.collect()
    if device.type == "cuda":
        torch_module.cuda.empty_cache()


def print_load_stats(label: str, loaded_keys: list[str], resized_keys: list[str], skipped_keys: list[str]) -> None:
    """打印模型权重加载统计信息

    参数含义:
        label: 模型阶段名称
        loaded_keys: 直接加载的参数名列表
        resized_keys: 因词表扩展而尺寸适配的参数名列表
        skipped_keys: 未加载的参数名列表

    返回值含义:
        无返回值，直接向控制台打印统计信息
    """

    print(f"{label}直接加载参数数: {len(loaded_keys):,}")
    print(f"{label}尺寸适配参数数: {len(resized_keys):,}")
    print(f"{label}跳过参数数: {len(skipped_keys):,}")


def print_comparison(
    config: InferenceConfig,
    pretrained_path: Path,
    finetuned_path: Path,
    device,
    prompt_style: str,
    eos_token: str,
    seed: int,
    before_answer: str,
    after_answer: str,
) -> None:
    """打印 MiniGPT 指令微调前后回答对比

    参数含义:
        config: MiniGPT 指令推理对比配置对象
        pretrained_path: 第 3 步预训练 checkpoint 路径
        finetuned_path: 第 4 步指令微调 checkpoint 路径
        device: 当前推理设备
        prompt_style: 实际使用的提示词模板风格
        eos_token: 实际使用的回答结束特殊 token
        seed: 实际使用的随机种子
        before_answer: 微调前模型生成回答
        after_answer: 微调后模型生成回答

    返回值含义:
        无返回值，直接向控制台打印对比内容
    """

    print("MiniGPT 指令微调前后回答对比")
    print(f"预训练 checkpoint: {pretrained_path.resolve()}")
    print(f"微调 checkpoint: {finetuned_path.resolve()}")
    print(f"运行设备: {device}")
    print(f"提示词模板: {prompt_style}")
    print(f"EOS token: {eos_token}")
    print(f"生成 token 数: {config.max_new_tokens}")
    print(f"temperature: {config.temperature}")
    print(f"top_k: {config.top_k}")
    print(f"seed: {seed}")
    print("\n指令:")
    print(config.instruction)
    print("\n输入:")
    print(config.input if config.input.strip() else "[空]")
    print("\n微调前模型输出:")
    print(before_answer or "[未生成可见文本]")
    print("\n微调后模型输出:")
    print(after_answer or "[未生成可见文本]")


def main() -> None:
    """运行 MiniGPT 指令微调前后回答对比流程

    参数含义:
        无参数

    返回值含义:
        无返回值，直接加载预训练模型和微调模型并打印回答对比
    """

    config = parse_args()
    validate_config(config)
    pretrained_path = Path(config.pretrained)
    finetuned_path = Path(config.finetuned)

    torch_module, mini_runtime = import_runtime_components()
    device = select_device(torch_module, config.device)
    pretrained_checkpoint = load_checkpoint(torch_module, pretrained_path, "预训练")
    finetuned_checkpoint = load_checkpoint(torch_module, finetuned_path, "微调")
    seed = resolve_seed(config, finetuned_checkpoint)
    set_seed(torch_module, seed)
    tokenizer = build_tokenizer(mini_runtime, finetuned_checkpoint)
    model_config = build_finetuned_model_config(mini_runtime, finetuned_checkpoint, len(tokenizer.itos))
    prompt_style, eos_token = resolve_generation_template(config, finetuned_checkpoint)
    eos_id = get_eos_id(tokenizer, eos_token)

    print("正在加载微调前模型并生成参照输出...")
    before_model, before_loaded, before_resized, before_skipped = load_model(
        mini_runtime,
        pretrained_checkpoint,
        model_config,
        device,
    )
    before_answer = generate_answer(
        mini_runtime=mini_runtime,
        model=before_model,
        tokenizer=tokenizer,
        instruction=config.instruction,
        input_text=config.input,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        eos_id=eos_id,
        prompt_style=prompt_style,
        device=device,
    )
    release_model(torch_module, before_model, device)
    before_model = None

    print("正在加载微调后模型并生成目标输出...")
    after_model, after_loaded, after_resized, after_skipped = load_model(
        mini_runtime,
        finetuned_checkpoint,
        model_config,
        device,
    )
    after_answer = generate_answer(
        mini_runtime=mini_runtime,
        model=after_model,
        tokenizer=tokenizer,
        instruction=config.instruction,
        input_text=config.input,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        eos_id=eos_id,
        prompt_style=prompt_style,
        device=device,
    )
    release_model(torch_module, after_model, device)
    after_model = None

    print()
    print_load_stats("微调前模型", before_loaded, before_resized, before_skipped)
    print_load_stats("微调后模型", after_loaded, after_resized, after_skipped)
    print()
    print_comparison(
        config=config,
        pretrained_path=pretrained_path,
        finetuned_path=finetuned_path,
        device=device,
        prompt_style=prompt_style,
        eos_token=eos_token,
        seed=seed,
        before_answer=before_answer,
        after_answer=after_answer,
    )


if __name__ == "__main__":
    main()
