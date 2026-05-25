# Qwen 训练前后回答对比脚本

## Goal

新增独立 CLI 程序 `qwen_lora_compare.py`，输入一条指令和可选输入后分别调用本地 Qwen2.5-0.5B base 模型和 LoRA 微调 adapter，打印训练前与训练后回答，方便直观看微调效果差异。

## Requirements

* 新增独立脚本 `qwen_lora_compare.py`，不修改现有训练脚本。
* 默认 base 模型路径为 `models/qwen2.5-0.5b`。
* 默认 LoRA adapter 路径为 `outputs_qwen_lora_finetune/qwen2_5_0_5b_lora_finetuned`。
* 支持位置参数输入：`instruction input`，其中 `input` 可为空。
* 支持命令行参数：`--instruction`、`--input`、`--model-dir`、`--adapter-dir`、`--device`、`--max-new-tokens`、`--temperature`、`--top-p`、`--do-sample`、`--repetition-penalty`、`--system-prompt`。
* 使用 Qwen `apply_chat_template` 构造 system/user/assistant prompt。
* 先使用 base 模型生成训练前回答，再用 `PeftModel.from_pretrained(base_model, adapter_dir)` 加载 LoRA adapter 生成训练后回答。
* 只解码新生成 token，不打印 prompt 模板。
* 输出包含指令、输入、训练前回答、训练后回答、base 模型路径和 adapter 路径。

## Acceptance Criteria

* [ ] `python -m py_compile G:\人工智能\qwen_lora_compare.py` 通过。
* [ ] 脚本可用 `--instruction "请用一句话解释什么是人工智能" --input "" --device cpu --max-new-tokens 80` 运行。
* [ ] 脚本可用位置参数 `"请用一句话解释什么是人工智能" ""` 运行。
* [ ] 控制台输出包含“指令”、“输入”、“训练前回答”和“训练后回答”。
* [ ] 输出不包含 `<|im_start|>`、`<|im_end|>` 等模板 token。
* [ ] 文件为 UTF-8 无 BOM，并使用 CRLF 行分隔符。

## Definition of Done

* 新脚本实现完成。
* 语法检查通过。
* 文件编码和行尾符合仓库要求。
* 若当前环境缺少 `torch`、`transformers` 或 `peft`，需在交付说明中明确未能执行实际推理测试的原因。

## Technical Approach

* 复用现有 Qwen 推理和训练脚本中的设备选择、chat template prompt 构造和新 token 解码思路。
* 使用 `transformers.AutoTokenizer`、`transformers.AutoModelForCausalLM` 和 `peft.PeftModel`。
* 为避免同时常驻两份模型权重，先运行 base 推理，释放 base 模型，再重新加载 base 模型并挂载 LoRA adapter 运行微调后推理。

## Out of Scope

* 不做批量评测。
* 不合并 LoRA 权重。
* 不修改 `.trellis/spec/` 结构。
* 不修改现有训练脚本。

## Technical Notes

* 本地 base 模型目录已确认存在：`models/qwen2.5-0.5b`。
* LoRA adapter 目录已确认存在：`outputs_qwen_lora_finetune/qwen2_5_0_5b_lora_finetuned`。
* 现有参考文件：`qwen_instruction_infer.py`、`mini_gpt_step4_instruction_finetune-gpt_chinese.py`。
