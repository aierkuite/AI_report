# 修复 Qwen LoRA 小规模训练

## Goal

修复 `mini_gpt_step4_instruction_finetune-gpt_chinese.py` 中导致 Qwen 指令微调输出破碎和重复的训练样本构造问题，并提供一个小规模训练预设，便于先快速观察修复后的效果。

## What I already know

* 用户使用该脚本训练 Qwen 后，训练后回答出现语序破碎、重复坍塌和无法正常结束的问题
* 脚本当前在构造 `labels` 时手动右移，而 Hugging Face causal LM 在计算 loss 时通常会内部执行 shift
* 当前默认训练配置较重，`max_steps=6000` 且 `generate_tokens=128`，不利于快速验证问题是否修复

## Requirements

* 修复监督微调样本的 `input_ids` 和 `labels` 对齐方式
* 保留 prompt 区域 label mask，只训练 assistant 回答区域
* 增加一个小规模训练入口，方便用户先验证效果
* 小规模训练默认写入独立输出目录，避免覆盖全量训练结果
* 保持现有命令行参数兼容

## Acceptance Criteria

* [ ] 脚本能通过 Python 语法编译检查
* [ ] `--quick-test` 能自动应用小规模训练参数
* [ ] 小规模训练产物保存到独立目录
* [ ] 样例输出不再出现明显的长串重复坍塌

## Out of Scope

* 不更换基础模型
* 不重写完整训练框架
* 不引入新的训练依赖

## Technical Notes

* 主要修改文件：`mini_gpt_step4_instruction_finetune-gpt_chinese.py`
* 项目要求：Python 文件中文注释，新增函数需说明总体作用、参数含义和返回值含义
