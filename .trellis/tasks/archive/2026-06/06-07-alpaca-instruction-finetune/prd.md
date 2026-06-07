# Alpaca MiniGPT 指令微调

## Goal

使用 `data_instruction/alpaca_gpt4_data_zh.json` 对 `outputs_true_pretrain_tinystories/mini_gpt_pretrained.pt` 中的 MiniGPT 预训练模型进行指令微调，让第 4 步 MiniGPT 微调入口默认加载这次真实 TinyStories-Zh 预训练产物，并保存新的微调结果用于后续推理对比和报告证据。

## What I Already Know

* 用户指定数据集为 `data_instruction/alpaca_gpt4_data_zh.json`
* 用户指定预训练模型目录为 `outputs_true_pretrain_tinystories/`
* 该目录包含 `mini_gpt_pretrained.pt`、`char_tokenizer.json`、`true_pretrain_config.json`、`loss_log.csv` 和 `sample_outputs.txt`
* `true_pretrain_config.json` 显示预训练模型结构为 `context_length=192`、`emb_dim=256`、`n_heads=8`、`n_layers=6`、`vocab_size=2626`
* `mini.py` 已经是 MiniGPT 指令微调入口，支持加载 MiniGPT checkpoint、读取 Alpaca JSON、扩展字符级 tokenizer、保存 `mini_gpt_instruction_finetuned.pt`
* `mini_gpt_step4_instruction_finetune-gpt_chinese.py` 当前是 Qwen2.5 LoRA 指令微调入口，不适合直接加载 MiniGPT checkpoint
* `mini_gpt_instruction_compare.py` 已经支持对比 MiniGPT 预训练模型和指令微调模型，但默认路径仍指向旧的 `outputs_pretrain` 和 `outputs_instruction_finetune`

## Assumptions

* 本任务应沿用 MiniGPT 路线，而不是 Qwen LoRA 路线
* 默认训练入口应避免覆盖已有 `outputs_instruction_finetune_v2/`，使用新的输出目录保存这次 Alpaca 微调产物
* 当前 Codex 环境不一定有完整 ML 运行库，验证以语法检查和 CLI 帮助为主，真实训练由用户虚拟环境执行

## Requirements

* MiniGPT 指令微调入口默认从 `outputs_true_pretrain_tinystories/mini_gpt_pretrained.pt` 加载预训练模型
* 默认指令数据使用 `data_instruction/alpaca_gpt4_data_zh.json`
* 默认输出目录使用新的、可区分本次真实预训练来源的目录
* 默认训练配置应适配 Alpaca 输入条件生成任务，避免短答强化样本压过原始 Alpaca 样本
* 推理对比工具默认路径应与新的预训练和微调产物保持一致
* 保持 JSON / 文本 artifact UTF-8 写入和现有 Windows 路径习惯
* 不安装缺失 ML 依赖，不强制运行长训练

## Acceptance Criteria

* [x] `python -m py_compile mini.py mini_gpt_instruction_compare.py` 通过
* [x] `python mini_gpt_instruction_compare.py --help` 可正常展示 CLI
* [x] `mini.py` 默认指令数据、预训练 checkpoint 和输出目录已静态核对
* [x] `mini.py` 默认关闭短答 focus 混入，默认不截断 Alpaca 输出
* [x] 用户可在虚拟环境中直接运行 `python mini.py` 执行 Alpaca 指令微调
* [x] 微调产物默认保存到新的输出目录，不覆盖旧实验结果
* [ ] `python mini.py --help` 可在安装 `torch` 的虚拟环境中展示 CLI；当前 Codex 环境缺少 `torch`，入口会在 argparse 前导入失败

## Definition of Done

* 相关 Python 文件修改完成
* 最小语法和 CLI smoke 检查完成
* 不回滚用户已有未提交改动
* 如发现新的长期约定，再评估是否更新 `.trellis/spec/`

## Out of Scope

* 不运行完整长时间训练
* 不改 Qwen LoRA 微调脚本
* 不新增专用封装脚本
* 不下载或安装模型依赖
* 不重构项目目录结构

## Decision (ADR-lite)

**Context**: `mini.py` 已经具备 MiniGPT 指令微调所需的 checkpoint 加载、Alpaca JSON 读取、字符词表扩展、监督掩码训练和产物保存逻辑；用户本次明确指定的是 MiniGPT 预训练输出目录和 Alpaca 数据集

**Decision**: 复用 `mini.py` 作为训练入口，只更新默认预训练 checkpoint、默认输出目录和 `mini_gpt_instruction_compare.py` 的默认对比路径，不新增专用封装脚本

**Consequences**: 用户可以直接运行 `python mini.py` 开始本次 Alpaca 指令微调；旧实验目录不会被覆盖；后续如需多组实验，仍可通过 CLI 参数覆盖路径

## Follow-up Diagnosis

用户使用指令 `根据输入生成一段故事` 和输入 `我来自海外` 对比时，旧微调模型输出了泛化拒答模板，没有利用输入。诊断结论是旧默认配置更适合课程短答问答，不适合通用 Alpaca 条件生成：`focus_repeat=120` 会把 `minigpt_focus_short_zh.json` 的短答样本压过原始 Alpaca 任务，`max_output_chars=36` 又会截断故事、总结、改写等长回答样本。

已调整默认配置：`focus_data=""`、`focus_repeat=0`、`focus_augment=False`、`max_output_chars=0`，并把训练前后样例改为 `根据输入生成一段故事` / `我来自海外`。该调整需要重新运行 `python mini.py` 生成新的微调 checkpoint，旧 checkpoint 不会自动改变。

用户随后反馈 `outputs_true_pretrain_alpaca_instruction_finetune/` 效果仍不好。已检查该目录中的 `finetune_config.json`、`loss_log.csv` 和 `sample_outputs.txt`：训练损失从 6.70 降到 3.18，验证损失从 7.10 降到 3.32，但生成样例仍围绕“海洋/生活”泛化重复，没有稳定利用输入。这说明训练链路能下降 loss，但小字符级 MiniGPT 无法有效吸收全量 48,818 条杂任务 Alpaca 数据。

用户进一步指出验证集 loss 本身仍然很大，并询问训练样本是否太少。重新量化后确认：仅使用 `story_input` 会把训练目标压缩到约 135 条，训练集约 121 条、验证集约 14 条，不足以支撑 6 层字符级 MiniGPT 做全参数故事指令微调；训练 loss 很快接近 0 更像记忆训练样本，不能代表泛化。

已将 `mini.py` 默认改为更宽的同任务筛选 `data_filter="story_generation"`，排除采访问题、文学元素列表、标题、总结、分析等非故事生成任务，默认输出目录改为 `outputs_true_pretrain_alpaca_story_generation_finetune`。静态统计显示新默认筛选保留 1,154 条故事生成样本，按 0.9 切分后训练 1,038 条、验证 116 条；训练集中 117 条带 input 样本默认额外重复 2 次，因此默认混合训练样本约 1,272 条，验证集不重复。`story_input` 仍保留为 CLI 对照筛选策略，旧目录保留作为失败对照。后续如需全量 Alpaca 对照，可运行 `python mini.py --data-filter none --output-dir outputs_true_pretrain_alpaca_instruction_finetune_full`。

同时已补充早停和最佳验证权重恢复：`loss_log.csv` 现在记录 `best_valid_loss`、`bad_eval_count`、`is_best`、`early_stopped`，训练结束保存 checkpoint 前会恢复验证集最佳 step 的权重，避免保存末尾过拟合权重。

## Technical Notes

* 已读取 `.trellis/spec/python/index.md`、`directory-structure.md`、`error-handling.md`、`logging-guidelines.md`、`quality-guidelines.md`
* 已读取 `llm-coursework-loop` 技能，采用训练路径检查和证据留存思路
* `mini.py` 已包含 `load_pretrained_checkpoint`、`tokenizer_from_checkpoint`、`expand_tokenizer`、`load_state_dict_with_resize` 等适配 MiniGPT checkpoint 的函数
* `mini.py` 当前默认 `pretrained=outputs_true_pretrain_tinystories/mini_gpt_pretrained.pt`、`data_filter=story_generation`、`output_dir=outputs_true_pretrain_alpaca_story_generation_finetune`
* `mini_gpt_instruction_compare.py` 当前默认 `DEFAULT_PRETRAINED_CHECKPOINT=outputs_true_pretrain_tinystories/mini_gpt_pretrained.pt`、`DEFAULT_FINETUNED_CHECKPOINT=outputs_true_pretrain_alpaca_story_generation_finetune/mini_gpt_instruction_finetuned.pt`

## Verification Notes

* 已运行 `rtk python -m py_compile mini.py mini_gpt_instruction_compare.py`，结果通过
* 已运行 `rtk python mini_gpt_instruction_compare.py --help`，结果通过
* 已运行 `rtk python mini.py --help`，当前 Codex 环境因 `ModuleNotFoundError: No module named 'torch'` 失败；按项目说明，用户虚拟环境提供训练依赖
* 已静态核对 `mini.py` 与 `mini_gpt_instruction_compare.py` 的默认路径
* 已更新 `.trellis/spec/python/quality-guidelines.md`，记录通用 Alpaca 输入条件生成不应默认启用短答 focus 和全局输出截断
* 已运行 `rtk python -m py_compile mini.py mini_short_reply.py mini_gpt_instruction_compare.py`，结果通过
* 已静态核对 `story_generation` 筛选规则，当前数据集会筛出 1,154 条故事生成样本，默认混合训练样本约 1,272 条
* 已静态核对 `story_input` 筛选规则，当前数据集会筛出约 129 条带输入的故事生成样本，仅作为小样本对照
