# 新建 Qwen2.5 分类微调程序

## Goal

新增并调整一个独立 Python 程序，用本地 `models/qwen2.5-0.5b` 对 `data_instruction/waimai_10k.csv` 外卖评论数据进行中文文本分类微调，产出可用于课程报告的分类训练日志、标签映射、LoRA 适配器和样例预测

## What I Already Know

* 用户要求改为适合 `G:\人工智能\models\qwen2.5-0.5b` 的分类微调
* 本地存在 `data_instruction/waimai_10k.csv`
* `waimai_10k.csv` 表头为 `label,review`，其中 `0/1` 可映射为 `差评/好评`
* 项目已有 Qwen LoRA 指令微调和比较脚本，依赖 `torch`、`transformers`、`peft`
* 新增/重写文件需使用 UTF-8 无 BOM 和 CRLF，中文读写需显式 UTF-8

## Requirements

* 新增主脚本 `qwen_classification_finetune.py`
* 保留 `mini_gpt_classification_finetune.py` 作为兼容入口，转调 Qwen 分类微调主逻辑
* 默认模型路径为 `models/qwen2.5-0.5b`
* 默认数据文件为 `data_instruction/waimai_10k.csv`
* 直接支持 `label,review` 字段，不要求用户手动改 CSV 表头
* 将外卖评论标签 `0/1` 规范化为中文类别 `差评/好评`
* 使用 `AutoModelForSequenceClassification` 执行真正的序列分类微调
* 默认使用 LoRA，任务类型为 `SEQ_CLS`，并保存分类头 `score`
* 支持训练/验证划分、交叉熵损失、验证准确率评估、梯度裁剪、样本数限制
* 保存 LoRA adapter、tokenizer、分类配置、标签映射、指标 CSV、数据统计和样例预测

## Acceptance Criteria

* [ ] `python -m py_compile qwen_classification_finetune.py mini_gpt_classification_finetune.py` 通过
* [ ] `python qwen_classification_finetune.py --help` 通过
* [ ] 脚本能在不加载模型的 smoke check 中读取 `data_instruction/waimai_10k.csv`
* [ ] 标签映射保持 `差评 -> 0`、`好评 -> 1`
* [ ] 默认路径与项目现有 `models/`、`data_instruction/`、`outputs_*` 约定一致
* [ ] 新增函数和方法包含中文文档注释，说明作用、参数和返回值

## Definition Of Done

* Qwen 分类微调脚本完成
* 数据读取 smoke check 通过
* 语法检查和 `--help` 检查通过
* 不触碰无关未提交改动
* 项目 Python 规范同步到 Qwen 分类微调约定

## Technical Approach

使用 `AutoModelForSequenceClassification.from_pretrained()` 加载本地 Qwen2.5-0.5B，并按 `label_to_id` 初始化二分类头。默认通过 PEFT `LoraConfig(task_type=TaskType.SEQ_CLS)` 挂载 LoRA，目标模块沿用 Qwen LoRA 指令微调常用的 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`，同时用 `modules_to_save=["score"]` 保存序列分类头。数据集按 tokenizer `max_length` 截断和 padding，训练时直接把 `input_ids`、`attention_mask`、`labels` 传给模型。

## Decision (ADR-lite)

**Context**: 最初任务是 MiniGPT 分类头微调，但用户后续指定本地 Qwen2.5-0.5B 和 `waimai_10k.csv`  
**Decision**: 主脚本改为 Qwen 序列分类 LoRA 微调，旧 MiniGPT 文件名保留为兼容入口  
**Consequences**: 实验效果更适合中文评论分类，训练依赖 `transformers` 和 `peft`；脚本不再使用 MiniGPT 字符级 checkpoint

## Out Of Scope

* 不下载外部模型
* 不运行长时间正式训练
* 不修改 `waimai_10k.csv` 原始表头
* 不改 Qwen 指令微调脚本

## Technical Notes

* 相关代码：`qwen_classification_finetune.py`、`mini_gpt_classification_finetune.py`、`qwen_lora_compare.py`、`mini_gpt_step4_instruction_finetune-gpt_chinese.py`
* 相关规范：`.trellis/spec/python/index.md`、`.trellis/spec/python/directory-structure.md`、`.trellis/spec/python/database-guidelines.md`、`.trellis/spec/python/quality-guidelines.md`
* 本次检查不跑长训练，只做语法、CLI、数据读取 smoke check
