# AI_report

本仓库是一个课程式大模型实验项目，围绕 MiniGPT 从零实现、中文指令微调、Qwen2.5 LoRA 微调和外卖评论分类展开。

本文档按远程仓库当前已追踪文件整理，只说明远程仓库中实际存在的文件，不把本地运行后生成的模型、日志、checkpoint、输出目录写入文件清单。

## 远程仓库文件结构

```text
.
|-- .gitignore
|-- data_instruction/
|   |-- alpaca_gpt4_data_zh.json
|   |-- minigpt_focus_short_zh.json
|   `-- waimai_10k.csv
|-- mini.py
|-- mini_gpt_instruction_compare.py
|-- mini_gpt_step2.py
|-- mini_gpt_step3_pretrain.py
|-- mini_gpt_step4_instruction_finetune-gpt_chinese.py
|-- mini_gpt_true_pretrain.py
|-- mini_short_reply.py
|-- qwen_classification_finetune.py
|-- qwen_classification_infer.py
|-- qwen_instruction_infer.py
`-- qwen_lora_compare.py
```

## 数据文件说明

| 文件 | 作用 |
| --- | --- |
| `data_instruction/alpaca_gpt4_data_zh.json` | 中文 Alpaca/GPT4 风格指令数据，用于 MiniGPT 或 Qwen2.5 的指令微调实验 |
| `data_instruction/minigpt_focus_short_zh.json` | MiniGPT 短回答聚焦数据，用于强化课程问答类短输出能力 |
| `data_instruction/waimai_10k.csv` | 外卖评论情感分类数据，用于 Qwen2.5 二分类 LoRA 微调 |

## Python 脚本说明

| 文件 | 作用 |
| --- | --- |
| `mini_gpt_step2.py` | MiniGPT 模型结构验证脚本。定义 `GPTConfig`、因果自注意力、前馈网络、Transformer Block、`MiniGPT`、参数统计和随机生成函数，用于验证模型结构、前向传播和生成流程 |
| `mini_gpt_step3_pretrain.py` | MiniGPT 小规模预训练脚本。实现字符级分词器、语言模型数据集、训练循环、损失评估和 checkpoint 保存逻辑，为后续指令微调提供基础模型 |
| `mini_gpt_true_pretrain.py` | 真实中文语料 MiniGPT 预训练脚本。支持 TinyStories、Wiki、JSON、Parquet、文本和比例混合语料，包含低频字符过滤、学习率调度、早停和 dry-run 检查 |
| `mini_short_reply.py` | MiniGPT 短回答指令微调脚本。加载 MiniGPT 预训练权重，对中文指令数据做监督微调，并默认结合 `data_instruction/minigpt_focus_short_zh.json` 强化短回答表现 |
| `mini.py` | MiniGPT 故事生成方向指令微调脚本。加载 MiniGPT 预训练权重，默认从 `data_instruction/alpaca_gpt4_data_zh.json` 中筛选故事生成样本，训练后对比微调前后生成效果 |
| `mini_gpt_instruction_compare.py` | MiniGPT 指令微调前后回答对比脚本。输入一条指令和可选输入内容，分别加载预训练模型和指令微调模型，输出两者回答差异 |
| `mini_gpt_step4_instruction_finetune-gpt_chinese.py` | Qwen2.5 中文指令 LoRA 微调脚本。文件名保留了 MiniGPT step4 命名，但代码实际使用 Hugging Face/Qwen2.5 因果语言模型与 PEFT LoRA，对中文指令数据做监督微调 |
| `qwen_lora_compare.py` | Qwen2.5 LoRA 微调前后回答对比脚本。输入指令和可选输入，分别调用 base 模型和 LoRA adapter，比较两者回答 |
| `qwen_classification_finetune.py` | Qwen2.5 外卖评论情感分类 LoRA 微调脚本。使用 `AutoModelForSequenceClassification` 做二分类，读取 `data_instruction/waimai_10k.csv`，并保存标签映射、指标日志和样例预测 |
| `qwen_classification_infer.py` | Qwen2.5 外卖评论分类推理脚本。加载分类微调得到的 LoRA adapter 和标签映射，对单条评论或批量评论判断“差评/好评” |
| `qwen_instruction_infer.py` | 本地 MiniGPT 文本续写脚本。文件名包含 `qwen`，但代码实际加载 MiniGPT checkpoint，读取文本开头并继续生成内容 |

## 建议阅读顺序

MiniGPT 从零构建路线：

```bash
python mini_gpt_step2.py
python mini_gpt_step3_pretrain.py --help
python mini_gpt_true_pretrain.py --help
python mini_short_reply.py --help
python mini.py --help
python mini_gpt_instruction_compare.py --help
python qwen_instruction_infer.py --help
```

Qwen2.5 LoRA 路线：

```bash
python mini_gpt_step4_instruction_finetune-gpt_chinese.py --help
python qwen_lora_compare.py --help
python qwen_classification_finetune.py --help
python qwen_classification_infer.py --help
```

## 运行说明

远程仓库只包含代码和少量指令/分类数据，不包含模型权重、训练 checkpoint 或运行输出。训练和推理前，需要在自己的虚拟环境中准备脚本所需的机器学习依赖，并按脚本参数准备本地模型或上一步训练得到的权重。

若只是查看脚本参数，可以先运行：

```bash
python <script>.py --help
```
