# 改进 MiniGPT 指令微调生成稳定性

## Goal

修复 `mini.py` 指令微调后生成中文开头正常但后续字符发散的问题，让小型字符级 MiniGPT 的训练目标和生成边界更稳定，能够作为课程实验展示更可靠的微调前后对比。

## Requirements

* 指令微调样本需要显式学习回答结束边界
* 生成时需要在回答结束 token 或下一个提示段落标记处停止
* 生成结果应只展示完整提示词加回答内容，不输出隐藏结束 token
* 默认训练和生成参数需要更适合小型字符级模型，降低采样随机性
* 过长样本不能因为截断提示词而误导回答监督，应过滤无法容纳回答的样本
* 需要混入高质量短中文问答样本，提升小模型对课程常见问题和短回答格式的学习频率
* 需要为重点短回答样本自动生成等价问法，降低模型只记住固定指令句式的风险
* 重点短回答重复次数不能过高，避免模型在少量样本上形成背题式输出
* 重点短回答数据文件需要保持 UTF-8 无 BOM 和 CRLF，避免 Windows 中文读写与版本差异问题
* 保存的 checkpoint、tokenizer、配置、loss 日志和样例输出格式保持兼容

## Acceptance Criteria

* [x] `python -m py_compile mini.py mini_gpt_step2.py mini_gpt_step3_pretrain.py` 通过
* [ ] `python mini.py --help` 能正常展示新增或调整后的参数（当前代理环境缺少 `torch`，导入阶段提前失败，需要在训练虚拟环境中复核）
* [x] `build_supervised_sample` 生成的 labels 只监督回答区域并包含结束 token
* [x] `generate_instruction_answer` 能只解码回答区域并在结束 token 处停止
* [x] 默认运行参数相比原先更保守，减少长文本随机发散
* [x] 新增 `data_instruction/minigpt_focus_short_zh.json`，训练集按 `--focus-repeat` 重复混入，验证集混入一次
* [x] `--focus-augment` 默认开启，为重点样本生成多种等价问法
* [x] `--focus-augment` / `--no-focus-augment` 使用 Python 3.8 兼容的 `store_true` / `store_false`
* [x] 默认 `focus_repeat` 从 500 降到 120，配合问法扩增减少固定句式过拟合
* [x] `data_instruction/minigpt_focus_short_zh.json` 通过 JSON 结构检查，并保持 UTF-8 无 BOM + CRLF

## Definition of Done

* Python 代码保持现有扁平脚本结构
* 新增函数和方法使用中文 docstring 说明作用、参数和返回值
* 中文文件读写继续显式使用 UTF-8
* 不提交模型权重、输出目录或运行缓存

## Technical Approach

在不替换外部大模型、不改变课程脚本结构的前提下改进核心链路：为字符级 tokenizer 动态补齐独立 EOS token id，训练样本在标准回答后手动追加该 id，生成循环使用低随机性默认参数并支持 greedy 解码与 EOS 停止，最后只返回提示词和回答文本。对过长样本增加过滤，避免提示词截断后回答监督失真。新增短中文课程问答数据，并通过 `--focus-data`、`--focus-repeat` 和 `--focus-augment` 提高小模型对目标回答风格和等价问法的采样频率。

## Out of Scope

* 不把 MiniGPT 替换为 Hugging Face 或 Qwen 模型
* 不重新设计 BPE 或 SentencePiece tokenizer
* 不执行长时间完整训练
* 不修改已有预训练 checkpoint 内容

## Technical Notes

* 相关代码文件：`mini.py`、`mini_gpt_step2.py`、`mini_gpt_step3_pretrain.py`
* 失败现象：微调后能生成“人工智能（AI”开头，但后续混入英文字符碎片并长距离发散
* 根因方向：字符级小模型能力有限、缺少回答结束边界、采样温度和 top-k 偏随机、生成长度偏长、过长样本可能削弱监督质量、通用 Alpaca 数据对短中文课程问答的采样密度不足、少量重点样本容易造成固定句式背诵
* 新增重点数据：`data_instruction/minigpt_focus_short_zh.json`，包含短中文问答样本，默认 `focus_repeat=120`
* 问法扩增：默认开启 `--focus-augment`，把“什么是 X”“请解释 X”“请用一句话解释什么是 X”等短问法互相补齐
* 适用规范：`.trellis/spec/python/index.md` 及其 Python CLI、数据和 artifact 约定
