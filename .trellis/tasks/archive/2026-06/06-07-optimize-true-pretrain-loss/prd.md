# 优化 true pretrain loss

## Goal

优化 `mini_gpt_true_pretrain.py` 的真实语料预训练默认策略，让从头预训练时默认使用 TinyStories-Zh 语料，并保留中文维基和混合比例作为显式对照入口

## Requirements

* 默认从 `data_pretrain/true_pretrain` 中只抽取 TinyStories-Zh 语料，不再默认混入中文维基
* 仍保留 `--source wiki` 与 `--source ratio --mix-ratio ...`，需要时可以显式启用中文维基或混合语料
* 支持从头训练，默认不需要兼容旧的中文维基或混合 checkpoint
* 增加或调整能降低 loss 噪声的训练前处理，优先处理字符级长尾词表问题
* 保持脚本的 Windows 本地训练体验，中文文件读写显式使用 UTF-8
* 保持 `--help` 可用，不要求当前 Codex 环境安装完整训练依赖

## Acceptance Criteria

* [x] `python mini_gpt_true_pretrain.py --help` 能看到默认 TinyStories-Zh 语料策略和相关参数
* [x] 默认配置的 `source` 为 `tinystories`，默认 ratio 不包含中文维基
* [x] 从头训练时使用新的 TinyStories-Zh 默认配置，无需指定 `--init-checkpoint`
* [x] 低频字符可按阈值映射到 `<UNK>`，默认阈值能减少字符级词表长尾
* [x] `python -m py_compile mini_gpt_true_pretrain.py` 通过
* [ ] 尽可能运行小规模 `--dry-run` 验证数据抽取、分词器和模型配置；当前 Codex 环境缺少 `torch`，需在你的虚拟环境中执行

## Definition of Done

* Python 规范已检查
* 脚本改动范围集中在真实预训练流程
* 语法和 CLI 检查通过
* 若当前环境缺少训练依赖，明确说明未能执行的验证项

## Technical Approach

将默认 `--source` 改为 `tinystories`，将默认 `--mix-ratio` 改为 `tinystories=1.0`，使中文维基只在用户显式选择时参与。保留 `--min-char-frequency` 参数，并用本脚本内的 tokenizer 构建函数按字符频次过滤低频字符，减少字符级 LM 在长尾字符上的损失噪声

## Decision (ADR-lite)

**Context**: Wiki-only 输出的验证 loss 高于旧混合训练，且用户希望改为使用 TinyStories-Zh 进行训练

**Decision**: 默认从头进行 TinyStories-Zh 预训练，并默认合并只出现一次的低频字符

**Consequences**: 新默认配置不再直接兼容旧 Wiki-only 或混合语料 checkpoint 的 tokenizer；如需复现实验或继续旧权重，用户需要显式传入旧数据策略并把低频字符阈值设回 1

## Out of Scope

* 不运行完整 12,000 step 长训练
* 不重构 MiniGPT 模型结构
* 不改动 Qwen 微调或推理脚本
* 不提交生成的模型权重或输出目录

## Technical Notes

* 相关脚本: `mini_gpt_true_pretrain.py`
* 已查看输出: `outputs_true_pretrain_continue/true_pretrain_config.json`、`outputs_true_pretrain_continue/loss_log.csv`、`outputs_true_pretrain_continue/sample_outputs.txt`
* 已读规范: `.trellis/spec/python/index.md`、`directory-structure.md`、`error-handling.md`、`logging-guidelines.md`、`quality-guidelines.md`、`database-guidelines.md`
