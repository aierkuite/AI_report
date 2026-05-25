# Journal - aierkuitr (Part 1)

> AI development session journal
> Started: 2026-05-24

---



## Session 1: Qwen LoRA 回答对比脚本

**Date**: 2026-05-25
**Task**: Qwen LoRA 回答对比脚本
**Branch**: `main`

### Summary

新增 qwen_lora_compare.py，用于对比 Qwen2.5 base 模型与 LoRA adapter 在同一指令和输入下的回答；补充 CLI 合约到 Trellis spec，并完成质量检查、提交与推送。

### Main Changes

- Replaced the scaffolded backend/frontend spec split with a single `.trellis/spec/python/` layer that matches the actual Python ML script repository.
- Documented script layout, file-based persistence, CLI validation, dependency handling, console output, and quality checks with references to the existing Python scripts.
- Kept shared Trellis thinking guides under `.trellis/spec/guides/`.
- Archived `00-bootstrap-guidelines` after the spec work commit.

### Git Commits

| Hash | Message |
|------|---------|
| `4ff82d4` | (see git log) |

### Testing

- [OK] `get_context.py --mode packages` reports only `Spec layers: python`
- [OK] Template filler search returned no remaining scaffold markers in `.trellis/spec/`
- [OK] `task.py validate .trellis/tasks/00-bootstrap-guidelines` passed before archive
- [OK] `git diff --check` passed for the Trellis spec and task changes

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 2: Bootstrap Python Trellis Guidelines

**Date**: 2026-05-25
**Task**: Bootstrap Python Trellis Guidelines
**Branch**: `main`

### Summary

Filled Trellis specs for the Python ML script repository, replaced the scaffolded fullstack layers with a python layer, and archived 00-bootstrap-guidelines.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `a2d05fb` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 3: Qwen LoRA 微调修复与 sub-agent 切换

**Date**: 2026-05-26
**Task**: Qwen LoRA 微调修复与 sub-agent 切换
**Branch**: `main`

### Summary

修复 Qwen LoRA 指令微调样本 label 对齐，新增 quick-test 小规模训练预设，启用 Codex sub-agent dispatch，并补充相关 Python 质量规范。

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `0891714` | (see git log) |
| `3a7b289` | (see git log) |
| `251f30c` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
