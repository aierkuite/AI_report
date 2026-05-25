# Python Development Guidelines

> Project-specific guidance for Python machine-learning scripts in this repository.

---

## Overview

This repository is a single-package Python project for course-style LLM experiments:

- `mini.py`, `mini_gpt_step2.py`, `mini_gpt_step3_pretrain.py`, and `mini_gpt_step4_instruction_finetune-gpt_chinese.py` implement a staged MiniGPT and Qwen LoRA workflow
- `qwen_instruction_infer.py` and `qwen_lora_compare.py` are local inference / comparison CLI tools
- `data_pretrain/` and `data_instruction/` hold input corpora and instruction data
- `models/`, `outputs_pretrain/`, `outputs_instruction_finetune/`, and `outputs_qwen_lora_finetune/` are local heavyweight runtime artifacts and are git-ignored

There is no web server, API layer, database service, frontend application, package manager config, or test suite at the moment. Treat this spec layer as the Python script / ML workflow layer.

---

## Guidelines Index

| Guide | Description | Status |
|-------|-------------|--------|
| [Directory Structure](./directory-structure.md) | Python script, data, model, and output layout | Active |
| [Database Guidelines](./database-guidelines.md) | Current persistence model and data-file conventions | Active |
| [Error Handling](./error-handling.md) | CLI validation, dependency checks, and model/data error patterns | Active |
| [Quality Guidelines](./quality-guidelines.md) | Python style, documentation, verification, and review checklist | Active |
| [Logging Guidelines](./logging-guidelines.md) | Console progress output and artifact logs | Active |

---

## Pre-Development Checklist

Before modifying Python workflow code, read:

- `python/directory-structure.md`
- `python/error-handling.md`
- `python/logging-guidelines.md`
- `python/quality-guidelines.md`

Also read `python/database-guidelines.md` when the task touches data files, checkpoints, generated outputs, or training artifacts.

---

## Quality Check

For Python script changes, run the narrowest reliable checks:

- `python -m py_compile <changed .py files>`
- `<script> --help` for scripts that expose `argparse`
- Any direct smoke command that does not require downloading models or running long training

Do not install missing ML dependencies just to satisfy local checks. The developer runs generated `.py` programs inside a virtual environment, and missing packages in the current agent environment are not a project failure.
