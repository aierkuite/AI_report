# Logging Guidelines

> Console output and artifact logging conventions for local Python ML scripts.

---

## Overview

The project currently uses direct `print()` calls for human-readable CLI progress and writes structured run artifacts to JSON / CSV files. It does not use Python's `logging` module, structured log aggregation, or a remote observability service.

This is appropriate for the current local training and inference workflows. Do not add a logging framework unless a task introduces long-running service code or repeated logging needs that `print()` cannot handle clearly.

---

## Console Output

Use `print()` for:

- Selected configuration
- Runtime device
- Data source counts
- Model parameter counts
- Train / validation loss checkpoints
- Before / after generation samples
- Output directory or file locations

Reference files:

- `mini_gpt_step3_pretrain.py` prints pretraining config, device, corpus statistics, parameter counts, generation samples, step losses, and final output directory
- `mini_gpt_step4_instruction_finetune-gpt_chinese.py` prints LoRA config, data statistics, effective batch size, scheduler settings, generation samples, and final output directory
- `qwen_lora_compare.py` prints instruction, optional input, base model answer, and LoRA adapter answer

---

## Artifact Logs

Training scripts persist important run state outside the console:

- JSON config and tokenizer metadata
- CSV loss logs
- Data statistics
- Sample outputs before and after training
- Model checkpoints or LoRA adapter directories

Use UTF-8, `ensure_ascii=False` for JSON, and CRLF line endings for new Windows-authored text artifacts.

---

## Log Levels

There are no formal log levels in the current codebase. Keep console messages plain and operational.

If a future change introduces `logging`, define level semantics in this file before using it broadly.

---

## What To Log

Print enough context for a local experiment to be reproducible:

- CLI config object or key parameter values
- File and directory paths used for model, data, and outputs
- Counts: samples, files, tokens, train / validation split sizes, parameter totals
- Training progress at configured evaluation intervals
- Final save location

---

## What Not To Log

- Do not print entire large datasets or model tensors
- Do not dump full model weights, adapter matrices, or tokenizer internals to the console
- Do not print secrets, tokens, API keys, or private credentials if future scripts add remote downloads
- Do not add noisy debug prints that obscure the training progress output
