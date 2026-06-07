# Quality Guidelines

> Code quality standards for Python ML scripts in this repository.

---

## Overview

The project favors readable, standalone Python scripts with explicit CLI parameters, typed helper functions, dataclass configs, Chinese documentation comments / docstrings, and deterministic validation before expensive ML work begins.

There is currently no configured formatter, linter, type checker, or unit test suite. Use syntax checks, CLI help checks, and targeted function-level checks as the default verification baseline.

---

## Required Patterns

- Start Python files with a module docstring that explains the script purpose
- Use `from __future__ import annotations` in new or substantially rewritten scripts
- Use `argparse` for CLI scripts
- Use `@dataclass` for cohesive training / inference configuration objects
- Use `pathlib.Path` for filesystem paths
- Use explicit UTF-8 for all Chinese file reads and writes
- Keep new or rewritten files UTF-8 without BOM and CRLF on Windows
- Keep code comments and docstrings in Chinese
- For newly added functions or methods, document the overall purpose, parameter meanings, and return value meanings
- Keep comments and Chinese docstring explanatory lines without a trailing Chinese full stop when they are written as comments
- Use `torch.no_grad()` for inference / evaluation helpers
- Use `model.eval()` for inference and evaluation paths
- Set random seeds for training workflows where reproducibility matters
- 给 Hugging Face `AutoModelForCausalLM(..., labels=labels)` 传入监督微调样本时，不要在数据集里手动右移 `labels`，模型内部会执行 causal LM shift，数据集只需要让 `input_ids` 和 `labels` 对齐并把非监督区域设为 `-100`

Reference files:

- `mini_gpt_step2.py` demonstrates typed PyTorch modules, a `GPTConfig` dataclass, shape validation, and generation under `@torch.no_grad()`
- `mini_gpt_step3_pretrain.py` demonstrates dataclass config parsing, seed setup, UTF-8 corpus loading, train / validation split, gradient clipping, and checkpoint saving
- `mini_gpt_step4_instruction_finetune-gpt_chinese.py` demonstrates rich config validation, JSON / JSONL data validation, tokenizer special-token handling, scheduler handling, LoRA application, and artifact saving
- `qwen_lora_compare.py` demonstrates lazy dependency imports and a local CLI comparison contract

---

## Forbidden Patterns

- Do not rely on the system default encoding when reading or writing Chinese text
- Do not commit generated model weights, downloaded model directories, `outputs_*`, `models/`, logs, or `__pycache__/`
- Do not add an `--output` argument to `qwen_lora_compare.py`; model answers are generated outputs, not expected-answer inputs
- Do not silently ignore malformed training records
- Do not silently fall back from explicitly requested CUDA to CPU
- Do not introduce a web frontend, service framework, ORM, or package structure unless the task asks for that architectural change
- Do not install or vendor missing ML packages as part of normal code edits; the developer's virtual environment supplies runtime dependencies

---

## CLI Contracts

All CLI scripts should support `--help` through `argparse`.

Prefer Python 3.8-compatible `argparse` actions such as `store_true` and `store_false` for boolean switches. Do not use `argparse.BooleanOptionalAction` unless the project runtime is explicitly guaranteed to be Python 3.9 or newer.

For local inspection tools, keep help available without requiring heavy model loads. `qwen_lora_compare.py` does this by lazily importing `torch`, `transformers`, and `peft`.

The current Qwen LoRA comparison contract is:

- Positional form: `python qwen_lora_compare.py [instruction] [input]`
- Option form: `python qwen_lora_compare.py --instruction <text> --input <text>`
- Runtime flags include model / adapter paths, device, generation length, sampling parameters, repetition penalty, and system prompt
- `instruction` is required before inference; `input` is optional
- Prompt construction must use the Qwen chat template
- Decode only newly generated tokens
- Do not print chat template markers such as `<|im_start|>` or `<|im_end|>`
- Print the instruction, input, base-model answer, and LoRA-adapter answer

---

## Testing Requirements

Use the lowest-cost checks that match the change:

- Syntax: `python -m py_compile <changed .py files>`
- CLI shape: `python <script>.py --help`
- Argument parsing: inspect `parse_args()` behavior when changing CLI fields
- Encoding: verify changed Python files are UTF-8 without BOM and CRLF when the task rewrites files
- Data loaders: test malformed records directly when changing JSON / JSONL parsing
- Generation helpers: test prompt construction and token slicing separately from full model loading when possible
- Hugging Face causal LM 微调样本：检查 `input_ids` 与 `labels` 长度一致，prompt 区域为 `-100`，assistant 回答区域保留原 token，不做额外右移

Long training runs and real model inference are not required for every code edit. Prefer smoke checks that do not download models or require large local artifacts unless the task specifically targets training behavior.

---

## MiniGPT Instruction Tuning Checks

When improving `mini.py` instruction fine-tuning quality, check the full training and generation boundary before adding more steps:

- Supervised samples should append an explicit answer-end token and mask only the prompt region with `-100`
- Generation should stop on the answer-end token and strip repeated prompt markers from the decoded answer
- For tiny character-level models, prefer compact Chinese prompt templates and short target answers over long Alpaca-style prompts when the task is short-answer coursework
- For broad Alpaca-style instruction tuning with input-conditioned generation, keep focused short-answer mixing disabled by default and avoid global output truncation; otherwise the model can learn generic short replies or refusal templates instead of using the `input` field
- For Alpaca story generation on MiniGPT, do not make a tiny input-only subset the default training set; prefer a broader same-task `story_generation` filter and, if needed, repeat input-conditioned training samples while keeping validation unrepeated
- If a broad instruction dataset underfits target questions, add a small focused UTF-8 JSON / JSONL dataset under `data_instruction/`, expand equivalent phrasings for the same concept, and mix it into training with an explicit repeat count, while keeping validation mixed only once
- When focused data starts to behave like memorized answers, lower the repeat count before adding more training steps
- Decode with conservative defaults for coursework comparisons, then use greedy decoding such as `--temperature 0 --top-k 0` when checking whether training learned a stable answer

---

## MiniGPT True Pretraining Checks

When improving `mini_gpt_true_pretrain.py`, treat the data-source defaults and tokenizer settings as part of the CLI contract:

### 1. Scope / Trigger

- Trigger: changing pretraining loss behavior, data source defaults, tokenizer vocabulary construction, or saved training metadata

### 2. Signatures

- CLI entrypoint: `python mini_gpt_true_pretrain.py [options]`
- Data-source flags: `--source {ratio,wiki,tinystories,json,parquet,text,all}` and `--mix-ratio <source=ratio,...>`
- Tokenizer flag: `--min-char-frequency <int>`

### 3. Contracts

- Default pretraining source should remain TinyStories-Zh unless a task explicitly asks to use Wiki or mixed corpora
- Wiki may be used through `--source wiki` or `--source ratio --mix-ratio tinystories=...,wiki=...`
- `--min-char-frequency` controls the minimum corpus frequency required for a character to enter the vocabulary; lower-frequency characters map to `<UNK>`
- Saved config should include tokenizer statistics when tokenizer construction changes

### 4. Validation & Error Matrix

- `min_char_frequency <= 0` -> `ValueError`
- Missing selected data source files -> `FileNotFoundError`
- Ratio source with no available selected files -> `FileNotFoundError`
- Unsupported ratio key -> `ValueError`

### 5. Good/Base/Bad Cases

- Good: TinyStories-Zh from-scratch training with low-frequency characters mapped to `<UNK>`
- Base: explicit ratio mix for experiments that intentionally compare Wiki with TinyStories-Zh
- Bad: silently adding Wiki to the default run when optimizing TinyStories-Zh pretraining loss

### 6. Tests Required

- Run `python -m py_compile mini_gpt_true_pretrain.py`
- Run `python mini_gpt_true_pretrain.py --help` and verify the data-source and tokenizer flags are visible
- Run a small `--dry-run` when the current environment has the required ML dependencies

### 7. Wrong vs Correct

#### Wrong

```bash
python mini_gpt_true_pretrain.py
# 默认隐式混合 TinyStories-Zh 和 Wiki，导致训练目标和验证分布混杂
```

#### Correct

```bash
python mini_gpt_true_pretrain.py
# 默认 TinyStories-Zh，从头预训练时保留一致的故事语料分布
```

---

## Qwen Classification Fine-Tuning Checks

When adding or changing Qwen classification fine-tuning scripts, check that the classifier is a true sequence-classification workflow rather than a generation prompt workaround:

- Use `AutoModelForSequenceClassification` for Qwen classification fine-tuning
- For LoRA classification, use `TaskType.SEQ_CLS` and save the classification head with `modules_to_save=["score"]`
- `waimai_10k.csv` should work directly with `label,review`; normalize `0/1` labels to human-readable `差评/好评`
- Build and save a deterministic `label_to_id` / `id_to_label` mapping with the classification artifacts
- Save classification metrics such as train / validation loss and accuracy, data stats, plus sample predictions before and after training
- Keep `--help` usable in the current agent environment when possible, especially if the script can avoid importing heavy ML dependencies before argument help exits

---

## Code Review Checklist

Reviewers should check:

- Does the script still run from the repository root with relative default paths?
- Are Chinese reads / writes explicit UTF-8?
- Are new functions documented with purpose, parameters, and return value?
- Are failure modes clear and raised before expensive model work begins?
- Are model / adapter / output directories consistent with existing names?
- Are generated artifacts still ignored by git?
- Are CLI flags backwards compatible unless the task explicitly changes the interface?
- Are dependency imports placed so `--help` remains usable for lightweight tools?
- Are tensor shapes, device movement, and `eval()` / `no_grad()` usage correct?
