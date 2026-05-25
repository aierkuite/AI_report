# Error Handling

> How Python CLI scripts validate inputs, report missing dependencies, and fail safely.

---

## Overview

The project uses standard Python exceptions at script boundaries. There is no API error response format and no custom exception hierarchy.

Use:

- `ValueError` for invalid user input, unsupported config values, malformed data records, or tokenizer states
- `FileNotFoundError` for missing input files, model directories, or adapter directories
- `RuntimeError` for unavailable runtime dependencies, unavailable CUDA, or failed model loading

Reference files:

- `mini_gpt_step2.py` raises `ValueError` when `emb_dim` is not divisible by `n_heads` or input sequence length exceeds `context_length`
- `mini_gpt_step3_pretrain.py` raises `RuntimeError` when CUDA is requested but unavailable and `ValueError` when token data is too short
- `mini_gpt_step4_instruction_finetune-gpt_chinese.py` validates instruction records, template style, LoRA target modules, dtype, scheduler config, and generation parameters
- `qwen_lora_compare.py` wraps missing `torch`, `transformers`, and `peft` imports in dependency-specific `RuntimeError`

---

## Dependency Handling

Do not assume every ML dependency is installed in the agent's current environment. The developer runs generated `.py` programs inside a virtual environment.

For local inspection tools that should support `--help` without importing large ML libraries, use lazy imports:

- `qwen_lora_compare.py` defines `import_torch`, `import_transformers`, and `import_peft_model`
- Each helper catches `ImportError` and raises `RuntimeError` naming the missing dependency

For training scripts where imports are part of normal execution, direct top-level imports are acceptable, as shown in `mini_gpt_step3_pretrain.py` and `mini_gpt_step4_instruction_finetune-gpt_chinese.py`.

---

## CLI Validation

Parse command-line arguments with `argparse`, then convert raw arguments into a dataclass config or validated local variables before work begins.

Local patterns:

- `mini_gpt_step3_pretrain.py` returns `PretrainConfig` from `parse_args`
- `mini_gpt_step4_instruction_finetune-gpt_chinese.py` returns `FinetuneConfig` from `parse_args` and validates it with `validate_config`
- `qwen_lora_compare.py` resolves positional and named instruction fields before inference begins

Validate early:

- `--device cuda` must fail with `RuntimeError` if CUDA is unavailable
- Empty required instruction text must fail with `ValueError`
- Numeric training and generation settings must be checked before allocating models or datasets

---

## File And Model Loading

Check paths before loading or writing expensive artifacts:

- Missing corpus or input files -> `FileNotFoundError`
- Missing local model directory -> `FileNotFoundError`
- Missing LoRA adapter directory -> `FileNotFoundError`
- Unsupported instruction data suffix -> `ValueError`

When re-raising model loading errors, preserve the original exception with `raise ... from exc` and include the likely cause in the message, as done in `mini_gpt_step4_instruction_finetune-gpt_chinese.py`.

---

## API Error Responses

Not applicable. This repository currently has no HTTP API or service boundary.

If an API layer is added later, create a new spec or update this one with the actual response contract after the API exists.

---

## Common Mistakes

- Do not catch broad exceptions just to print a message and continue training
- Do not silently fall back from CUDA to CPU when the user explicitly requested `--device cuda`
- Do not let missing `instruction`, missing model directories, or malformed JSON records reach the training / generation loop
- Do not use current-environment import failures as proof that a script is invalid; verify syntax and CLI behavior separately when dependencies are intentionally virtualenv-provided
