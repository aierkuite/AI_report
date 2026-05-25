# Directory Structure

> How Python scripts, data, model files, and training artifacts are organized.

---

## Repository Shape

The project currently uses flat, task-oriented Python scripts rather than a `src/` package. New code should follow the existing script layout unless a task explicitly asks for a package refactor.

```text
.
|-- mini.py
|-- mini_gpt_step2.py
|-- mini_gpt_step3_pretrain.py
|-- mini_gpt_step4_instruction_finetune-gpt_chinese.py
|-- qwen_instruction_infer.py
|-- qwen_lora_compare.py
|-- data_pretrain/
|-- data_instruction/
|-- artical/
|-- models/
|-- outputs_pretrain/
|-- outputs_instruction_finetune/
`-- outputs_qwen_lora_finetune/
```

Reference files:

- `mini_gpt_step2.py` defines the reusable MiniGPT model architecture
- `mini_gpt_step3_pretrain.py` imports `GPTConfig`, `MiniGPT`, `count_parameters`, and `generate` from `mini_gpt_step2.py`
- `mini_gpt_step4_instruction_finetune-gpt_chinese.py` owns the Qwen LoRA instruction tuning workflow
- `qwen_instruction_infer.py` and `qwen_lora_compare.py` are standalone command-line inspection tools

---

## Module Organization

Keep each major experiment as one readable script with this local order:

1. Module docstring describing the script purpose
2. `from __future__ import annotations`
3. Standard-library imports
4. Third-party imports
5. Local imports
6. Constants and default examples
7. `@dataclass` configuration / record types
8. Dataset or model classes
9. Small helper functions
10. Training, inference, save, and `main()` orchestration functions
11. `if __name__ == "__main__": main()`

Examples:

- `mini_gpt_step3_pretrain.py` uses `PretrainConfig`, `CharTokenizer`, `LanguageModelDataset`, then helper functions such as `set_seed`, `select_device`, `load_corpus`, `train_model`, `save_checkpoint`, and `main`
- `mini_gpt_step4_instruction_finetune-gpt_chinese.py` uses separate dataclasses for `InstructionExample`, encoding statistics, and `FinetuneConfig`, then keeps parsing, loading, training, generation, and saving in separate functions
- `qwen_lora_compare.py` keeps dependency imports lazy in `import_torch`, `import_transformers`, and `import_peft_model` so help and argument parsing can run without all ML packages installed

---

## Data And Artifact Directories

Use these directories consistently:

- `data_pretrain/` for plain text pretraining corpora
- `data_instruction/` for JSON / JSONL instruction tuning data
- `models/` for local downloaded or cached model directories
- `outputs_pretrain/` for MiniGPT pretraining checkpoints, tokenizer JSON, config JSON, and loss logs
- `outputs_qwen_lora_finetune/` for LoRA adapters, finetune configs, loss logs, sample outputs, and data stats
- `artical/` for generated article / continuation text outputs

Heavy runtime directories are ignored in `.gitignore`. Do not commit model weights, generated checkpoints, cache directories, logs, or `__pycache__/`.

---

## Naming Conventions

- Keep script filenames descriptive and task-oriented, for example `qwen_lora_compare.py`
- Use `snake_case` for functions, variables, and Python filenames
- Use `PascalCase` for dataclasses and model/dataset classes, for example `PretrainConfig`, `InstructionDataset`, and `MiniGPT`
- Use uppercase constants for defaults, for example `DEFAULT_MODEL_DIR`, `DEFAULT_ADAPTER_DIR`, and `DEFAULT_SYSTEM_PROMPT`
- Keep command-line flags kebab-cased and map them to snake_case `argparse` fields, for example `--max-new-tokens` -> `args.max_new_tokens`

---

## When To Add A New File

Add a new top-level script when it represents a distinct course step, local workflow, or CLI inspection tool. Prefer extending an existing script when the change only adds an option to the same workflow.

Avoid introducing a package structure, service layer, or shared utilities module unless there are repeated helpers in at least two active scripts and the task explicitly benefits from extraction.
