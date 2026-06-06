# Data And Persistence Guidelines

> The repository does not use a database. Persistence is file-based: corpora, instruction datasets, model directories, checkpoints, JSON configs, CSV loss logs, and generated text files.

---

## Current Persistence Model

There is no ORM, migration system, SQL database, or transaction layer. Do not add database concepts to normal script work.

Use local files and directories:

- Pretraining text input lives under `data_pretrain/`
- Instruction tuning input lives under `data_instruction/`
- Local model directories live under `models/`
- Training and evaluation outputs live under `outputs_pretrain/`, `outputs_instruction_finetune/`, or `outputs_qwen_lora_finetune/`
- Qwen classification outputs live under `outputs_qwen_classification_finetune/`
- Generated article continuations live under `artical/`

Reference files:

- `mini_gpt_step3_pretrain.py` reads all `*.txt` files from `data_pretrain/` in `load_corpus`
- `mini_gpt_step4_instruction_finetune-gpt_chinese.py` reads JSON arrays and JSONL records in `load_json_instruction_file` and `load_jsonl_instruction_file`
- `qwen_instruction_infer.py` reads one UTF-8 text file and optionally writes a generated continuation file

---

## Text And JSON Reading

Always read Chinese text data with explicit UTF-8.

Local patterns:

- `qwen_instruction_infer.py` uses `file_path.read_text(encoding="utf-8").strip()`
- Training data loaders use `open(file_path, "r", encoding="utf-8", errors="replace")` when robustness against imperfect corpora matters
- JSON instruction files must be arrays of objects; JSONL files must have one object per non-empty line
- Classification fine-tuning files may be JSON arrays, JSONL records, or CSV tables. Each record must expose one non-empty text field from `text`, `review`, `content`, `input`, `sentence`, `句子`, `文本`, `评论` and one non-empty label field from `label`, `category`, `class`, `target`, `标签`, `类别`

When adding new data readers, validate the structure near the read boundary and include the source file path plus row / item number in error messages.

---

## Output Files

Use explicit UTF-8 for all generated text, JSON, and CSV files. On Windows, preserve CRLF for newly written text artifacts.

Local patterns:

- JSON files are written with `json.dump(..., ensure_ascii=False, indent=2)`
- Text outputs use `open(output_path, "w", encoding="utf-8", newline="\r\n")`
- CSV loss logs use `csv.DictWriter` with `lineterminator="\r\n"`
- Parent directories are created with `Path(...).parent.mkdir(parents=True, exist_ok=True)` or `output_dir.mkdir(parents=True, exist_ok=True)`

Do not rely on the platform default encoding for Chinese data or generated Chinese text.

---

## Model And Checkpoint Artifacts

Model and adapter outputs are local runtime artifacts, not source code.

Current examples:

- `mini_gpt_step3_pretrain.py` saves `mini_gpt_pretrained.pt`, tokenizer metadata, config JSON, and loss logs under `outputs_pretrain/`
- `mini_gpt_step4_instruction_finetune-gpt_chinese.py` saves LoRA adapter artifacts and training metadata under `outputs_qwen_lora_finetune/`
- `qwen_classification_finetune.py` saves a Qwen classification LoRA adapter, tokenizer metadata, classification config, label mapping, metrics CSV, data stats, and sample predictions under `outputs_qwen_classification_finetune/`
- `qwen_lora_compare.py` loads the base model from `models/qwen2.5-0.5b` and the adapter from `outputs_qwen_lora_finetune/qwen2_5_0_5b_lora_finetuned`

Keep these directories git-ignored. Do not add generated checkpoints, downloaded models, or adapter binaries to commits.

---

## Data Validation

Validate file-backed data before training or inference:

- Missing model or adapter directories should raise `FileNotFoundError`
- Missing input files should raise `FileNotFoundError`
- Unsupported file extensions should raise `ValueError`
- Instruction records must contain non-empty `instruction` and `output`; `input` may be empty
- Classification records must contain non-empty text and label fields; malformed records should include the source path and row / item number in the error
- `waimai_10k.csv` uses `label,review`; Qwen classification fine-tuning should accept this file directly and normalize labels `0/1` to `差评/好评`
- Tokenizer special tokens must be checked before generation or padding

Prefer deterministic validation failures over silent fallback when malformed data would make training results misleading.
