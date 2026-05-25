# Bootstrap Task: Fill Project Development Guidelines

**You (the AI) are running this task. The developer does not read this file.**

The developer ran `trellis init` on this project for the first time. `.trellis/`
now exists with spec scaffolding, and this bootstrap task exists under
`.trellis/tasks/`.

**Your job**: populate `.trellis/spec/` with the team's real coding
conventions. Every future AI coding task can load these spec files so agents
match the project's actual patterns instead of writing generic code.

---

## Status

- [x] Replace the generic fullstack scaffold with a Python ML script spec layer
- [x] Remove the non-applicable frontend spec layer
- [x] Add source-backed code examples and anti-patterns

---

## Spec files populated

### Python guidelines

| File | What it documents |
|------|-------------------|
| `.trellis/spec/python/index.md` | Python ML script layer overview, reading order, and checks |
| `.trellis/spec/python/directory-structure.md` | Script, data, model, and output directory conventions |
| `.trellis/spec/python/database-guidelines.md` | File-based persistence, data input, and artifact output conventions |
| `.trellis/spec/python/error-handling.md` | CLI validation, dependency checks, model/data loading errors |
| `.trellis/spec/python/logging-guidelines.md` | Console progress output and JSON/CSV artifact logs |
| `.trellis/spec/python/quality-guidelines.md` | Python style, documentation, CLI contracts, and review checks |

### Thinking guides

`.trellis/spec/guides/` contains the shared Trellis thinking guides. They are
kept as generic guides and excluded from layer detection.

---

## Evidence used

The spec is based on the current repository shape and real source examples:

- `mini_gpt_step2.py`
- `mini_gpt_step3_pretrain.py`
- `mini_gpt_step4_instruction_finetune-gpt_chinese.py`
- `qwen_instruction_infer.py`
- `qwen_lora_compare.py`
- `AGENTS.md`
- `.gitignore`

---

## Completion

When the spec contains real examples and no template filler, archive this task:

```bash
python ./.trellis/scripts/task.py archive 00-bootstrap-guidelines
```

After archive, every new developer who joins this project will get a
`00-join-<slug>` onboarding task instead of this bootstrap task.
