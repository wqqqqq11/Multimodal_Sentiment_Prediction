# Problem 3 reproducible runbook

Use the project virtual environment from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/problem3 -q
.\.venv\Scripts\python.exe -m data_progressing.problem3_pipeline --config configs/problem3.yaml
.\.venv\Scripts\python.exe -m data_progressing.problem3_validate --config configs/problem3.yaml
```

`problem3_pipeline` is the recommended entry point. It uses the pinned fast tokenizer
when the complete tokenizer is present locally. If only model weights are cached, it
keeps the organizer token IDs unchanged and activates an audited low-confidence
proportional character mapping. It never downloads tokenizer files implicitly.

The default configuration refuses to overwrite an existing non-empty output directory.
For a deliberate rebuild, archive the previous Problem 3 result and set
`output.overwrite` to `true` in a reviewed configuration copy.

