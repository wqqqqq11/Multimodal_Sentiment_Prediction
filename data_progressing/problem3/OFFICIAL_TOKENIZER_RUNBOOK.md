# Official-tokenizer runbook

The competition-grade pipeline uses the official `google-bert/bert-base-uncased`
WordPiece tokenizer pinned to revision
`86b5e0934494bd15c9632b12f734a8a67f723594`.

First cache the tokenizer once with network access:

```powershell
.\.venv\Scripts\python.exe -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google-bert/bert-base-uncased', revision='86b5e0934494bd15c9632b12f734a8a67f723594', use_fast=True)"
```

Then all preprocessing is offline and reproducible:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/problem3 -q
.\.venv\Scripts\python.exe -m data_progressing.problem3_preprocess --config configs/problem3.yaml
.\.venv\Scripts\python.exe -m data_progressing.problem3_validate --config configs/problem3.yaml
.\.venv\Scripts\python.exe -m data_progressing.problem3_analyze --config configs/problem3.yaml
```

This route has no approximate character-offset fallback. The preprocessing report must
show zero tokenizer mismatches before the data are admitted to Problem 3 modeling.

