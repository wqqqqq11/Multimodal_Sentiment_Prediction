# Problem 3 preprocessing

This package prepares data for the sparse-evidence bottleneck, prototype-memory,
and counterfactual-consistency solution. It does not implement the predictive model.

Run from the repository root:

```powershell
python -m data_progressing.problem3_preprocess --config configs/problem3.yaml
python -m data_progressing.problem3_validate --config configs/problem3.yaml
python -m data_progressing.problem3_analyze --config configs/problem3.yaml
```

The pipeline reads Problem 2 artifacts but never changes them. Its primary outputs are
under `datasets/preprocessed_data/problem3`, while reports, tables, and figures are under
`outputs/problem3/data_analysis`.

Important safeguards:

- no `privileged_text` or `modality_reliability` appears in main Problem 3 inputs;
- the Problem 2 scaler is reused read-only and its hash is recorded;
- prototype candidates come only from the training split;
- Attachment 4 text duplicates are audit-only and never receive copied labels;
- sample 13 visual reconstruction is a separate repair candidate, not a replacement;
- time/frame localization is explicitly marked as a low-confidence proportional fallback.

