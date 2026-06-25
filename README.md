# SD2 ColabFold Homodimer Batch

This repository prepares and runs AlphaFold2 homodimer predictions for the `SD2` sheet in `41467_2024_54452_MOESM4_ESM.xlsx`.

For each row in `SD2`, the `Cloned_AA` sequence is converted into a homodimer query:

```text
sequence:sequence
```

The workflow preserves the original sequence, removes gap placeholder characters (`-`) from model inputs, and records all metadata in a manifest.

## Files

- `41467_2024_54452_MOESM4_ESM.xlsx` - source workbook.
- `scripts/extract_sd2_queries.py` - extracts `SD2!Cloned_AA` into ColabFold-ready homodimer inputs.
- `scripts/run_sd2_colabfold.py` - runs ColabFold/AlphaFold2 multimer predictions.
- `scripts/summarize_sd2_colabfold.py` - summarizes ranked model scores after predictions complete.
- `outputs/sd2_colabfold/sd2_homodimer_manifest.csv` - audited row-level manifest.
- `outputs/sd2_colabfold/sd2_homodimer_queries.csv` - ColabFold query table.
- `outputs/sd2_colabfold/sd2_colabfold_scores_summary.csv` - model-level summary table.
- `outputs/sd2_colabfold/sd2_colabfold_scores_summary.txt` - text progress summary.

Heavy prediction folders, model parameters, and result zip packages are ignored by Git by default.

## Colab Run

Mount Google Drive and change into this project:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/AlphaFold
```

Install ColabFold:

```python
!pip install -q --no-warn-conflicts "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"
```

Apply the TensorFlow Lite workaround used by the public ColabFold notebook:

```python
!rm -f /usr/local/lib/python3.*/dist-packages/tensorflow/core/kernels/libtfkernel_sobol_op.so
!rm -f /usr/local/lib/python3.*/dist-packages/tensorflow/lite/python/*/*.so
```

Run a single smoke test:

```python
!python scripts/run_sd2_colabfold.py --start-at 1 --limit 1
```

Run or resume a batch:

```python
!python scripts/run_sd2_colabfold.py --start-at 1
```

Run only one row:

```python
!python scripts/run_sd2_colabfold.py --start-at 28 --limit 1
```

Summarize results:

```python
!python scripts/summarize_sd2_colabfold.py
```

## Full Prediction Outputs

Full AlphaFold2 prediction outputs are available as a GitHub Release: v1.0.0

## Notes

- Completed jobs are skipped automatically unless `--overwrite` is supplied.
- If a Colab session disconnects mid-job, rerun that `List_No` with `--overwrite`.
- Result packages are written under `outputs/sd2_colabfold/results/` and are intentionally excluded from Git because they can be large.
