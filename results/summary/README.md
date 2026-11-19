# Saved experiment summary

`metrics.csv` is an export of local saved measurements, not a new training run.

## Provenance

- Twelve validation rows come from completed checkpoints in `data/weights/`.
- One test row comes from an existing `predictions.npz` for the frozen 7,000-image run with learning rate 0.0001. No test evaluation was run during repository preparation.
- One interrupted fine-tuning checkpoint lacks completion metadata and is excluded. Its filename requests 100,000 epochs; that is not evidence that those epochs completed.
- The source column records the relative path of each original local artifact. Large source artifacts are intentionally excluded from Git.
- Older galleries without matching local checkpoints are not part of this table.

## Reading the table

- `dataset`: total subset size before the 70/15/15 split. The 700-image subset has 490 training and 105 validation images; the 7,000-image subset has 4,900 training and 1,050 validation images.
- `split`: `validation` or a previously recorded `test` measurement.
- `accuracy` and `macro_f1`: fractions on a 0–1 scale. Macro-F1 includes all 101 classes, with zero for undefined scores.
- `epochs`: the budget of the current stage, not the cumulative parent + fine-tuning budget.
- `learning_rate`: classifier learning rate; `backbone_learning_rate` is populated for fine-tuning.
- `stage_seconds`: current-stage training time; `total_training_seconds` additionally includes parent-stage time for fine-tuning.
- `best_epoch`: selected by highest validation macro-F1, breaking ties with lower validation loss.
- `device`: training device, not the CPU used for reading checkpoint metadata during export.

The table reflects one seed and different per-method budgets. Changing subset size also changes the holdouts. It supports a descriptive comparison of saved experiments, not statistical significance or an official Food-101 benchmark claim.

## Regeneration

With the original local checkpoints and diagnostic files present, run:

```bash
python -m src.export_results
```

The exporter replaces the CSV deterministically. It never trains a model or evaluates the test set. A fresh clone without checkpoints can read the published table but cannot regenerate the original measurements until the experiments have been rerun.
