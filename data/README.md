# Fictional observations

`observations.csv` contains 600 synthetic, independent observations, generated for these lessons by [`generate_data.py`](generate_data.py). It contains no real people or sensitive measurements. The features are simulated numeric inputs without physical units; `label` is a noisy binary outcome derived from a nonlinear combination of those inputs. The data are intentionally learnable, not a benchmark for real-world claims.

| Column | Meaning |
| --- | --- |
| `feature_1`, `feature_2`, `feature_3` | Numeric simulated predictors |
| `label` | Binary class index, 0 or 1 |
| `split` | Preassigned `train` (420), `validation` (90), or `test` (90) partition |

The fixed split makes examples comparable; it was randomly shuffled, not stratified. Fit all preprocessing using training rows only. Validation is for development; reserve the test rows for the final assessment. For separate exploratory notebooks that introduce test evaluation, treat their reported test results as demonstrations, not independent evaluations of repeated model selection.

To regenerate the CSV from the standard library, run `uv run python data/generate_data.py` from the repository root. This overwrites the CSV with the same values for the same Python random-number implementation. The included CSV means students do not need to generate or download data.
