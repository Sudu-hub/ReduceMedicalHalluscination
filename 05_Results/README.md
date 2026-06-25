# Results

Outputs of all experiments.

| Subfolder | Contents |
|---|---|
| `figures/` | Plots reproducing Fig. 3 (static vs adaptive) and Fig. 4 (hallucination rate under noise) |
| `tables/` | CSV / markdown exports of Table I (illustrative comparison) |
| `logs/` | Raw training and inference logs (per-epoch, per-query) |

Key reported numbers (from paper):
- Answer Accuracy: Baseline 65–70% → Fixed PNCD 72–75% → **Proposed 85–88%**
- Hallucination Rate: High → Moderate → **Low**
- ~20–30% hallucination reduction under noisy conditions
- ~15% factual correctness gain over baseline
- ~25% reduction in unsupported medical statements in skewed scenarios
