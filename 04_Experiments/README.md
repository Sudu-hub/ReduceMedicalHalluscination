# Experiments

Three experimental conditions evaluated in the paper (Section V):

| Subfolder | Setup | Description |
|---|---|---|
| `baselines/` | Baseline LLM | Vanilla LLM decoding, no expert / non-expert separation |
| `fixed_pncd/` | Fixed-weight PNCD | PNCD fusion with static α and β values (no adaptation) |
| `adaptive_pncd/` | Proposed Adaptive PNCD | Full framework with learned α–β predictor |

Each subfolder should contain:
- `train.py` / `run.py` — execution entry point
- `config.yaml` — hyperparameters
- `outputs/` — predictions, metrics, logs
