# Models

Saved model artifacts (excluded from version control via `.gitignore`).

| Subfolder | Artifact |
|---|---|
| `expert_lora/` | Adapter weights fine-tuned on curated PubMed biomedical content |
| `non_expert_lora/` | Adapter weights fine-tuned on synthetic misinformation |
| `alpha_beta_predictor/` | Trained MLP weights and feature-scaler state for the adaptive predictor |

Base LLM weights are **not** stored here — they are loaded from HuggingFace or a local path. Only the lightweight LoRA / MLP adapters live in this folder.
