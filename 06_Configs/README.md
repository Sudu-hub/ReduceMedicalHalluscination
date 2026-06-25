# Configs

YAML/JSON configuration files for every experiment and module:

- `base_llm.yaml` — base LLM identifier (e.g., LLaMA-2-7B, Mistral-7B)
- `lora.yaml` — LoRA rank, alpha, dropout, target modules
- `faiss.yaml` — index type (Flat / IVF), embedding model, top-K
- `alpha_beta_predictor.yaml` — MLP hidden dims, learning rate, uncertainty features
- `pncd_fusion.yaml` — stability constraints, α/β clipping ranges
- `experiment_baseline.yaml`, `experiment_fixed_pncd.yaml`, `experiment_adaptive_pncd.yaml`
