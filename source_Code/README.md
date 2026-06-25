# Source Code

Modular implementation of the Adaptive α–β Guided PNCD framework. Each subfolder maps to one architectural component from Fig. 1 of the paper.

| Subfolder | Component | Description |
|---|---|---|
| `retrieval/` | Dual FAISS Retrieval Layer | Build & query two FAISS indexes: expert (PubMed) and non-expert (noisy) |
| `lora_modules/` | Expert-LoRA & Non-Expert-LoRA | Parameter-efficient fine-tuning adapters on a single base LLM |
| `logit_extraction/` | Logit Extraction Engine | Produce `z` (baseline), `z+` (expert-guided), `z−` (non-expert) logit distributions |
| `alpha_beta_predictor/` | Adaptive α–β Predictor | Lightweight MLP that maps uncertainty features to α and β |
| `pncd_fusion/` | PNCD Fusion Module | Implements `z_final = z + α·z+ − β·z−` |
| `decoding/` | Decoding & Response Generation | Greedy / nucleus sampling on the fused logits |
| `pipeline/` | End-to-End Pipeline | `main.py` entry point that wires all modules together |
