# Adaptive α–β Guided PNCD Framework

Implementation of the paper: **"Learning-Based α–β Guided PNCD: An Adaptive Framework for Hallucination Mitigation in Medical LLMs"** (IEEE ESCI 2026).

## Paper Summary

A framework that reduces hallucinations in medical LLMs by:
- **Dual FAISS retrieval** (expert PubMed passages + non-expert noisy passages)
- **Dual LoRA fine-tuning** on a single base LLM (Expert + Non-Expert adapters)
- **Logit extraction** producing baseline, expert-guided, and non-expert logit streams
- **Adaptive α–β predictor** (lightweight MLP) that estimates dynamic weights
- **Positive–Negative Contrastive Decoding (PNCD)** fusion: `z_final = z + α·z+ − β·z−`

## Folder Structure

| Folder | Purpose |
|---|---|
| `01_Documentation/` | Paper notes, architecture diagrams, methodology docs |
| `02_Data/` | Expert and non-expert corpora (raw + processed) |
| `03_Source_Code/` | Modular code: retrieval, LoRA, logit extraction, α–β predictor, PNCD fusion, decoding, end-to-end pipeline |
| `04_Experiments/` | Baseline, fixed-weight PNCD, and proposed adaptive PNCD experiment runs |
| `05_Results/` | Figures, tables, training/inference logs |
| `06_Configs/` | YAML/JSON config files for all experiments |
| `07_Models/` | Saved Expert LoRA, Non-Expert LoRA, and α–β predictor weights |
| `08_Evaluation/` | Metric implementations and benchmark loaders (PubMedQA, TruthfulQA, MedHallu, MedHallBench) |
| `09_References/` | Cited papers and external resources |

## Pipeline Flow

```
User Query → Dual FAISS Retrieval → Base LLM + Dual LoRA → Logit Extraction (z, z+, z−)
                                                       ↓
                                            α–β Predictor (MLP)
                                                       ↓
                                          PNCD Fusion (z + α·z+ − β·z−)
                                                       ↓
                                        Decoding (greedy / nucleus)
                                                       ↓
                                          Hallucination-reduced answer
```
