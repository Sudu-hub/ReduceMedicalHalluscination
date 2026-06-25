# Data

## Subfolders
- `raw/` — Original downloaded data before any preprocessing
- `expert_corpus/` — Curated PubMed biomedical literature used to build the expert FAISS index (S+)
- `non_expert_corpus/` — Synthetic / noisy medical passages used for the non-expert FAISS index (S−)
- `processed/` — Tokenized, chunked, and indexed artifacts ready for training/retrieval

## Datasets used in the paper
- **PubMedQA** — biomedical yes/no/maybe QA grounded in PubMed abstracts
- **TruthfulQA** — measures imitative falsehoods
- **MedHallu** — real-world hallucinated answers to clinical queries
- **MedHallBench** — benchmark for medical LLM hallucination assessment
- **Med-HALT** — memory traps, fake knowledge, NOTA distractors
