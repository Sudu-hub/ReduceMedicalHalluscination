import torch

# =====================================================
# MODEL PATHS
# =====================================================

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

EXPERT_LORA_PATH = "07_Models/expert_lora"

NONEXPERT_LORA_PATH = "07_Models/nonexpert_lora"

ADAPTIVE_MODEL_PATH = "07_Models/adaptive_predictor.pt"


# =====================================================
# TRAINING
# =====================================================

LEARNING_RATE = 1e-3

BATCH_SIZE = 4

NUM_EPOCHS = 10

WEIGHT_DECAY = 1e-4


# =====================================================
# ADAPTIVE PREDICTOR
# =====================================================

INPUT_DIM = 7

HIDDEN_DIM_1 = 32

HIDDEN_DIM_2 = 16

OUTPUT_DIM = 2


# =====================================================
# PNCD
# =====================================================

TOP_K = 2


# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


# =====================================================
# RANDOM SEED
# =====================================================

SEED = 42