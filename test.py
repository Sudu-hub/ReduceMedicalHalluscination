from source_Code.pncd.inference_engine import InferenceEngine

engine = InferenceEngine()

result = engine.inference(
    "Can aspirin help stroke patients?"
)

print("\nAlpha:")
print(result["alpha"])

print("\nBeta:")
print(result["beta"])

print("\nFinal Logits:")
print(result["final_logits"].shape)