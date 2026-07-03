import torch

from source_Code.retrieval.retriever import DualRetriever

from source_Code.pncd.feature_extractor import (
    FeatureExtractor
)

from source_Code.pncd.adaptive_predictor import (
    AdaptivePredictor
)

from source_Code.pncd.logit_extractor import (
    LogitExtractor
)

from source_Code.pncd.decoder import (
    PNCDDecoder
)

from source_Code.pncd.config import (
    ADAPTIVE_MODEL_PATH,
    DEVICE
)


class InferenceEngine:

    def __init__(self):

        print("Loading Retriever...")

        self.retriever = DualRetriever()

        print("Loading Logit Extractor...")

        self.logit_extractor = LogitExtractor()

        print("Loading Adaptive Predictor...")

        self.predictor = AdaptivePredictor().to(DEVICE)

        self.predictor.load_state_dict(
            torch.load(
                ADAPTIVE_MODEL_PATH,
                map_location=DEVICE
            )
        )

        self.predictor.eval()

        print("Loading PNCD Decoder...")

        self.decoder = PNCDDecoder()

        print("Inference Engine Ready")

    def inference(
        self,
        question
    ):

        # =================================================
        # Retrieval
        # =================================================

        retrieval = self.retriever.retrieve(
            question
        )

        # =================================================
        # Feature Extraction
        # =================================================

        features = FeatureExtractor.extract(

            retrieval["expert_scores"],

            retrieval["nonexpert_scores"]

        )

        if not isinstance(
            features,
            torch.Tensor
        ):

            features = torch.tensor(

                features,

                dtype=torch.float32

            )

        if features.dim() == 1:

            features = features.unsqueeze(0)

        features = features.to(DEVICE)

        # =================================================
        # Adaptive Predictor
        # =================================================

        with torch.no_grad():

            alpha, beta = self.predictor.predict(
                features
            )

        # =================================================
        # Logit Extraction
        # =================================================

        logits = self.logit_extractor.get_logits(
            question
        )

        # =================================================
        # PNCD Decoder
        # =================================================

        final_logits = self.decoder.decode(

            logits["base"],

            logits["expert"],

            logits["nonexpert"],

            alpha,

            beta

        )

        # =================================================
        # Return
        # =================================================

        return {

            "retrieval": retrieval,

            "features": features.cpu(),

            "alpha": alpha.item(),

            "beta": beta.item(),

            "logits": logits,

            "final_logits": final_logits

        }