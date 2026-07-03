import torch


class PNCDDecoder:

    def __init__(self):

        pass

    def decode(
        self,
        base_logits,
        expert_logits,
        nonexpert_logits,
        alpha,
        beta
    ):

        # --------------------------------------------------
        # Move alpha and beta to same device as logits
        # --------------------------------------------------

        device = base_logits.device

        alpha = alpha.to(device)

        beta = beta.to(device)

        # --------------------------------------------------
        # Reshape for broadcasting
        # --------------------------------------------------

        alpha = alpha.view(1, 1, 1)

        beta = beta.view(1, 1, 1)

        # --------------------------------------------------
        # Adaptive PNCD Equation
        # z = zb + α(ze-zb) - β(zn-zb)
        # --------------------------------------------------

        final_logits = (

            base_logits

            + alpha *

            (expert_logits - base_logits)

            - beta *

            (nonexpert_logits - base_logits)

        )

        return final_logits

    def __call__(
        self,
        base_logits,
        expert_logits,
        nonexpert_logits,
        alpha,
        beta
    ):

        return self.decode(
            base_logits,
            expert_logits,
            nonexpert_logits,
            alpha,
            beta
        )