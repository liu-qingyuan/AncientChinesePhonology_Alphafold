from typing import Any, Dict, Tuple, cast

from .diffusion import DiffusionHead
from .embedder import GraphEmbedder
from .pairformer import PairformerLite


class PGDNv0:
    def __init__(self, ablation: str = "none") -> None:
        self.ablation = ablation
        self.embedder = GraphEmbedder(input_dim=32, single_dim=64, pair_dim=32)
        self.pairformer = PairformerLite(single_dim=64, pair_dim=32)
        self.diffusion = DiffusionHead(target_dim=32, cond_dim=64)

    def forward_loss(self, batch: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        reps = self.embedder.forward(batch)
        single = cast(list[list[float]], reps["single"])
        pair = cast(list[list[list[float]]], reps["pair"])
        if self.ablation != "no_pairformer":
            single = cast(
                list[list[float]], self.pairformer.forward(single, pair)["single"]
            )

        if self.ablation == "no_diffusion":
            total = 0.0
            denom = 0
            for i in range(len(single)):
                for j in range(32):
                    d = single[i][j] - batch["target_vector"][i][j]
                    total += d * d
                    denom += 1
            denoise = total / max(denom, 1)
            loss = denoise
        else:
            loss, loss_terms = self.diffusion.loss(
                cast(list[list[float]], batch["target_vector"]), single
            )
            denoise = loss_terms["denoise_mse"]

        if self.ablation == "no_constraint_loss":
            constraint = 0.0
        else:
            constraint = 0.0
            count = 0
            for i in range(len(batch["slot_mask"])):
                for s, slot in enumerate(("I", "M", "N", "C")):
                    m = float(batch["slot_mask"][i][slot])
                    for j in range(8):
                        constraint += abs(batch["target_vector"][i][s * 8 + j]) * (
                            1.0 - m
                        )
                        count += 1
            constraint = constraint / max(count, 1)

        total_loss = loss + 0.1 * constraint
        return total_loss, {
            "denoise_mse": denoise,
            "constraint_loss": constraint,
            "total_loss": total_loss,
        }
