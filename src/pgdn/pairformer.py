from typing import Any, Dict, List


class PairformerLite:
    def __init__(self, single_dim: int = 64, pair_dim: int = 32) -> None:
        self.single_dim = single_dim
        self.pair_dim = pair_dim

    def forward(
        self, single: List[List[float]], pair: List[List[List[float]]]
    ) -> Dict[str, Any]:
        pair_updated = []
        for row in pair:
            pair_row = []
            for p in row:
                pair_row.append([v * 0.95 for v in p])
            pair_updated.append(pair_row)

        single_updated = []
        for i, s in enumerate(single):
            pair_bias = [0.0] * self.pair_dim
            for j in range(len(pair_updated[i])):
                for k in range(self.pair_dim):
                    pair_bias[k] += pair_updated[i][j][k]
            n = max(len(pair_updated[i]), 1)
            pair_bias = [v / n for v in pair_bias]
            single_new = [
                0.9 * s[k] + 0.1 * pair_bias[k % self.pair_dim]
                for k in range(self.single_dim)
            ]
            single_updated.append(single_new)
        return {"single": single_updated, "pair": pair_updated}
