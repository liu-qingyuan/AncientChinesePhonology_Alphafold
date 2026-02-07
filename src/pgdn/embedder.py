from typing import Any, Dict, List


class GraphEmbedder:
    def __init__(
        self, input_dim: int = 32, single_dim: int = 64, pair_dim: int = 32
    ) -> None:
        self.input_dim = input_dim
        self.single_dim = single_dim
        self.pair_dim = pair_dim

    def _project_single(self, vec: List[float]) -> List[float]:
        base = vec[: self.input_dim]
        if len(base) < self.single_dim:
            base = base + [0.0] * (self.single_dim - len(base))
        return base[: self.single_dim]

    def _project_pair(self, left: List[float], right: List[float]) -> List[float]:
        merged = [(l + r) * 0.5 for l, r in zip(left, right)]
        if len(merged) < self.pair_dim:
            merged = merged + [0.0] * (self.pair_dim - len(merged))
        return merged[: self.pair_dim]

    def forward(self, batch: Dict[str, List[List[float]]]) -> Dict[str, Any]:
        single = [self._project_single(v) for v in batch["target_vector"]]
        pair = []
        for i in range(len(single)):
            row = []
            for j in range(len(single)):
                row.append(self._project_pair(single[i], single[j]))
            pair.append(row)
        return {"single": single, "pair": pair}
