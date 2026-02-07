import random
from typing import Dict, List, Tuple


class DiffusionHead:
    def __init__(
        self, target_dim: int = 32, cond_dim: int = 64, timesteps: int = 100
    ) -> None:
        self.timesteps = timesteps
        self.target_dim = target_dim
        self.cond_dim = cond_dim

    def loss(
        self, target: List[List[float]], cond: List[List[float]]
    ) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        denom = 0
        for i in range(len(target)):
            for j in range(self.target_dim):
                pred = 0.8 * target[i][j] + 0.2 * cond[i][j % self.cond_dim]
                d = pred - target[i][j]
                total += d * d
                denom += 1
        mse = total / max(denom, 1)
        return mse, {"denoise_mse": mse}

    def sample(
        self,
        cond: List[List[float]],
        seed: int,
        num_steps: int = 20,
        noise_scale: float = 0.3,
    ) -> List[List[float]]:
        rng = random.Random(seed)
        out = []
        for row in cond:
            x = [rng.gauss(0.0, noise_scale) for _ in range(self.target_dim)]
            for step in range(num_steps, 0, -1):
                alpha = float(step) / float(num_steps)
                for i in range(self.target_dim):
                    pred = 0.7 * x[i] + 0.3 * row[i % self.cond_dim]
                    x[i] = alpha * x[i] + (1.0 - alpha) * pred
            out.append(x)
        return out
