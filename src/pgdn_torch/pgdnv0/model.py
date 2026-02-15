from __future__ import annotations

from dataclasses import dataclass
import math
from typing import cast

import torch
from torch import nn

from .articulatory_constraints import articulatory_constraint_loss


@dataclass(frozen=True)
class LossTerms:
    denoise_mse: float
    constraint_loss: float
    constraint_I: float
    constraint_M: float
    constraint_N: float
    constraint_C: float
    artic_constraint_mean: float
    total_loss: float


class GraphEmbedder(nn.Module):
    def __init__(self, input_dim: int = 32, single_dim: int = 64, pair_dim: int = 32):
        super().__init__()
        self.single = nn.Linear(input_dim, single_dim)
        self.pair_dim = pair_dim
        self.pair = nn.Linear(single_dim, pair_dim, bias=False)

    def forward(self, target_vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # target_vector: [B, 32]
        single = self.single(target_vector)  # [B, 64]
        # Pair representation: simple outer-add then project. [B, B, 64] -> [B, B, pair_dim]
        pair_base = 0.5 * (single[:, None, :] + single[None, :, :])
        pair = self.pair(pair_base)
        return single, pair


class TriangleMultiplicativeUpdate(nn.Module):
    """Tiny AF-style triangular update over pair reps.

    This is intentionally lightweight and uses only a single channel-wise
    multiplication path, implemented in two directions (outgoing + incoming):

      outgoing: out[i,j,c] = sum_k a[i,k,c] * b[k,j,c]
      incoming: out[i,j,c] = sum_k a[k,i,c] * b[j,k,c]

    Complexity is O(B^3 * C) where B is the "graph size" (here: batch size).
    Keep B modest (e.g. 16-64) for CPU runs.
    """

    def __init__(self, pair_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(pair_dim)
        self.proj_a_out = nn.Linear(pair_dim, pair_dim, bias=False)
        self.proj_b_out = nn.Linear(pair_dim, pair_dim, bias=False)
        self.gate_out = nn.Linear(pair_dim, pair_dim)

        self.proj_a_in = nn.Linear(pair_dim, pair_dim, bias=False)
        self.proj_b_in = nn.Linear(pair_dim, pair_dim, bias=False)
        self.gate_in = nn.Linear(pair_dim, pair_dim)

        # Final projection after summing the two directional paths.
        self.out = nn.Linear(pair_dim, pair_dim)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        # pair: [B, B, C]
        x = self.norm(pair)
        bsz = max(int(pair.shape[0]), 1)

        # Outgoing path: i->k, k->j
        a_out = self.proj_a_out(x)
        b_out = self.proj_b_out(x)
        g_out = torch.sigmoid(self.gate_out(x))
        tri_out = torch.einsum("ikc,kjc->ijc", a_out, b_out) / bsz

        # Incoming path: k->i, j->k
        a_in = self.proj_a_in(x)
        b_in = self.proj_b_in(x)
        g_in = torch.sigmoid(self.gate_in(x))
        tri_in = torch.einsum("kic,jkc->ijc", a_in, b_in) / bsz

        tri = g_out * tri_out + g_in * tri_in
        return self.out(tri)


class PairToSingleUpdate(nn.Module):
    def __init__(self, pair_dim: int, single_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(pair_dim)
        self.proj = nn.Linear(pair_dim, single_dim)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        # pair: [B, B, C]
        x = self.norm(pair)
        pooled = x.mean(dim=1)  # [B, C]
        return self.proj(pooled)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class PairformerLite(nn.Module):
    """A small pairmixer-ish trunk.

    Not a faithful AlphaFold3 implementation; it exists to embed the key
    inductive bias from the paper: triangular (transitivity-like) propagation
    over pair representations, with a cheap single update from pair context.
    """

    def __init__(
        self,
        single_dim: int = 64,
        pair_dim: int = 32,
        blocks: int = 2,
        recycle: int = 1,
    ) -> None:
        super().__init__()
        self.single_dim = single_dim
        self.pair_dim = pair_dim
        self.blocks = int(blocks)
        self.recycle = int(recycle)

        self.tri_mul = nn.ModuleList(
            [TriangleMultiplicativeUpdate(pair_dim) for _ in range(self.blocks)]
        )
        self.pair_ff = nn.ModuleList([FeedForward(pair_dim, hidden=pair_dim * 4) for _ in range(self.blocks)])
        self.pair_to_single = nn.ModuleList(
            [PairToSingleUpdate(pair_dim, single_dim) for _ in range(self.blocks)]
        )
        self.single_ff = nn.ModuleList(
            [FeedForward(single_dim, hidden=single_dim * 4) for _ in range(self.blocks)]
        )

        self.single_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.prev_single_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.prev_pair_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def _forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        prev_single: torch.Tensor | None,
        prev_pair: torch.Tensor | None,
        collect_deltas: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        deltas: list[float] = []
        r = max(self.recycle, 1)

        ps = prev_single
        pp = prev_pair
        for _cycle in range(r):
            single_prev = single
            if ps is not None:
                single = single + self.prev_single_scale * ps
            if pp is not None:
                pair = pair + self.prev_pair_scale * pp

            for i in range(self.blocks):
                pair = pair + self.tri_mul[i](pair)
                pair = pair + self.pair_ff[i](pair)
                single = single + self.single_scale * self.pair_to_single[i](pair)
                single = single + self.single_ff[i](single)

            if collect_deltas:
                with torch.no_grad():
                    d = torch.sqrt(torch.mean((single - single_prev) ** 2))
                    deltas.append(float(d.detach().cpu().item()))
            ps = single
            pp = pair

        return single, pair, deltas

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        prev_single: torch.Tensor | None = None,
        prev_pair: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # single: [B, single_dim]
        # pair: [B, B, pair_dim]
        single, pair, _d = self._forward(
            single,
            pair,
            prev_single=prev_single,
            prev_pair=prev_pair,
            collect_deltas=False,
        )
        return single, pair

    def forward_with_deltas(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        prev_single: torch.Tensor | None = None,
        prev_pair: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        return self._forward(
            single,
            pair,
            prev_single=prev_single,
            prev_pair=prev_pair,
            collect_deltas=True,
        )


class DiffusionHead(nn.Module):
    def __init__(
        self,
        target_dim: int = 32,
        cond_dim: int = 64,
        hidden: int = 128,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule: str = "linear",
        pred_type: str = "eps",
        cond_dropout: float = 0.0,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.cond_dim = cond_dim
        self.timesteps = int(timesteps)
        self.schedule = str(schedule)
        self.pred_type = str(pred_type)
        self.cond_dropout = float(cond_dropout)

        if self.pred_type not in {"eps", "v"}:
            raise ValueError(f"unsupported pred_type: {self.pred_type}")

        betas: torch.Tensor
        if self.schedule == "cosine":
            betas = self._cosine_beta_schedule(self.timesteps)
        elif self.schedule == "fast":
            # A slightly more aggressive linear schedule that tends to denoise faster.
            betas = torch.linspace(float(beta_start), float(max(beta_end, 5e-2)), steps=self.timesteps)
        else:
            betas = torch.linspace(float(beta_start), float(beta_end), steps=self.timesteps)

        betas = torch.clamp(betas, min=1e-8, max=0.999)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        # Buffers are moved with the module device.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

        # Predict epsilon from (x_t, cond, t_scalar).
        self.net = nn.Sequential(
            nn.Linear(target_dim + cond_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, target_dim),
        )

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = int(timesteps) + 1
        x = torch.linspace(0, int(timesteps), steps=steps, dtype=torch.float32)
        t = x / max(int(timesteps), 1)
        f = torch.cos(((t + float(s)) / (1.0 + float(s))) * (math.pi / 2.0)) ** 2
        alpha_bar = f / f[0]
        betas = 1.0 - (alpha_bar[1:] / torch.clamp(alpha_bar[:-1], min=1e-8))
        return torch.clamp(betas, min=1e-4, max=0.999)

    def loss(self, target: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # target: [B, D], cond: [B, C]
        b = int(target.shape[0])
        device = target.device

        t_int = torch.randint(0, self.timesteps, (b,), device=device)
        t = (t_int.float() / max(self.timesteps - 1, 1)).unsqueeze(1)  # [B, 1]

        eps = torch.randn_like(target)
        sqrt_ab = cast(torch.Tensor, self.sqrt_alpha_bar)
        sqrt_1m = cast(torch.Tensor, self.sqrt_one_minus_alpha_bar)
        s_ab = sqrt_ab[t_int].unsqueeze(1)
        s_1m = sqrt_1m[t_int].unsqueeze(1)
        x_t = s_ab * target + s_1m * eps

        if self.cond_dropout > 0.0:
            drop = (torch.rand((b, 1), device=device) < float(self.cond_dropout)).float()
            cond = cond * (1.0 - drop)

        inp = torch.cat([x_t, cond, t], dim=1)
        y_hat = self.net(inp)
        if self.pred_type == "eps":
            y = eps
            eps_hat = y_hat
            x0_hat = (x_t - s_1m * eps_hat) / torch.clamp(s_ab, min=1e-8)
        else:
            # v = sqrt(ab) * eps - sqrt(1-ab) * x0
            y = s_ab * eps - s_1m * target
            v_hat = y_hat
            x0_hat = s_ab * x_t - s_1m * v_hat
            eps_hat = s_1m * x_t + s_ab * v_hat

        mse = torch.mean((y_hat - y) ** 2)

        return mse, mse, x0_hat

    @staticmethod
    def range_contract(x: torch.Tensor, enabled: bool = True, scale: float = 1.0) -> torch.Tensor:
        if not enabled:
            return x
        s = float(scale)
        if s <= 0.0:
            s = 1.0
        return torch.tanh(x / s)

    @torch.no_grad()
    def precompute_shared_uncond_y_by_step(
        self,
        x_T: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Precompute unconditional net outputs per diffusion step.

        This is used for `shared_denoise` coupling: all records in the same group
        reuse the same unconditional prediction y_u at each step.

        Note: this uses a deterministic trajectory update (no per-step noise).
        """
        device = x_T.device
        x = x_T
        gsz = int(x.shape[0])
        if tuple(x.shape) != (gsz, int(self.target_dim)):
            raise ValueError(
                f"x_T has wrong shape: got={tuple(x.shape)} want={(gsz, int(self.target_dim))}"
            )

        betas = cast(torch.Tensor, self.betas)
        alphas = cast(torch.Tensor, self.alphas)
        alpha_bar = cast(torch.Tensor, self.alpha_bar)

        zeros_cond = torch.zeros((gsz, int(self.cond_dim)), device=device, dtype=x.dtype)
        outs: list[torch.Tensor] = []

        t_grid = torch.linspace(self.timesteps - 1, 0, steps=int(num_steps), device=device)
        for t_val in t_grid:
            t_int = int(t_val.item())
            t = torch.full((gsz, 1), float(t_int) / max(self.timesteps - 1, 1), device=device, dtype=x.dtype)

            beta_t = betas[t_int]
            alpha_t = alphas[t_int]
            ab_t = alpha_bar[t_int]
            s_ab = torch.sqrt(ab_t)
            s_1m = torch.sqrt(1.0 - ab_t)

            inp_u = torch.cat([x, zeros_cond, t], dim=1)
            y_u = self.net(inp_u)
            outs.append(y_u)

            if self.pred_type == "eps":
                eps_hat = y_u
                x0_hat = (x - s_1m * eps_hat) / torch.clamp(s_ab, min=1e-8)
            else:
                v_hat = y_u
                x0_hat = s_ab * x - s_1m * v_hat
                eps_hat = s_1m * x + s_ab * v_hat

            coef = beta_t / torch.clamp(torch.sqrt(1.0 - ab_t), min=1e-8)
            mean = (x - coef * eps_hat) / torch.clamp(torch.sqrt(alpha_t), min=1e-8)
            x = mean
            x = torch.clamp(x, min=-6.0, max=6.0)
            if t_int == 0:
                x = 0.5 * x + 0.5 * x0_hat

        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        seed: int,
        num_steps: int = 20,
        noise_scale: float = 0.3,
        enforce_range: bool = True,
        range_scale: float = 1.0,
        cfg_scale: float = 1.0,
        x_T: torch.Tensor | None = None,
        shared_denoise_uncond_y_by_step: torch.Tensor | None = None,
        shared_denoise_group_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # cond: [B, C]
        device = cond.device
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

        if x_T is None:
            x = (
                torch.randn((cond.shape[0], self.target_dim), generator=g, device=device)
                * float(noise_scale)
            )
        else:
            if tuple(x_T.shape) != (int(cond.shape[0]), int(self.target_dim)):
                raise ValueError(
                    f"x_T has wrong shape: got={tuple(x_T.shape)} want={(int(cond.shape[0]), int(self.target_dim))}"
                )
            x = x_T.to(device=device, dtype=cond.dtype)

        # Use a strided schedule if num_steps < timesteps.
        betas = cast(torch.Tensor, self.betas)
        alphas = cast(torch.Tensor, self.alphas)
        alpha_bar = cast(torch.Tensor, self.alpha_bar)

        t_grid = torch.linspace(self.timesteps - 1, 0, steps=int(num_steps), device=device)
        for step_idx, t_val in enumerate(t_grid):
            t_int = int(t_val.item())
            t = torch.full((cond.shape[0], 1), float(t_int) / max(self.timesteps - 1, 1), device=device)

            beta_t = betas[t_int]
            alpha_t = alphas[t_int]
            ab_t = alpha_bar[t_int]
            s_ab = torch.sqrt(ab_t)
            s_1m = torch.sqrt(1.0 - ab_t)

            if float(cfg_scale) == 1.0:
                inp = torch.cat([x, cond, t], dim=1)
                y_hat = self.net(inp)
            else:
                inp_c = torch.cat([x, cond, t], dim=1)
                y_c = self.net(inp_c)
                if shared_denoise_uncond_y_by_step is not None and shared_denoise_group_index is not None:
                    if step_idx >= int(shared_denoise_uncond_y_by_step.shape[0]):
                        raise ValueError("shared_denoise_uncond_y_by_step has insufficient steps")
                    y_u_step = shared_denoise_uncond_y_by_step[step_idx]
                    y_u = y_u_step[shared_denoise_group_index]
                else:
                    inp_u = torch.cat([x, torch.zeros_like(cond), t], dim=1)
                    y_u = self.net(inp_u)
                y_hat = y_u + float(cfg_scale) * (y_c - y_u)
            if self.pred_type == "eps":
                eps_hat = y_hat
                x0_hat = (x - s_1m * eps_hat) / torch.clamp(s_ab, min=1e-8)
            else:
                v_hat = y_hat
                x0_hat = s_ab * x - s_1m * v_hat
                eps_hat = s_1m * x + s_ab * v_hat
            x0_hat = self.range_contract(x0_hat, enabled=bool(enforce_range), scale=float(range_scale))

            # DDPM mean for p(x_{t-1} | x_t).
            coef = beta_t / torch.clamp(torch.sqrt(1.0 - ab_t), min=1e-8)
            mean = (x - coef * eps_hat) / torch.clamp(torch.sqrt(alpha_t), min=1e-8)

            if t_int > 0:
                z = torch.randn(x.shape, device=device, generator=g)
                x = mean + torch.sqrt(beta_t) * z
            else:
                x = mean

            # Keep x within a reasonable numeric range.
            x = torch.clamp(x, min=-6.0, max=6.0)
            # Encourage the final sample to be close to the denoised estimate.
            if t_int == 0:
                x = 0.5 * x + 0.5 * x0_hat

        x = self.range_contract(x, enabled=bool(enforce_range), scale=float(range_scale))

        return x


class PGDNTorchV0(nn.Module):
    def __init__(
        self,
        ablation: str = "none",
        pairformer_blocks: int = 2,
        recycle: int = 1,
        diffusion_timesteps: int = 100,
        diffusion_schedule: str = "linear",
        diffusion_pred: str = "eps",
        cond_dropout: float = 0.0,
        enforce_range: bool = True,
        range_scale: float = 1.0,
        constraint_slot_weights: list[float] | None = None,
        constraint_dim_weights: list[float] | None = None,
        artic_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.ablation = ablation
        self.enforce_range = bool(enforce_range)
        self.range_scale = float(range_scale)
        self.artic_loss_weight = float(artic_loss_weight)
        if self.artic_loss_weight < 0.0:
            raise ValueError("artic_loss_weight must be >= 0")
        self.embedder = GraphEmbedder(input_dim=32, single_dim=64, pair_dim=32)
        self.pairformer = PairformerLite(
            single_dim=64,
            pair_dim=32,
            blocks=int(pairformer_blocks),
            recycle=int(recycle),
        )
        self.diffusion = DiffusionHead(
            target_dim=32,
            cond_dim=64,
            timesteps=int(diffusion_timesteps),
            schedule=str(diffusion_schedule),
            pred_type=str(diffusion_pred),
            cond_dropout=float(cond_dropout),
        )

        slot_w = [1.0, 1.0, 1.0, 1.0] if constraint_slot_weights is None else list(constraint_slot_weights)
        if len(slot_w) != 4:
            raise ValueError("constraint_slot_weights must have 4 values for I,M,N,C")
        dim_w = [1.0] * 8 if constraint_dim_weights is None else list(constraint_dim_weights)
        if len(dim_w) != 8:
            raise ValueError("constraint_dim_weights must have 8 values")

        self.register_buffer("constraint_slot_weights", torch.tensor(slot_w, dtype=torch.float32))
        self.register_buffer("constraint_dim_weights", torch.tensor(dim_w, dtype=torch.float32))

    def forward_loss(
        self,
        target_vector: torch.Tensor,
        slot_mask: torch.Tensor,
        articulatory_vector: torch.Tensor | None = None,
        articulatory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LossTerms]:
        # target_vector: [B, 32]
        # slot_mask: [B, 4] values in {0,1} for I,M,N,C
        single, pair = self.embedder(target_vector)

        if self.ablation != "no_pairformer":
            single, _pair = self.pairformer(single, pair)

        if self.ablation == "no_diffusion":
            denoise_mse_t = torch.mean((single[:, :32] - target_vector) ** 2)
            loss_t = denoise_mse_t
            x0_hat = single[:, :32]
        else:
            loss_t, denoise_mse_t, x0_hat = self.diffusion.loss(target_vector, single)

        x0_hat = self.diffusion.range_contract(
            x0_hat,
            enabled=bool(self.enforce_range),
            scale=float(self.range_scale),
        )

        if self.ablation == "no_constraint_loss":
            constraint_t = torch.tensor(0.0, device=target_vector.device)
            slot_losses = torch.zeros((4,), device=target_vector.device)
        else:
            # Hierarchical constraints over 4 slots (I,M,N,C) x 8 dims each.
            x = x0_hat.reshape(-1, 4, 8)
            viol = torch.abs(x) * (1.0 - slot_mask[:, :, None])
            viol = viol * cast(torch.Tensor, self.constraint_dim_weights)[None, None, :]
            slot_losses = torch.mean(viol, dim=(0, 2))  # [4]
            w = cast(torch.Tensor, self.constraint_slot_weights)
            denom = torch.clamp(torch.sum(w), min=1e-8)
            constraint_t = torch.sum(slot_losses * w) / denom

        if self.artic_loss_weight > 0.0:
            artic_constraint_t = articulatory_constraint_loss(
                predicted_vector=x0_hat,
                target_vector=target_vector,
                slot_mask=slot_mask,
                articulatory_vector=articulatory_vector,
                articulatory_mask=articulatory_mask,
            )
        else:
            artic_constraint_t = torch.tensor(0.0, device=target_vector.device)

        total = loss_t + 0.1 * constraint_t + self.artic_loss_weight * artic_constraint_t
        terms = LossTerms(
            denoise_mse=float(denoise_mse_t.detach().cpu().item()),
            constraint_loss=float(constraint_t.detach().cpu().item()),
            constraint_I=float(slot_losses[0].detach().cpu().item()),
            constraint_M=float(slot_losses[1].detach().cpu().item()),
            constraint_N=float(slot_losses[2].detach().cpu().item()),
            constraint_C=float(slot_losses[3].detach().cpu().item()),
            artic_constraint_mean=float(artic_constraint_t.detach().cpu().item()),
            total_loss=float(total.detach().cpu().item()),
        )
        return total, terms
