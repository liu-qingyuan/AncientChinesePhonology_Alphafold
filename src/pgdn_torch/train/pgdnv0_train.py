from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from ..pgdnv0.data import ACPJsonlDataset, GraphBatchSampler, SyntheticPGDNDataset, collate_pgdn
from ..pgdnv0.splits import load_split_ids
from ..pgdnv0.model import PGDNTorchV0


def _default_num_workers() -> int:
    try:
        import os

        return min(4, max(os.cpu_count() or 0, 1))
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PGDN v0 (torch)")
    parser.add_argument("--targets", type=Path, default=None, help="ACP targets JSONL")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Split manifest JSON produced by pgdn.data.build_acp (data/splits/manifest.json)",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="random",
        choices=["random", "temporal"],
        help="Which split strategy to use from the manifest",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "dev", "test"],
        help="Which split partition to train on",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows read from targets")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset instead of JSONL")
    parser.add_argument("--synthetic-n", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--batching",
        type=str,
        default="graph",
        choices=["flat", "graph"],
        help="Batching strategy. 'graph' yields single-graph batches for Pairformer coupling.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=_default_num_workers())
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=["none", "no_pairformer", "no_diffusion", "no_constraint_loss"],
    )
    parser.add_argument("--pairformer-blocks", type=int, default=2)
    parser.add_argument("--recycle", type=int, default=1)
    parser.add_argument("--diffusion-timesteps", type=int, default=100)
    parser.add_argument(
        "--diffusion-schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine", "fast"],
        help="Diffusion beta schedule.",
    )
    parser.add_argument(
        "--diffusion-pred",
        type=str,
        default="eps",
        choices=["eps", "v"],
        help="Diffusion prediction type.",
    )
    parser.add_argument(
        "--cond-dropout",
        type=float,
        default=0.1,
        help="Conditioning dropout probability for CFG training (0 disables).",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="EMA decay for diffusion net weights (0 disables).",
    )
    parser.add_argument(
        "--constraint-slot-weights",
        type=float,
        nargs=4,
        default=[1.0, 1.0, 1.0, 1.0],
        metavar=("WI", "WM", "WN", "WC"),
        help="Per-slot constraint weights for I,M,N,C.",
    )
    parser.add_argument(
        "--constraint-dim-weights",
        type=float,
        nargs=8,
        default=None,
        metavar=("D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"),
        help="Optional per-dimension weights within a slot (8 values).",
    )
    parser.add_argument(
        "--artic-loss-weight",
        type=float,
        default=0.0,
        help="Weight for optional articulatory constraint term (>= 0).",
    )
    parser.add_argument(
        "--enforce-range",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply tanh range contract to x0_hat and diffusion samples (default: enabled)",
    )
    parser.add_argument(
        "--range-scale",
        type=float,
        default=1.0,
        help="Scale factor for tanh range contract (x -> tanh(x/scale))",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/pgdn_torch_v0"))
    args = parser.parse_args()

    if float(args.artic_loss_weight) < 0.0:
        parser.error("--artic-loss-weight must be >= 0")

    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.synthetic:
        ds = SyntheticPGDNDataset(n=int(args.synthetic_n), seed=int(args.seed))
    else:
        if args.targets is None:
            raise SystemExit("--targets is required unless --synthetic is set")
        include_ids: set[str] | None = None
        if args.split_manifest is not None:
            include_ids = load_split_ids(
                Path(args.split_manifest),
                strategy=str(args.split_strategy),
                split=str(args.split),
            )
        ds = ACPJsonlDataset(Path(args.targets), limit=args.limit, include_ids=include_ids)

    pin_memory = device.type == "cuda"
    num_workers = max(int(args.num_workers), 0)
    common_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_pgdn,
    }
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = 2
        common_kwargs["persistent_workers"] = True

    graph_sampler: GraphBatchSampler | None = None
    if (not args.synthetic) and str(args.batching) == "graph":
        graph_ids = getattr(ds, "graph_ids", None)
        if not isinstance(graph_ids, list) or len(graph_ids) != len(ds):
            raise RuntimeError("graph batching requires dataset.graph_ids list")
        graph_sampler = GraphBatchSampler(
            graph_ids=graph_ids,
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            drop_last=False,
            shuffle_graphs=True,
            shuffle_within_graph=True,
        )
        dl = DataLoader(ds, batch_sampler=graph_sampler, **common_kwargs)
    else:
        dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, **common_kwargs)

    model = PGDNTorchV0(
        ablation=str(args.ablation),
        pairformer_blocks=int(args.pairformer_blocks),
        recycle=int(args.recycle),
        diffusion_timesteps=int(args.diffusion_timesteps),
        diffusion_schedule=str(args.diffusion_schedule),
        diffusion_pred=str(args.diffusion_pred),
        cond_dropout=float(args.cond_dropout),
        enforce_range=bool(args.enforce_range),
        range_scale=float(args.range_scale),
        constraint_slot_weights=list(args.constraint_slot_weights),
        constraint_dim_weights=list(args.constraint_dim_weights)
        if args.constraint_dim_weights is not None
        else None,
        artic_loss_weight=float(args.artic_loss_weight),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    ema_decay = float(args.ema_decay)
    use_ema = ema_decay > 0.0 and ema_decay < 1.0
    ema_state: dict[str, torch.Tensor] | None = None
    if use_ema:
        ema_state = {k: v.detach().clone() for k, v in model.diffusion.net.state_dict().items()}

    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / "train_metrics.jsonl"
    ckpt_path = args.out / f"checkpoint_{args.ablation}.pt"

    t_start = time.time()
    step = 0
    artic_constraint_sum = 0.0
    artic_constraint_count = 0
    articulatory_batches_seen = 0
    articulatory_mask_sum = 0.0
    articulatory_mask_nonzero_batches = 0
    for epoch in range(int(args.epochs)):
        if graph_sampler is not None:
            graph_sampler.set_epoch(epoch)
        for batch in dl:
            step += 1
            if graph_sampler is not None:
                gids = cast(list[str], batch["graph_id"])
                if gids and any(g != gids[0] for g in gids[1:]):
                    raise RuntimeError("graph batching violated: batch contains multiple graph_id values")
            target = cast(torch.Tensor, batch["target_vector"]).to(device, non_blocking=True)
            slot_mask = cast(torch.Tensor, batch["slot_mask"]).to(device, non_blocking=True)
            articulatory_vector_obj = batch.get("articulatory_vector")
            articulatory_mask_obj = batch.get("articulatory_mask")
            articulatory_vector = (
                articulatory_vector_obj.to(device, non_blocking=True)
                if isinstance(articulatory_vector_obj, torch.Tensor)
                else None
            )
            articulatory_mask = (
                articulatory_mask_obj.to(device, non_blocking=True)
                if isinstance(articulatory_mask_obj, torch.Tensor)
                else None
            )
            batch_has_articulatory = articulatory_vector is not None and articulatory_mask is not None
            batch_articulatory_mask_sum = 0.0
            if batch_has_articulatory and articulatory_mask is not None:
                batch_articulatory_mask_sum = float(torch.sum(articulatory_mask.detach()).cpu().item())
                articulatory_batches_seen += 1
                articulatory_mask_sum += batch_articulatory_mask_sum
                if batch_articulatory_mask_sum > 0.0:
                    articulatory_mask_nonzero_batches += 1

            opt.zero_grad(set_to_none=True)
            loss, terms = model.forward_loss(
                target,
                slot_mask,
                articulatory_vector=articulatory_vector,
                articulatory_mask=articulatory_mask,
            )
            loss.backward()
            opt.step()

            artic_constraint_sum += float(terms.artic_constraint_mean)
            artic_constraint_count += 1

            if use_ema and ema_state is not None:
                # Maintain EMA weights for the diffusion net only.
                net_state = model.diffusion.net.state_dict()
                for k, v in net_state.items():
                    ema_state[k] = ema_decay * ema_state[k] + (1.0 - ema_decay) * v.detach()

            if step % 20 == 0 or step == 1:
                row = {
                    "epoch": epoch + 1,
                    "step": step,
                    "ablation": args.ablation,
                    "device": str(device),
                    "loss": terms.total_loss,
                    "denoise_mse": terms.denoise_mse,
                    "constraint_loss": terms.constraint_loss,
                    "artic_loss_weight": float(args.artic_loss_weight),
                    "artic_constraint_mean": terms.artic_constraint_mean,
                    "batch_has_articulatory": bool(batch_has_articulatory),
                    "batch_articulatory_mask_sum": float(batch_articulatory_mask_sum),
                    "constraint_I": terms.constraint_I,
                    "constraint_M": terms.constraint_M,
                    "constraint_N": terms.constraint_N,
                    "constraint_C": terms.constraint_C,
                    "seconds": time.time() - t_start,
                }
                print(json.dumps(row, ensure_ascii=False), flush=True)
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save checkpoint.
    observed_artic_constraint_mean = artic_constraint_sum / max(artic_constraint_count, 1)
    torch.save(
        {
            "ablation": args.ablation,
            "seed": args.seed,
            "pairformer_blocks": int(args.pairformer_blocks),
            "recycle": int(args.recycle),
            "diffusion_timesteps": int(args.diffusion_timesteps),
            "diffusion_schedule": str(args.diffusion_schedule),
            "diffusion_pred": str(args.diffusion_pred),
            "cond_dropout": float(args.cond_dropout),
            "ema_decay": float(args.ema_decay),
            "enforce_range": bool(args.enforce_range),
            "range_scale": float(args.range_scale),
            "constraint_slot_weights": [float(x) for x in args.constraint_slot_weights],
            "constraint_dim_weights": [float(x) for x in args.constraint_dim_weights]
            if args.constraint_dim_weights is not None
            else None,
            "artic_loss_weight": float(args.artic_loss_weight),
            "artic_constraint_mean": float(observed_artic_constraint_mean),
            "articulatory_batches_seen": int(articulatory_batches_seen),
            "articulatory_mask_sum": float(articulatory_mask_sum),
            "articulatory_mask_nonzero_batches": int(articulatory_mask_nonzero_batches),
            "batching": str(args.batching),
            "split_manifest": str(args.split_manifest) if args.split_manifest else None,
            "split_strategy": str(args.split_strategy) if args.split_manifest else None,
            "split": str(args.split) if args.split_manifest else None,
            "model_state": model.state_dict(),
            "diffusion_net_ema": {
                k: v.detach().cpu() for k, v in ema_state.items()
            }
            if use_ema and ema_state is not None
            else None,
        },
        ckpt_path,
    )
    print(f"checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
