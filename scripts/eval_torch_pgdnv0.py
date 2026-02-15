import argparse
import json
from pathlib import Path
from statistics import mean
from typing import cast

import torch


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise RuntimeError(f"unexpected checkpoint format: {path}")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval PGDN v0 (torch) on ACP dev split")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=None)
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
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev", "test"],
    )
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batching", type=str, default="graph", choices=["flat", "graph"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use EMA diffusion net weights if present (default: enabled when available)",
    )
    parser.add_argument(
        "--enforce-range",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply tanh range contract (default: from checkpoint, else enabled)",
    )
    parser.add_argument("--range-scale", type=float, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/pgdn_torch_v0_eval"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.targets is None:
        raise SystemExit("--targets is required")

    from pgdn.confidence import ranking_terms_for_record
    from pgdn_torch.pgdnv0.data import ACPJsonlDataset, GraphBatchSampler, collate_pgdn
    from pgdn_torch.pgdnv0.model import PGDNTorchV0
    from pgdn_torch.pgdnv0.splits import load_split_ids
    from torch.utils.data import DataLoader

    include_ids: set[str] | None = None
    if args.split_manifest is not None:
        include_ids = load_split_ids(
            Path(args.split_manifest),
            strategy=str(args.split_strategy),
            split=str(args.split),
        )
    ds = ACPJsonlDataset(Path(args.targets), limit=int(args.limit), include_ids=include_ids)

    ckpt = _load_checkpoint(Path(args.checkpoint), device)
    ablation = str(ckpt.get("ablation", "none"))

    ckpt_enforce_range = ckpt.get("enforce_range")
    ckpt_range_scale = ckpt.get("range_scale")
    enforce_range = (
        bool(args.enforce_range)
        if args.enforce_range is not None
        else bool(ckpt_enforce_range) if ckpt_enforce_range is not None else True
    )
    range_scale = (
        float(args.range_scale)
        if args.range_scale is not None
        else float(ckpt_range_scale) if ckpt_range_scale is not None else 1.0
    )

    model = PGDNTorchV0(
        ablation=ablation,
        pairformer_blocks=int(ckpt.get("pairformer_blocks", 2)),
        recycle=int(ckpt.get("recycle", 1)),
        diffusion_timesteps=int(ckpt.get("diffusion_timesteps", 100)),
        diffusion_schedule=str(ckpt.get("diffusion_schedule", "linear")),
        diffusion_pred=str(ckpt.get("diffusion_pred", "eps")),
        cond_dropout=float(ckpt.get("cond_dropout", 0.0)),
        enforce_range=bool(enforce_range),
        range_scale=float(range_scale),
        constraint_slot_weights=cast(list[float] | None, ckpt.get("constraint_slot_weights")),
        constraint_dim_weights=cast(list[float] | None, ckpt.get("constraint_dim_weights")),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    have_ema = bool(ckpt.get("diffusion_net_ema"))
    use_ema = bool(args.use_ema) if args.use_ema is not None else bool(have_ema)
    if use_ema and have_ema:
        ema_state = ckpt.get("diffusion_net_ema")
        if isinstance(ema_state, dict):
            model.diffusion.net.load_state_dict(ema_state)

    model.eval()

    graph_sampler: GraphBatchSampler | None = None
    if str(args.batching) == "graph":
        graph_ids = getattr(ds, "graph_ids", None)
        if not isinstance(graph_ids, list) or len(graph_ids) != len(ds):
            raise RuntimeError("graph batching requires dataset.graph_ids list")
        graph_sampler = GraphBatchSampler(
            graph_ids=graph_ids,
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            drop_last=False,
            shuffle_graphs=False,
            shuffle_within_graph=False,
        )
        dl = DataLoader(ds, batch_sampler=graph_sampler, num_workers=0, collate_fn=collate_pgdn)
    else:
        dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_pgdn)

    # Aggregate metrics.
    max_abs_overall = 0.0
    abs_gt_1 = 0
    abs_total = 0
    abs_gt_1_penalty_sum = 0.0

    constraint_penalties: list[float] = []
    ranking_scores: list[float] = []
    uncertainties: list[float] = []
    impossible_penalties: list[float] = []

    with torch.no_grad():
        for batch in dl:
            record_ids = cast(list[str], batch["record_id"])
            gids = cast(list[str], batch["graph_id"])
            if graph_sampler is not None and gids and any(g != gids[0] for g in gids[1:]):
                raise RuntimeError("graph batching violated: batch contains multiple graph_id values")

            target = cast(torch.Tensor, batch["target_vector"]).to(device)
            slot_mask = cast(torch.Tensor, batch["slot_mask"]).to(device)

            single, pair = model.embedder(target)
            if model.ablation != "no_pairformer":
                single, _pair = model.pairformer(single, pair)

            # Sample S vectors.
            sample_vecs: list[torch.Tensor] = []
            for s in range(int(args.samples)):
                v = model.diffusion.sample(
                    single,
                    seed=int(args.seed) + s,
                    num_steps=int(args.num_steps),
                    noise_scale=0.3,
                    enforce_range=bool(enforce_range),
                    range_scale=float(range_scale),
                    cfg_scale=float(args.cfg_scale),
                )
                v_cpu = v.detach().cpu()
                sample_vecs.append(v_cpu)

                max_abs = float(torch.max(torch.abs(v_cpu)).item())
                max_abs_overall = max(max_abs_overall, max_abs)
                abs_gt_1 += int((torch.abs(v_cpu) > 1.0).sum().item())
                abs_total += int(v_cpu.numel())
                abs_gt_1_penalty_sum += float(torch.relu(torch.abs(v_cpu) - 1.0).sum().item())

            # Per-record confidence terms over S samples.
            # Shape: [S, B, 32]
            stack = torch.stack(sample_vecs, dim=0)
            avg = torch.mean(stack, dim=0)  # [B, 32]

            # Constraint violation penalty: mean(|v| * (1-mask)).
            expanded = slot_mask[:, :, None].repeat(1, 1, 8).reshape(-1, 32).detach().cpu()
            viol = torch.abs(avg.detach().cpu()) * (1.0 - expanded)
            constraint_penalties.extend(viol.mean(dim=1).tolist())

            # Ranking summary.
            stack_list = stack.tolist()
            for i, rid in enumerate(record_ids):
                mask = {
                    "I": int(slot_mask[i, 0].item()),
                    "M": int(slot_mask[i, 1].item()),
                    "N": int(slot_mask[i, 2].item()),
                    "C": int(slot_mask[i, 3].item()),
                }
                terms = ranking_terms_for_record([stack_list[s][i] for s in range(int(args.samples))], mask)
                ranking_scores.append(float(terms["ranking_score"]))
                uncertainties.append(float(terms["uncertainty_mean"]))
                impossible_penalties.append(float(terms["penalty_impossible_combo"]))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "eval.json"

    eval_obj = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "split": str(args.split),
        "limit": int(args.limit),
        "samples": int(args.samples),
        "num_steps": int(args.num_steps),
        "batching": str(args.batching),
        "batch_size": int(args.batch_size),
        "diffusion_schedule": str(ckpt.get("diffusion_schedule", "linear")),
        "diffusion_pred": str(ckpt.get("diffusion_pred", "eps")),
        "cfg_scale": float(args.cfg_scale),
        "have_ema": bool(have_ema),
        "use_ema": bool(use_ema),
        "ema_decay": float(ckpt.get("ema_decay", 0.0)),
        "enforce_range": bool(enforce_range),
        "range_scale": float(range_scale),
        "max_abs": float(max_abs_overall),
        "frac_abs_gt_1": float(abs_gt_1 / max(abs_total, 1)),
        "abs_gt_1_penalty_mean": float(abs_gt_1_penalty_sum / max(abs_total, 1)),
        "constraint_penalty_mean": float(mean(constraint_penalties) if constraint_penalties else 0.0),
        "constraint_penalty_min": float(min(constraint_penalties) if constraint_penalties else 0.0),
        "constraint_penalty_max": float(max(constraint_penalties) if constraint_penalties else 0.0),
        "ranking_score_mean": float(mean(ranking_scores) if ranking_scores else 0.0),
        "uncertainty_mean": float(mean(uncertainties) if uncertainties else 0.0),
        "impossible_penalty_mean": float(mean(impossible_penalties) if impossible_penalties else 0.0),
    }

    eval_path.write_text(json.dumps(eval_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(eval_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
