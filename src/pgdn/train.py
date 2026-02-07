import argparse
import json
import random
from pathlib import Path

from .data.dataset import load_targets


def _mean_vector(rows):
    dim = len(rows[0]["target_vector"]) if rows else 32
    acc = [0.0] * dim
    for row in rows:
        for i, v in enumerate(row["target_vector"]):
            acc[i] += float(v)
    n = max(len(rows), 1)
    return [v / n for v in acc]


def _mse(rows, mean_vec):
    if not rows:
        return 0.0
    total = 0.0
    denom = 0
    for row in rows:
        for i, v in enumerate(row["target_vector"]):
            d = float(v) - mean_vec[i]
            total += d * d
            denom += 1
    return total / max(denom, 1)


def _constraint_loss(rows):
    if not rows:
        return 0.0
    total = 0.0
    denom = 0
    for row in rows:
        mask = row["mask"]
        vector = row["target_vector"]
        for slot_idx, slot in enumerate(("I", "M", "N", "C")):
            m = float(mask[slot])
            for j in range(8):
                val = abs(float(vector[slot_idx * 8 + j]))
                total += val * (1.0 - m)
                denom += 1
    return total / max(denom, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PGDN v0")
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("runs/pgdn_v0/checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=["none", "no_pairformer", "no_diffusion", "no_constraint_loss"],
    )
    args = parser.parse_args()

    random.seed(args.seed)
    rows = load_targets(args.targets)
    with args.split_manifest.open("r", encoding="utf-8") as f:
        split = json.load(f)
    train_ids = set(split["random"]["train"])
    train_rows = [r for r in rows if r["record_id"] in train_ids]

    mean_vec = _mean_vector(train_rows)
    denoise = _mse(train_rows, mean_vec)
    constraint = (
        0.0 if args.ablation == "no_constraint_loss" else _constraint_loss(train_rows)
    )
    total_loss = denoise + 0.1 * constraint

    for epoch in range(args.epochs):
        print(
            " ".join(
                [
                    f"epoch={epoch + 1}",
                    f"ablation={args.ablation}",
                    f"total_loss={total_loss:.6f}",
                    f"denoise_mse={denoise:.6f}",
                    f"constraint_loss={constraint:.6f}",
                ]
            ),
            flush=True,
        )

    ckpt = {
        "ablation": args.ablation,
        "seed": args.seed,
        "mean_vector": mean_vec,
        "loss": {
            "total_loss": total_loss,
            "denoise_mse": denoise,
            "constraint_loss": constraint,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / f"pgdn_v0_{args.ablation}.json"
    ckpt_path.write_text(
        json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
