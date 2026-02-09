from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.file_dataset import SimpleFileDataset
from ..data.hub import resolve_hub_dataset
from ..models.mlp import MLPClassifier


def _default_num_workers() -> int:
    # Safe default: avoid oversubscription in constrained environments.
    try:
        import os

        return min(4, max(os.cpu_count() or 0, 1))
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Torch smoke training: 1 optimizer step on hub-backed dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="phoible",
        help="Dataset key under data/external/links (default: phoible)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-files", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=_default_num_workers())
    parser.add_argument("--out", type=Path, default=Path("runs/torch_smoke"))
    args = parser.parse_args()

    ref = resolve_hub_dataset(args.dataset)
    ds = SimpleFileDataset(ref.root, max_files=args.max_files)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    # Prefetch/persistent workers only make sense with workers > 0.
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": max(args.num_workers, 0),
        "pin_memory": pin_memory,
    }
    if args.num_workers and args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    dl = DataLoader(ds, **loader_kwargs)

    model = MLPClassifier(in_dim=3, hidden=64, num_classes=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    t0 = time.time()
    batch = next(iter(dl))
    x = batch["x"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)

    opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    opt.step()
    dt = time.time() - t0

    metrics = {
        "dataset": ref.key,
        "device": str(device),
        "batch_size": int(x.shape[0]),
        "loss": float(loss.detach().cpu().item()),
        "seconds": float(dt),
        "cuda_available": bool(torch.cuda.is_available()),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    print(f"wrote={metrics_path}")


if __name__ == "__main__":
    main()
