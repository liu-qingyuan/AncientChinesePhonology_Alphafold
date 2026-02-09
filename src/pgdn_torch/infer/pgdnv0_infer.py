from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader

from ..pgdnv0.data import ACPJsonlDataset, GraphBatchSampler, SyntheticPGDNDataset, collate_pgdn
from ..pgdnv0.model import PGDNTorchV0
from ..pgdnv0.splits import load_split_ids


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise RuntimeError(f"unexpected checkpoint format: {path}")
    return ckpt


def _character_from_record_id(record_id: str) -> tuple[str, bool]:
    rid = str(record_id)
    if ":" not in rid:
        return rid, True
    return rid.split(":", 1)[0], False


def _derive_shared_noise_seed(base_seed: int, group_key: str, sample_index: int) -> int:
    h = hashlib.sha256()
    h.update(str(int(base_seed)).encode("ascii"))
    h.update(b"\0")
    h.update(str(int(sample_index)).encode("ascii"))
    h.update(b"\0")
    h.update(str(group_key).encode("utf-8"))
    seed64 = int.from_bytes(h.digest()[:8], "little", signed=False)
    # torch.Generator seeds are expected to fit in signed 64-bit range.
    return int(seed64 % (2**63 - 1))


def _fingerprint_noise_row(x_row_cpu_f32: torch.Tensor) -> str:
    x = x_row_cpu_f32.detach().float().cpu().contiguous()
    b = x.numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


def _fingerprint_row(x_row: torch.Tensor) -> str:
    return _fingerprint_noise_row(x_row)


def _build_shared_x_T(
    keys: list[str],
    base_seed: int,
    sample_index: int,
    target_dim: int,
    noise_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, str]]:
    uniq: dict[str, torch.Tensor] = {}
    fps: dict[str, str] = {}
    for k in keys:
        if k in uniq:
            continue
        seed_i = _derive_shared_noise_seed(int(base_seed), k, int(sample_index))
        g = torch.Generator(device=torch.device("cpu"))
        g.manual_seed(int(seed_i))
        row = torch.randn((int(target_dim),), generator=g, device="cpu", dtype=torch.float32) * float(noise_scale)
        uniq[k] = row
        fps[k] = _fingerprint_noise_row(row)
    x_t = torch.stack([uniq[k] for k in keys], dim=0)
    return x_t.to(device=device, dtype=dtype), fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer PGDN v0 (torch)")
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
        help="Which split strategy to use from the manifest",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev", "test"],
        help="Which split partition to run inference on",
    )
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--batching",
        type=str,
        default="graph",
        choices=["flat", "graph"],
        help="Batching strategy. 'graph' yields single-graph batches for Pairformer coupling.",
    )
    parser.add_argument(
        "--enforce-range",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply tanh range contract to diffusion outputs (default: from checkpoint, else enabled)",
    )
    parser.add_argument(
        "--range-scale",
        type=float,
        default=None,
        help="Scale factor for tanh range contract (x -> tanh(x/scale)); default from checkpoint",
    )
    parser.add_argument(
        "--range-threshold",
        type=float,
        default=1.2,
        help="Threshold for reporting max_abs range violations",
    )
    parser.add_argument(
        "--fail-on-range-violation",
        action="store_true",
        help="Exit non-zero if enforce-range is enabled and max_abs exceeds --range-threshold",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale (1.0 disables CFG).",
    )
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use EMA diffusion net weights if present in checkpoint (default: enabled when available)",
    )
    parser.add_argument(
        "--group-coupling",
        type=str,
        default="none",
        choices=["none", "shared_noise", "shared_denoise"],
        help="Group coupling mode for diffusion sampling (default: none).",
    )
    parser.add_argument(
        "--group-key",
        type=str,
        default="character",
        choices=["character"],
        help="Grouping key for coupling. For ACP, 'character' is derived from record_id '<character>:<language>'.",
    )
    parser.add_argument(
        "--fail-on-coupling-mismatch",
        action="store_true",
        help="Exit non-zero if coupling is enabled and a key maps to multiple x_T fingerprints within the run.",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/pgdn_torch_v0_infer"))
    args = parser.parse_args()

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
        ds = ACPJsonlDataset(Path(args.targets), limit=int(args.limit), include_ids=include_ids)

    ckpt = _load_checkpoint(args.checkpoint, device)
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
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    have_ema = bool(ckpt.get("diffusion_net_ema"))
    use_ema = bool(args.use_ema) if args.use_ema is not None else bool(have_ema)
    if use_ema and have_ema:
        ema_state = ckpt.get("diffusion_net_ema")
        if isinstance(ema_state, dict):
            model.diffusion.net.load_state_dict(ema_state)
    model.eval()

    # Build DataLoader for inference.
    num_workers = 0
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
            shuffle_graphs=False,
            shuffle_within_graph=False,
        )
        dl = DataLoader(ds, batch_sampler=graph_sampler, num_workers=num_workers, collate_fn=collate_pgdn)
    else:
        dl = DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_pgdn,
        )

    # Precompute conditioning per batch (deterministic) so sampling loop is cheap.
    batches: list[dict[str, object]] = []
    coupling_malformed_record_id = 0
    coupling_unique_keys: set[str] = set()
    recycle_deltas_sum: list[float] = []
    recycle_batches = 0
    with torch.no_grad():
        for batch in dl:
            record_ids = cast(list[str], batch["record_id"])
            group_keys: list[str] = []
            for rid in record_ids:
                k, malformed = _character_from_record_id(rid)
                if malformed:
                    coupling_malformed_record_id += 1
                group_keys.append(k)
                coupling_unique_keys.add(k)
            if graph_sampler is not None:
                gids = cast(list[str], batch["graph_id"])
                if gids and any(g != gids[0] for g in gids[1:]):
                    raise RuntimeError("graph batching violated: batch contains multiple graph_id values")
            target = cast(torch.Tensor, batch["target_vector"]).to(device)
            slot_mask = cast(torch.Tensor, batch["slot_mask"]).to(device)

            single, pair = model.embedder(target)
            if model.ablation != "no_pairformer":
                single, _pair, deltas = model.pairformer.forward_with_deltas(single, pair)
                if not recycle_deltas_sum:
                    recycle_deltas_sum = [0.0 for _ in deltas]
                if len(deltas) == len(recycle_deltas_sum):
                    for i, d in enumerate(deltas):
                        recycle_deltas_sum[i] += float(d)
                recycle_batches += 1

            batches.append(
                {
                    "record_id": record_ids,
                    "group_key": group_keys,
                    "slot_mask": slot_mask.detach().cpu(),
                    "cond": single.detach(),
                }
            )

    args.out.mkdir(parents=True, exist_ok=True)
    samples_dir = args.out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Optional ranking using existing src/pgdn/confidence.py (read-only).
    try:
        from pgdn.confidence import ranking_terms_for_record

        have_ranking = True
    except Exception:
        ranking_terms_for_record = None
        have_ranking = False

    ranking_rows: list[dict[str, object]] = []
    range_threshold = float(args.range_threshold)
    max_abs_overall = 0.0
    abs_gt_1 = 0
    abs_total = 0

    coupling_mode = str(args.group_coupling)
    coupling_shared_noise = coupling_mode in {"shared_noise", "shared_denoise"}
    coupling_shared_denoise = coupling_mode == "shared_denoise"
    coupling_noise_scale = 0.3
    coupling_max_keys_store = 2048
    coupling_mismatch_count = 0
    coupling_fingerprints_by_sample: list[dict[str, str]] = []
    coupling_truncated_by_sample: list[bool] = []
    coupling_digest_xor64_by_sample: list[int] = []

    shared_denoise_mismatch_count = 0
    shared_denoise_fingerprints_by_sample: list[dict[str, str]] = []
    shared_denoise_truncated_by_sample: list[bool] = []
    shared_denoise_digest_xor64_by_sample: list[int] = []

    for s in range(int(args.samples)):
        all_record_ids: list[str] = []
        all_vectors: list[torch.Tensor] = []
        fp_seen: dict[str, str] = {}
        fp_truncated = False
        digest_xor64 = 0
        digest_seen: set[str] = set()
        # Precompute shared unconditional denoise predictions per group (global, batch-order independent).
        shared_y_u_by_step: torch.Tensor | None = None
        key_to_group_index: dict[str, int] = {}
        if coupling_shared_denoise and float(args.cfg_scale) != 1.0:
            uniq_keys_sorted = sorted(coupling_unique_keys)
            key_to_group_index = {k: i for i, k in enumerate(uniq_keys_sorted)}
            x_T_groups, _fps_groups = _build_shared_x_T(
                keys=uniq_keys_sorted,
                base_seed=int(args.seed),
                sample_index=int(s),
                target_dim=int(model.diffusion.target_dim),
                noise_scale=float(coupling_noise_scale),
                device=device,
                dtype=torch.float32,
            )
            shared_y_u_by_step = model.diffusion.precompute_shared_uncond_y_by_step(
                x_T=x_T_groups,
                num_steps=int(args.num_steps),
            )

            # Fingerprint only the first step's unconditional prediction per key (bounded size).
            fp_seen_d: dict[str, str] = {}
            fp_truncated_d = False
            digest_xor64_d = 0
            digest_seen_d: set[str] = set()
            step0 = shared_y_u_by_step[0].detach().cpu()
            for i, k in enumerate(uniq_keys_sorted):
                fp = _fingerprint_row(step0[i])
                if (not fp_truncated_d) and (len(fp_seen_d) < int(coupling_max_keys_store)):
                    fp_seen_d[k] = fp
                else:
                    fp_truncated_d = True
                if k not in digest_seen_d:
                    hh = hashlib.sha256()
                    hh.update(k.encode("utf-8"))
                    hh.update(b"\0")
                    hh.update(fp.encode("ascii"))
                    digest_xor64_d ^= int.from_bytes(hh.digest()[:8], "little", signed=False)
                    digest_seen_d.add(k)

            shared_denoise_fingerprints_by_sample.append(fp_seen_d)
            shared_denoise_truncated_by_sample.append(bool(fp_truncated_d))
            shared_denoise_digest_xor64_by_sample.append(int(digest_xor64_d))
        else:
            shared_denoise_fingerprints_by_sample.append({})
            shared_denoise_truncated_by_sample.append(False)
            shared_denoise_digest_xor64_by_sample.append(0)

        for b in batches:
            b_record_ids = cast(list[str], b["record_id"])
            b_keys = cast(list[str], b["group_key"])
            b_slot_mask = cast(torch.Tensor, b["slot_mask"])
            b_cond = cast(torch.Tensor, b["cond"])

            x_T: torch.Tensor | None = None
            if coupling_shared_noise:
                x_T, fps = _build_shared_x_T(
                    keys=b_keys,
                    base_seed=int(args.seed),
                    sample_index=int(s),
                    target_dim=int(model.diffusion.target_dim),
                    noise_scale=float(coupling_noise_scale),
                    device=device,
                    dtype=b_cond.dtype,
                )
                for k, fp in fps.items():
                    prev = fp_seen.get(k)
                    if prev is not None and prev != fp:
                        coupling_mismatch_count += 1
                        if bool(args.fail_on_coupling_mismatch):
                            raise SystemExit(3)
                    if prev is None:
                        if (not fp_truncated) and (len(fp_seen) < int(coupling_max_keys_store)):
                            fp_seen[k] = fp
                        else:
                            fp_truncated = True
                    if k not in digest_seen:
                        hh = hashlib.sha256()
                        hh.update(k.encode("utf-8"))
                        hh.update(b"\0")
                        hh.update(fp.encode("ascii"))
                        digest_xor64 ^= int.from_bytes(hh.digest()[:8], "little", signed=False)
                        digest_seen.add(k)

            shared_group_index: torch.Tensor | None = None
            if shared_y_u_by_step is not None and key_to_group_index:
                shared_group_index = torch.tensor(
                    [key_to_group_index.get(k, 0) for k in b_keys],
                    device=device,
                    dtype=torch.long,
                )

            vec = model.diffusion.sample(
                b_cond,
                seed=int(args.seed) + s,
                num_steps=int(args.num_steps),
                noise_scale=float(coupling_noise_scale),
                enforce_range=bool(enforce_range),
                range_scale=float(range_scale),
                cfg_scale=float(args.cfg_scale),
                x_T=x_T,
                shared_denoise_uncond_y_by_step=shared_y_u_by_step,
                shared_denoise_group_index=shared_group_index,
            )
            vec_cpu = vec.detach().cpu()
            all_record_ids.extend(b_record_ids)
            all_vectors.append(vec_cpu)

            with torch.no_grad():
                max_abs = float(torch.max(torch.abs(vec_cpu)).item())
                max_abs_overall = max(max_abs_overall, max_abs)
                abs_gt_1 += int((torch.abs(vec_cpu) > 1.0).sum().item())
                abs_total += int(vec_cpu.numel())

            if have_ranking and ranking_terms_for_record is not None:
                vec_list = vec_cpu.tolist()
                for i, rid in enumerate(b_record_ids):
                    mask = {
                        "I": int(b_slot_mask[i, 0].item()),
                        "M": int(b_slot_mask[i, 1].item()),
                        "N": int(b_slot_mask[i, 2].item()),
                        "C": int(b_slot_mask[i, 3].item()),
                    }
                    terms = ranking_terms_for_record([vec_list[i]], mask)
                    ranking_rows.append({"record_id": rid, "sample": s, **terms})

        out_path = samples_dir / f"sample_{s:03d}.pt"
        torch.save(
            {"record_id": all_record_ids, "vector": torch.cat(all_vectors, dim=0)},
            out_path,
        )

        coupling_fingerprints_by_sample.append(fp_seen)
        coupling_truncated_by_sample.append(bool(fp_truncated))
        coupling_digest_xor64_by_sample.append(int(digest_xor64))

    if have_ranking and ranking_rows:
        csv_path = args.out / "ranking_scores.csv"
        fieldnames = list(ranking_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(ranking_rows)

    meta = {
        "checkpoint": str(args.checkpoint),
        "ablation": ablation,
        "device": str(device),
        "samples": int(args.samples),
        "num_steps": int(args.num_steps),
        "batch_size": int(args.batch_size),
        "batching": str(args.batching),
        "ranking": bool(have_ranking),
        "diffusion_schedule": str(ckpt.get("diffusion_schedule", "linear")),
        "diffusion_pred": str(ckpt.get("diffusion_pred", "eps")),
        "cond_dropout": float(ckpt.get("cond_dropout", 0.0)),
        "cfg_scale": float(args.cfg_scale),
        "have_ema": bool(have_ema),
        "use_ema": bool(use_ema),
        "ema_decay": float(ckpt.get("ema_decay", 0.0)),
        "enforce_range": bool(enforce_range),
        "range_scale": float(range_scale),
        "range_threshold": float(range_threshold),
        "range_max_abs": float(max_abs_overall),
        "range_frac_abs_gt_1": float(abs_gt_1 / max(abs_total, 1)),
        "range_ok": bool((not enforce_range) or (max_abs_overall <= range_threshold)),
        "recycle_deltas": [
            float(d / max(recycle_batches, 1)) for d in recycle_deltas_sum
        ],
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "split_strategy": str(args.split_strategy) if args.split_manifest else None,
        "split": str(args.split) if args.split_manifest else None,
    }

    coupling_block = {
        "enabled": bool(coupling_shared_noise),
        "mode": str(args.group_coupling),
        "group_key": str(args.group_key),
        "scope": "global_infer_run",
        "x_T_only": True,
        "base_seed": int(args.seed),
        "noise_scale": float(coupling_noise_scale),
        "target_dim": int(model.diffusion.target_dim),
        "group_count": int(len(coupling_unique_keys)),
        "malformed_record_id_count": int(coupling_malformed_record_id),
        "fingerprint_algo": "sha256(float32_cpu_bytes)",
        "max_keys_stored": int(coupling_max_keys_store),
        "mismatch_count": int(coupling_mismatch_count),
        "fail_fast": bool(args.fail_on_coupling_mismatch),
        "fingerprints_truncated_by_sample": coupling_truncated_by_sample,
        "fingerprints_digest_xor64_by_sample": coupling_digest_xor64_by_sample,
        "fingerprints_by_key": coupling_fingerprints_by_sample[0] if coupling_fingerprints_by_sample else {},
        "fingerprints_by_key_by_sample": coupling_fingerprints_by_sample,
    }
    coupling_meta: dict[str, object] = {"shared_noise": coupling_block}

    shared_denoise_block = {
        "enabled": bool(coupling_shared_denoise),
        "effective": bool(coupling_shared_denoise and float(args.cfg_scale) != 1.0),
        "mode": str(args.group_coupling),
        "group_key": str(args.group_key),
        "scope": "global_infer_run",
        "unconditional_branch_only": True,
        "base_seed": int(args.seed),
        "noise_scale": float(coupling_noise_scale),
        "target_dim": int(model.diffusion.target_dim),
        "group_count": int(len(coupling_unique_keys)),
        "fingerprint_algo": "sha256(float32_cpu_bytes)",
        "max_keys_stored": int(coupling_max_keys_store),
        "mismatch_count": int(shared_denoise_mismatch_count),
        "fail_fast": bool(args.fail_on_coupling_mismatch),
        "fingerprints_truncated_by_sample": shared_denoise_truncated_by_sample,
        "fingerprints_digest_xor64_by_sample": shared_denoise_digest_xor64_by_sample,
        "fingerprints_by_key": shared_denoise_fingerprints_by_sample[0]
        if shared_denoise_fingerprints_by_sample
        else {},
        "fingerprints_by_key_by_sample": shared_denoise_fingerprints_by_sample,
    }
    coupling_meta["shared_denoise"] = shared_denoise_block
    meta["coupling"] = coupling_meta
    if bool(enforce_range) and float(max_abs_overall) > float(range_threshold):
        print(
            json.dumps(
                {
                    "warning": "range_violation",
                    "max_abs": float(max_abs_overall),
                    "threshold": float(range_threshold),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if bool(args.fail_on_range_violation):
            raise SystemExit(2)
    (args.out / "infer_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
