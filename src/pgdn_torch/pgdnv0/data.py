from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class PGDNSample:
    record_id: str
    graph_id: str
    target_vector: torch.Tensor  # [32]
    slot_mask: torch.Tensor  # [4]


def _mask_to_tensor(mask: dict) -> torch.Tensor:
    return torch.tensor([float(mask.get(k, 0.0)) for k in ("I", "M", "N", "C")], dtype=torch.float32)


class ACPJsonlDataset(Dataset[PGDNSample]):
    def __init__(
        self,
        path: Path,
        limit: int | None = None,
        include_ids: set[str] | None = None,
    ) -> None:
        self.rows: list[dict] = []
        self.graph_ids: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if include_ids is not None:
                    rid = row.get("record_id")
                    if rid is None or str(rid) not in include_ids:
                        continue
                self.rows.append(row)
                self.graph_ids.append(str(row.get("graph_id", "")))
                if limit is not None and len(self.rows) >= int(limit):
                    break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> PGDNSample:
        r = self.rows[idx]
        vec = torch.tensor([float(x) for x in r["target_vector"]], dtype=torch.float32)
        mask = _mask_to_tensor(r.get("mask", {}))
        return PGDNSample(
            record_id=str(r.get("record_id", idx)),
            graph_id=str(r.get("graph_id", "")),
            target_vector=vec,
            slot_mask=mask,
        )


class SyntheticPGDNDataset(Dataset[PGDNSample]):
    def __init__(self, n: int = 2048, seed: int = 0) -> None:
        self.n = n
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> PGDNSample:
        rng = random.Random(self.seed + idx)
        vec = torch.tensor([rng.uniform(-1.0, 1.0) for _ in range(32)], dtype=torch.float32)
        # slot mask: randomly mask out some slots to exercise constraint loss
        mask = torch.tensor([float(rng.random() > 0.25) for _ in range(4)], dtype=torch.float32)
        return PGDNSample(
            record_id=f"synthetic-{idx}",
            graph_id="graph:synthetic",
            target_vector=vec,
            slot_mask=mask,
        )


def collate_pgdn(batch: list[PGDNSample]) -> dict[str, object]:
    return {
        "record_id": [b.record_id for b in batch],
        "graph_id": [b.graph_id for b in batch],
        "target_vector": torch.stack([b.target_vector for b in batch], dim=0),
        "slot_mask": torch.stack([b.slot_mask for b in batch], dim=0),
    }


class GraphBatchSampler(Sampler[list[int]]):
    """Yield batches that contain indices from a single graph_id."""

    def __init__(
        self,
        graph_ids: list[str],
        batch_size: int,
        seed: int = 0,
        drop_last: bool = False,
        shuffle_graphs: bool = True,
        shuffle_within_graph: bool = True,
    ) -> None:
        self.graph_ids = list(graph_ids)
        self.batch_size = max(int(batch_size), 1)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.shuffle_graphs = bool(shuffle_graphs)
        self.shuffle_within_graph = bool(shuffle_within_graph)
        self.epoch = 0

        graph_to_indices: dict[str, list[int]] = {}
        for idx, gid in enumerate(self.graph_ids):
            graph_to_indices.setdefault(str(gid), []).append(int(idx))
        self.graph_to_indices = graph_to_indices

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        graphs = list(self.graph_to_indices.keys())
        if self.shuffle_graphs:
            rng.shuffle(graphs)

        for gid in graphs:
            idxs = list(self.graph_to_indices[gid])
            if self.shuffle_within_graph:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for idxs in self.graph_to_indices.values():
            if self.drop_last:
                total += len(idxs) // self.batch_size
            else:
                total += (len(idxs) + self.batch_size - 1) // self.batch_size
        return total
