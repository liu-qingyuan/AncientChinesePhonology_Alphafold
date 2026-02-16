from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MochaUtterance:
    speaker: str
    stem: str
    wav: Path | None
    ema: Path | None
    lab: Path | None


def index_mocha_timit(root: Path, speakers: list[str]) -> list[MochaUtterance]:
    """Index extracted Mocha-TIMIT v1_1 directory.

    Phase 2 placeholder: we only build an index of common modalities.
    No feature engineering is performed here.

    Expected layout:
      <root>/v1_1/<speaker>/**/*.(wav|ema|lab)
    """

    out: list[MochaUtterance] = []
    for speaker in speakers:
        base = root / "v1_1" / speaker
        if not base.exists():
            continue
        wavs = {p.stem: p for p in base.rglob("*.wav") if p.is_file()}
        emas = {p.stem: p for p in base.rglob("*.ema") if p.is_file()}
        labs = {p.stem: p for p in base.rglob("*.lab") if p.is_file()}
        stems = sorted(set(wavs) | set(emas) | set(labs))
        for stem in stems:
            out.append(
                MochaUtterance(
                    speaker=speaker,
                    stem=stem,
                    wav=wavs.get(stem),
                    ema=emas.get(stem),
                    lab=labs.get(stem),
                )
            )
    return out


def _hash_to_unit_floats(key: str, n: int) -> list[float]:
    out: list[float] = []
    counter = 0
    while len(out) < n:
        digest = hashlib.sha256(f"{key}|{counter}".encode("utf-8")).digest()
        for b in digest:
            out.append((float(b) / 127.5) - 1.0)
            if len(out) >= n:
                break
        counter += 1
    return out


def _slot_vector(modality_key: str | None) -> list[float]:
    if modality_key is None:
        return [0.0] * 8
    return _hash_to_unit_floats(modality_key, 8)


def _row_from_utterance(utt: MochaUtterance) -> dict[str, object]:
    rid = f"mocha:{utt.speaker}:{utt.stem}"
    wav_key = str(utt.wav) if utt.wav is not None else None
    ema_key = str(utt.ema) if utt.ema is not None else None
    lab_key = str(utt.lab) if utt.lab is not None else None
    aux_key = f"{utt.speaker}:{utt.stem}:aux"

    target_vector = (
        _slot_vector(wav_key)
        + _slot_vector(ema_key)
        + _slot_vector(lab_key)
        + _slot_vector(aux_key)
    )
    articulatory_vector = _slot_vector(ema_key)
    articulatory_mask = [1.0 if ema_key is not None else 0.0]
    mask = {
        "I": 1 if wav_key is not None else 0,
        "M": 1 if ema_key is not None else 0,
        "N": 1 if lab_key is not None else 0,
        "C": 1,
    }
    return {
        "record_id": rid,
        "graph_id": f"speaker:{utt.speaker}",
        "source": "mocha_timit_sidecar_v1",
        "speaker": utt.speaker,
        "stem": utt.stem,
        "target_vector": target_vector,
        "mask": mask,
        "articulatory_vector": articulatory_vector,
        "articulatory_mask": articulatory_mask,
        "paths": {
            "wav": wav_key,
            "ema": ema_key,
            "lab": lab_key,
        },
    }


def build_mocha_sidecar(
    root: Path,
    speakers: list[str],
    out_dir: Path,
    limit: int | None = None,
) -> dict[str, object]:
    rows = [_row_from_utterance(u) for u in index_mocha_timit(root, speakers)]
    if limit is not None:
        rows = rows[: max(int(limit), 0)]

    if not rows:
        raise ValueError("no Mocha rows found for sidecar build")

    out_dir.mkdir(parents=True, exist_ok=True)
    targets_path = out_dir / "mocha_sidecar_targets.jsonl"
    split_path = out_dir / "mocha_sidecar_splits.json"
    meta_path = out_dir / "mocha_sidecar_meta.json"

    with targets_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ids = [str(r["record_id"]) for r in rows]
    n = len(ids)
    n_train = max(1, int(n * 0.7))
    n_dev = max(1, int(n * 0.15))
    if n_train + n_dev >= n:
        n_dev = 1 if n > 1 else 0
    random_splits = {
        "train": ids[:n_train],
        "dev": ids[n_train : n_train + n_dev],
        "test": ids[n_train + n_dev :],
    }
    if not random_splits["test"]:
        random_splits["test"] = ids[-1:]

    split_obj = {
        "random": random_splits,
        "temporal": random_splits,
    }
    split_path.write_text(json.dumps(split_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    speakers_seen = sorted({str(r.get("speaker", "")) for r in rows})
    articulatory_rows = 0
    for r in rows:
        raw_mask = r.get("articulatory_mask")
        if isinstance(raw_mask, list) and raw_mask:
            articulatory_rows += int(float(raw_mask[0]) > 0.0)
    meta = {
        "source_root": str(root),
        "speakers_requested": speakers,
        "speakers_indexed": speakers_seen,
        "rows": len(rows),
        "articulatory_rows": articulatory_rows,
        "targets_path": str(targets_path),
        "split_manifest_path": str(split_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "targets": targets_path,
        "splits": split_path,
        "meta": meta_path,
        "rows": len(rows),
    }
