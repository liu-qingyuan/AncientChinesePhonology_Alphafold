from __future__ import annotations

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
