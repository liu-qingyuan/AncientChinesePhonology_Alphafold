from __future__ import annotations

import torch


def _resolve_proxy_articulatory_reference(
    target_vector: torch.Tensor,
    slot_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    slots = target_vector.reshape(-1, 4, 8)
    slot_weights = slot_mask.to(device=target_vector.device, dtype=target_vector.dtype)

    ema_ref = slots[:, 1, :]
    ema_present = slot_weights[:, 1:2]

    fallback_slots = slots[:, (0, 2, 3), :]
    fallback_weights = slot_weights[:, (0, 2, 3)]
    fallback_denom = torch.clamp(torch.sum(fallback_weights, dim=1, keepdim=True), min=1.0)
    fallback_ref = torch.sum(fallback_slots * fallback_weights[:, :, None], dim=1) / fallback_denom

    ref = ema_ref * ema_present + fallback_ref * (1.0 - ema_present)
    ref_mask = torch.ones_like(ema_present)
    return ref, ref_mask


def articulatory_constraint_loss(
    predicted_vector: torch.Tensor,
    target_vector: torch.Tensor,
    slot_mask: torch.Tensor,
    articulatory_vector: torch.Tensor | None = None,
    articulatory_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    predicted_slots = predicted_vector.reshape(-1, 4, 8)
    predicted_artic = predicted_slots[:, 1, :]

    if articulatory_vector is not None:
        ref = articulatory_vector.to(device=predicted_vector.device, dtype=predicted_vector.dtype)
        if tuple(ref.shape) != tuple(predicted_artic.shape):
            raise ValueError(
                f"articulatory_vector has wrong shape: got={tuple(ref.shape)} want={tuple(predicted_artic.shape)}"
            )
        if articulatory_mask is None:
            ref_mask = torch.ones((int(ref.shape[0]), 1), device=ref.device, dtype=ref.dtype)
        else:
            raw_mask = articulatory_mask.to(device=ref.device, dtype=ref.dtype)
            ref_mask = raw_mask.reshape(int(ref.shape[0]), -1)
            if int(ref_mask.shape[1]) > 1:
                ref_mask = ref_mask[:, :1]
    else:
        ref, ref_mask = _resolve_proxy_articulatory_reference(target_vector=target_vector, slot_mask=slot_mask)
        ref = ref.to(device=predicted_vector.device, dtype=predicted_vector.dtype)
        ref_mask = ref_mask.to(device=predicted_vector.device, dtype=predicted_vector.dtype)

    per_row = torch.mean(torch.abs(predicted_artic - ref), dim=1, keepdim=True)
    weighted = per_row * ref_mask
    denom = torch.clamp(torch.sum(ref_mask), min=1.0)
    return torch.sum(weighted) / denom
