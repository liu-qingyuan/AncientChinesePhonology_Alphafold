from pgdn.model import PGDNv0


def main() -> None:
    model = PGDNv0()
    batch = {
        "target_vector": [[0.1] * 32, [0.2] * 32, [0.3] * 32],
        "slot_mask": [
            {"I": 1, "M": 1, "N": 1, "C": 0},
            {"I": 1, "M": 0, "N": 1, "C": 0},
            {"I": 1, "M": 1, "N": 1, "C": 1},
        ],
    }
    loss, terms = model.forward_loss(batch)
    print(
        f"loss={loss:.6f} denoise_mse={terms['denoise_mse']:.6f} constraint_loss={terms['constraint_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
