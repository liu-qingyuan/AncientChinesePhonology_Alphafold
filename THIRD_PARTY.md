# Third-Party Components

This workspace contains third-party code and datasets in subdirectories.
Each component retains its own upstream license.

If you publish this repository to GitHub, avoid re-licensing third-party code.
Prefer linking to upstream (git submodules) or excluding these directories
from the top-level repo and providing fetch instructions.

## Components observed in this workspace

| Path | License file | License (observed) | Notes |
|------|--------------|--------------------|-------|
| `alphafold3/` | `alphafold3/LICENSE` | CC BY-NC-SA 4.0 | Non-commercial + ShareAlike; be careful mixing with other code. |
| `Ancient-Chinese-Phonology/` | `Ancient-Chinese-Phonology/LICENSE` | GPL-3.0 | Copyleft; redistributing/modifying may impose GPL terms on derivatives. |
| `cldf/` | `cldf/LICENSE` | Apache-2.0 | Third-party specification/docs code. |
| `denoising-diffusion-pytorch/` | `denoising-diffusion-pytorch/LICENSE` | MIT | Third-party library. |
| `phoible/` | `phoible/LICENSE` | Apache-2.0 | Third-party dataset + code; also has `phoible/data/LICENSE`. |
| `wikihan/` | `wikihan/LICENSE` | CC0-1.0 | Public-domain style dedication. |
| `wikipron/` | `wikipron/LICENSE.txt` | Apache-2.0 | Third-party tool + data. |

## External datasets

- Mocha-TIMIT is not yet downloaded in this workspace.
- When you download it, keep it under `data/external/mocha_timit/` and do not
  commit it to Git by default. Its license is typically non-commercial.
