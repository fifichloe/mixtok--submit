# MixTok training code

This repository contains the core training code for MixTok. It has two top-level components: reconstruction (`recon/`) and generation (`gene/`). Weights, datasets, training caches, evaluation outputs, and qualitative images are not included.

## Layout

| Directory | Contents |
| --- | --- |
| `recon/tokenizer/` | Spatial expansion and reconstruction tokenizer training, with the adapter, fixed-budget, and adaptive-budget configurations. |
| `recon/router/` | Reconstruction feature preparation, grid scoring, image-level utility, and their training code. |
| `gene/adapter/` | Spatial context embedding adapter and paired-token data handling. |
| `gene/router/` | Generation router distillation, spatial and budget training, and model export. |
| `gene/maskgit/` | Generation route-cache construction and MaskGIT training. |
| `gene/support/` | Shared asset loading, paths, and latent model components. |

The released routing policy scores the grid once, computes an image-specific token count from a threshold, clamps that count to 48–144, and keeps the top-scoring locations. Its parameters are recorded in `gene/configs/method.json` and `gene/router/bundle/config.json`.

## External inputs

The code needs ImageNet 256×256, the official TiTok-L32, LlamaGen VQ-16, and Halton-MaskGIT source and pretrained weights, and intermediate feature and paired-code caches. Point the scripts to local assets through their CLI arguments and configuration files. The principal environment variables are:

```bash
export MIXTOK_REPO=/path/to/this/repository
export MIXTOK_TITOK_ROOT=/path/to/TiTok
export MIXTOK_LLAMAGEN_ROOT=/path/to/LlamaGen
export HALTON_UPSTREAM_ROOT=/path/to/Halton-MaskGIT
export MIXTOK_RECON_CODEC=/path/to/reconstruction_checkpoint.pt
export MIXTOK_TITOK_WEIGHTS=/path/to/tokenizer_titok_l32.bin
export MIXTOK_LLAMAGEN_WEIGHTS=/path/to/vq_ds16_c2i.pt
export MIXTOK_B0_WEIGHTS=/path/to/efficientnet_b0_weights.pth
export MIXTOK_SCORER_STAGE_CONFIG=/path/to/filled_stage_config.json
export MIXTOK_ROUTER_BUNDLE=/path/to/this/repository/gene/router/bundle
export MIXTOK_PAIRED_CODES=/path/to/paired_codes
export MIXTOK_WEIGHTS=/path/to/generation_weights
export MOTAR_ASSETS=/path/to/generation_assets
export MOTAR_RESULTS=/path/to/output_directory
```

Fill in `gene/configs/scorer_stage.example.json` for the generation-router stages. The router bundle also requires a trained `student_state.pt` next to the included model and config. The source checks require the corresponding pretrained artifacts and matching checkpoint hashes.

## Training order

1. Run `recon/tokenizer/train_base.py` with `recon/tokenizer/configs/adapter_only_2p5epoch.yaml`.
2. Run `recon/tokenizer/train_refinement.py` with the fixed-budget configuration, then with `top_p_train.yaml`. Set dataset and checkpoint paths in the YAML files.
3. Train reconstruction scoring with `recon/router/training/fit_gain.py` and `fit_utility.py` using cached supervision and feature statistics.
4. Train the generation router with `gene/router/distill/train.py`, then `gene/router/spatial/train.py` and `gene/router/budget/train.py`.
5. Train the spatial context adapter with `gene/adapter/train_adapter.py`.
6. Build the generation route cache with `gene/maskgit/cache_routes.py` (`prepare`, `extract`, `finalize`) and train MaskGIT with `gene/maskgit/train.py`.

The reconstruction training configs use offline WandB logging. This minimal release does not include the data or intermediate caches required for a full run from an empty machine.
