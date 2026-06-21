# User Guide

## CLI Reference

```
python run.py <model> <mode> [--push-hf] [--from-hf]
python run.py list
```

| Argument | Description |
|---|---|
| `model` | One of the 17 model aliases (see table below) |
| `mode` | `train` or `eval` |
| `--push-hf` | After training, upload the checkpoint to HuggingFace (default: off) |
| `--from-hf` | During eval, download checkpoint from HuggingFace instead of loading local (default: off) |

## Model Aliases

| Alias | Description |
|---|---|
| `baseline` | 9-class behavioral SupCon baseline (SSM encoder, BN projection head) |
| `baseline_nosupcon` | 6-class baseline without contrastive learning — SSM encoder + linear head only |
| `baseline_6class` | 6-class baseline SupCon (original CESNET classes) |
| `held_out_6class` | Held-out generalisation: train on 5 classes, test zero-shot on File sharing |
| `v1` | V1 — joint QUIC+TLS training for protocol invariance |
| `v2` | V2 — V1 + GatedFiLM conditioning on 6-feature flow context |
| `v3` | V3 — V1 + cross-protocol balanced batch sampler |
| `v5_dann` | V5z — DANN adversarial discriminator on z-embedding (GRL) |
| `v6` | V6 — centroid uniformity loss for inter-class separation |
| `v7` | V7 — FiLM + centroid separation, all pairs (best TLS accuracy) |
| `v7_dann` | V7+DANN — fine-tuned V7 with domain adversarial training |
| `hierarchical` | Hierarchical variant with per-category sub-heads |
| `transformer` | Transformer encoder ablation (replaces SSM blocks) |
| `v_cond` | Condition-invariant training with timing augmentation + GatedFiLM |
| `mjkan` | Full MJKAN: SSM + GatedFiLM + Tiny FasterKAN reasoning layer |
| `mjkan_temporal` | MJKAN temporal robustness: QUIC(W44-45)+TLS(M1-9) train, W46-47+M10-12 test |
| `realtime` | Real-time joint QUIC+TLS with temporal split |
| `realtime_finetune` | Fine-tune of `realtime` model with lower LR + cosine decay |

## Per-Model Commands

Each model section in [architecture.md](architecture.md) includes the exact `eval` and `train` commands for that model. Go there if you want to run a specific model — no need to look up the alias separately.

## Common Workflows

### Reproduce published KPIs (eval from HuggingFace)

```bash
python run.py baseline eval --from-hf
python run.py mjkan eval --from-hf
python run.py v7 eval --from-hf
python run.py realtime eval --from-hf
```

### Run unseen-traffic generalization eval (flagship model)

```bash
python3 src/supcon+9class/MJKAN_variants/temporal_robust/unseen_handling/novel_arg.py
```

Loads the MJKAN Temporal checkpoint and the unseen-app benchmark directly from HuggingFace. No token or local data needed. Prints per-app classification, novelty detection AUROC, and the known-vs-unseen confidence gap.

### Train a model locally

```bash
python run.py baseline train
python run.py mjkan train
```

Checkpoints are saved to `./checkpoints/<model>/` by default. Override with `OUT_DIR` in `.env`.

### Train and push to HuggingFace in one step

```bash
python run.py baseline train --push-hf
```

### Push an existing local checkpoint to HuggingFace

```bash
python push_to_hf.py baseline
python push_to_hf.py mjkan
```

This is the safe default — review your checkpoint locally before uploading.

## Output Directory

All checkpoints and result JSON files are written to `OUT_DIR/<model>/` (defaults to `./checkpoints/<model>/`).
Set `OUT_DIR` in `.env` to redirect to a different path (e.g. an NFS mount or SSD).

## What Gets Created

### After `train`

```
checkpoints/<model>/
├── <model>_best.pt        # best checkpoint (saved when val loss improves)
└── train.log              # full training output — loss, accuracy, KPIs per epoch
```

### After `eval` (or `eval --from-hf`)

```
checkpoints/<model>/
├── <model>_eval.json      # accuracy, balanced accuracy, per-class breakdown,
│                          #   intra/inter cosine, latency
├── tsne_embeddings.png    # t-SNE plot of z-embeddings coloured by class
└── cosine_matrix.png      # inter-class cosine heatmap
```

`--from-hf` downloads the checkpoint from HuggingFace into `checkpoints/<model>/` before running eval, so the same output files are produced regardless of where the checkpoint came from.

### After dataset collection scripts

```
data/
├── baseline6classdata/            # collect_rich_pool_for_baseline
│   ├── pool_train.npz
│   ├── pool_val.npz
│   ├── pool_test.npz
│   └── pool_meta.json             # norm stats + class info (required at eval time)
│
├── behavioral9classdata/          # collect_rich_pool.py
│   ├── pool_train.npz
│   ├── pool_val.npz
│   ├── pool_test.npz
│   └── pool_meta.json
│
└── combined_temporal/             # temporal_quic_data_coll + temporal_tls_data_coll
    ├── quic_temporal.npz          # seq, ctx, label, week, app, protocol
    └── tls_temporal.npz           # seq, ctx, label, month, app, protocol
```

On Kaggle these land in `/kaggle/working/` instead of `./data/`.

### `configs/` (already in repo — read-only)

```
configs/
├── mjkan_temporal.yaml    # one file per model alias
├── v7.yaml
└── ...                    # 18 files total
```

Each YAML is read by `run.py` to resolve script paths and HF locations. You never need to edit these unless you are adding a new model.

## HuggingFace Artifacts

All checkpoints and preprocessed datasets live at **[donbosoc/shigan-mjkan-baseline](https://huggingface.co/donbosoc/shigan-mjkan-baseline)**. The repo is public — no token needed to download.

```
donbosoc/shigan-mjkan-baseline/
│
├── ★ combined_temporal/                   ← FLAGSHIP (MJKAN Temporal)
│   ├── combined_temporal_best.pt          ← model checkpoint
│   └── data/
│       ├── quic_temporal.npz              ← QUIC flows, week-tagged (W44–W47)
│       └── tls_temporal.npz              ← TLS flows, month-tagged (M1–M12)
│
├── baseline_nosupcon/
│   └── baseline_best.pt                  ← 6-class, no SupCon
│
├── baseline6classdata/                   ← shared data for all 6-class models
│   ├── pool_train.npz
│   ├── pool_val.npz
│   ├── pool_test.npz
│   └── pool_meta.json                    ← norm stats (required at eval time)
│
├── baseline+supcon/                      ← 6-class SupCon baseline
│   └── supcon_best.pt                    ← baseline_6class
│
├── behavioral9+supcon/                   ← 9-class SupCon baseline
│   ├── supcon_best.pt
│   └── data/
│       ├── meta.json
│       ├── train.npz
│       ├── val.npz
│       └── test.npz
│
├── protocol_invariance/                  ← V-series models + shared data
│   ├── data/
│   │   ├── joint_train.npz               ← joint QUIC+TLS training data
│   │   ├── joint_val.npz
│   │   ├── joint_test.npz
│   │   ├── joint_meta.json
│   │   └── fiveg_youtube.npz             ← 5G condition-transfer test data
│   ├── v1_joint/supcon_v1_best.pt        ← V1
│   ├── v2_film/supcon_v2_best.pt         ← V2
│   ├── v3_joint/supcon_v3_best.pt        ← V3
│   ├── v5z_dann/supcon_v5z_best.pt       ← V5z DANN
│   └── v7_sep/supcon_v7_best.pt          ← V7
│
├── condition_invariance/
│   └── v_cond/v_cond_best.pt             ← V-COND
│
├── mjkan/
│   └── mjkan_best.pt                     ← MJKAN (non-temporal)
│
├── realtime_joint_temporal/              ← real-time variant
│   ├── rtjt_best.pt                      ← realtime (base)
│   ├── rtjt_best_v2.pt                   ← realtime_finetune
│   ├── rtjt_meta.json
│   ├── rtjt_train.npz
│   └── rtjt_test.npz
│
└── generalization/
    ├── unknown_apps_combined.npz         ← unseen-app benchmark (19 apps, 800 flows each)
    └── novel_apps.npz                    ← additional novel-app flows
```

`--from-hf` on any eval command fetches the relevant `.pt` from the path shown above. Data files are fetched at eval time by the script itself — you never need to manually download anything.

### Models without saved checkpoints

The following 5 models do **not** have checkpoints on HuggingFace and cannot be run with `--from-hf`:

| Alias | Reason |
|---|---|
| `held_out_6class` | Trained and evaluated, but did not meet KPI thresholds — checkpoint not published |
| `v6` | Degenerate collapse during training (uniformity loss drove inter → −0.018) — discarded |
| `v7_dann` | Fine-tuned V7 with DANN — results generated but no improvement over V7; not saved |
| `hierarchical` | Exploratory variant — training results captured but checkpoint not published |
| `transformer` | Ablation (replaces SSM with Transformer encoder) — results captured but not saved |

These models were all trained and their outputs were recorded. Training screenshots and result logs are provided separately as supplementary material. To run them yourself, train from scratch first:

```bash
python run.py held_out_6class train
python run.py v6 train
python run.py v7_dann train
python run.py hierarchical train
python run.py transformer train
```

Then eval without `--from-hf` (uses the locally trained checkpoint):

```bash
python run.py held_out_6class eval
python run.py transformer eval
# etc.
```

## Dataset Collection (Reproducibility)

The raw CESNET datasets are ~90 GB and are not practical to download locally. **The recommended way to run these scripts is on Kaggle**, where the datasets are already available as mounted inputs — no download required.

### Kaggle datasets

| Dataset | Kaggle link | Used by |
|---|---|---|
| CESNET-QUIC22 | [kaggle.com/datasets/anishanandhan/cesnet](https://www.kaggle.com/datasets/anishanandhan/cesnet) | baseline, 9-class, temporal QUIC |
| CESNET-TLS-Year22 | [kaggle.com/datasets/pranjalkar99/cesnet-22](https://www.kaggle.com/datasets/pranjalkar99/cesnet-22) | temporal TLS |

Add the relevant dataset as a Kaggle input, then run the script as a notebook cell or `%%bash`. The scripts default to the Kaggle mount paths (`/kaggle/input/datasets/...`) so no configuration is needed. Output goes to `/kaggle/working/`.

### Script overview

| Script | Produces |
|---|---|
| `dataset_collection/collect_rich_pool_for_baseline` | `baseline6classdata/` — 6-class QUIC, balanced pools + norm stats |
| `src/.../baseline/dataset_collection_script/collect_rich_pool.py` | `behavioral9classdata/` — 9-class behavioral, capped pools + class weights + norm |
| `src/.../temporal_robust/.../temporal_quic_data_coll` | `combined_temporal/quic_temporal.npz` — 9 QUIC classes with week tags |
| `src/.../temporal_robust/.../temporal_tls_data_coll` | `combined_temporal/tls_temporal.npz` — 7 TLS classes with month tags |

### How to run on Kaggle

1. Create a new Kaggle notebook and attach the dataset(s) above as inputs.
2. Add the repo as an input or upload the script file.
3. Run the script — no env vars needed, paths are pre-configured:

```python
# In a Kaggle notebook cell
!python3 dataset_collection/collect_rich_pool_for_baseline
```

To push output to your own HuggingFace repo, set two Kaggle secrets (`HF_TOKEN`, `HF_DATASET_REPO`) and add `MJKAN_PUSH_HF=1`:

```python
import os
os.environ["MJKAN_PUSH_HF"] = "1"
!python3 dataset_collection/collect_rich_pool_for_baseline
```

### Running locally (optional)

If you have the data locally, point the env vars at your download:

```bash
# .env
CESNET_DATA_DIR=/path/to/cesnet-quic22       # baseline + 9-class
CESNET_QUIC_DATA_DIR=/path/to/cesnet-quic22  # temporal QUIC
CESNET_TLS_DATA_DIR=/path/to/CESNET-TLS-Year22
DATA_DIR=./data
HF_DATASET_REPO=yourname/your-repo           # optional push target
```

Then run directly:
```bash
python3 dataset_collection/collect_rich_pool_for_baseline
python3 src/supcon+9class/baseline/dataset_collection_script/collect_rich_pool.py
# temporal (QUIC first, then TLS):
python3 src/supcon+9class/MJKAN_variants/temporal_robust/dataset_collection_tls+quic/temporal_quic_data_coll
python3 src/supcon+9class/MJKAN_variants/temporal_robust/dataset_collection_tls+quic/temporal_tls_data_coll
```

### What each script produces

Each collection script runs in two phases: **collect** (reads CSVs, extracts PPI sequences + 6 context features, maps labels) then **normalize** (fits z-score stats on train only, applies to val/test, saves `pool_*.npz` + `pool_meta.json`). `pool_meta.json` carries the norm stats and is required at eval time.

The temporal scripts tag each flow with its week (QUIC) or month (TLS). The `mjkan_temporal` model trains on QUIC W44–W45 + TLS months 1–9 and tests on W46–W47 + months 10–12.

## Environment Variables (advanced)

These are set automatically by `run.py` but can be set manually for direct script execution:

| Variable | Values | Purpose |
|---|---|---|
| `MJKAN_MODE` | `train` / `eval` | Passed to inline scripts that handle both modes |
| `MJKAN_PUSH_HF` | `1` / `0` | Enable HF push at end of training |
| `MJKAN_FROM_HF` | `1` / `0` | Download checkpoint from HF before eval |
| `OUT_DIR` | path | Base directory for checkpoints |
| `HF_TOKEN` | token string | HuggingFace write token |
| `CESNET_DATA_DIR` | path | CESNET-QUIC22 root — only needed for local runs (Kaggle uses default path) |
| `CESNET_QUIC_DATA_DIR` | path | CESNET-QUIC22 root for temporal extraction |
| `CESNET_TLS_DATA_DIR` | path | CESNET-TLS-Year22 root for temporal extraction |
| `DATA_DIR` | path | Output folder for dataset collection (default: `/kaggle/working` on Kaggle) |
| `HF_DATASET_REPO` | `user/repo` | Your own HF repo to push reproduced datasets into |
