# Installation

## Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (CPU works for inference but training is slow)
- Git

## Clone and Install

```bash
git clone <repo-url>
cd MJKAN_Net
pip install -r requirements.txt
```

## HuggingFace Setup

All pre-trained checkpoints and preprocessed data are hosted publicly at `donbosoc/shigan-mjkan-baseline`.

- **Eval (`--from-hf`)** — no token needed. The repo is public; `hf_hub_download` works anonymously.
- **Training** — no token needed unless you want to push your trained checkpoint to HuggingFace.
- **Push (`--push-hf`)** — requires a write token.

If you plan to push, create a `.env` from the example:

```bash
cp .env.example .env
# open .env and set HF_TOKEN to your HuggingFace write token
```

`.env` format (see `.env.example` for all options):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx   # write token — only needed for --push-hf
OUT_DIR=./checkpoints              # where checkpoints are saved locally
```

## Optional: Weights & Biases

Several training scripts log metrics to W&B. To enable, set `WANDB_API_KEY` in your environment or add it to `.env`. To disable, set `USE_WANDB=False` inside the training script or leave the key unset — logging is skipped silently when the key is absent.

## Optional: Dataset Collection

To reproduce the training datasets from raw CESNET data, the scripts are designed to run on **Kaggle** where the ~90 GB datasets are already available as mounted inputs — no local download needed. See the [Dataset Collection](user_guide.md#dataset-collection-reproducibility) section of the user guide for Kaggle links and instructions.


