"""
push_to_hf.py — manually push a trained checkpoint to HuggingFace.

Usage:
    python push_to_hf.py <model>

Reads configs/<model>.yaml for the HF repo and folder, then uploads
the checkpoint from ./checkpoints/<model>/<checkpoint_name> to HF.
"""
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

load_dotenv()

CONFIGS_DIR = Path(__file__).parent / "configs"


def main():
    if len(sys.argv) < 2:
        print("Usage: python push_to_hf.py <model>")
        print("       python push_to_hf.py list")
        sys.exit(1)

    model = sys.argv[1]

    if model == "list":
        for p in sorted(CONFIGS_DIR.glob("*.yaml")):
            print(f"  {p.stem}")
        return

    cfg_path = CONFIGS_DIR / f"{model}.yaml"
    if not cfg_path.exists():
        print(f"[error] No config for '{model}'. Run 'python push_to_hf.py list'.")
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text())
    hf_repo = cfg["hf_repo"]
    hf_folder = cfg["hf_model_folder"]
    ckpt_name = cfg["checkpoint_name"]

    out_base = os.environ.get("OUT_DIR", "./checkpoints")
    local_path = Path(out_base) / model / ckpt_name

    if not local_path.exists():
        print(f"[error] Checkpoint not found: {local_path}")
        print(f"        Train the model first: python run.py {model} train")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[error] HF_TOKEN not set. Add it to .env or export it.")
        sys.exit(1)

    login(token=hf_token)
    api = HfApi()

    remote_path = f"{hf_folder}/{ckpt_name}"
    print(f"[push] {local_path} -> {hf_repo}/{remote_path}")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=remote_path,
        repo_id=hf_repo,
    )
    print(f"[done] https://huggingface.co/{hf_repo}/tree/main/{hf_folder}")


if __name__ == "__main__":
    main()
