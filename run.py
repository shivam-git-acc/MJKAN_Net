"""
MJKAN-Net CLI — run any model in train or eval mode.

Usage:
    python run.py <model> train [--push-hf]
    python run.py <model> eval  [--from-hf]
    python run.py list
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIGS_DIR = Path(__file__).parent / "configs"


def load_config(model: str) -> dict:
    cfg_path = CONFIGS_DIR / f"{model}.yaml"
    if not cfg_path.exists():
        print(f"[error] No config found for model '{model}'. Run 'python run.py list' to see available models.")
        sys.exit(1)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def list_models():
    print(f"\n{'Model':<20} {'Description'}")
    print("-" * 80)
    for cfg_path in sorted(CONFIGS_DIR.glob("*.yaml")):
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            print(f"  {cfg_path.stem:<18} {cfg.get('description', '')}")
        except Exception:
            print(f"  {cfg_path.stem:<18} (could not parse config)")
    print()


def resolve_script(cfg: dict, mode: str) -> str:
    key = "train_script" if mode == "train" else "eval_script"
    script = cfg.get(key)
    if not script:
        print(f"[error] Config has no '{key}' entry.")
        sys.exit(1)
    script_path = Path(__file__).parent / script
    if not script_path.exists():
        print(f"[error] Script not found: {script_path}")
        sys.exit(1)
    return str(script_path)


def build_env(cfg: dict, mode: str, push_hf: bool, from_hf: bool) -> dict:
    env = os.environ.copy()

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        env["HF_TOKEN"] = hf_token

    out_base = os.environ.get("OUT_DIR", "./checkpoints")
    env["OUT_DIR"] = out_base

    env["MJKAN_MODE"] = mode
    env["MJKAN_PUSH_HF"] = "1" if push_hf else "0"
    env["MJKAN_FROM_HF"] = "1" if from_hf else "0"

    return env


def main():
    # Handle 'list' before argparse so it doesn't conflict with positional args
    if len(sys.argv) >= 2 and sys.argv[1] == "list":
        list_models()
        return

    parser = argparse.ArgumentParser(
        description="MJKAN-Net — run any model in train or eval mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model", help="Model alias (run 'python run.py list' to see all)")
    parser.add_argument("mode", choices=["train", "eval"], help="Mode: train or eval")
    parser.add_argument(
        "--push-hf",
        action="store_true",
        default=False,
        help="After training, push checkpoint to HuggingFace (default: off)",
    )
    parser.add_argument(
        "--from-hf",
        action="store_true",
        default=False,
        help="For eval: download checkpoint from HuggingFace instead of loading local",
    )

    args = parser.parse_args()

    cfg = load_config(args.model)
    script = resolve_script(cfg, args.mode)
    env = build_env(cfg, args.mode, args.push_hf, args.from_hf)

    print(f"[run] model={args.model}  mode={args.mode}  push_hf={args.push_hf}  from_hf={args.from_hf}")
    print(f"[run] script: {script}")
    print(f"[run] OUT_DIR: {env['OUT_DIR']}")

    result = subprocess.run([sys.executable, script], env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
