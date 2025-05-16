#!/usr/bin/env python3
"""
Main dispatcher for IndEgo VQA benchmarking.

Usage examples
--------------
python vqa_test.py --vlm qwen_2_5
python vqa_test.py --vlm videollama3 --config config/paths_example.yaml
"""
import argparse, importlib, yaml, os, sys

DEFAULT_CFG = {
    "json_file_path": "<YOUR_PATH_HERE>/indego_vqa.json",
    "mp4_dir":        "<YOUR_PATH_HERE>/mp4_480",
    "output_path":    "<YOUR_PATH_HERE>/output.json",
    "mistral_key":    "<YOUR_MISTRAL_KEY>"
}

def load_cfg(path):
    if path and os.path.isfile(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return DEFAULT_CFG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm",
                        required=True,
                        choices=["qwen_2_5", "videollama3", "internvl2", "custom"],
                        help="Vision-Language Model to benchmark")
    parser.add_argument("--config", default=None,
                        help="YAML file with paths & API key (optional)")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    # dynamic import of the selected runner
    if args.vlm == "custom":
        print("Please implement your runner in vlm_runners/custom_runner.py "
              "with a `run(cfg)` function.")
        sys.exit(1)

    runner_module = importlib.import_module(f"vlm_runners.{args.vlm}_runner")
    runner_module.run(cfg)

if __name__ == "__main__":
    main()
