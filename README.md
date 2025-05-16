# IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants

**IndEgo** is a large-scale multimodal dataset for research in egocentric AI, collaborative work, mistake detection, task understanding, and vision-language reasoning in industrial scenarios.

## 🔗 Dataset Access

The dataset is available here:  
👉 [Hugging Face: IndEgo_Demo](https://huggingface.co/datasets/vivek9chavan/IndEgo_Demo)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qCZnFQNRjBuy3vBlkMy7sMTcYkTNOzgg?usp=sharing)

---

## 🔍 Key Features
- 3000+ egocentric videos, 1000+ exocentric videos
- Task steps, audio narration, SLAM, gaze, motion data
- Reasoning-based video QA benchmark
- Annotated collaborative sequences with tools and workspace layout

---

## 📦 Dataset Structure
Each task includes:
- Egocentric + Exocentric videos
- Gaze & motion CSV logs
- Narrations (where applicable)
- Task steps and mistakes (if any)
- 3D room layout metadata (for some sequences)

---

## 🛠️ Environment Setup

### Create and install from `requirements.txt`
```bash
# Create a new virtual environment
python3 -m venv $HOME/indego_env
source $HOME/indego_env/bin/activate

# Install dependencies
pip install -r requirements.txt

---

# Reasoning-based Video-Question-Answering (VQA) Benchmark

| Supported VLMs (pre-configured) | Quick start |
|---------------------------------|-------------|
| **Qwen 2.5-VL**, **VideoLLaMA-3**, **InternVL 2.5** (add your own ↗︎) | ```bash  # install once  pip install -r requirements.txt  # copy & edit paths  cp config/paths_example.yaml my_paths.yaml  # choose a model  python vqa_test.py --vlm qwen_2_5    --config my_paths.yaml  python vqa_test.py --vlm videollama3 --config my_paths.yaml  python vqa_test.py --vlm internvl2   --config my_paths.yaml  ``` |

* The script streams each video, asks the question, records the VLM’s answer, and uses **Mistral-Large** to grade it (set `mistral_key` in `my_paths.yaml`).
* Results are appended incrementally to `output_path`, so you can interrupt and resume.
* **Bring your own model:** drop a `run(cfg)` function into `vlm_runners/your_model_runner.py`, then run  
  `python vqa_test.py --vlm custom --config my_paths.yaml`.
