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
Each Category includes:
- Egocentric + Exocentric videos
- Gaze, motion, hand-pose logs
- Narrations (where applicable)
- Keysteps and mistakes (if any)
- SLAM data (missing for some sequences)

---

## 🛠️ Environment Setup

### Create and install from `requirements.txt`
```bash
# Create a new virtual environment
python3 -m venv $HOME/indego_env
source $HOME/indego_env/bin/activate

# Install dependencies
pip install -r requirements.txt
