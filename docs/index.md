# IndEgo

---

<h3>🎥 Industrial Scenarios</h3>

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">

  <div style="flex: 1; min-width: 220px; text-align: center;">
    <strong>Assembly/Disassembly</strong><br/>
    <iframe width="100%" height="200" 
      src="https://www.youtube.com/embed/xvs_uFhwrvs?autoplay=1&mute=1&loop=1&playlist=xvs_uFhwrvs" 
      title="Assembly Disassembly" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
  </div>

  <div style="flex: 1; min-width: 220px; text-align: center;">
    <strong>Inspection/Repair</strong><br/>
    <iframe width="100%" height="200" 
      src="https://www.youtube.com/embed/yVn7pm8EPig?autoplay=1&mute=1&loop=1&playlist=yVn7pm8EPig" 
      title="Inspection Repair" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
  </div>

  <div style="flex: 1; min-width: 220px; text-align: center;">
    <strong>Logistics</strong><br/>
    <iframe width="100%" height="200" 
      src="https://www.youtube.com/embed/Euxye9HInk4?autoplay=1&mute=1&loop=1&playlist=Euxye9HInk4" 
      title="Logistics" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
  </div>

  <div style="flex: 1; min-width: 220px; text-align: center;">
    <strong>Woodworking</strong><br/>
    <iframe width="100%" height="200" 
      src="https://www.youtube.com/embed/UVzIk0A3OQ4?autoplay=1&mute=1&loop=1&playlist=UVzIk0A3OQ4" 
      title="Woodworking" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
  </div>

  <div style="flex: 1; min-width: 220px; text-align: center;">
    <strong>Miscellaneous</strong><br/>
    <iframe width="100%" height="200" 
      src="https://www.youtube.com/embed/JPehcSF_tGc?autoplay=1&mute=1&loop=1&playlist=JPehcSF_tGc" 
      title="Miscellaneous" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
  </div>

</div>

---

Welcome to **IndEgo**, a **NeurIPS 2025 Datasets & Benchmarks Track** accepted dataset and open-source framework for **industrial egocentric vision** — designed to support **training, real-time guidance, process improvement, and collaboration**.

🎓 **Paper:** [IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants (NeurIPS 2025)](https://neurips.cc/virtual/2025/poster/121501)  
👉 [GitHub Repo](https://github.com/Vivek9Chavan/IndEgo)  
🤗 [Hugging Face Dataset](https://huggingface.co/datasets/vivek9chavan/IndEgo_Demo)  
🚀 [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qCZnFQNRjBuy3vBlkMy7sMTcYkTNOzgg?usp=sharing)

<p align="left">
  <img src="https://github.com/user-attachments/assets/fcf2e236-768a-4348-9762-28f4fa62d405" alt="IndEgo Logo" width="500"/>
</p>

---

## 📘 About

**IndEgo** introduces a multimodal **egocentric + exocentric** video dataset capturing common industrial activities such as assembly/disassembly, inspection, repair, logistics, and woodworking.

It includes **3,460 egocentric videos (~197h)** and **1,092 exocentric videos (~97h)** with synchronized **eye gaze**, **audio narration**, **hand pose**, **motion**, and **semi-dense point clouds**.

IndEgo enables research on:
- **Procedural & collaborative task understanding**
- **Mistake detection** and **process deviation recognition**
- **Reasoning-based Video Question Answering (VQA)**

---

## ⚙️ Technology

IndEgo combines:
- **Egocentric Computer Vision** for context-aware task understanding  
- **Vision-Language Models (VLMs)** for multimodal reasoning  
- **Smart Glasses Integration** for on-site, real-time assistance  

![tech_concept](https://github.com/user-attachments/assets/692c196c-c842-4467-9cf2-e78b0e005c27)

---

## 🎬 Demo Video

<iframe width="100%" height="400" src="https://www.youtube.com/embed/ric5f6jH7AI?autoplay=1&loop=1&mute=1&playlist=ric5f6jH7AI" 
title="IndEgo Demo Video" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

---

## 🚀 Try It – No Setup Required

[Launch Colab Notebook](https://colab.research.google.com/drive/1mC-W5czouMFgICMktrffOU7sSjMBXENO?usp=sharing)  
Run IndEgo’s core logic directly in your browser with Google Colab — no installation needed.

---

## 📊 Dataset

🔗 [Open Dataset on Hugging Face](https://huggingface.co/datasets/vivek9chavan/IndEgo_Demo)

The IndEgo dataset includes annotated **egocentric and exocentric** videos of real-world industrial scenarios with:
- Action & narration annotations  
- Mistake labels and summaries  
- Eye-gaze and 3D mapping data  
- Benchmarks for procedural reasoning and collaborative task understanding  

---

## 🧩 Citation

If you use **IndEgo**, please cite:

```bibtex
@inproceedings{Chavan2025IndEgo,
  author    = {Vivek Chavan and Yasmina Imgrund and Tung Dao and Sanwantri Bai and Bosong Wang and Ze Lu and Oliver Heimann and J{\"o}rg Kr{\"u}ger},
  title     = {IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year      = {2025},
  url       = {https://neurips.cc/virtual/2025/poster/121501}
}
