# IndEgo

## AI Assistant for Industrial Automation and Robotics | Augmenting Human Skills with AI

**💬 Interested? Reach out at: [vivek.chavan@ipk.fraunhofer.de](mailto:vivek.chavan@ipk.fraunhofer.de)**

Welcome to the IndEgo, a privacy-preserving, open-source framework that learns from experts to support training, real-time guidance, process improvement, and collaboration.

👉 [GitHub Repo](https://github.com/Vivek9Chavan/IndEgo)

🤗 [Hugging Face: IndEgo_Demo](https://huggingface.co/datasets/vivek9chavan/IndEgo_Demo)

🤝 [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qCZnFQNRjBuy3vBlkMy7sMTcYkTNOzgg?usp=sharing)

<p align="left">
  <img src="https://github.com/user-attachments/assets/fcf2e236-768a-4348-9762-28f4fa62d405" alt="IndEgo Logo" width="500"/>
</p>

---

## About

**IndEgo** combines Egocentric AI, Vision-Language Models, and Robotics-aware reasoning to understand human actions and provide real-time guidance. It is designed for use in industrial settings through smart glasses or mobile devices — enabling contextual assistance, task verification, and human-robot collaboration.

IndEgo supports use cases such as:
- Industrial task training and onboarding  
- Real-time guidance and error prevention  
- Knowledge transfer from experts to new operators  
- AI-augmented collaboration between workers and machines  

Built as a privacy-first, open-source project, IndEgo gives organisations full control over data, deployment, and customisation.

---

## Technology

IndEgo runs on an adaptable framework that leverages:
- **Egocentric computer vision** for task context understanding  
- **Vision-language models (VLMs)** for multi-modal reasoning and interaction  
- **Robotics-aware logic** to align human guidance with automated systems  
- **Smart glasses or mobile devices** for real-time, on-site use

![tech_concept](https://github.com/user-attachments/assets/692c196c-c842-4467-9cf2-e78b0e005c27)

---

## Features

- 🧠 Learns from expert demonstrations  
- 📱 Works with smart glasses or mobile devices  
- 🛠️ Customizable and extensible for various industrial domains  
- 🔐 Privacy-preserving — no cloud upload required  
- 🤖 Compatible with robotics workflows and task handoffs

---

## Demo Video


<iframe width="100%" height="400" src="https://www.youtube.com/embed/ric5f6jH7AI?autoplay=1&loop=1&mute=1&playlist=ric5f6jH7AI" 
title="IndEgo Demo Video" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

---

## Try It – No Setup Required!

👉 [Launch Colab Notebook](https://colab.research.google.com/drive/1mC-W5czouMFgICMktrffOU7sSjMBXENO?usp=sharing)  
Run IndEgo's core logic directly in your browser using Google Colab — no install or setup needed.

---

## Dataset

🔗 [Open-source Dataset on Hugging Face](https://huggingface.co/datasets/vivek9chavan/IndEgo_Demo)  
The IndEgo demo dataset includes annotated egocentric videos from real-world industrial tasks, ready for VLM, task graph, and mistake detection research.

---

## Get Involved

We are currently looking for:
 - ⚙️**Industry partners** for pilot studies and collaboration.
 - 🎓**Research Institutes** for forming a Consortium.
 - 💼**Investors** to discuss the commercialisationn potential.

---
**💬 Interested? Reach out at: [vivek.chavan@ipk.fraunhofer.de](mailto:vivek.chavan@ipk.fraunhofer.de)**
---

## 🧪 Mistake Detection Challenge: Can You Spot the Error?

In this example, two workers perform the same task: **Loading a trolley into a hatch**.  
One performs it correctly — the other makes a mistake.

### ✅ Expected Steps:
1. Open hatch  
2. Put on gloves  
3. Load trolley securely  
4. Close hatch  
5. Check if loaded securely

---

### 🎥 Watch the Comparison:

<table>
<tr>
  <th>✅ Correct Execution</th>
  <th>⚠️ Mistake Case</th>
</tr>
<tr>
  <td align="center">
    <a href="https://huggingface.co/datasets/vivek9chavan/IndEgo_Demo/resolve/main/Mistake_Detection/Task_10/mp4_480/User_1_H_10c_3_480.mp4" target="_blank">
      <img src="https://img.icons8.com/ios-filled/100/000000/play-button-circled--v1.png" width="60"/><br/>
      <strong>Watch Correct Execution</strong>
    </a>
  </td>
  <td align="center">
    <a href="https://huggingface.co/datasets/vivek9chavan/IndEgo_Demo/resolve/main/Mistake_Detection/Task_10/mp4_480/User_16_0611_10m_3_480.mp4" target="_blank">
      <img src="https://img.icons8.com/ios-filled/100/000000/play-button-circled--v1.png" width="60"/><br/>
      <strong>Watch Mistake Execution</strong>
    </a>
  </td>
</tr>
</table>

**🤔 Can you identify which step was skipped or done incorrectly in the second video?**  
This challenge illustrates how IndEgo can be used to analyze procedural tasks and detect deviations automatically using egocentric AI.

---

## License

Apache 2.0 License – see the [GitHub repo](https://github.com/Vivek9Chavan/IndEgo) for details.
