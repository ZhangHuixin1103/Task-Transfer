# Unlocking the Boundaries of Cross-Task Visual In-Context Learning via Implicit Text-Driven VLMs
## 📌 Project Overview
This project introduces T2T-VICL, a collaborative pipeline designed to explore cross-task visual in-context learning (VICL). Unlike traditional single-task approaches, our focus is on transfer across heterogeneous low-level vision tasks — asking whether VLMs (Vision-Language Models) can still enable VICL when the visual prompt and target images come from different tasks. We design a mechanism for generating and selecting implicit text prompts, construct the first cross-task VICL dataset, and propose a training strategy to transfer knowledge from large VLMs to lightweight sVLMs, followed by a deployment framework from sVLM back to large models.

## 🚀 Key Contributions
**Cross-task VICL dataset:** The first dataset with implicit text descriptions for visual tasks, enabling systematic exploration of task boundaries.

**VLM↔sVLM framework:** A bidirectional pipeline for knowledge transfer and prompt generation between large-scale and compact models.

**Inference & evaluation scheme:** An automatic framework combining VIE Score and image quality assessment metrics to perform cross-task VICL without additional training or fine-tuning.

Our results show that T2T-VICL achieves stable performance across diverse low-level task pairs, demonstrating the feasibility of cross-task VICL within VLMs and highlighting its potential as a cost-effective paradigm for generalizable vision-language reasoning.

<img width="400" height="470" alt="figure1" src="https://github.com/user-attachments/assets/0de690b6-b38d-4492-9344-7c30028a5910" />


## Task2task pairs

| Task | colorization | deblurring | dehazing | demoireing | denoising | deraining | harmonization | inpainting | light enhancement | reflection removal | shadow removal | style transfer |
|:----:|:------------:|:----------:|:--------:|:----------:|:---------:|:---------:|:-------------:|:----------:|:-----------------:|:------------------:|:--------------:|:--------------:|
| **colorization**       | - |   |   |   |   |   | ✔ |   |   |   |   | ✔ |
| **deblurring**         |   | - | ✔ | ✔ |   | ✔ |   |   |   |   |   |   |
| **dehazing**           |   |   | - |   | ✔ | ✔ |   |   |   |   |   |   |
| **demoireing**         |   |   | ✔ | - |   |   |   |   |   |   |   |   |
| **denoising**          |   | ✔ |   |   | - |   |   |   | ✔ |   |   |   |
| **deraining**          |   |   |   | ✔ | ✔ | - |   |   |   |   |   | ✔ |
| **harmonization**      |   |   |   |   |   |   | - |   | ✔ |   |   | ✔ |
| **inpainting**         | ✔ |   |   |   |   |   | ✔ | - | ✔ |   |   | ✔ |
| **light enhancement**  | ✔ |   |   |   |   | ✔ |   |   | - |   | ✔ |   |
| **reflection removal** |   |   | ✔ |   |   |   |   |   |   | - |   |   |
| **shadow removal**     |   |   |   |   |   | ✔ |   |   |   | ✔ | - |   |
| **style transfer**     |   |   |   |   |   |   |   |   | ✔ |   |   | - |
