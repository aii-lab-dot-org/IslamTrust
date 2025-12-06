<h1>IslamTrust: A Benchmark for LLMs Alignment with Islamic Values</h1> 


<a href="#">Leaderboard (Coming Soon)</a> 📊 |
<a href="https://huggingface.co/datasets/Abderraouf000/IslamTrust-benchmark">Evaluation Dataset</a> 📂 |
<a href="https://openreview.net/pdf?id=PBcv90iKFB">Paper 📝</a> |
<a href="https://neurips.cc/virtual/2025/loc/san-diego/133012">Poster 📌</a>


<p align="center">
  <img src="Assets/IslamTrust_Logo.png" alt="Logo" width="400"/>
</p>


<p align="center">
  <a href="https://github.com/aii-lab-dot-org/IslamTrust/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/aii-lab-dot-org/IslamTrust">
  </a>
</p>


**IslamTrust** is a multilingual benchmark designed to evaluate the alignment of LLMs with Islamic values and principles.
This repository contains code to evaluate different models on the IslamTrust benchmark. The full set of benchmark questions and choices is found in <a href = "https://huggingface.co/datasets/Abderraouf000/IslamTrust-benchmark">HuggingFace</a>.


**Authors**: 
<a href="https://scholar.google.com/citations?view_op=list_works&hl=en&user=G-u4OPAAAAAJ">Abderraouf Lahmar</a>, 
<a href="https://scholar.google.com/citations?user=RRMOwEMAAAAJ&hl=en&oi=ao">Arafat Md Easin</a>, 
<a href = "https://scholar.google.com/citations?hl=fr&user=FM-8ArsAAAAJ&view_op=list_works&sortby=pubdate">Farou Zakaria</a> , 
<a href = "https://scholar.google.com/citations?user=L8em2YoAAAAJ&hl=en">Mufti Mahmud </a>.


## <img src="https://github.com/user-attachments/assets/1abcf195-ad44-4500-a14b-f1a4bef9b748" width="40" height="40" />Latest Updates
 🔥  **[31 August 2025]** IslamTrust is **1st** Multilingual benchmark designed to evaluate the alignment of LLMs with General Islamic values and principles.<br>
 🔥  **[15 October 2025]** IslamTrust Paper is accepted to be presented at the 5th Muslims in Machine Learning (MusIML) Workshop at NeurIPS.<br>


### IslamTrust evaluation scope
---
The IslamTrust benchmark dataset is available in both English and Arabic, spanning six different categories.

### Tasks
---
For now, our benchmark consists of one task, which is a multiple-choice MC1 task.

#### Multiple-choice:
---
To select among the candidate answers, we did not employ a prompt-based approach. Instead, we relied on the mean of log-probabilities to identify the most confident answer produced by each model for a given question–answer pair independently from other choices.

For HuggingFace-supported models, the multiple-choice benchmark score can be calculated using `evaluate.py`.


## <img src="https://github.com/user-attachments/assets/08e47f66-e0aa-49b5-b886-ad65ae7a6faa" width="30" height="30" /> Citation
If you use the IslamTrust dataset in your research, please consider citing:

**Lahmar, A., Arafat, M.E., Farou, Z., & Mahmud, M.** (2025). *IslamTrust: A Benchmark for LLMs Alignment with Islamic Values*. In **5th Muslims in ML Workshop**, co-located with **NeurIPS 2025**.

@inproceedings{
lahmar2025islamtrust,
title={IslamTrust: A Benchmark for {LLM}s Alignment with Islamic Values},
author={Abderraouf Lahmar and Md Easin Arafat and Zakarya Farou and Mufti Mahmud},
booktitle={5th Muslims in ML Workshop co-located with NeurIPS 2025},
year={2025},
url={https://openreview.net/forum?id=PBcv90iKFB}
}
