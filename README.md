# Multimodal Disaster Identification and Assessment

<div align="center">
  <h3>A Noise-Resilient Fusion Architecture for Low-Resource and High-Resource Environments</h3>
</div>

---

## 📌 Overview

This repository contains the codebase and methodology for a highly robust, dual-encoder multimodal fusion architecture designed for real-time disaster identification. The system fuses Unmanned Aerial Vehicle (UAV) visual surveillance with crowd-sourced social media text. 

It is specifically engineered to mitigate noise and data scarcity in low-resource computational environments (such as edge deployment) and linguistically low-resource languages (e.g., Bengali), while maintaining zero-shot generalization capabilities globally.

## 🧠 Architectural Highlights

- **Dual-Encoder Late Fusion:** Bridges a computationally lightweight **Swin-Tiny** vision transformer (`swin_tiny_patch4_window7_224`) with domain-specific BERT language models (`csebuetnlp/banglabert` and `xlm-roberta-large`).
- **Targeted Fast Gradient Method (FGM):** Restricts adversarial perturbations exclusively to the text embedding space to prevent visual manifold degradation while heavily regularizing text noise (typos, slang).
- **Dynamic Pseudo-Labeling Strategy (DIPS):** Mines the unannotated test stream to organically augment the training distribution by vetting high-confidence ($\ge 0.85$), low-variance ($\le 0.15$) predictions.
- **Optuna-Calibrated LightGBM Stacking:** Employs Level-2 stacking on Out-of-Fold features to correct long-tail class imbalances natively found in natural disaster distributions.

## 📊 Benchmarking Phases

### Phase 1: Local Low-Resource Benchmark (`BanglaCalamityMMD`)
An 8-class Disaster Identification task (Earthquake, Flood, Landslides, Wildfires, Tropical Storms, Droughts, Human Damage, Non-Disaster) across 7,903 multimodal Bengali instances.
- **Script:** `phase1_benchmark.py`
- Runs an extensive ablation study proving the mathematical efficacy of Base vs. FGM vs. FGM+DIPS+Stacking.

### Phase 2: Global Generalization Benchmark (`CrisisMMD v2.0`)
A generalization sanity check demonstrating the pipeline's robustness on a globally recognized high-resource multilingual dataset.
- **Script:** `phase2_benchmark.py`

### Phase 3: Results Visualization
A clean presentation layer built with `seaborn` and `matplotlib` to parse the CSV outputs of Phase 1 and 2 into publication-ready bar charts.
- **Script:** `phase3_results_visualization.py`

## 🚀 Running on Kaggle

This project is optimized to run flawlessly within a Kaggle Notebook utilizing dual-GPU support (T4 x2).

**Setup Instructions:**
1. Create a new Kaggle Notebook.
2. Attach the `BanglaCalamityMMD` and `CrisisMMD v2.0` datasets to the kernel.
3. Configure the accelerator to **GPU T4 x2**.
4. **Cell 1:** Paste the contents of `phase1_benchmark.py` and execute to run the Ablation Study.
5. **Cell 2:** Paste the contents of `phase2_benchmark.py` and execute to run the Generalization Test.
6. **Cell 3:** Paste the contents of `phase3_results_visualization.py` and execute to visualize your F1 scores!

## 📜 Research Paper
The LaTeX source code for the associated academic research paper detailing the mathematical background of the targeted FGM and DIPS strategy is available in `research_paper.tex`. It is natively formatted for single-file compilation in Overleaf.
