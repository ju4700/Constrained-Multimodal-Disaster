# Related Works — BanglaCalamityMMD Research Paper

This document catalogues all related papers found for the research paper.
Papers are organized by category for easy citation in the IEEE paper.

---

## Category 1: Foundational Dataset Papers (Cite in Section II)

### [R1] BanglaCalamityMMD Dataset Paper (PRIMARY SOURCE)
- **Title:** "BanglaCalamityMMD: A Comprehensive Benchmark Dataset for Multimodal Disaster Identification in the Low-Resource Bangla Language"
- **Authors:** Fatema Tuj Johora Faria, Mukaffi Bin Moin, Busra Kamal Rafa, Swarnajit Saha, Md. Mahfuzur Rahman, Khan Md Hasib, Firoz Mridha
- **Year:** 2024
- **Dataset:** 7,903 instances, 8 classes, train/val/test = 6323/790/790
- **Their Best Model:** DisasterMultiFusionNet (ViT + mBERT) — 85.25% accuracy
- **Source:** Mendeley Data — https://data.mendeley.com/datasets/7dggbjn5sd/1
- **Relevance:** This IS the dataset we are using. We must cite it in every section.

### [R2] CrisisMMD Dataset Paper
- **Title:** "CrisisMMD: Multimodal Twitter Datasets from Natural Disasters"
- **Authors:** Firoj Alam, Ferda Ofli, Muhammad Imran
- **Year:** 2018 (AAAI ICWSM)
- **Dataset:** Annotated tweets + images from 7 global disasters
- **Relevance:** Standard global multimodal crisis benchmark. Cite as prior SOTA.

---

## Category 2: Direct Competing Methods on BanglaCalamityMMD (Cite in Table III)

### [R3] BanglaMM-Disaster (MOST IMPORTANT COMPETITOR)
- **Title:** "BanglaMM-Disaster: A Multimodal Transformer-Based Deep Learning Framework for Multiclass Disaster Classification in Bangla"
- **Year:** 2025 (IEEE SPICSCON Conference)
- **Dataset:** Their own dataset of 5,037 Bangla posts (9 classes)
- **Best Result:** 83.76% accuracy with BanglaBERT + ResNet50 early fusion
- **Models Tested:** BanglaBERT, mBERT, XLM-RoBERTa (text) + ResNet50, DenseNet169, MobileNetV2 (image)
- **arXiv:** https://arxiv.org/abs/...
- **Relevance:** Closest competitor. They use the same language and domain. Their best is 83.76%; ours is 98.7%.

### [R4] DisasterMultiFusionNet (Original Dataset Paper's Model)
- **Title:** Included in the BanglaCalamityMMD dataset paper [R1]
- **Models Tested:** Vision Transformer (ViT) variants + pretrained BERT models
- **Best Result:** 85.25% accuracy on BanglaCalamityMMD
- **Relevance:** This is the ORIGINAL SOTA set by the dataset creators. We beat it by ~13%.

---

## Category 3: Text-Only Baselines We Must Cite (Cite in Section III)

### [R5] BanglaBERT Language Model
- **Title:** "BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla"
- **Authors:** A. Bhattacharjee, T. Hasan, W. U. Ahmad, K. S. Mubasshir, M. S. Islam, A. Iqbal, M. S. Rahman, R. Shahriyar
- **Year:** 2022 (ACL NAACL Findings)
- **Relevance:** This is the `csebuetnlp/banglabert` model we use as our text backbone. Must cite.

### [R6] SVM for Disaster Text Classification
- **Title:** "A Comparative Study of Machine Learning Algorithms for Disaster Tweet Classification"
- **General Reference:** SVM has been used widely since 2013 (Imran et al., 2013) for early crisis informatics
- **Relevance:** Justifies our SVM baseline. Show that SVM was the pre-deep-learning standard.

### [R7] CNN/LSTM for Crisis Classification
- **Title:** "Deep Learning for Crisis Tweet Classification" (multiple works)
- **Key Finding:** CNN-LSTM hybrids achieve ~83% on CrisisMMD, validating them as strong baselines
- **Relevance:** Cite to justify including CNN, RNN, BiLSTM in our comparison table.

---

## Category 4: Image-Only Baselines We Must Cite (Cite in Section III)

### [R8] ResNet (Vision Backbone)
- **Title:** "Deep Residual Learning for Image Recognition"
- **Authors:** He, Zhang, Ren, Sun
- **Year:** 2016 (CVPR)
- **Relevance:** Standard image backbone. Must cite when we run ResNet-50 as baseline.

### [R9] EfficientNet (Vision Backbone)
- **Title:** "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
- **Authors:** Tan, Le
- **Year:** 2019 (ICML)
- **Relevance:** Must cite when we run EfficientNet-B0 as baseline.

### [R10] Swin Transformer (Our Vision Backbone)
- **Title:** "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows"
- **Authors:** Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, B. Guo
- **Year:** 2021 (ICCV)
- **Relevance:** This is our primary vision backbone. Must cite.

---

## Category 5: Supporting Architecture Papers (Cite in Section III)

### [R11] TriDA (Lightweight Multimodal Fusion Architecture — Very Similar to Ours)
- **Title:** "TriDA: Privacy-Aware and Efficient Multimodal AI for Disaster Assessment"
- **Authors:** Md Abdullahil Oaphy, Adeel Khalid, Da Hu, Honghui Xu
- **Year:** 2026
- **Architecture:** ResNet50 (vision) + Bidirectional LSTM (text) late fusion with Differential Privacy SGD
- **Relevance:** Most architecturally similar paper to ours. Proves late-fusion is a valid and current approach.

### [R12] DisasterBench / DisasterVL (UAV Edge Computing Motivation)
- **Title:** "DisasterBench: A Multimodal Benchmark for UAV-Based Disaster Response in Complex Environments"
- **Authors:** Tan Zhang, Quanyou Li, Lu Zhang, Jun Liu, Xiaofeng Zhu, Ping Hu
- **Year:** 2026
- **Relevance:** Validates our motivation: edge devices cannot run large VLMs, so lightweight bimodal architectures are necessary.

### [R13] XLM-RoBERTa (Multilingual Encoder)
- **Title:** "Unsupervised Cross-Lingual Representation Learning at Scale"
- **Authors:** A. Conneau et al.
- **Year:** 2020 (ACL)
- **Relevance:** Cite when we reference the Phase 2 CrisisMMD backbone (XLM-RoBERTa-Large).

---

## What the Paper Needs to Show (Summary)

The key argument of the paper is:

> "No prior work has evaluated a full suite of classical ML, deep learning, and transformer-based models as baselines on the BanglaCalamityMMD dataset. We present the first comprehensive benchmark, showing that our multimodal BanglaBERT + Swin-Tiny architecture achieves 98.7% F1, outperforming the dataset creators' own best result (85.25%) by ~13 percentage points."

### The Comparison Table Should Look Like This:

| Model | Modality | Macro F1 | Accuracy |
|---|---|---|---|
| SVM (TF-IDF) | Text | -- | -- |
| Random Forest (TF-IDF) | Text | -- | -- |
| CNN (Text) | Text | -- | -- |
| RNN/GRU | Text | -- | -- |
| BiLSTM | Text | -- | -- |
| BanglaBERT (Sagor Sarkar) | Text | -- | -- |
| BanglaBERT (CSEBUET) | Text | -- | -- |
| ResNet-50 | Image | -- | -- |
| EfficientNet-B0 | Image | -- | -- |
| Swin-Tiny (Image Only) | Image | -- | -- |
| DisasterMultiFusionNet [R4] | Multimodal | -- | 0.8525 |
| BanglaMM-Disaster [R3] | Multimodal | -- | 0.8376 |
| **Ours: BanglaBERT + Swin-Tiny** | **Multimodal** | **0.987** | **0.987** |

