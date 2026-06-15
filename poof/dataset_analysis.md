# Multimodal Bangla Disaster Severity Detection Dataset Analysis

Here is a detailed breakdown of the dataset provided for the datathon.

## 1. Overview & Dataset Splits

The dataset consists of **Train**, **Validation**, and **Test** sets. 
Each sample comes with an image and a Bangla text describing the disaster context.

| Split | Samples | Columns | Description |
| :--- | :--- | :--- | :--- |
| **Train** | `2,800` | `5` | `image_id`, `context`, `category`, `image_name`, `label` |
| **Validation** | `790` | `5` | `image_id`, `context`, `category`, `image_name`, `label` |
| **Test** | `790` | `4` | `image_id`, `context`, `category`, `image_name` (Target `label` is hidden) |

> [!TIP]
> There are absolutely **zero missing values** across any columns in all splits. `image_id` values are also 100% unique within each split.

---

## 2. Category Distribution

The dataset covers eight disaster categories. The training set is perfectly balanced across all categories.

| Category | Train Count | Validation Count |
| :--- | :--- | :--- |
| **Landslides** | 350 | 100 |
| **Wildfire** | 350 | 90 |
| **Tropical Storm** | 350 | 100 |
| **Drought** | 350 | 100 |
| **Flood** | 350 | 100 |
| **Earthquake** | 350 | 100 |
| **Non Disaster** | 350 | 100 |
| **Human Damage** | 350 | 100 |

---

## 3. Label (Severity) Distribution

The task is to classify the severity into one of 5 labels: `Minimal`, `Mild`, `Moderate`, `Severe`, and `Catastrophic`. 
This is an imbalanced classification problem, especially with `Catastrophic`.

| Label | Train Count | Validation Count |
| :--- | :--- | :--- |
| **Moderate** | 883 | 206 |
| **Severe** | 655 | 243 |
| **Mild** | 629 | 100 |
| **Minimal** | 448 | 165 |
| **Catastrophic** | 185 | 76 |

> [!IMPORTANT]
> The severity classes are skewed. You might need to use techniques like class weighting, oversampling, or focal loss to improve performance on minority classes such as `Catastrophic`.

---

## 4. Category vs. Severity (Train Set)

Understanding how severities map to specific categories gives good insight. Notably, all "Non Disaster" samples are strictly labeled as "Minimal".

| Category | Catastrophic | Severe | Moderate | Mild | Minimal |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Drought** | 41 | 115 | 95 | 69 | 30 |
| **Earthquake** | 34 | 82 | 165 | 63 | 6 |
| **Flood** | 29 | 83 | 165 | 72 | 1 |
| **Human Damage** | 3 | 63 | 193 | 88 | 3 |
| **Landslides** | 34 | 125 | 77 | 83 | 31 |
| **Non Disaster** | 0 | 0 | 0 | 0 | 350 |
| **Tropical Storm** | 2 | 47 | 86 | 191 | 24 |
| **Wildfire** | 42 | 140 | 102 | 63 | 3 |

> [!NOTE]
> - `Non Disaster` cases represent the entirety of the 350 `Minimal` labels outside of some edge cases in other categories.
> - `Catastrophic` labels are mostly found in `Wildfire`, `Drought`, `Earthquake`, and `Landslides`.
> - `Tropical Storm` and `Human Damage` rarely hit `Catastrophic`.

---

## 5. Text (Context) Modality Analysis

An analysis of the Bangla text descriptions (word counts) from the training set:

* **Mean Length**: ~17 words
* **Median Length (50%)**: 12 words
* **75th Percentile**: 24 words
* **Max Length**: 147 words
* **Min Length**: 1 word

> [!TIP]
> Since 75% of the texts are under 24 words and the median is 12, a maximum sequence length (`max_len`) of **64 to 128** for transformers like XLM-RoBERTa will be more than sufficient while saving significant memory and compute during training.

---

## 6. Directory Structure

* **Images**: The visual features are divided into `Train/`, `Validation/`, and `Test/` folders. They match one-to-one with the CSV data.
* **CSVs**: 
  * The `image_name` column in the CSV directly maps to the file name inside the image directories.
  * The `context` column contains the Bangla sentence text.
  * The `label` column contains the ground truth (e.g., `Moderate`) which must be predicted for the hidden `Test` set.
