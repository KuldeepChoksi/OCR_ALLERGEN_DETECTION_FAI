# OCR Allergen Detection & Complex Ingredient Simplifier

**Custom OCR for Food Safety**  
Group 15:  Harshith Umesh, Kuldeep Choksi, Ronit Naik, Kiran Deav

---

## Table of Contents

- [Introduction](#introduction)  
- [Solution Overview](#solution-overview)  
- [Features](#features)  
- [Directory Structure](#directory-structure)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Data Preparation](#data-preparation)  
- [Usage](#usage)  
- [Model Training & Evaluation](#model-training--evaluation)  
- [Future Work](#future-work)  

---

## Introduction

Millions of people suffer from food allergies, yet ingredient labels often contain complex, hard‑to‑read chemical or botanical names. Our project builds a tailored OCR pipeline to:

1. **Extract** ingredient text from label images using a custom CNN.  
2. **Post‑process** via fuzzy matching (Levenshtein distance) and heuristic rules.  
3. **Map** complex ingredient names to simpler, user‑friendly synonyms.  
4. **Flag** known allergens at a glance.

---


## Solution Overview

- **Data Pipeline**  
  - ~174 000 generated character images (A–Z, a–z) with rotations and noise  
  - Real‑world photos for robustness  
- **Preprocessing**  
  - Grayscale conversion  
  - Binarization (Otsu’s thresholding)  
  - Morphological dilation to detect word boundaries  
- **Custom CNN**  
  - Achieved **95 % character‑level accuracy**  
- **Post‑Processing**  
  - Candidate trimming & ambiguous‑character replacement (e.g. `I`↔`l`, `r`↔`f`)  
  - Fuzzy matching against two CSV datasets:  
    - `FoodData.csv` (allergens list)  
    - `ComplexIngredients.csv` (complex→simple map)  
- **Output**  
  - Highlighted allergens  
  - Simplified ingredient list  

---

## Features

- 🔍 **Accurate OCR** on noisy, rotated, colored‑background images  
- 🧹 **Heuristic cleanup** for punctuation & font‑variation issues  
- 🔗 **Fuzzy matching** to handle small OCR errors  
- ⚠️ **Allergen alerts** and **ingredient simplification**  

---



## Directory Structure

```
.
├── Presentation.pdf
├── README.md
├── data
│   ├── ComplexIngredients.csv
│   ├── FoodData.csv
│   ├── download_kaggle_dataset.py
│   └── generate_chars.py
├── models
│   ├── custom_cnn_model_all.keras
│   ├── model_accuracy.txt
│   ├── Training_validation_loss.png
│   ├── model_statistics.py
│   └── ocr_cnn.py
├── requirements.txt
└── src
    ├── gui.py
    └── ocr.py
```

> **Note:** Any `old-approach/` subdirectories have been deprecated and can be ignored.

---

## Getting Started

### Prerequisites

- Python 3.8+  
- pip (or conda)

### Installation

1. Clone this repository  
   ```bash
   git clone https://github.com/KuldeepChoksi/OCR_ALLERGEN_DETECTION_FAI.git
   cd OCR_ALLERGEN_DETECTION_FAI
   ```
2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

- **Download real‑world ingredient images dataset** 
  ```bash
  python data/download_kaggle_dataset.py
  ```
- **Generate additional character samples** (A–Z, a–z):  
  ```bash
  python data/generate_chars.py
  ```

---

## Usage

1. **Run the OCR engine** from the command line:  
   ```bash
   python src/ocr.py --input path/to/image.jpg
   ```
2. **Launch the simple GUI**:  
   ```bash
   python src/gui.py
   ```
3. **View output**  
   - Detected text & bounding boxes  
   - Allergen highlights  
   - Simplified ingredient list  

---

## Model Training & Evaluation

- **Training script:** `models/ocr_cnn.py`  
- **Statistics & plots:**  
  - `models/model_statistics.py`  
  - `models/Training_validation_loss.png`  
- **Final weights:** `models/custom_cnn_model_all.keras`  
- **Test accuracy:** recorded in `models/model_accuracy.txt`  

To re‑train, modify hyperparameters in `models/ocr_cnn.py` and run:

```bash
python models/ocr_cnn.py --epochs 20 --batch-size 64
```

---

## Future Work

- 📱 **Mobile app** for on‑the‑fly label scanning  
- 📦 Expand allergen & synonym datasets  
- 🌐 Deploy as a web service with REST API  

---


