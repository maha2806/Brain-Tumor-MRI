# 🧠 Brain Tumor MRI — Exploratory Data Analysis

An exploratory data analysis project on brain MRI images belonging to four classes:

- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

The project focuses on understanding the structure, distribution, visual characteristics, intensity patterns, and potential data-quality issues within the Brain Tumor MRI dataset.

> ⚠️ **Research / Education Only:**  
> This project is intended for educational and research purposes. It is not a medical diagnostic system and should not be used to diagnose, treat, or make clinical decisions about patients.

---

## 📌 Project Overview

Brain MRI datasets can contain differences in image characteristics that are important to understand before applying machine-learning techniques.

This project performs an exploratory analysis of the MRI dataset to investigate:

- Dataset composition
- Class distribution
- Train, validation, and test splits
- Image dimensions and image properties
- Pixel-intensity characteristics
- Image brightness and contrast
- Foreground characteristics
- Pixel-intensity distributions
- File-size characteristics
- Relationships between image-level features
- Correlations between numerical features
- Potential source-image overlap across dataset splits
- Dataset-quality considerations and limitations

The analysis is performed at the **dataset level** and is intended to provide a clearer understanding of the data before further machine-learning work.

---

## 📂 Project Structure

```text
Brain-Tumor-MRI/
│
├── App.py
├── Brain_Tumor_MRI_.ipynb
├── Config.py
├── Data.py
├── Evaluate.py
├── LICENSE
├── Make group split.py
├── Models.py
├── README.md
├── Requirements.txt
├── Technical Tags.txt
└── Train.py
