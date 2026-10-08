"""
Brain Tumor MRI - Streamlit EDA Application

Run:
    streamlit run App.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor MRI Analysis",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🧠 Brain Tumor MRI Analysis")

st.markdown(
    """
    **Exploratory Data Analysis of Brain Tumor MRI Images**

    This application presents the key findings from the Brain Tumor MRI
    dataset, including class distribution, image characteristics,
    intensity statistics, and dataset-level observations.
    """
)


# --------------------------------------------------
# Dataset information
# --------------------------------------------------

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Images", "2,443")

with col2:
    st.metric("Image Size", "640 × 640")

with col3:
    st.metric("Image Type", "RGB")

with col4:
    st.metric("Classes", "4")


# --------------------------------------------------
# Classes
# --------------------------------------------------

st.header("🏷️ Tumor Classes")

classes = pd.DataFrame(
    {
        "Class": [
            "Glioma",
            "Meningioma",
            "Pituitary",
            "No Tumor"
        ],
        "Description": [
            "MRI images associated with glioma tumors.",
            "MRI images associated with meningioma tumors.",
            "MRI images associated with pituitary tumors.",
            "MRI images without a detected tumor."
        ]
    }
)

st.dataframe(
    classes,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# Dataset structure
# --------------------------------------------------

st.header("📁 Dataset Structure")

st.code(
    """
Tumour/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
│
├── valid/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
│
└── test/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/
""",
    language="text"
)


# --------------------------------------------------
# EDA findings
# --------------------------------------------------

st.header("🔍 Key EDA Findings")

findings = [
    "The dataset contains four MRI image classes.",
    "The images are RGB images with a resolution of 640 × 640 pixels.",
    "The dataset is divided into training, validation, and test sets.",
    "Class distribution and split distribution should be considered when evaluating model performance.",
    "Image intensity and contrast characteristics were examined during the EDA.",
    "Source-aware splitting is important when augmented or related images originate from the same source image.",
]

for finding in findings:
    st.markdown(f"• {finding}")


# --------------------------------------------------
# Medical disclaimer
# --------------------------------------------------

st.divider()

st.warning(
    """
    **Research / Education Only**

    This project is intended for educational and research purposes.
    It is not a medical diagnostic system and should not be used to
    diagnose, treat, or make clinical decisions about patients.
    """
)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Brain Tumor MRI Analysis | Exploratory Data Analysis Project"
)
