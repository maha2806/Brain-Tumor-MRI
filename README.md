

# 🧠 Brain Tumor MRI Image Classification

Classifies brain MRI slices into **glioma, meningioma, pituitary tumour, or no tumour** using a custom CNN and
transfer-learning models (ResNet50, MobileNetV2, InceptionV3, EfficientNetB0), with a Streamlit app for
real-time predictions.

> ⚠️ Research / education project. Not a medical device; outputs must not replace review by a radiologist.

## Project structure
```
app.py                      Streamlit app (upload MRI -> class + confidence)
Brain_Tumor_MRI_EDA.ipynb   EDA notebook (runs top-to-bottom)
src/config.py               paths, image size, class names
src/data.py                 tf.data loading + augmentation + class weights
src/models.py               custom CNN + transfer-learning factories
src/train.py                training (callbacks, two-stage fine-tuning)
src/evaluate.py             test metrics, confusion matrices, history plots, comparison table
src/make_group_split.py     leakage-free re-split (see "Dataset notes")
models/                     saved .h5 models        reports/   plots + model_comparison.csv
```

## Setup
```bash
pip install -r requirements.txt
# unzip the dataset so that these folders exist: data/Tumour/{train,valid,test}/{glioma,meningioma,no_tumor,pituitary}
```

## Run
```bash
python src/train.py --model custom_cnn --epochs 40
python src/train.py --model all --epochs 15 --finetune-epochs 10   # all 4 backbones (+ custom CNN)
python src/evaluate.py          # -> reports/model_comparison.csv, *_cm.png, *_history.png
streamlit run app.py
```
Use a GPU (e.g. a free Colab T4); on CPU the transfer models take hours.

## Method
- **Preprocessing:** resize to 224x224; 0-1 scaling (custom CNN) or backbone-specific scaling, built into each saved model so the app feeds raw pixels.
- **Augmentation:** flips, rotation, zoom, shifts, contrast and brightness jitter (training only).
- **Custom CNN:** 4 blocks of (Conv-BN-ReLU)x2 + MaxPool + Dropout, GAP, Dense head.
- **Transfer learning:** frozen ImageNet backbone + new Dense head, then optional fine-tuning of the top 30 layers at LR 1e-5 (BatchNorm kept frozen).
- **Training:** Adam, balanced class weights, EarlyStopping + ModelCheckpoint on `val_loss`, ReduceLROnPlateau.
- **Evaluation:** accuracy, macro precision/recall/F1, confusion matrix, size and latency.

## Dataset notes (from the EDA)
- 2,443 images (train 1,695 / valid 502 / test 246), all 640x640 RGB; mild imbalance (glioma largest, no_tumor smallest).
- **Brightness differs strongly by class** (mean grey level: glioma ≈ 28, meningioma ≈ 46, pituitary ≈ 50, no_tumor ≈ 60), likely an acquisition artefact. Brightness/contrast augmentation is used to reduce shortcut learning.
- **Possible leakage:** 164 source images appear in more than one split as augmented copies. Test scores on the original split may be optimistic. Run `python src/make_group_split.py` and point `DATA_DIR` in `src/config.py` to `data/Tumour_grouped` for an honest estimate.

## Results
Run `python src/evaluate.py` and paste `reports/model_comparison.csv` here.

| Model | Accuracy | Macro F1 | Params (M) | Size (MB) | ms/image |
|---|---|---|---|---|---|
| _fill in after training_ | | | | | |
