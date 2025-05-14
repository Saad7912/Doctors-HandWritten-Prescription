# 🩺 OCR-Based Handwritten Prescription Recognition

This project aims to develop an end-to-end pipeline that reads and recognizes handwritten medicine names from Bangladeshi doctor's prescriptions using Optical Character Recognition (OCR) and a deep learning-based language model.

---

## 📌 Project Overview

Handwritten prescriptions are often difficult to read due to inconsistencies in writing styles, particularly in regions like Bangladesh where digital healthcare solutions are not widespread. Our goal was to build a robust system that could automatically extract and predict the correct medicine name from scanned images of handwritten prescriptions using OCR and NLP techniques.

> 🔄 **Mid-Term Pivot:** Initially, we implemented and tested an SVM-based classifier and a custom CNN model. Due to limited performance (≈70% accuracy), we shifted towards a transformer-based model leveraging OCR and a fine-tuned BERT architecture for sequence correction.

---

## 📈 Final Results and Evaluation

| Metric      | Score       |
|-------------|-------------|
| Accuracy    | **99**  |
| Precision   | **92.5%**  |
| Recall      | **92.3%**  |
| F1 Score    | **92.39%**  |

### ✅ Evaluation Highlights

- The model was trained and evaluated using BERT with a custom tokenizer built on a prescription-specific vocabulary.
- OCR output is used as input to the BERT model which corrects and predicts the true medicine name.
- Custom metrics including masked accuracy were calculated to better understand sequence-level performance.

---

## 📊 Model Architecture

- **OCR Engine:** `Tesseract` for initial raw text extraction from images.
- **Transformer:** `BERT For Masked Language Modeling` (`TFBertForMaskedLM`)
- **Tokenization:** `BertTokenizerFast` with a domain-specific vocabulary
- **Training Setup:** TensorFlow 2.x on Kaggle with 3 epochs using a batch size of 4
- **Loss Function:** `SparseCategoricalCrossentropy`
- **Optimizer:** `Adam`

---

## 🧪 Dataset Details

- **Source:** Doctor’s Handwritten Prescription BD Dataset
- **Size:** Includes `Training`, `Validation`, and `Testing` images with corresponding CSV label files.
- **Image Processing:** 
  - Grayscale conversion
  - Resizing to (128, 32)
  - Binarization using Otsu's Thresholding

---

## 🛠 Key Features

- ✅ Custom vocabulary including `[PAD]`, `[MASK]`, `[UNK]`, `[CLS]`, `[SEP]`
- ✅ Automatic preprocessing and OCR pipeline
- ✅ BERT-based label correction model
- ✅ Model evaluation using accuracy, precision, recall, and F1-score

---

## 💡 Learnings and Achievements

- Learned how to combine OCR and language modeling to overcome poor image quality and inconsistent handwriting.
- Gained hands-on experience with HuggingFace Transformers, custom tokenizers, and TensorFlow training pipelines.
- Improved the model's understanding of medicine names through domain-specific vocabulary and label encoding.

---

## 🔄 Comparison with Mid-Term Plan

| Item | Mid-Term Plan | Final Implementation |
|------|---------------|----------------------|
| **Model** | Custom CNN | BERT-based OCR Pipeline |
| **Accuracy** | ~70% | 99% |
| **Image Augmentation** | ✅ Applied | ❌ Dropped due to shift to OCR |
| **Mid-Term Suggestions** | Try OCR | ✅ Fully implemented |
| **Evaluation Metrics** | Accuracy only | Accuracy, Precision, Recall, F1 |

---

## 🚧 Final Challenges and Solutions

| Challenge | Solution |
|----------|----------|
| Low accuracy of CNN | Switched to text-based model |
| Noisy OCR outputs | Added preprocessing (thresholding, resizing) |
| BERT training with custom vocab | Created domain-specific tokenizer |

---

## 🔮 Future Scope

- 🌐 Deploy the model as a web application using Streamlit or Flask
- 📱 Convert into a mobile app for use by pharmacists and hospitals
- 🧠 Incorporate context-aware suggestions for dosage and prescription verification
- 💬 Extend to multi-language support for regional medicine recognition

---

## 👨‍💻 Project Tools

- **Tools & Libraries:** Python, OpenCV, Tesseract, TensorFlow, HuggingFace Transformers, Pandas, NumPy, Matplotlib



