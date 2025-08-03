# 🚫 Spam Detection Twitter - Deteksi Spam Bahasa Indonesia

Project machine learning untuk deteksi spam pada teks bahasa Indonesia menggunakan transformer models. Project ini mengintegrasikan berbagai sumber data spam (email, SMS, dan data umum) untuk membangun model yang robust dalam mendeteksi konten spam.

## 📋 Deskripsi Project

Sistem deteksi spam ini dirancang khusus untuk teks bahasa Indonesia dengan menggunakan model transformer yang telah dilatih pada dataset gabungan dari berbagai sumber. Model v3 terbaru telah di-fine-tune dengan dataset yang diperbaharui dan mencapai akurasi **95%** pada validation set.

## 🗂️ Struktur Project

```
spam-detection-twitter/
├── 📊 data/                     # Dataset dan file data
│   ├── combined_dataset.csv     # Dataset gabungan (6415 entri)
│   ├── email_spam_indo.csv      # Data spam email Indonesia (2636 entri)
│   ├── sms_spam_indo.csv        # Data spam SMS Indonesia (1143 entri)
│   ├── spam.csv                 # Data spam umum (2636 entri)
│   ├── instagram_posts.csv      # Data postingan Instagram
│   ├── twitter_posts.csv        # Data postingan Twitter
│   └── mail_data.csv            # Data email untuk fine-tuning
├── 🤖 models/v3/                # Model terlatih versi 3 (TERBARU)
│   ├── config.json             # Konfigurasi model
│   ├── model.safetensors       # File model weights
│   ├── tokenizer_config.json   # Konfigurasi tokenizer
│   ├── tokenizer.json          # File tokenizer
│   ├── vocab.txt               # Vocabulary
│   └── README.md               # Dokumentasi model v3
├── 📓 notebooks                # Jupyter notebooks untuk development
│   ├── 0_dataset.ipynb         # Preprocessing dan penggabungan dataset
│   ├── 1_model_inspect.ipynb   # Inspeksi dan analisis model
│   ├── 2_dev.ipynb             # Development dan eksplorasi
│   ├── 3_train.ipynb           # Training model dasar
│   ├── 4_fine_tune.ipynb       # Fine-tuning model v2 (eksperimen)
│   └── 5_fine_tune.ipynb       # Fine-tuning model v3 (TERBARU)
├── 🖼️ public/                   # File gambar dan aset publik
│   └── cf_1.png                # Confusion matrix atau visualisasi
├── requirements.txt            # Dependencies Python
└── README.md                   # Dokumentasi ini
```

## 🚀 Instalasi dan Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd spam-detection-twitter
```

### 2. Buat Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# atau
.venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📦 Dependencies

Project ini menggunakan library berikut:

- **Machine Learning**: `transformers`, `torch`, `scikit-learn`, `xgboost`, `lightgbm`
- **Data Processing**: `pandas`, `numpy`, `datasets`, `imbalanced-learn`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`, `wordcloud`
- **Utilities**: `humanize`, `joblib`, `nbformat`, `accelerate`

## 📊 Dataset

### 🤗 Hugging Face Dataset

Model v3 ini menggunakan dataset yang tersedia di Hugging Face:

- **Dataset**: `[USERNAME]/[DATASET_NAME]` <!-- Ganti dengan nama dataset Hugging Face Anda -->
- **Total Samples**: 1,123 pesan email
- **Distribution**: 988 HAM, 135 SPAM
- **Language**: Bahasa Indonesia
- **Format**: CSV dengan kolom `Message` dan `Category`

```python
# Cara menggunakan dataset dari Hugging Face
from datasets import load_dataset

dataset = load_dataset("[USERNAME]/[DATASET_NAME]")
# Ganti [USERNAME]/[DATASET_NAME] dengan dataset Anda
```

### Sumber Data Lainnya

- **Email Spam Indonesia**: 2,636 entri spam email berbahasa Indonesia
- **SMS Spam Indonesia**: 1,143 entri spam SMS berbahasa Indonesia
- **Data Spam Umum**: 2,636 entri data spam dari berbagai sumber
- **Data Media Sosial**: Twitter dan Instagram posts untuk konteks

### Format Data

Dataset gabungan memiliki struktur:

- `Kategori`: Label spam/tidak spam
- `Pesan`: Teks pesan yang akan diklasifikasi
- `source`: Sumber data (email_spam_indo, sms_spam_indo, spam)

## 🔬 Workflow Development

### 1. Dataset Processing (`0_dataset.ipynb`)

- Menggabungkan dataset dari berbagai sumber
- Cleaning dan preprocessing teks
- Balancing dataset untuk mengatasi imbalanced data
- Eksplorasi karakteristik data

### 2. Development & Exploration (`2_dev.ipynb`)

- Analisis eksploratori data (EDA)
- Feature engineering
- Pengujian berbagai pendekatan preprocessing
- Visualisasi distribusi data

### 3. Model Inspection (`1_model_inspect.ipynb`)

- Inspeksi arsitektur model transformer
- Analisis konfigurasi model
- Perbandingan performa berbagai model
- Evaluasi detail dengan confusion matrix

### 4. Model Training (`3_train.ipynb`)

- Fine-tuning model transformer
- Hyperparameter optimization
- Training dengan dataset gabungan
- Evaluasi dan validasi model

### 5. Fine-Tuning v3 (`5_fine_tune.ipynb`) - TERBARU

- Re-training model dengan dataset `mail_data.csv`
- Fine-tuning menggunakan base model `nahiar/spam-detection-bert-v2` untuk menghasilkan model v3
- Training dengan 3 epochs, batch size 16
- Optimasi dengan AdamW optimizer dan learning rate 2e-5
- Evaluasi komprehensif dengan confusion matrix dan classification report

## 🤖 Model

### Model v3 (TERBARU)

- **Base Model**: nahiar/spam-detection-bert-v2 (fine-tuned menjadi model v3)
- **Task**: Sequence Classification (Binary: Spam/Not Spam)
- **Language**: Bahasa Indonesia
- **Version**: v3 (tersimpan di `models/v3/`)
- **Training Data**: mail_data.csv
- **Max Sequence Length**: 128 tokens

### Performance v3

Model v3 telah dilatih dengan hasil sebagai berikut:

- **Validation Accuracy**: 95.10% (epoch terakhir)
- **Training Epochs**: 3
- **Final Training Loss**: 0.0159
- **Dataset Size**: 1,123 samples (setelah train/validation split)

#### Confusion Matrix (Validation Set)

![Confusion Matrix](public/confusion_matrix_v3.png)

<!-- Placeholder untuk confusion matrix image - ganti path sesuai dengan file gambar Anda -->

#### Classification Report

```
              precision    recall  f1-score   support

    not spam       0.98      0.96      0.97       988
        spam       0.77      0.85      0.81       135

    accuracy                           0.95      1123
   macro avg       0.87      0.91      0.89      1123
weighted avg       0.95      0.95      0.95      1123
```

### Training Configuration v3

- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Max Length**: 128 tokens
- **Train/Validation Split**: 80/20
- **GPU**: NVIDIA GeForce RTX 3080

## 💻 Cara Penggunaan

### 1. Jalankan Preprocessing Data

```bash
jupyter lab 0_dataset.ipynb
```

### 2. Eksplorasi Data (Opsional)

```bash
jupyter lab 2_dev.ipynb
```

### 3. Training Model

```bash
jupyter lab 3_train.ipynb
```

### 4. Inspeksi Model

```bash
jupyter lab 1_model_inspect.ipynb
```

### 5. Prediksi dengan Model v3 (TERBARU)

#### Menggunakan Model Lokal

```python
from transformers import pipeline

# Load model v3 dari folder lokal
classifier = pipeline(
    "text-classification",
    model="./models/v3/",
    tokenizer="./models/v3/"
)

# Prediksi
text = "lacak hp hilang by no hp / imei lacak penipu/scammer/tabrak lari/terror/revengeporn sadap"
result = classifier(text)
print(result)
```

#### Menggunakan Model dari Hugging Face

```python
from transformers import pipeline

# Load model v3 dari Hugging Face
classifier = pipeline(
    "text-classification",
    model="[USERNAME]/[MODEL_NAME]"  # Ganti dengan model Hugging Face Anda
)

# Prediksi
text = "lacak hp hilang by no hp / imei lacak penipu/scammer/tabrak lari/terror/revengeporn sadap"
result = classifier(text)
print(result)
```

#### Contoh Prediksi dengan Model v3

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model dan tokenizer v3
tokenizer = AutoTokenizer.from_pretrained("./models/v3/")
model = AutoModelForSequenceClassification.from_pretrained("./models/v3/")

def predict_spam(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    predicted_label = torch.argmax(probs, dim=1).item()
    label_map = {0: "HAM", 1: "SPAM"}
    return label_map[predicted_label]

# Test dengan berbagai contoh
test_cases = [
    "lacak hp hilang by no hp / imei lacak penipu/scammer/tabrak lari/terror/revengeporn sadap",  # SPAM
    "Senin, 21 Juli 2025, Samapta Polsek Ngaglik melaksanakan patroli stasioner balong jalan palagan donoharjo",  # HAM
    "Mari berkontribusi terhadap gerakan rakyat dengan membeli baju ini seharga Rp 160.000. Hubungi kami melalui WA 08977472296"  # SPAM
]

for text in test_cases:
    result = predict_spam(text, model, tokenizer)
    print(f"Text: {text[:50]}...")
    print(f"Prediction: {result}\n")
```

## 📈 Evaluasi Model

Model dievaluasi menggunakan berbagai metrik:

- **Accuracy**: Akurasi keseluruhan
- **Precision**: Precision untuk kelas spam
- **Recall**: Recall untuk kelas spam
- **F1-Score**: Harmonic mean dari precision dan recall
- **Confusion Matrix**: Visualisasi performa klasifikasi

Detail evaluasi dapat dilihat di `1_model_inspect.ipynb`.

## 🛠️ Kontribusi

1. Fork repository ini
2. Buat branch fitur baru (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## �� Catatan

- Model v3 saat ini dioptimalkan untuk teks bahasa Indonesia dengan akurasi 95%
- Dataset mencakup berbagai domain (email, SMS, media sosial)
- Model telah di-fine-tune menggunakan dataset email terbaru untuk performa optimal
- Menggunakan base model BERT yang telah dilatih khusus untuk bahasa Indonesia
- Model dapat di-fine-tune lebih lanjut dengan data spesifik domain

## 🤗 Model di Hugging Face

Model v3 juga tersedia di Hugging Face untuk kemudahan akses:

- **Model**: `[USERNAME]/[MODEL_NAME]` <!-- Ganti dengan nama model Hugging Face Anda -->
- **Dataset**: `[USERNAME]/[DATASET_NAME]` <!-- Ganti dengan nama dataset Hugging Face Anda -->
- **Model Card**: Dokumentasi lengkap tersedia di Hugging Face
- **Inference API**: Tersedia untuk testing langsung

```python
# Quick test di Hugging Face
from transformers import pipeline

classifier = pipeline("text-classification", model="[USERNAME]/[MODEL_NAME]")
result = classifier("Teks yang ingin diuji")
```

## 🆕 Update Terbaru (v3)

- ✅ **Fine-tuned** dengan dataset `mail_data.csv`
- ✅ **Improved accuracy** mencapai 95% pada validation set
- ✅ **Better spam detection** untuk konten berbahasa Indonesia
- ✅ **Enhanced performance** pada deteksi spam email dan media sosial
- ✅ **GPU-accelerated training** menggunakan NVIDIA RTX 3080
- ✅ **Comprehensive evaluation** dengan confusion matrix dan classification report
- ✅ **Available on Hugging Face** untuk akses yang mudah

## 🔧 Customization Guide

### Mengganti Placeholder

Untuk mengkustomisasi README ini dengan informasi Anda:

1. **Ganti `[USERNAME]`** dengan username Hugging Face Anda
2. **Ganti `[MODEL_NAME]`** dengan nama model Anda di Hugging Face
3. **Ganti `[DATASET_NAME]`** dengan nama dataset Anda di Hugging Face
4. **Upload confusion matrix** ke folder `public/` dengan nama `confusion_matrix_v3.png`
5. **Update URL repository** di bagian bawah

### Template untuk Hugging Face

```
Model: [USERNAME]/spam-detection-bert-v3
Dataset: [USERNAME]/indonesian-spam-detection-dataset
```

## 🐛 Issues dan Bug Report

Jika menemukan bug atau ingin request fitur baru, silakan buat issue di repository ini dengan detail yang jelas.

## 📄 Lisensi

Project ini menggunakan lisensi [MIT License](LICENSE) - lihat file LICENSE untuk detail lengkap.

## 👥 Tim Pengembang

- **Data Scientist**: Pengembangan model dan preprocessing
- **ML Engineer**: Optimasi dan deployment model

---

**Terakhir diupdate**: Januari 2025 (Model v3)

🔗 **Repository**: [spam-detection-twitter](repository-url)
