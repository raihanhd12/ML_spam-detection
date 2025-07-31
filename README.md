# 🚫 Spam Detection Twitter - Deteksi Spam Bahasa Indonesia

Project machine learning untuk deteksi spam pada teks bahasa Indonesia menggunakan transformer models. Project ini mengintegrasikan berbagai sumber data spam (email, SMS, dan data umum) untuk membangun model yang robust dalam mendeteksi konten spam.

## 📋 Deskripsi Project

Sistem deteksi spam ini dirancang khusus untuk teks bahasa Indonesia dengan menggunakan model transformer yang telah dilatih pada dataset gabungan dari berbagai sumber. Model ini dapat mengklasifikasikan teks sebagai spam atau tidak spam dengan akurasi tinggi.

## 🗂️ Struktur Project

```
spam-detection-twitter/
├── 📊 data/                     # Dataset dan file data
│   ├── combined_dataset.csv     # Dataset gabungan (6415 entri)
│   ├── email_spam_indo.csv      # Data spam email Indonesia (2636 entri)
│   ├── sms_spam_indo.csv        # Data spam SMS Indonesia (1143 entri)
│   ├── spam.csv                 # Data spam umum (2636 entri)
│   ├── instagram_posts.csv      # Data postingan Instagram
│   └── twitter_posts.csv        # Data postingan Twitter
├── 🤖 models/v1/                # Model terlatih versi 1
│   ├── config.json             # Konfigurasi model
│   ├── tokenizer_config.json   # Konfigurasi tokenizer
│   ├── tokenizer.json          # File tokenizer
│   ├── vocab.txt               # Vocabulary
│   └── README.md               # Dokumentasi model
├── 📓 Notebooks                # Jupyter notebooks untuk development
│   ├── 0_dataset.ipynb         # Preprocessing dan penggabungan dataset
│   ├── 1_model_inspect.ipynb   # Inspeksi dan analisis model
│   ├── 2_dev.ipynb             # Development dan eksplorasi
│   └── 3_train.ipynb           # Training model
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

### Sumber Data

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

## 🤖 Model

### Arsitektur

- **Base Model**: Transformer-based model (BERT/DistilBERT/IndoBERT-like)
- **Task**: Sequence Classification (Binary: Spam/Not Spam)
- **Language**: Bahasa Indonesia
- **Version**: v1 (tersimpan di `models/v1/`)

### Performance

Model telah dilatih pada 6,415 sampel data dengan berbagai metrik evaluasi yang tersedia di notebook inspeksi.

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

### 5. Prediksi dengan Model Terlatih

```python
from transformers import pipeline

# Load model
classifier = pipeline(
    "text-classification",
    model="./models/v1/",
    tokenizer="./models/v1/"
)

# Prediksi
text = "Selamat! Anda memenangkan hadiah 1 milyar rupiah!"
result = classifier(text)
print(result)
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

## 📝 Catatan

- Model saat ini dioptimalkan untuk teks bahasa Indonesia
- Dataset mencakup berbagai domain (email, SMS, media sosial)
- Model dapat di-fine-tune lebih lanjut dengan data spesifik domain

## 🐛 Issues dan Bug Report

Jika menemukan bug atau ingin request fitur baru, silakan buat issue di repository ini dengan detail yang jelas.

## 📄 Lisensi

Project ini menggunakan lisensi [MIT License](LICENSE) - lihat file LICENSE untuk detail lengkap.

## 👥 Tim Pengembang

- **Data Scientist**: Pengembangan model dan preprocessing
- **ML Engineer**: Optimasi dan deployment model

---

**Terakhir diupdate**: $(date +%Y-%m-%d)

🔗 **Repository**: [spam-detection-twitter](repository-url)
