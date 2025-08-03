---
language:
  - id
license: mit
tags:
  - text-classification
  - bert
  - spam-detection
  - indonesian
  - twitter
  - retrained
datasets:
  - nahiar/spam_detection_v2
pipeline_tag: text-classification
inference: true
base_model: nahiar/spam-detection-bert-v1
model_type: bert
library_name: transformers
widget:
  - text: "lacak hp hilang by no hp / imei lacak penipu/scammer/tabrak lari/terror/revengeporn sadap / hack / pulihkan akun"
    example_title: "Spam Example"
  - text: "Senin, 21 Juli 2025, Samapta Polsek Ngaglik melaksanakan patroli stasioner balong jalan palagan donoharjo"
    example_title: "Ham Example"
  - text: "Mari berkontribusi terhadap gerakan rakyat dengan membeli baju ini seharga Rp 160.000. Hubungi kami melalui WA 08977472296"
    example_title: "Obvious Spam"
model-index:
  - name: spam-detection-bert
    results:
      - task:
          type: text-classification
          name: Text Classification
        dataset:
          name: Indonesian Spam Detection Dataset v2
          type: nahiar/spam_detection_v2
        metrics:
          - name: Accuracy
            type: accuracy
            value: 0.99
          - name: F1 Score (Weighted)
            type: f1
            value: 0.99
          - name: Precision (HAM)
            type: precision
            value: 0.99
          - name: Recall (HAM)
            type: recall
            value: 1.00
          - name: Precision (SPAM)
            type: precision
            value: 1.00
          - name: Recall (SPAM)
            type: recall
            value: 0.83
---

# Indonesian Spam Detection BERT

Model BERT untuk deteksi spam dalam bahasa Indonesia dengan akurasi **99%**. Model ini telah di-retrain dengan dataset yang telah diperbarui dan dilabeli ulang untuk performa yang optimal pada konten Indonesia.

## Quick Start

```python
from transformers import pipeline

# Cara termudah menggunakan model
classifier = pipeline("text-classification",
                     model="nahiar/spam-detection-bert",
                     tokenizer="nahiar/spam-detection-bert")

# Test dengan teks
texts = [
    "lacak hp hilang by no hp / imei lacak penipu/scammer/tabrak lari/terror/revengeporn sadap / hack / pulihkan akun",
    "Senin, 21 Juli 2025, Samapta Polsek Ngaglik melaksanakan patroli stasioner balong jalan palagan donoharjo",
    "Mari berkontribusi terhadap gerakan rakyat dengan membeli baju ini seharga Rp 160.000. Hubungi kami melalui WA 08977472296"
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Result: {result['label']} (confidence: {result['score']:.4f})")
    print("---")
```

## Model Details

- **Base Model**: nahiar/spam-detection-bert-v1 (fine-tuned from cahya/bert-base-indonesian-1.5G)
- **Task**: Binary Text Classification (Spam vs Ham)
- **Language**: Indonesian (Bahasa Indonesia)
- **Model Size**: ~110M parameters
- **Max Sequence Length**: 512 tokens
- **Training Epochs**: 3
- **Batch Size**: 16
- **Learning Rate**: 2e-5

## Performance

| Metric               | HAM  | SPAM | Overall |
| -------------------- | ---- | ---- | ------- |
| Precision            | 99%  | 100% | 99%     |
| Recall               | 100% | 83%  | 99%     |
| F1-Score             | 99%  | 91%  | 99%     |
| **Overall Accuracy** | -    | -    | **99%** |

### Confusion Matrix

- True HAM correctly predicted: 430/430 (100%)
- True SPAM correctly predicted: 25/30 (83%)
- False Positives (HAM predicted as SPAM): 0
- False Negatives (SPAM predicted as HAM): 5

## Dataset

Model v2 ini dilatih ulang menggunakan dataset yang telah diperbarui dan dilabeli ulang secara manual:

- **Dataset**: spam_re_labelled_vNew.csv
- **Total Samples**: 460 pesan
- **Distribution**: 430 HAM, 30 SPAM
- **Encoding**: Latin-1
- **Quality**: Manual re-labeling untuk akurasi yang lebih tinggi

**Updated**: Januari 2025

## Key Features

✅ **Re-trained** dengan dataset yang telah dilabeli ulang secara manual
✅ **High accuracy** (99%) pada deteksi spam dengan konteks Indonesia
✅ **Better handling** untuk pesan dengan format yang kompleks
✅ **Enhanced performance** pada teks dengan campuran formal dan informal
✅ **Optimized** untuk konten media sosial Indonesia

## Label Mapping

```
0: "HAM" (tidak spam)
1: "SPAM" (spam)
```

## Training Process

Model ini di-retrain menggunakan:

- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Epochs**: 3
- **Batch Size**: 16
- **Max Length**: 128 tokens
- **Train/Validation Split**: 80/20

## Usage Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model dan tokenizer
tokenizer = AutoTokenizer.from_pretrained("nahiar/spam-detection-bert")
model = AutoModelForSequenceClassification.from_pretrained("nahiar/spam-detection-bert")

def predict_spam(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    predicted_label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_label].item()
    label_map = {0: "HAM", 1: "SPAM"}
    return label_map[predicted_label], confidence

# Test
text = "Dapatkan uang dengan mudah! Klik link ini sekarang!"
result, confidence = predict_spam(text)
print(f"Prediksi: {result} (Confidence: {confidence:.4f})")
```

## Citation

```bibtex
@misc{nahiar_spam_detection_bert,
  title={Indonesian Spam Detection BERT},
  author={Raihan Hidayatullah Djunaedi},
  year={2025},
  url={https://huggingface.co/nahiar/spam-detection-bert}
}
```

## Changelog

### Current Version (January 2025)

- Re-trained model dengan dataset yang telah dilabeli ulang secara manual
- Enhanced handling untuk konten Indonesia yang kompleks
- Better performance pada deteksi spam dengan konteks lokal Indonesia
- Optimized untuk konten media sosial (Twitter, Instagram, dll)
- Improved accuracy dengan distribusi dataset yang lebih balanced
