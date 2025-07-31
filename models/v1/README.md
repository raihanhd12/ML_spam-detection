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
datasets:
  - nahiar/spam_detection
pipeline_tag: text-classification
inference: true
base_model: cahya/bert-base-indonesian-1.5G
model_type: bert
library_name: transformers
widget:
  - text: "Dapatkan hadiah gratis dengan klik link ini!"
    example_title: "Spam Example"
  - text: "Selamat pagi, bagaimana kabarnya hari ini?"
    example_title: "Ham Example"
  - text: "GRATIS! Klik link ini untuk mendapat hadiah jutaan rupiah!"
    example_title: "Obvious Spam"
model-index:
  - name: spam-detection-bert-v1
    results:
      - task:
          type: text-classification
          name: Text Classification
        dataset:
          name: Indonesian Spam Detection Dataset
          type: nahiar/spam_detection
        metrics:
          - name: Accuracy
            type: accuracy
            value: 0.99
          - name: F1 Score
            type: f1
            value: 0.99
          - name: Precision
            type: precision
            value: 0.99
          - name: Recall
            type: recall
            value: 0.99
---

# Indonesian Spam Detection BERT v1

Model BERT yang di-fine-tune untuk deteksi spam dalam bahasa Indonesia dengan akurasi **99%**.

## Quick Start

```python
from transformers import pipeline

# Cara termudah menggunakan model
classifier = pipeline("text-classification",
                     model="nahiar/spam-detection-bert-v1",
                     tokenizer="nahiar/spam-detection-bert-v1")

# Test dengan teks
texts = [
    "Dapatkan hadiah gratis dengan klik link ini!",
    "Selamat pagi, bagaimana kabarnya hari ini?",
    "GRATIS! Klik link ini untuk mendapat hadiah jutaan rupiah!"
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Result: {result['label']} (confidence: {result['score']:.4f})")
    print("---")
```

## Model Details

- **Base Model**: cahya/bert-base-indonesian-1.5G
- **Task**: Binary Text Classification (Spam vs Ham)
- **Language**: Indonesian (Bahasa Indonesia)
- **Model Size**: ~110M parameters
- **Max Sequence Length**: 512 tokens

## Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 99.00% |
| F1 Score  | 99.00% |
| Precision | 99.00% |
| Recall    | 99.00% |

## Dataset

Model ini dilatih menggunakan dataset gabungan dari:

- SMS spam Indonesia
- Email spam Indonesia
- Twitter posts
- Instagram posts

**Total**: 6,415 pesan (3,310 spam, 3,105 ham)

**Updated**: January 2025

## Label Mapping

```
0: "ham" (tidak spam)
1: "spam" (spam)
```

## Citation

```bibtex
@misc{nahiar_spam_detection_bert_v1,
  title={Indonesian Spam Detection BERT v1},
  author={Raihan Hidayatullah Djunaedi},
  year={2025},
  url={https://huggingface.co/nahiar/spam-detection-bert-v1}
}
```
