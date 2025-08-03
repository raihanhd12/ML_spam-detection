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
  - nahiar/mail_data
pipeline_tag: text-classification
inference: true
base_model: nahiar/spam-detection-bert-v2
model_type: bert
library_name: transformers
widget:
  - text: "Senin, 21 Juli 2025, Samapta Polsek Ngaglik melaksanakan patroli stasioner balong jalan palagan donoharjo"
    example_title: "Ham Example"
  - text: "Mari berkontribusi terhadap gerakan rakyat dengan membeli baju ini seharga Rp 160.000. Hubungi kami melalui WA 08977472296"
    example_title: "Spam Example"
model-index:
  - name: spam-detection-bert-v3
    results:
      - task:
          type: text-classification
          name: Text Classification
        dataset:
          name: Mail Data Indonesian Spam Detection
          type: csv
        metrics:
          - name: Accuracy
            type: accuracy
            value: 0.95
          - name: F1 Score (Weighted)
            type: f1
            value: 0.95
          - name: Precision (HAM)
            type: precision
            value: 0.98
          - name: Recall (HAM)
            type: recall
            value: 0.96
          - name: Precision (SPAM)
            type: precision
            value: 0.77
          - name: Recall (SPAM)
            type: recall
            value: 0.85
---

# Indonesian Spam Detection BERT

BERT model for spam detection in Indonesian with **95% accuracy**. This v3 model has been fine-tuned from v2 model with email dataset for optimal performance on Indonesian content.

## Quick Start

```python
from transformers import pipeline

# The easiest way to use the model
classifier = pipeline("text-classification",
                     model="nahiar/spam-detection-bert-v3",
                     tokenizer="nahiar/spam-detection-bert-v3")

# Test with text
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

- **Base Model**: nahiar/spam-detection-bert-v2
- **Task**: Binary Text Classification (Spam vs Ham)
- **Language**: Indonesian (Bahasa Indonesia)
- **Model Size**: ~110M parameters
- **Max Sequence Length**: 512 tokens
- **Training Epochs**: 3
- **Batch Size**: 16
- **Learning Rate**: 2e-5

## Performance

| Metric               | HAM | SPAM | Overall |
| -------------------- | --- | ---- | ------- |
| Precision            | 98% | 77%  | 95%     |
| Recall               | 96% | 85%  | 95%     |
| F1-Score             | 97% | 81%  | 95%     |
| **Overall Accuracy** | -   | -    | **95%** |

### Confusion Matrix

- True HAM correctly predicted: 953/988 (96%)
- True SPAM correctly predicted: 115/135 (85%)
- False Positives (HAM predicted as SPAM): 35
- False Negatives (SPAM predicted as HAM): 20

## Key Features

✅ **Fine-tuned** from v2 model with email dataset
✅ **Good accuracy** (95%) on spam detection with Indonesian context
✅ **Better handling** for spam email content
✅ **Enhanced performance** on Indonesian email text
✅ **Optimized** for Indonesian email and social media spam detection

## Label Mapping

```
0: "HAM" (not spam)
1: "SPAM" (spam)
```

## Training Process

This model was retrained using:

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

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nahiar/spam-detection-bert-v3")
model = AutoModelForSequenceClassification.from_pretrained("nahiar/spam-detection-bert-v3")

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
  url={https://huggingface.co/nahiar/spam-detection-bert-v3}
}
```

## Changelog

### Current Version v3 (August 2025)

- Fine-tuned from v2 model with email dataset (mail_data.csv)
- Enhanced handling for Indonesian spam email content
- Good performance (95% accuracy) on email spam detection
- Optimized for Indonesian email and social media content
- Improved with GPU-accelerated training using RTX 3080
