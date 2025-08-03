# DistilBERT Emotion Classifier  

This repository fine-tunes **distilbert-base-uncased** on the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) (Labeled English Tweets) dataset for lightweight emotion recognition.

---
## Emotion Label Classes

|    SADNESS    |      JOY      |      LOVE     |     ANGER     |      FEAR     |    SURPRISE   |

---

## Training results
| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| No log        | 1.0   | 125  | 0.2232          | 0.9215   |
| No log        | 2.0   | 250  | 0.1552          | 0.9385   |
| No log        | 3.0   | 375  | 0.1469          | 0.9375   |
| 0.2724        | 4.0   | 500  | 0.1395          | 0.933    |

---

## Quick Demo
```python
from transformers import pipeline
classifier = pipeline(
    task="text-classification",
    model="mehmet0sahinn/distilbert-emotion",
)

text = "I'm absolutely thrilled this works like a charm!"
print(classifier(text))
```

---

## Dataset

- Source: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
- Language: English
- Train Size: 16K tweets
- Validation Size: 2K
- Test Size: 2K

---

## Resources

- [Hugging Face Model](https://huggingface.co/mehmet0sahinn/distilbert-emotion)
- [Kaggle Notebook](https://www.kaggle.com/code/mehmet0sahinn/distilbert-emotion-classifier)

---

## License

This repository is licensed under the MIT License.
