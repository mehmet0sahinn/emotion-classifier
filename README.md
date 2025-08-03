# DistilBERT Emotion Classifier  
Fine-tune **distilBERT-base-uncased** on the `dair-ai/emotion` (English Tweets) dataset (6 labels) for lightweight, real-time emotion recognition.

| sadness | joy | love | anger | fear | surprise |

## Quick Demo
```python
from transformers import pipeline
classifier = pipeline(
    task="text-classification",
    model="mehmet0sahinn/distilbert-emotion",
    top_k=None         # return all labels
)

text = "I'm absolutely thrilled this works like a charm!"
print(classifier(text))
```

## Dataset

- Source: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
- Language: English
- Train Size: 16,000
- Validation Size: 2,000
- Test Size: 2,000

## Resources

- [Hugging Face Model](https://huggingface.co/mehmet0sahinn/distilbert-emotion)
- [Kaggle Notebook](https://www.kaggle.com/code/mehmet0sahinn/emotion-classifier-w-distilbert)

## License

This repository is licensed under the MIT License.
