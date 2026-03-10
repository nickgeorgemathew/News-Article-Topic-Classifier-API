# 📰 Text Classification Project

A complete, production-ready text classification system built over 5 days.
Uses TF-IDF vectorization with Naive Bayes and Logistic Regression on the AG News dataset.

---

## Project Structure

```
text-classification-project/
├── config.yaml              # All tunable settings
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Dataset loading (AG News, 20 Newsgroups)
│   ├── preprocessor.py      # Tokenization, stopword removal, lemmatization
│   ├── model_trainer.py     # Training + GridSearch / RandomizedSearch
│   ├── evaluator.py         # Metrics, confusion matrices, error analysis
│   ├── explainability.py    # Feature importance + LIME
│   └── pipeline.py          # End-to-end orchestration
├── tests/
│   ├── test_preprocessor.py # Unit tests for preprocessing
│   └── test_pipeline.py     # Integration tests with synthetic data
├── app/
│   ├── streamlit_app.py     # Interactive web UI
│   └── fastapi_app.py       # REST API
├── notebooks/
│   └── end_to_end_demo.ipynb
├── data/
└── models/
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional, for spaCy lemmatization
python -m nltk.downloader punkt stopwords wordnet
```

### 2. Run the full pipeline

```python
from src.pipeline import TextClassificationPipeline

pipeline = TextClassificationPipeline()
pipeline.run()            # loads data → preprocesses → trains → evaluates → saves
```

Or from the command line:

```bash
cd src
python pipeline.py
```

### 3. Run tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 4. Launch the web apps

```bash
# Interactive Streamlit UI
streamlit run app/streamlit_app.py

# REST API (visit http://localhost:8000/docs for Swagger UI)
uvicorn app.fastapi_app:app --reload
```

---

## 5-Day Plan

| Day | Focus | Key files |
|-----|-------|-----------|
| 1 | Data loading + baseline preprocessing | `data_loader.py`, `preprocessor.py` |
| 2 | TF-IDF + model training | `model_trainer.py`, `config.yaml` |
| 3 | Hyperparameter tuning + evaluation | `evaluator.py` |
| 4 | Explainability + error analysis | `explainability.py` |
| 5 | Testing + deployment (Streamlit / FastAPI) | `tests/`, `app/` |

---

## Configuration

All settings live in `config.yaml`:

```yaml
dataset:
  name: "ag_news"      # or "20newsgroups"
  test_size: 0.2

preprocessing:
  lowercase: true
  remove_stopwords: true
  lemmatize: true
  max_features: 10000
  ngram_range: [1, 2]

models:
  naive_bayes:
    alpha: [0.1, 0.5, 1.0]
  logistic_regression:
    C: [0.1, 1.0, 10.0]
```

---

## API Reference

### POST `/predict`

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Tesla reports record earnings this quarter."}'
```

Response:
```json
{
  "predicted_label": "Business",
  "confidence": 0.9823,
  "probabilities": {"World": 0.01, "Sports": 0.003, "Business": 0.982, "Sci/Tech": 0.005}
}
```

### POST `/predict/batch`

```bash
curl -X POST http://localhost:8000/predict/batch \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Lakers win the NBA finals!", "New Mars rover launched."]}'
```

---

## Results (AG News, typical run)

| Model | Accuracy | F1 (macro) |
|-------|----------|------------|
| Baseline NB | ~0.89 | ~0.89 |
| Naive Bayes (tuned) | ~0.90 | ~0.90 |
| Logistic Regression (tuned) | ~0.92 | ~0.92 |

---

## License

MIT
