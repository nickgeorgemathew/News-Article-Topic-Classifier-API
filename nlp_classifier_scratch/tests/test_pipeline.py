"""
Integration tests for the end-to-end pipeline using synthetic data.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_data(n=200, n_classes=4):
    """Create simple labelled text data."""
    templates = [
        "sports football game player score team win match",
        "business market stock economy profit company revenue",
        "science technology research discovery innovation",
        "world politics government country leader election",
    ]
    texts, labels = [], []
    per_class = n // n_classes
    for label, template in enumerate(templates[:n_classes]):
        words = template.split()
        for _ in range(per_class):
            sample = " ".join(np.random.choice(words, 15, replace=True))
            texts.append(sample)
            labels.append(label)
    return pd.DataFrame({"text": texts, "target": labels})


def make_minimal_model(df):
    """Train a tiny model on synthetic data."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=100)),
        ("clf", LogisticRegression(max_iter=200, random_state=42))
    ])
    pipeline.fit(df["text"], df["target"])
    return pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_df():
    return make_synthetic_data()


@pytest.fixture(scope="module")
def trained_model(synthetic_df):
    return make_minimal_model(synthetic_df)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModelPipeline:
    def test_pipeline_has_correct_steps(self, trained_model):
        assert "tfidf" in trained_model.named_steps
        assert "clf" in trained_model.named_steps

    def test_prediction_output_shape(self, trained_model, synthetic_df):
        preds = trained_model.predict(synthetic_df["text"])
        assert len(preds) == len(synthetic_df)

    def test_predictions_in_valid_range(self, trained_model, synthetic_df):
        preds = trained_model.predict(synthetic_df["text"])
        assert set(preds).issubset({0, 1, 2, 3})

    def test_predict_proba_sums_to_one(self, trained_model, synthetic_df):
        probas = trained_model.predict_proba(synthetic_df["text"][:10])
        row_sums = probas.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_shape(self, trained_model, synthetic_df):
        probas = trained_model.predict_proba(synthetic_df["text"][:5])
        assert probas.shape == (5, 4)


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_evaluate_returns_metrics(self, trained_model, synthetic_df):
        from src.evaluator import ModelEvaluator
        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        evaluator.config = {}
        evaluator.class_names = ["World", "Sports", "Business", "Sci/Tech"]
        evaluator.results = {}

        metrics = evaluator.evaluate(
            trained_model,
            synthetic_df["text"],
            synthetic_df["target"],
            model_name="test_model"
        )
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_macro"] <= 1.0

    def test_error_analysis_returns_dataframe(self, trained_model, synthetic_df):
        from src.evaluator import ModelEvaluator
        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        evaluator.config = {}
        evaluator.class_names = None
        evaluator.results = {}

        errors = evaluator.error_analysis(
            trained_model,
            synthetic_df["text"],
            synthetic_df["target"],
            n_errors=5
        )
        assert isinstance(errors, pd.DataFrame)
        assert "true_label" in errors.columns
        assert "predicted_label" in errors.columns
        assert len(errors) <= 5


# ---------------------------------------------------------------------------
# Explainability tests
# ---------------------------------------------------------------------------

class TestExplainer:
    def test_get_top_features_logistic_regression(self, trained_model):
        from src.explainability import ModelExplainer
        explainer = ModelExplainer.__new__(ModelExplainer)
        explainer.model = trained_model
        explainer.vectorizer = trained_model.named_steps["tfidf"]
        explainer.classifier = trained_model.named_steps["clf"]
        explainer.class_names = ["World", "Sports", "Business", "Sci/Tech"]

        df = explainer.get_top_features(class_idx=0, n=5)
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "weight" in df.columns
        assert len(df) > 0

    def test_get_top_features_naive_bayes(self, synthetic_df):
        from src.explainability import ModelExplainer
        nb_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=100)),
            ("clf", MultinomialNB())
        ])
        nb_pipeline.fit(synthetic_df["text"], synthetic_df["target"])

        explainer = ModelExplainer.__new__(ModelExplainer)
        explainer.model = nb_pipeline
        explainer.vectorizer = nb_pipeline.named_steps["tfidf"]
        explainer.classifier = nb_pipeline.named_steps["clf"]
        explainer.class_names = ["World", "Sports", "Business", "Sci/Tech"]

        df = explainer.get_top_features(class_idx=1, n=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Smoke test: full mini-pipeline
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_train_evaluate_smoke(self, synthetic_df):
        """Minimal end-to-end smoke test without I/O dependencies."""
        from src.evaluator import ModelEvaluator

        train = synthetic_df.sample(frac=0.8, random_state=42)
        test = synthetic_df.drop(train.index)

        model = make_minimal_model(train)
        preds = model.predict(test["text"])

        assert len(preds) == len(test)

        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        evaluator.config = {}
        evaluator.class_names = None
        evaluator.results = {}

        metrics = evaluator.evaluate(model, test["text"], test["target"], "smoke_test")
        assert metrics["accuracy"] > 0.5, "Expected accuracy > 50% on synthetic data"
