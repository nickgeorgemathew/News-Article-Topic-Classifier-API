"""
End-to-end text classification pipeline.
Orchestrates data loading, preprocessing, training, and evaluation.
"""

import os
import yaml
import joblib
import pandas as pd
from typing import Optional, List, Dict, Any
import os
from pathlib import Path
from data_loader import DataLoader
from preprocessor import preprocess as pre
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from explainability import ModelExplainer


# AG News label mapping
AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


class TextClassificationPipeline:
    """Full end-to-end text classification pipeline."""

    
    def __init__(self, config: str = "config.yaml"):
         # 1. Check if CONFIG_PATH environment variable is set
            env_path = os.getenv("CONFIG_PATH")

            if env_path:
                full_path = Path(env_path)
            else:
                # 2. Default: look in project root
                base_dir = Path(__file__).parent
                project_root = base_dir.parent
                full_path = project_root / config

                # 3. Fallback: look inside src/
                if not full_path.exists():
                    full_path = base_dir / config

            # 4. Final check
            if not full_path.exists():
                raise FileNotFoundError(
                    f"Config file not found. Tried: {env_path or project_root/config} and {base_dir/config}"
                )

            # Load config
            with open(full_path, "r") as f:
                self.config_ = yaml.safe_load(f)
            self.config_path =self.config_
            self.class_names: Optional[List[str]] = None
            self.best_model = None
            self.best_model_name: Optional[str] = None

            # Ensure output directories exist
            for d in self.config_path["paths"].values():
                os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    def load_data(self) -> tuple:
        #load data from local
        if self.config_path["dataset"]["flag"]=="custom":
            self.train_df=pd.read_csv(self.config_path["dataset"]["paths"]["train"])
            self.test_df=pd.read_csv(self.config_path["dataset"]["paths"]["test"])
        else:
            #load data from hugging face
            loader = DataLoader(self.config_path)
            self.train_df, self.test_df = loader.load_data()
        

        self.class_names=[AG_NEWS_LABELS[i] for i in sorted(AG_NEWS_LABELS)]

        # Resolve class names for known datasets
        # if self.config["dataset"]["name"] == "ag_news":
        #     self.class_names = [AG_NEWS_LABELS[i] for i in sorted(AG_NEWS_LABELS)]
        

        print(f"\nData loaded. Train: {len(self.train_df)}, Test: {len(self.test_df)}")
        return self.train_df, self.test_df

    # ------------------------------------------------------------------
    # Step 2: Preprocess
    # ------------------------------------------------------------------
    def preprocess(self) -> tuple:
        print("\nPreprocessing texts...")
        text_cols=self.train_df.select_dtypes(include=['object','string'])
        self.X_train = (self.train_df["Title"].astype(str) + " " + self.train_df["Description"].astype(str)).tolist()
        self.X_train= [pre(i) for i in self.X_train]
        self.X_test = (self.test_df["Title"].astype(str) + " " + self.test_df["Description"].astype(str)).tolist()
        self.X_test=[pre(i) for i in self.X_test]
        self.y_train = self.train_df["Class Index"]
        self.y_test = self.test_df["Class Index"]

        print(f"Sample preprocessed text:\n  {self.X_train[450]}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    # ------------------------------------------------------------------
    # Step 3: Train & tune models
    # ------------------------------------------------------------------
    def train(self) -> Dict[str, Any]:
        
        trainer=ModelTrainer()
        print("\n--- Baseline ---")
        baseline = trainer.train_baseline(self.X_train, self.y_train)

        print("\n--- Naive Bayes ---")
        nb = trainer.tune_naive_bayes(self.X_train, self.y_train)

        print("\n--- Logistic Regression ---")
        lr = trainer.tune_logistic_regression(self.X_train, self.y_train)

        self.trained_models = {
            "Baseline (NB)": baseline["model"],
            "Naive Bayes (tuned)": nb["best_model"],
            "Logistic Regression (tuned)": lr["best_model"],
        }
        
        self.trainer=trainer
        return self.trained_models

    # ------------------------------------------------------------------
    # Step 4: Evaluate
    # ------------------------------------------------------------------
    def evaluate(self) -> pd.DataFrame:
        evaluator = ModelEvaluator(self.config_path, class_names=self.class_names)

        for name, model in self.trained_models.items():
            evaluator.evaluate(model, self.X_test, self.y_test, model_name=name)

        comparison = evaluator.compare_models(
            save_path=os.path.join(self.config_path["paths"]["results_dir"], "model_comparison_temp.png")
        )

        # Pick best model by F1 macro
        best_name = comparison["f1_macro"].idxmax()
        self.best_model = self.trained_models[best_name]
        self.best_model_name = best_name
        print(f"\nBest model: {best_name} (F1 macro = {comparison.loc[best_name, 'f1_macro']:.4f})")

        # Confusion matrix for best model
        evaluator.plot_confusion_matrix(
            best_name,
            save_path=os.path.join(self.config_path["paths"]["results_dir"], "confusion_matrix_temp.png")
        )

        self.evaluator = evaluator
        return comparison

    # ------------------------------------------------------------------
    # Step 5: Explain
    # ------------------------------------------------------------------
    def explain(self):
        if self.best_model is None:
            raise RuntimeError("Run evaluate() first.")

        explainer = ModelExplainer(
            self.best_model,
            class_names=self.class_names
        )

        print(f"\nGenerating feature importance plots for: {self.best_model_name}")
        explainer.explain_dataset_features(
            n=15,
            save_path=os.path.join(self.config_path["paths"]["results_dir"], "feature_importance.png")
        )

        # Explain a single example
        sample_text = self.test_df["Description"].iloc[0]
        print(f"\nExample explanation for:\n  {sample_text[:200]}")
        try:
            explainer.explain_prediction_LIME(sample_text, num_features=10, num_samples=300)
        except ImportError:
            print("  (Install `lime` for per-instance explanations: pip install lime)")

    # ------------------------------------------------------------------
    # Step 6: Save best model
    # ------------------------------------------------------------------
    def save_best_model(self):
        if self.best_model is None:
            raise RuntimeError("Run evaluate() first.")

        path = os.path.join(self.config_path["paths"]["models_dir"], "best_model.joblib")
        joblib.dump(self.best_model, path)
        print(f"\nBest model saved to: {path}")
        return path

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict labels for a list of raw texts using the best model.

        Args:
            texts: List of raw (unprocessed) text strings

        Returns:
            List of dicts with prediction and probabilities
        """
        if self.best_model is None:
            raise RuntimeError("No model available. Run the full pipeline first.")

        
        processed = [pre(t) for t in texts]
        preds = self.best_model.predict(processed)
        probas = self.best_model.predict_proba(processed)

        results = []
        for text, pred, proba in zip(texts, preds, probas):
            label = self.class_names[pred] if self.class_names else str(pred)
            results.append({
                "text": text[:100],
                "predicted_class": int(pred),
                "predicted_label": label,
                "confidence": float(proba.max()),
                "probabilities": {
                    (self.class_names[i] if self.class_names else str(i)): float(p)
                    for i, p in enumerate(proba)
                }
            })
        return results

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------
    def run(self):
        """Run the complete pipeline end-to-end."""
        print("=" * 60)
        print("TEXT CLASSIFICATION PIPELINE")
        print("=" * 60)

        self.load_data()
        self.preprocess()
        self.train()
        self.evaluate()
        self.explain()
        self.save_best_model()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Best model : {self.best_model_name}")
        print(f"Models dir : {self.config_path['paths']['models_dir']}")
        print(f"Results dir: {self.config_path['paths']['results_dir']}")


if __name__ == "__main__":
    pipeline = TextClassificationPipeline()
    pipeline.run()
