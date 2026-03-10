

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from typing import Dict, List, Any, Optional
import os
import yaml


class ModelEvaluator:
   

    def __init__(self, config_path: str = "config.yaml", class_names: Optional[List[str]] = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.class_names = class_names
        self.results = {}

    def evaluate(self, model: Pipeline, X_test: pd.Series, y_test: pd.Series,
                 model_name: str = "model") -> Dict[str, Any]:
        
        print(f"\nEvaluating: {model_name}")
        print("=" * 50)

        y_pred = model.predict(X_test)

        # Try to get probabilities
        try:
            y_proba = model.predict_proba(X_test)
            has_proba = True
        except AttributeError:
            has_proba = False

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average='macro'),
            "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
            "precision_macro": precision_score(y_test, y_pred, average='macro'),
            "recall_macro": recall_score(y_test, y_pred, average='macro'),
        }

        if has_proba and len(np.unique(y_test)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
        elif has_proba:
            try:
                metrics["roc_auc_ovr"] = roc_auc_score(
                    y_test, y_proba, multi_class='ovr', average='macro'
                )
            except Exception:
                pass

        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"F1 (macro):        {metrics['f1_macro']:.4f}")
        print(f"F1 (weighted):     {metrics['f1_weighted']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):    {metrics['recall_macro']:.4f}")

        report = classification_report(y_test, y_pred, target_names=self.class_names)
        print(f"\nClassification Report:\n{report}")

        self.results[model_name] = {
            **metrics,
            "y_pred": y_pred,
            "y_test": y_test,
            "classification_report": report
        }

        return metrics

    def plot_confusion_matrix(self, model_name: str, save_path: Optional[str] = None):
        
        if model_name not in self.results:
            raise ValueError(f"No results found for {model_name}. Run evaluate() first.")

        y_test = self.results[model_name]["y_test"]
        y_pred = self.results[model_name]["y_pred"]
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def compare_models(self, save_path: Optional[str] = None) -> pd.DataFrame:
        
        if not self.results:
            raise ValueError("No models evaluated yet. Run evaluate() first.")

        rows = []
        for model_name, result in self.results.items():
            row = {k: v for k, v in result.items()
                   if k not in ("y_pred", "y_test", "classification_report")}
            row["model"] = model_name
            rows.append(row)

        df = pd.DataFrame(rows).set_index("model")

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(df.to_string())

        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ["accuracy", "f1_macro", "f1_weighted"]
        colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

        for ax, metric in zip(axes, metrics):
            bars = ax.bar(df.index, df[metric], color=colors)
            ax.set_title(metric.replace("_", " ").title(), fontsize=12)
            ax.set_ylim(max(0, df[metric].min() - 0.05), min(1.0, df[metric].max() + 0.05))
            ax.set_ylabel("Score")
            ax.set_xlabel("Model")
            for bar, val in zip(bars, df[metric]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{val:.4f}", ha='center', va='bottom', fontsize=9)

        plt.suptitle("Model Performance Comparison", fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()
        return df

    def error_analysis(self, model: Pipeline, X_test: pd.Series, y_test: pd.Series,
                       n_errors: int = 10) -> pd.DataFrame:
        
        y_pred = model.predict(X_test)
        mask = y_pred != y_test

        error_df = pd.DataFrame({
            "text": X_test[mask].values,
            "true_label": y_test[mask].values,
            "predicted_label": y_pred[mask]
        })

        # Add confidence if available
        try:
            proba = model.predict_proba(X_test)
            error_df["confidence"] = proba[mask].max(axis=1)
            error_df = error_df.sort_values("confidence", ascending=False)
        except AttributeError:
            pass

        print(f"\nTotal errors: {mask.sum()} / {len(y_test)} "
              f"({100 * mask.mean():.2f}%)")
        print(f"\nSample misclassified examples:")
        print(error_df.head(n_errors).to_string(max_colwidth=80))

        return error_df.head(n_errors)
