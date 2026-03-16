import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import os
from sklearn.pipeline import Pipeline
from typing import List,Optional,Dict


class ModelExplainer:


    def __init__(self, model: Pipeline, class_names: Optional[List[str]] = None,
                 config_path: str = "config.yaml"):
            env_path=os.getenv("CONFIG_PATH")
            if env_path:
                full_path=Path(env_path)
            else:
                base_dir=Path(__file__).parent
                project_root=base_dir.parent
                full_path=project_root  /  config_path
                if not full_path.exists():
                    full_path=base_dir / config_path
            if not full_path.exists():
                raise FileNotFoundError(
                    f"Config file not found. Tried: {env_path or project_root/config_path} and {base_dir/config_path}"
                )



            with open(full_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
            self.model=model
            self.class_names=class_names

            self.vectorizer=model.named_steps.get('tfidf')
            self.classifier=model.named_steps.get('clf')

            if  self.vectorizer is None or self.classifier is None:
                raise ValueError("Pipeline must have steps named 'tfidf' and 'clf'.")
            



    def get_top_feature(self,class_idx:int,n:int=20)->pd.DataFrame:
        feature_names=np.array(self.vectorizer.get_feature_names_out())
        clf=self.classifier
        if hasattr(clf,"coef_"):
            weights=clf.coef_[class_idx]
        elif hasattr(clf,'feature_log_prob'):
            weights=clf.feature_log_prob[class_idx]
        else:
             raise ValueError("classifier does not expose ")
            
        top_idx=np.argsort(weights)[::-1][:n]
        bottom_idx=np.argsort(weights)[:n]

        label=self.class_names[class_idx] if self.class_names else f"class{class_idx}"
        top_df=pd.DataFrame({
            "feature":feature_names[top_idx],
            "weights":weights[top_idx],
            "direction":"positive"
        })
        bottom_df=pd.DataFrame({
             "feature":feature_names[bottom_idx],
            "weights":weights[bottom_idx],
            "direction":"negative"
        })
        
        df=pd.concat([top_df,bottom_df],ignore_index=True)
        df["class"]=label
        return df
    def plot_top_features(self,class_idx:int,n:int=15,save_path:Optional[str]=None):
        df=self.get_top_feature(class_idx,n)
        label=df["class"].iloc[0]

        pos=df[df["direction"]=="positive"].head(n)
        neg=df[df["direction"]=="negtive"].head(n)

        combined=pd.concat([pos,neg]).sort_values("weights")

        fig,ax=plt.subplots(figsize=(10, max(6, n // 2)))
        colors= ["#d73027" if w < 0 else "#1a9850" for w in combined["weights"]]
        ax.barh(combined["features"],combined["weights"],color=colors)
        ax.axvline(0,color="black",width=0.8)
        ax.set_title(f"Top Features — {label}", fontsize=13)
        ax.set_xlabel("Feature Weight")
        ax.set_ylabel("Feature (word / n-gram)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Feature importance plot saved: {save_path}")
        else:
            plt.show()
        plt.close()

    def explain_prediction_LIME(self,text:str,num_features:int=10,num_samples:int=500):
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            raise ImportError("Install lime: pip install lime")
        explainer=LimeTextExplainer(class_names=self.class_names)
        exp=explainer.explain_instance(
            text,
            self.model.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
        prediction=self.model.predict([text])[0]
        proba=self.model.predict_proba([text])[0]
        label = self.class_names[prediction] if self.class_names else str(prediction)
        print(f"\nText (first 200 chars): {text[:200]}")
        print(f"Predicted class:        {label}")
        print(f"Confidence:             {proba.max():.4f}")
        print("\nLIME explanation (top features):")
        for feat, weight in exp.as_list():
            direction = "→ supports" if weight > 0 else "→ against"
            print(f"  {feat:<30} {weight:+.4f}  {direction} prediction")

        return {
            "text": text,
            "prediction": prediction,
            "predicted_label": label,
            "probabilities": dict(zip(self.class_names or range(len(proba)), proba)),
            "lime_explanation": exp.as_list()
        }
    def explain_dataset_features(self, n: int = 15, save_path: Optional[str] = None):
       
        n_classes = len(self.classifier.classes_)
        cols = min(3, n_classes)
        rows = (n_classes + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
        axes = np.array(axes).flatten()

        for class_idx in range(n_classes):
            ax = axes[class_idx]
            df = self.get_top_feature(class_idx, n)
            label = df["class"].iloc[0]

            pos = df[df["direction"] == "positive"].head(n)
            neg = df[df["direction"] == "negative"].head(n)
           
            combined = pd.concat([pos, neg]).sort_values("weights")

            colors = ["#d73027" if w < 0 else "#1a9850" for w in combined["weights"]]
            ax.barh(combined["feature"], combined["weights"], color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(f"{label}", fontsize=11)
            ax.set_xlabel("Weight")

        # Hide unused axes
        for ax in axes[n_classes:]:
            ax.set_visible(False)

        plt.suptitle("Top Features per Class", fontsize=14, y=1.01)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"All-class feature plot saved: {save_path}")
        else:
            plt.show()
        plt.close()

