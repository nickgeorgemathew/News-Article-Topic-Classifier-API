import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,f1_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
from sklearn.pipeline import Pipeline
from typing import Dict,List,Any,Optional
import yaml
import os
import matplotlib as plt
import seaborn as sns


results={}

def evaluate(model:Pipeline,X_test:pd.Series,y_test:pd.Series,model_name:str="model")->Dict[str,Any]:
    print(f"Evaluating{model_name}")
    print("="*50)

    y_pred=model.predict(X_test)
    try:
        y_proba=model.predict_proba(X_test)
        has_proba=True
    except AttributeError:
        has_proba=False
    metrics={

        "accuracy":accuracy_score(y_test,y_pred),
        "precision_macro":precision_score(y_test,y_pred,average='macro'),
        "recall_macro":recall_score(y_test,y_pred,average='macro'),
        "f1_macro":f1_score(y_test,y_pred,average='macro'),
        "f1_weight":f1_score(y_test,y_pred,average='weighted')
    }
    if has_proba and len(np.unique(y_test))==2:
        metrics["roc_auc"]=roc_auc_score(y_test,y_proba[:,1])
    elif has_proba:
        try:
            metrics["roc_auc_ovr"]=roc_auc_score(y_test,y_proba,multi_class='ovr',average='macro')
        except Exception:
            pass
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted):     {metrics['f1_weighted']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    report=classification_report(y_test,y_pred,target_names=["World","Sports","Business","Sci/Tech"])
    print(f"classification report :\n{report}")
    results[model_name] = {
            **metrics,
            "y_pred": y_pred,
            "y_test": y_test,
            "classification_report": report
        }

    return metrics






def plot_confusion_matrix(model_name:str,save_path:Optional[str]=None):
    if model_name not in results:
        raise ValueError(f"no result found for{model_name}.Run evaluate() first")
    y_test=results[model_name]["y_test"]
    y_pred = results[model_name]["y_pred"]
    cm=confusion_matrix(y_test,y_pred)

    fig,ax=plt.subplots(figsize=(10,6))
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["World","Sports","Business","Sci/Tech"])
    disp.plot(ax=ax,cmap='blues',colorbar=True)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot