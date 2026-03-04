import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.pipeline import Pipeline
from typing import List,Optional,Dict


class ModelExplainer:


    def __init__(self, model: Pipeline, class_names: Optional[List[str]] = None,
                 config_path: str = "config.yaml"):
        
        self.model=model
        self.class_names=class_names
        with open(config_path,'r') as f:
            self.config=yaml.safe_load(f)
        self.vectorizer=model.named_steps.get('tfidf')
        self.classifier=model.named_steps.get('clf')

        if  self.vectorizer is None or self.classifier is None:
            raise ValueError("Pipeline must have steps named 'tfidf' and 'clf'.")




    

        