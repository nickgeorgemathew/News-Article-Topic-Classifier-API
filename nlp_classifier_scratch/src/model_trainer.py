import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
import joblib
import yaml
from typing import Dict,Tuple,Any
import time




models={}
best_models={}
results={}




tfidvec=TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,2),
            max_df=0.85,
            min_df=2,
            sublinear_tf=True,  # Use log scaling
            use_idf=True
        )

def train_base_model(X_train:pd.Series,y_train:pd.Series)->Dict[str,Any]:
    print("training baseline model(TF-IDF + MultinominalNB)")
    start_time=time.time()
    base_pipeline=Pipeline(
        [
            ('tfidf', tfidvec),
            ('clf', MultinomialNB())
        ]
    )
    base_pipeline.fit(X_train,y_train)
    training_time=time.time()-start_time
    print(f"Baseline training completed in {training_time:.2f} seconds")
    return {
            'model': base_pipeline,
            'training_time': training_time
        }
    
def tune_NB(X_train:pd.Series,y_train:pd.Series)->Dict[str,Any]:
    print("Tuning Naive Bayes")
    nb_pipeline=Pipeline(
        [
            ('tfidf', tfidvec),
            ('clf', MultinomialNB())
        ]
    )
    
    param_grid={'clf__alpha':[0.1, 0.5, 1.0]}
    cv= StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    search=RandomizedSearchCV(
        estimator=nb_pipeline,
        param_distributions=param_grid,
        cv=cv,
        n_iter=10,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        random_state=42
)
    start_time=time.time()
    search.fit(X_train,y_train)
    training_time=time.time()-start_time
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    best_models['naive_bayes'] = search.best_estimator_
    
    return {
        'best_model': search.best_estimator_,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': pd.DataFrame(search.cv_results_),
        'training_time': training_time
    }


def tune_log_reg(X_train:pd.Series,y_train:pd.Series)->Dict[str,Any]:
    print(f"\nTuning Logistic Regression )...")
    lr_pipeline=Pipeline([
         ('tfidf', tfidvec),
         ('lr',LogisticRegression(random_state=42))
       ])
    param_grid={
        'lr__C': [0.1, 1.0, 10.0],
        'lr__max_iter': [1000],
        'lr__solver':['lbfgs']
    }
    cv= StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    search=RandomizedSearchCV(
        estimator=lr_pipeline,
        param_distributions=param_grid,
        cv=cv,
        n_iter=10,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        random_state=42
        )
    start_time = time.time()
    search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    best_models['logistic_regression'] = search.best_estimator_
    
    return {
        'best_model': search.best_estimator_,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': pd.DataFrame(search.cv_results_),
        'training_time': training_time
    }
def save_model(model_name:str,model:Any,path:str=None):
    if path is None:
            path = f"{"./models"}/{model_name}.joblib"
        
    joblib.dump(model, path)
    print(f"Model saved to: {path}")

def load_model(self, path: str) -> Any:
        
        model = joblib.load(path)
        print(f"Model loaded from: {path}")
        return model


if __name__ == "__main__":
    # Demo usage
    
    from preprocessor import batch_preprocess
    
    # Load data
    
    train_df =pd.read_csv("nlp_classifier_scratch/train.csv")
    test_df = pd.read_csv("nlp_classifier_scratch/test.csv")
    
    # Preprocess
    preprocessor = batch_preprocess()
    X_train = train_df['text'].apply(preprocessor)
    y_train = train_df['target']
    
    # Train
    
    
    # Baseline
    baseline_results = train_base_model(X_train, y_train)
    
    # Tune models
    nb_results = tune_NB(X_train, y_train)
    lr_results = tune_log_reg(X_train, y_train)















