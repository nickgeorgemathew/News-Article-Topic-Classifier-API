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
from pathlib import Path
import os
import time



class ModelTrainer:
    """Train and tune text classification models."""
    
    def __init__(self, config_path: str = "config.yaml"):
         # 1. Check if CONFIG_PATH environment variable is set
        env_path = os.getenv("CONFIG_PATH")

        if env_path:
            full_path = Path(env_path)
        else:
            # 2. Default: look in project root
            base_dir = Path(__file__).parent
            project_root = base_dir.parent
            full_path = project_root / config_path

            # 3. Fallback: look inside src/
            if not full_path.exists():
                full_path = base_dir / config_path

        # 4. Final check
        if not full_path.exists():
            raise FileNotFoundError(
                f"Config file not found. Tried: {env_path or project_root/config_path} and {base_dir/config_path}"
            )

        # Load config
        with open(full_path, "r") as f:
            self.config = yaml.safe_load(f)

        
        self.models = {}
        self.best_models = {}
        self.results = {}
    
    def create_tfidf_vectorizer(self) -> TfidfVectorizer:
        """Create TF-IDF vectorizer from config."""
        preproc_config = self.config['preprocessing']
        
        return TfidfVectorizer(
            max_features=preproc_config['max_features'],
            ngram_range=tuple(preproc_config['ngram_range']),
            max_df=preproc_config['max_df'],
            min_df=preproc_config['min_df'],
            sublinear_tf=True,  # Use log scaling
            use_idf=True
        )
    
    def train_baseline(self, X_train: pd.Series, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train a quick baseline model (TF-IDF + Naive Bayes).
        
        Args:
            X_train: Training texts
            y_train: Training labels
            
        Returns:
            Dictionary with model and training time
        """
        print("Training baseline model (TF-IDF + Multinomial NB)...")
        
        start_time = time.time()
        
        # Create pipeline
        baseline_pipeline = Pipeline([
            ('tfidf', self.create_tfidf_vectorizer()),
            ('clf', MultinomialNB())
        ])
        
        # Train
        baseline_pipeline.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        print(f"Baseline training completed in {training_time:.2f} seconds")
        
        self.models['baseline'] = baseline_pipeline
        
        return {
            'model': baseline_pipeline,
            'training_time': training_time
        }
    
    def tune_naive_bayes(self, X_train: pd.Series, y_train: pd.Series,
                        search_type: str = 'grid') -> Dict[str, Any]:
        """
        Tune Naive Bayes with cross-validation.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            search_type: 'grid' or 'random'
            
        Returns:
            Dictionary with best model and results
        """
        print(f"\nTuning Naive Bayes ...")
        
        # Create pipeline
        nb_pipeline = Pipeline([
            ('tfidf', self.create_tfidf_vectorizer()),
            ('clf', MultinomialNB())
        ])
        
        # Parameter grid
        param_grid = {
            'clf__alpha': self.config['models']['naive_bayes']['alpha']
        }
        
        # Cross-validation strategy
        cv = StratifiedKFold(
            n_splits=self.config['training']['cv_folds'],
            shuffle=True,
            random_state=self.config['dataset']['random_state']
        )
        
        # Search
        search = RandomizedSearchCV(
                nb_pipeline,
                param_grid,
                n_iter=10,
                cv=cv,
                scoring=self.config['training']['scoring'],
                n_jobs=self.config['training']['n_jobs'],
                verbose=self.config['training']['verbose'],
                random_state=self.config['dataset']['random_state']
            )
        
        start_time = time.time()
        search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV score: {search.best_score_:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        
        self.best_models['naive_bayes'] = search.best_estimator_
        
        return {
            'best_model': search.best_estimator_,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': pd.DataFrame(search.cv_results_),
            'training_time': training_time
        }
    
    def tune_logistic_regression(self, X_train: pd.Series, y_train: pd.Series,
                                 search_type: str = 'grid') -> Dict[str, Any]:
        """
        Tune Logistic Regression with cross-validation.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            search_type: 'grid' or 'random'
            
        Returns:
            Dictionary with best model and results
        """
        print(f"\nTuning Logistic Regression ({search_type} search)...")
        
        # Create pipeline
        lr_pipeline = Pipeline([
            ('tfidf', self.create_tfidf_vectorizer()),
            ('clf', LogisticRegression(random_state=self.config['dataset']['random_state']))
        ])
        
        # Parameter grid
        param_grid = {
            'clf__C': self.config['models']['logistic_regression']['C'],
            'clf__max_iter': self.config['models']['logistic_regression']['max_iter'],
            'clf__solver': self.config['models']['logistic_regression']['solver']
        }
        
        # Cross-validation strategy
        cv = StratifiedKFold(
            n_splits=self.config['training']['cv_folds'],
            shuffle=True,
            random_state=self.config['dataset']['random_state']
        )
        
        # Search
    
        search = RandomizedSearchCV(
                lr_pipeline,
                param_grid,
                n_iter=10,
                cv=cv,
                scoring=self.config['training']['scoring'],
                n_jobs=self.config['training']['n_jobs'],
                verbose=self.config['training']['verbose'],
                random_state=self.config['dataset']['random_state']
            )
        
        start_time = time.time()
        search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV score: {search.best_score_:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        
        self.best_models['logistic_regression'] = search.best_estimator_
        
        return {
            'best_model': search.best_estimator_,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': pd.DataFrame(search.cv_results_),
            'training_time': training_time
        }
    
    def save_model(self, model_name: str, model: Any, path: str = None):
        """Save model to disk."""
        if path is None:
            path = f"{self.config['paths']['models_dir']}/{model_name}.joblib"
        
        joblib.dump(model, path)
        print(f"Model saved to: {path}")
    
    def load_model(self, path: str) -> Any:
        """Load model from disk."""
        model = joblib.load(path)
        print(f"Model loaded from: {path}")
        return model


if __name__ == "__main__":
    # Demo usage
    
    from preprocessor import preprocess as pre
    
    # Load data
    train_df=pd.read_csv("nlp_classifier_scratch/train.csv")
    test_df=pd.read_csv("nlp_classifier_scratch/test.csv")
    #preprocess
    X_train = (train_df["Title"].astype(str) + " " + train_df["Description"].astype(str)).tolist()
    X_train= [pre(i) for i in X_train]
    y_train = train_df["Class Index"]
    X_train=X_train[:400]
    y_train=y_train[:400]
    
    
    # Train
    trainer = ModelTrainer()
    
    # Baseline
    baseline_results = trainer.train_baseline(X_train, y_train)
    
    # Tune models
    nb_results = trainer.tune_naive_bayes(X_train, y_train)
    lr_results = trainer.tune_logistic_regression(X_train, y_train)





