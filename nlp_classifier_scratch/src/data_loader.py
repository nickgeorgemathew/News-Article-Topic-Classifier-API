import os
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import yaml


class DataLoader:
    """Load and prepare datasets for text classification."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize DataLoader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_name = self.config['dataset']['name']
        self.test_size = self.config['dataset']['test_size']
        self.random_state = self.config['dataset']['random_state']
    
    def load_ag_news(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load AG News dataset."""
        print("Loading AG News dataset...")
        
        # Load from HuggingFace datasets
        dataset = load_dataset("ag_news")
        
        # Convert to pandas
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        
        # Rename columns for consistency
        train_df = train_df.rename(columns={'text': 'text', 'label': 'target'})
        test_df = test_df.rename(columns={'text': 'text', 'label': 'target'})
        
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        print(f"Classes: {train_df['target'].nunique()}")
        print(f"Class distribution:\n{train_df['target'].value_counts()}")
        
        return train_df, test_df
    
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load dataset based on configuration."""
        if self.dataset_name == 'ag_news':
            return self.load_ag_news()
        
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def get_sample_texts(self, n: int = 5) -> Dict:
        """Get sample texts for each class."""
        train_df, _ = self.load_data()
        
        samples = {}
        for label in train_df['target'].unique():
            samples[label] = train_df[train_df['target'] == label]['text'].head(n).tolist()
        
        return samples


if __name__ == "__main__":
    # Demo usage
    loader = DataLoader()
    train_df, test_df = loader.load_data()
    
    print("\n" + "="*50)
    print("SAMPLE TEXTS")
    print("="*50)
    
    for i in range(3):
        print(f"\nClass {train_df.iloc[i]['target']}:")
        print(f"{train_df.iloc[i]['text'][:200]}...")
