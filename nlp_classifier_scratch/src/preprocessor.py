"""
Text preprocessing pipeline for classification tasks.
Includes tokenization, stopword removal, and lemmatization.
"""
import os as o
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pathlib import Path
import spacy
from typing import List, Union
import yaml


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """Comprehensive text preprocessing for NLP tasks."""
    
    def __init__(self, config_path: str = "config.yaml", use_spacy: bool = False):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            use_spacy: Whether to use spaCy for lemmatization (more accurate but slower)
        """
        # 1. Check if CONFIG_PATH environment variable is set
        env_path = o.getenv("CONFIG_PATH")

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
            self.config = yaml.safe_load(f)['preprocessing']
        
        
        self.use_spacy = use_spacy
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize spaCy if requested
        if self.use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("Downloading spaCy model...")
                import os
                os.system('python -m spacy download en_core_web_sm')
                self.nlp = spacy.load('en_core_web_sm')
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        if self.config['lowercase']:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def remove_punctuation(self, tokens: List[str]) -> List[str]:
        """Remove punctuation from tokens."""
        return [token for token in tokens if token not in string.punctuation]
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens."""
        if not self.config['remove_stopwords']:
            return tokens
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_nltk(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens using NLTK."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def lemmatize_spacy(self, text: str) -> List[str]:
        """Lemmatize text using spaCy."""
        doc = self.nlp(text)
        return [token.lemma_ for token in doc]
    
    def filter_short_words(self, tokens: List[str]) -> List[str]:
        """Remove very short words."""
        min_len = self.config['min_word_length']
        return [token for token in tokens if len(token) >= min_len]
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text as a single string
        """
        # Clean text
        text = self.clean_text(text)
        
        if self.use_spacy and self.config['lemmatize']:
            # Use spaCy pipeline
            tokens = self.lemmatize_spacy(text)
            tokens = self.remove_punctuation(tokens)
            tokens = self.remove_stopwords(tokens)
            tokens = self.filter_short_words(tokens)
        else:
            # Use NLTK pipeline
            tokens = self.tokenize(text)
            tokens = self.remove_punctuation(tokens)
            tokens = self.remove_stopwords(tokens)
            
            if self.config['lemmatize']:
                tokens = self.lemmatize_nltk(tokens)
            
            tokens = self.filter_short_words(tokens)
        
        # Join tokens back into string
        return ' '.join(tokens)
    
    def batch_preprocess(self, texts: List[str], show_progress: bool = True) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            List of preprocessed texts
        """
        if show_progress:
            from tqdm import tqdm
            return [self.preprocess(text) for text in tqdm(texts, desc="Preprocessing")]
        else:
            return [self.preprocess(text) for text in texts]


if __name__ == "__main__":
    # Demo usage
    preprocessor = TextPreprocessor()
    
    sample_text = """
    Hello! This is a SAMPLE text with URLs like https://example.com 
    and emails like test@email.com. It has punctuation!!! And stopwords.
    We're testing the preprocessing pipeline here.
    """
    
    print("Original text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    processed = preprocessor.preprocess(sample_text)
    print("Processed text:")
    print(processed)















  