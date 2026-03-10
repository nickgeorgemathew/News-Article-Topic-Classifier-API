"""
Tests for the text preprocessor module.
"""

import pytest
from src.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor(tmp_path):
    """Create a preprocessor with a minimal config."""
    config_content = """
preprocessing:
  lowercase: true
  remove_stopwords: true
  lemmatize: true
  min_word_length: 2
  max_features: 10000
  ngram_range: [1, 2]
  max_df: 0.85
  min_df: 2
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return TextPreprocessor(config_path=str(config_file))


class TestCleanText:
    def test_lowercasing(self, preprocessor):
        result = preprocessor.clean_text("Hello WORLD")
        assert result == result.lower()

    def test_url_removal(self, preprocessor):
        result = preprocessor.clean_text("Visit https://example.com today!")
        assert "http" not in result
        assert "example.com" not in result

    def test_email_removal(self, preprocessor):
        result = preprocessor.clean_text("Send mail to user@domain.com please.")
        assert "@" not in result

    def test_html_removal(self, preprocessor):
        result = preprocessor.clean_text("<b>Bold</b> text here")
        assert "<b>" not in result
        assert "Bold" in result

    def test_extra_whitespace(self, preprocessor):
        result = preprocessor.clean_text("  too   many   spaces  ")
        assert "  " not in result
        assert result == result.strip()


class TestTokenize:
    def test_basic_tokenization(self, preprocessor):
        tokens = preprocessor.tokenize("hello world")
        assert isinstance(tokens, list)
        assert len(tokens) >= 2

    def test_punctuation_tokenization(self, preprocessor):
        tokens = preprocessor.tokenize("hello, world!")
        assert any(t in tokens for t in ["hello", "world"])


class TestStopwordRemoval:
    def test_removes_stopwords(self, preprocessor):
        tokens = ["this", "is", "a", "test", "sentence"]
        result = preprocessor.remove_stopwords(tokens)
        assert "this" not in result
        assert "is" not in result
        assert "test" in result

    def test_keeps_content_words(self, preprocessor):
        tokens = ["python", "machine", "learning"]
        result = preprocessor.remove_stopwords(tokens)
        assert result == tokens


class TestPreprocess:
    def test_output_is_string(self, preprocessor):
        result = preprocessor.preprocess("This is a test sentence for classification.")
        assert isinstance(result, str)

    def test_output_not_empty(self, preprocessor):
        result = preprocessor.preprocess("This is a meaningful sentence about AI.")
        assert len(result) > 0

    def test_removes_very_short_words(self, preprocessor):
        result = preprocessor.preprocess("I am a big deal")
        tokens = result.split()
        for t in tokens:
            assert len(t) >= 2

    def test_handles_empty_string(self, preprocessor):
        result = preprocessor.preprocess("")
        assert isinstance(result, str)

    def test_handles_only_stopwords(self, preprocessor):
        result = preprocessor.preprocess("this is the a an")
        # Should return empty or near-empty string
        assert isinstance(result, str)


class TestBatchPreprocess:
    def test_batch_length(self, preprocessor):
        texts = ["First text.", "Second text.", "Third text."]
        results = preprocessor.batch_preprocess(texts, show_progress=False)
        assert len(results) == len(texts)

    def test_batch_all_strings(self, preprocessor):
        texts = ["Hello world", "Machine learning is great"]
        results = preprocessor.batch_preprocess(texts, show_progress=False)
        assert all(isinstance(r, str) for r in results)
