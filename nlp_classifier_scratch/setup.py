from setuptools import setup, find_packages

setup(
    name="text-classification-project",
    version="1.0.0",
    description="End-to-end text classification with TF-IDF and sklearn",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn>=1.3",
        "numpy>=1.24",
        "pandas>=2.0",
        "nltk>=3.8",
        "datasets>=2.14",
        "joblib>=1.3",
        "tqdm>=4.66",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "streamlit>=1.28",
        "fastapi>=0.104",
        "uvicorn>=0.24",
        "pydantic>=2.4",
        "pyyaml>=6.0",
        "plotly>=5.18",
    ],
    extras_require={
        "spacy": ["spacy>=3.7"],
        "lime": ["lime>=0.2"],
        "dev": ["pytest>=7.4", "pytest-cov>=4.1", "jupyter>=1.0"],
    },
)
