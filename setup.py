# measure_hallucination/setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hallucination-measure",
    version="1.0.0",
    author="Tsegay Araya",
    author_email="",
    description="Comprehensive hallucination measurement and mitigation toolkit for LLM/RAG systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tgy-12/Neurostack-RAG-Copilot/tree/main/measure_hallucination",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "sentence-transformers>=2.2.0",
        "rouge-score>=0.1.2",
        "bert-score>=0.3.11",
        "nltk>=3.7",
        "pandas>=1.4.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "rank_bm25>=0.2.1",
        "datasets>=2.14.0",
        "plotly>=5.17.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hallucination-analyze=measure_hallucination.cli:main",
        ],
    },
)
