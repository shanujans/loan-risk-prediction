from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="loan-risk-predictor",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine learning model for loan default risk prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/loan-risk-prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "notebook": ["jupyter", "ipykernel"],
    },
    entry_points={
        "console_scripts": [
            "loan-predict=src.cli:main",
        ],
    },
)