from setuptools import setup, find_packages

setup(
    name="prophet_forecaster",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "prophet>=1.1.0",
        "yfinance>=0.2.3",
        "PyYAML>=6.0.0",
        "matplotlib>=3.4.0",
        "plotly>=5.3.0",
        "seaborn>=0.11.0",
        "holidays>=0.13",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.9.0",
        ],
    },
    python_requires=">=3.8",
) 