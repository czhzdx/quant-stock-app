"""安装脚本"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quant-stock-selection",
    version="1.0.0",
    author="Quant Team",
    author_email="quant@example.com",
    description="A quantitative stock selection system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quant-stock-selection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "yfinance>=0.2.0",
        "ta-lib>=0.4.0",
        "backtesting>=0.3.3",
        "plotly>=5.0.0",
        "streamlit>=1.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "sqlalchemy>=2.0.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quant-stock=main:main",
        ],
    },
    package_data={
        "src": ["config/*.yaml"],
    },
    include_package_data=True,
    keywords=[
        "quantitative",
        "stock",
        "trading",
        "investment",
        "backtest",
        "strategy",
        "finance",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/quant-stock-selection/issues",
        "Source": "https://github.com/yourusername/quant-stock-selection",
    },
)