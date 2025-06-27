"""
Setup script for XSS Detection with LSTM Networks.

This script installs the package and its dependencies.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements/requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="xss-detection-lstm",
    version="1.0.0",
    author="XSS Detection Team",
    author_email="your.email@example.com",
    description="XSS Detection with LSTM Networks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/xss-detection-lstm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "xss-train=src.train:main",
            "xss-preprocess=src.data.preprocessor:main",
            "xss-evaluate=src.evaluation.evaluator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "requirements/*.txt"],
    },
    keywords="xss detection lstm neural networks cybersecurity machine learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/xss-detection-lstm/issues",
        "Source": "https://github.com/yourusername/xss-detection-lstm",
        "Documentation": "https://github.com/yourusername/xss-detection-lstm#readme",
    },
) 