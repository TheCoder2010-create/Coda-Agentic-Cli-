#!/usr/bin/env python3

from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Coda - A local-first terminal-based AI coding assistant"

setup(
    name="coda",
    version="0.1.0",
    author="Coda AI Assistant",
    description="A local-first terminal-based AI coding assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "openai>=1.0.0",
        "anthropic>=0.68.0",
        "tiktoken>=0.11.0",
        "gitpython>=3.1.0",
        "pathspec>=0.12.0",
        "watchdog>=6.0.0",
        "requests>=2.32.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "coda=coda.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)