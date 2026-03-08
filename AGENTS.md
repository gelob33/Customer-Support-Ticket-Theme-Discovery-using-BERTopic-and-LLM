# AGENTS.md - Guidelines for AI Coding Agents

This document provides guidelines for AI coding agents working in the `bert-embedding-llm-insights` repository.

## Project Overview

- **Language**: Python 3.12+
- **Package Manager**: uv (recommended) or pip
- **Project Config**: pyproject.toml

## Build & Environment Commands

```bash
# Create/sync virtual environment
uv sync

# Add a dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>

# Run a Python script
uv run python main.py

# Or activate venv and run directly
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
python main.py
```

## Testing Commands

```bash
# Install pytest (if not already installed)
uv add --dev pytest pytest-cov

# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run a single test file
uv run pytest tests/test_module.py

# Run a single test function
uv run pytest tests/test_module.py::test_function_name

# Run a single test class
uv run pytest tests/test_module.py::TestClassName

# Run a single method in a test class
uv run pytest tests/test_module.py::TestClassName::test_method

# Run tests matching a pattern
uv run pytest -k "pattern"

# Run tests with coverage
uv run pytest --cov=src --cov-report=term-missing
```

## Linting & Formatting Commands

```bash
# Install ruff (if not already installed)
uv add --dev ruff

# Check for linting errors
uv run ruff check .

# Fix auto-fixable linting errors
uv run ruff check --fix .

# Format code
uv run ruff format .

# Check formatting without changing files
uv run ruff format --check .

# Type checking (if mypy is installed)
uv add --dev mypy
uv run mypy src/
```

## Code Style Guidelines

### Imports

- Group imports in this order, separated by blank lines:
  1. Standard library imports
  2. Third-party imports
  3. Local application imports
- Use absolute imports over relative imports
- Sort imports alphabetically within each group
- One import per line for clarity (or use `from x import a, b, c` for related items)

```python
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from src.embeddings import compute_embeddings
from src.utils import load_config
```

### Formatting

- **Line length**: 88 characters (Black/Ruff default)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings
- **Trailing commas**: Use in multi-line collections
- **Blank lines**: 2 between top-level definitions, 1 between methods

### Type Annotations

- Use type hints for all function signatures
- Use `from __future__ import annotations` for forward references
- Prefer built-in generics (`list[str]` over `List[str]`) for Python 3.12+
- Use `|` for union types (`str | None` over `Optional[str]`)

```python
from __future__ import annotations

def process_text(text: str, max_length: int | None = None) -> list[float]:
    """Process text and return embeddings."""
    ...
```

### Naming Conventions

- **Variables/functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`
- **Module-level "dunder" names**: `__all__`, `__version__`

```python
MAX_SEQUENCE_LENGTH = 512

class EmbeddingProcessor:
    def __init__(self, model_name: str) -> None:
        self._model = self._load_model(model_name)
    
    def _load_model(self, name: str) -> BertModel:
        ...
    
    def compute_embeddings(self, texts: list[str]) -> np.ndarray:
        ...
```

### Error Handling

- Use specific exception types over generic `Exception`
- Create custom exceptions for domain-specific errors
- Include context in error messages
- Use `raise ... from` to chain exceptions

```python
class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""
    pass

class ModelLoadError(EmbeddingError):
    """Raised when model loading fails."""
    pass

def load_model(path: str) -> BertModel:
    try:
        return torch.load(path)
    except FileNotFoundError as e:
        raise ModelLoadError(f"Model not found at {path}") from e
```

### Docstrings

- Use Google-style docstrings
- Include docstrings for all public modules, classes, and functions
- Document parameters, return values, and raised exceptions

```python
def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector.
        embedding2: Second embedding vector.
    
    Returns:
        Cosine similarity score between -1 and 1.
    
    Raises:
        ValueError: If embeddings have different dimensions.
    """
    ...
```

## Project Structure

```
bert-embedding-llm-insights/
├── src/                    # Source code
│   ├── __init__.py
│   ├── embeddings/         # Embedding computation
│   ├── models/             # Model definitions
│   └── utils/              # Utility functions
├── tests/                  # Test files
│   ├── conftest.py         # Pytest fixtures
│   ├── test_embeddings.py
│   └── test_models.py
├── main.py                 # Entry point
├── pyproject.toml          # Project configuration
└── README.md               # Documentation
```

## Best Practices for AI Agents

1. **Read before editing**: Always read the file before making changes
2. **Run tests**: After changes, run relevant tests to verify correctness
3. **Follow existing patterns**: Match the style of surrounding code
4. **Keep changes minimal**: Make focused, atomic changes
5. **Add tests**: Include tests for new functionality
6. **Update docs**: Update docstrings and README when adding features
7. **Type safety**: Ensure type hints are correct and complete
8. **Error messages**: Provide helpful, actionable error messages
