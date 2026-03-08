"""Configuration constants and settings for the topic modeling pipeline.

This module defines project paths, model constants, processing parameters,
and handles GPU/CPU detection for PyTorch operations.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# Project Paths
# =============================================================================

# Base directory is the project root (parent of src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Input/output directories
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
VIZ_DIR = BASE_DIR / "visualizations"

# Input files
TICKETS_FILE = INPUT_DIR / "customer_support_tickets.csv"

# Output files
OUTPUT_TICKETS_FILE = OUTPUT_DIR / "tickets_with_topics.csv"
OUTPUT_INSIGHTS_FILE = OUTPUT_DIR / "topic_insights.csv"
OUTPUT_MODEL_DIR = OUTPUT_DIR / "bertopic_model"

# Visualization files
VIZ_TOPIC_MAP = VIZ_DIR / "topic_map.html"
VIZ_BARCHART = VIZ_DIR / "barchart.html"
VIZ_HEATMAP = VIZ_DIR / "heatmap.html"
VIZ_TOPICS_PER_CLASS = VIZ_DIR / "topics_per_class.html"

# =============================================================================
# Model Constants
# =============================================================================

# Embedding model from sentence-transformers
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# BERTopic configuration
MIN_TOPIC_SIZE = 30

# OpenAI configuration
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_DOCS_PER_TOPIC = 10
OPENAI_DELAY_SECONDS = 2

# =============================================================================
# Processing Constants
# =============================================================================

# Minimum text length for valid tickets
MIN_TEXT_LENGTH = 20

# =============================================================================
# Environment Variables
# =============================================================================

# OpenAI API key (loaded from .env file)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =============================================================================
# Device Detection
# =============================================================================


def get_device() -> str:
    """Detect and return the appropriate device for PyTorch operations.

    Returns:
        Device string: 'cuda' if GPU is available, 'cpu' otherwise.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Current device for model operations
DEVICE = get_device()

# =============================================================================
# OpenAI Prompt Template
# =============================================================================

OPENAI_PROMPT_TEMPLATE = """I have a topic that contains the following customer support tickets:
[DOCUMENTS]

The topic is described by the following keywords: [KEYWORDS]

Based on the information above, provide a structured analysis in the following format:

Main Theme: <one sentence summary of the core issue>

Key Pain Points:
- <bullet points describing customer frustrations>

Common Root Causes:
- <bullet points identifying underlying causes>

Business Recommendations:
- <actionable items to reduce support tickets>
"""


def ensure_directories() -> None:
    """Create output and visualization directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def validate_openai_key() -> bool:
    """Check if OpenAI API key is configured.

    Returns:
        True if API key is set, False otherwise.
    """
    return OPENAI_API_KEY is not None and len(OPENAI_API_KEY) > 0
