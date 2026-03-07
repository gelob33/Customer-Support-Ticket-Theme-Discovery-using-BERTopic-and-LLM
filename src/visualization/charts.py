"""Visualization module for BERTopic results.

This module provides functions to generate interactive HTML visualizations
for topic modeling results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from bertopic import BERTopic

from src.config import (
    VIZ_BARCHART,
    VIZ_DIR,
    VIZ_HEATMAP,
    VIZ_TOPIC_MAP,
    VIZ_TOPICS_PER_CLASS,
)

# Configure logging
logger = logging.getLogger(__name__)


def generate_topic_map(
    topic_model: BERTopic,
    output_path: str | Path | None = None,
) -> Path:
    """Generate a 2D topic map showing intertopic distances.

    Args:
        topic_model: Fitted BERTopic model.
        output_path: Output file path. If None, uses default from config.

    Returns:
        Path to the saved HTML file.
    """
    save_path = Path(output_path) if output_path else VIZ_TOPIC_MAP
    logger.info(f"Generating topic map: {save_path}")

    fig = topic_model.visualize_topics()
    fig.write_html(str(save_path))

    logger.info("Topic map saved successfully")
    return save_path


def generate_barchart(
    topic_model: BERTopic,
    output_path: str | Path | None = None,
    top_n_topics: int = 10,
) -> Path:
    """Generate a bar chart showing top words per topic.

    Args:
        topic_model: Fitted BERTopic model.
        output_path: Output file path. If None, uses default from config.
        top_n_topics: Number of top topics to display.

    Returns:
        Path to the saved HTML file.
    """
    save_path = Path(output_path) if output_path else VIZ_BARCHART
    logger.info(f"Generating bar chart: {save_path}")

    fig = topic_model.visualize_barchart(top_n_topics=top_n_topics)
    fig.write_html(str(save_path))

    logger.info("Bar chart saved successfully")
    return save_path


def generate_heatmap(
    topic_model: BERTopic,
    output_path: str | Path | None = None,
) -> Path:
    """Generate a heatmap showing topic similarity matrix.

    Args:
        topic_model: Fitted BERTopic model.
        output_path: Output file path. If None, uses default from config.

    Returns:
        Path to the saved HTML file.
    """
    save_path = Path(output_path) if output_path else VIZ_HEATMAP
    logger.info(f"Generating heatmap: {save_path}")

    fig = topic_model.visualize_heatmap()
    fig.write_html(str(save_path))

    logger.info("Heatmap saved successfully")
    return save_path


def generate_topics_per_class(
    topic_model: BERTopic,
    topics: list[int],
    classes: list[str],
    output_path: str | Path | None = None,
    class_name: str = "Product",
) -> Path:
    """Generate a visualization of topics per class (e.g., product or ticket type).

    Args:
        topic_model: Fitted BERTopic model.
        topics: List of topic assignments.
        classes: List of class labels (e.g., product names).
        output_path: Output file path. If None, uses default from config.
        class_name: Name of the class dimension for the title.

    Returns:
        Path to the saved HTML file.
    """
    save_path = Path(output_path) if output_path else VIZ_TOPICS_PER_CLASS
    logger.info(f"Generating topics per class ({class_name}): {save_path}")

    fig = topic_model.visualize_topics_per_class(
        topics=topics,
        classes=classes,
    )
    fig.write_html(str(save_path))

    logger.info("Topics per class visualization saved successfully")
    return save_path


def save_all_visualizations(
    topic_model: BERTopic,
    df: pd.DataFrame,
    topics: list[int],
    topic_column: str = "topic",
) -> dict[str, Path]:
    """Generate and save all visualizations.

    Args:
        topic_model: Fitted BERTopic model.
        df: DataFrame with topic assignments and metadata.
        topics: List of topic assignments.
        topic_column: Name of the column containing topic IDs.

    Returns:
        Dictionary mapping visualization names to their file paths.
    """
    logger.info("Generating all visualizations...")

    # Ensure visualization directory exists
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # 1. Topic Map - 2D projection of topics
    try:
        saved_files["topic_map"] = generate_topic_map(topic_model)
    except Exception as e:
        logger.error(f"Failed to generate topic map: {e}")

    # 2. Bar Chart - Top words per topic
    try:
        saved_files["barchart"] = generate_barchart(topic_model)
    except Exception as e:
        logger.error(f"Failed to generate bar chart: {e}")

    # 3. Heatmap - Topic similarity
    try:
        saved_files["heatmap"] = generate_heatmap(topic_model)
    except Exception as e:
        logger.error(f"Failed to generate heatmap: {e}")

    # 4. Topics per Class - Distribution by Product
    try:
        classes = df["Product Purchased"].tolist()
        saved_files["topics_per_class"] = generate_topics_per_class(
            topic_model,
            topics=topics,
            classes=classes,
            class_name="Product Purchased",
        )
    except Exception as e:
        logger.error(f"Failed to generate topics per class: {e}")

    logger.info(f"Generated {len(saved_files)} visualizations")

    # Print summary of saved files
    print("\n" + "-" * 40)
    print("VISUALIZATIONS SAVED")
    print("-" * 40)
    for name, path in saved_files.items():
        print(f"  {name}: {path}")

    return saved_files
