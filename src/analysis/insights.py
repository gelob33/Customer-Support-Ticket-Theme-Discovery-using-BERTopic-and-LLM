"""Analysis and insights module for customer support ticket topics.

This module provides functions to analyze the distribution of topics
across various dimensions like product, ticket type, and channel.
"""

from __future__ import annotations

import logging

import pandas as pd
from bertopic import BERTopic

# Configure logging
logger = logging.getLogger(__name__)


def get_topic_frequency(df: pd.DataFrame, topic_column: str = "topic") -> pd.DataFrame:
    """Get the frequency of each topic.

    Args:
        df: DataFrame with topic assignments.
        topic_column: Name of the column containing topic IDs.

    Returns:
        DataFrame with topic counts, sorted by frequency descending.
    """
    logger.info("Calculating topic frequencies...")

    freq = (
        df[topic_column]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "topic", topic_column: "count"})
    )
    freq.columns = ["topic", "count"]
    freq["percentage"] = (freq["count"] / freq["count"].sum() * 100).round(2)

    logger.info(f"Found {len(freq)} unique topics")
    return freq


def get_topics_by_product(
    df: pd.DataFrame,
    topic_column: str = "topic",
    product_column: str = "Product Purchased",
) -> pd.DataFrame:
    """Get topic distribution by product.

    Args:
        df: DataFrame with topic assignments and product info.
        topic_column: Name of the column containing topic IDs.
        product_column: Name of the column containing product names.

    Returns:
        DataFrame with topic counts per product.
    """
    logger.info("Analyzing topics by product...")

    grouped = (
        df.groupby([topic_column, product_column])
        .size()
        .reset_index(name="count")
        .sort_values(["count"], ascending=False)
    )

    logger.info(f"Analyzed {len(grouped)} topic-product combinations")
    return grouped


def get_topics_by_type(
    df: pd.DataFrame,
    topic_column: str = "topic",
    type_column: str = "Ticket Type",
) -> pd.DataFrame:
    """Get crosstab of topics vs ticket type.

    Args:
        df: DataFrame with topic assignments and ticket type.
        topic_column: Name of the column containing topic IDs.
        type_column: Name of the column containing ticket types.

    Returns:
        Crosstab DataFrame of topics vs ticket types.
    """
    logger.info("Creating topic vs ticket type crosstab...")

    crosstab = pd.crosstab(df[topic_column], df[type_column])

    logger.info(
        f"Created crosstab with {len(crosstab)} topics and {len(crosstab.columns)} types"
    )
    return crosstab


def get_topics_by_channel(
    df: pd.DataFrame,
    topic_column: str = "topic",
    channel_column: str = "Ticket Channel",
) -> pd.DataFrame:
    """Get topic distribution by ticket channel.

    Args:
        df: DataFrame with topic assignments and channel info.
        topic_column: Name of the column containing topic IDs.
        channel_column: Name of the column containing channel names.

    Returns:
        Crosstab DataFrame of topics vs channels.
    """
    logger.info("Analyzing topics by channel...")

    crosstab = pd.crosstab(df[topic_column], df[channel_column])

    logger.info(
        f"Created crosstab with {len(crosstab)} topics and {len(crosstab.columns)} channels"
    )
    return crosstab


def get_top_products_per_topic(
    df: pd.DataFrame,
    topic_column: str = "topic",
    product_column: str = "Product Purchased",
    top_n: int = 3,
) -> pd.DataFrame:
    """Get the top N products for each topic.

    Args:
        df: DataFrame with topic assignments and product info.
        topic_column: Name of the column containing topic IDs.
        product_column: Name of the column containing product names.
        top_n: Number of top products to return per topic.

    Returns:
        DataFrame with top products per topic.
    """
    logger.info(f"Finding top {top_n} products per topic...")

    result = (
        df.groupby([topic_column, product_column])
        .size()
        .reset_index(name="count")
        .sort_values([topic_column, "count"], ascending=[True, False])
        .groupby(topic_column)
        .head(top_n)
    )

    return result


def print_analysis_summary(
    df: pd.DataFrame,
    topic_model: BERTopic,
    topic_column: str = "topic",
) -> None:
    """Print a summary of the analysis to the console.

    Args:
        df: DataFrame with topic assignments.
        topic_model: Fitted BERTopic model.
        topic_column: Name of the column containing topic IDs.
    """
    print("\n" + "=" * 60)
    print("TOPIC ANALYSIS SUMMARY")
    print("=" * 60)

    # Basic stats
    n_tickets = len(df)
    n_topics = len(topic_model.get_topic_info()) - 1  # Exclude outlier topic
    outliers = (df[topic_column] == -1).sum()
    outlier_pct = outliers / n_tickets * 100

    print(f"\nTotal tickets analyzed: {n_tickets:,}")
    print(f"Number of topics discovered: {n_topics}")
    print(f"Outliers remaining: {outliers:,} ({outlier_pct:.1f}%)")

    # Topic frequency
    print("\n" + "-" * 40)
    print("TOP 10 TOPICS BY FREQUENCY")
    print("-" * 40)

    topic_freq = get_topic_frequency(df, topic_column)
    topic_info = topic_model.get_topic_info()

    for i, row in topic_freq.head(10).iterrows():
        topic_id = row["topic"]
        count = row["count"]
        pct = row["percentage"]

        # Get topic name/keywords
        topic_row = topic_info[topic_info["Topic"] == topic_id]
        if not topic_row.empty and "Name" in topic_row.columns:
            name = topic_row["Name"].values[0]
        else:
            name = f"Topic {topic_id}"

        print(f"  {int(topic_id):3d}: {count:5,} tickets ({pct:5.1f}%) - {name[:50]}")

    # Top products with issues
    print("\n" + "-" * 40)
    print("TOP 10 PRODUCTS BY TICKET COUNT")
    print("-" * 40)

    product_counts = df["Product Purchased"].value_counts().head(10)
    for product, count in product_counts.items():
        pct = count / n_tickets * 100
        print(f"  {product}: {count:,} tickets ({pct:.1f}%)")

    # Ticket type distribution
    print("\n" + "-" * 40)
    print("TICKET TYPE DISTRIBUTION")
    print("-" * 40)

    type_counts = df["Ticket Type"].value_counts()
    for ticket_type, count in type_counts.items():
        pct = count / n_tickets * 100
        print(f"  {ticket_type}: {count:,} tickets ({pct:.1f}%)")

    # Channel distribution
    print("\n" + "-" * 40)
    print("CHANNEL DISTRIBUTION")
    print("-" * 40)

    channel_counts = df["Ticket Channel"].value_counts()
    for channel, count in channel_counts.items():
        pct = count / n_tickets * 100
        print(f"  {channel}: {count:,} tickets ({pct:.1f}%)")

    print("\n" + "=" * 60)


def run_analysis(
    df: pd.DataFrame,
    topic_model: BERTopic,
    topic_column: str = "topic",
) -> dict:
    """Run all analysis functions and return results.

    Args:
        df: DataFrame with topic assignments.
        topic_model: Fitted BERTopic model.
        topic_column: Name of the column containing topic IDs.

    Returns:
        Dictionary containing all analysis results.
    """
    logger.info("Running comprehensive analysis...")

    results = {
        "topic_frequency": get_topic_frequency(df, topic_column),
        "topics_by_product": get_topics_by_product(df, topic_column),
        "topics_by_type": get_topics_by_type(df, topic_column),
        "topics_by_channel": get_topics_by_channel(df, topic_column),
        "top_products_per_topic": get_top_products_per_topic(df, topic_column),
    }

    # Print summary to console
    print_analysis_summary(df, topic_model, topic_column)

    logger.info("Analysis complete")
    return results
