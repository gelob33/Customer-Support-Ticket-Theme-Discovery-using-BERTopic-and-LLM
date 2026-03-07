"""Main pipeline for Customer Support Ticket Theme Discovery.

This script orchestrates the complete pipeline:
1. Load and preprocess ticket data
2. Create and train BERTopic model
3. Reduce outliers for better topic assignments
4. Extract and save LLM-powered insights
5. Run analysis queries
6. Generate visualizations
7. Save the clustered dataset
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime

from src.analysis.insights import run_analysis
from src.config import (
    OUTPUT_MODEL_DIR,
    OUTPUT_TICKETS_FILE,
    ensure_directories,
    validate_openai_key,
)
from src.data.loader import load_and_preprocess_tickets
from src.models.topic_model import (
    create_topic_model,
    fit_and_reduce_outliers,
    save_model,
    save_topic_insights,
)
from src.visualization.charts import save_all_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the complete topic modeling pipeline."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("CUSTOMER SUPPORT TICKET THEME DISCOVERY PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Ensure output directories exist
    ensure_directories()

    # Check OpenAI API key
    if validate_openai_key():
        logger.info("OpenAI API key found - LLM insights will be generated")
    else:
        logger.warning(
            "OpenAI API key not found - LLM insights will be skipped. "
            "Set OPENAI_API_KEY in your .env file to enable this feature."
        )

    # =========================================================================
    # Step 1: Load and preprocess data
    # =========================================================================
    logger.info("-" * 40)
    logger.info("STEP 1: Loading and preprocessing data")
    logger.info("-" * 40)

    step_start = time.time()
    df = load_and_preprocess_tickets()
    documents = df["question"].tolist()
    logger.info(f"Step 1 completed in {time.time() - step_start:.2f}s")

    # =========================================================================
    # Step 2: Create topic model
    # =========================================================================
    logger.info("-" * 40)
    logger.info("STEP 2: Creating topic model")
    logger.info("-" * 40)

    step_start = time.time()
    topic_model = create_topic_model()
    logger.info(f"Step 2 completed in {time.time() - step_start:.2f}s")

    # =========================================================================
    # Step 3: Fit model and reduce outliers
    # =========================================================================
    logger.info("-" * 40)
    logger.info("STEP 3: Fitting model and reducing outliers")
    logger.info("-" * 40)

    step_start = time.time()
    topic_model, topics = fit_and_reduce_outliers(topic_model, documents)
    df["topic"] = topics
    logger.info(f"Step 3 completed in {time.time() - step_start:.2f}s")

    # =========================================================================
    # Step 4: Save model
    # =========================================================================
    logger.info("-" * 40)
    logger.info("STEP 4: Saving trained model")
    logger.info("-" * 40)

    step_start = time.time()
    save_model(topic_model)
    logger.info(f"Model saved to: {OUTPUT_MODEL_DIR}")
    logger.info(f"Step 4 completed in {time.time() - step_start:.2f}s")

    # =========================================================================
    # Step 5: Extract and save topic insights
    # =========================================================================
    logger.info("-" * 40)
    logger.info("STEP 5: Extracting topic insights")
    logger.info("-" * 40)

    step_start = time.time()
    insights = save_topic_insights(topic_model)
    logger.info(f"Extracted insights for {len(insights)} topics")
    logger.info(f"Step 5 completed in {time.time() - step_start:.2f}s")

    # =========================================================================
    # Step 6: Run analysis
    # =========================================================================
    logger.info("-" * 40)
    logger.info("STEP 6: Running analysis")
    logger.info("-" * 40)

    step_start = time.time()
    run_analysis(df, topic_model)
    logger.info(f"Step 6 completed in {time.time() - step_start:.2f}s")

    # =========================================================================
    # Step 7: Generate visualizations
    # =========================================================================
    logger.info("-" * 40)
    logger.info("STEP 7: Generating visualizations")
    logger.info("-" * 40)

    step_start = time.time()
    save_all_visualizations(topic_model, df, topics)
    logger.info(f"Step 7 completed in {time.time() - step_start:.2f}s")

    # =========================================================================
    # Step 8: Save clustered dataset
    # =========================================================================
    logger.info("-" * 40)
    logger.info("STEP 8: Saving clustered dataset")
    logger.info("-" * 40)

    step_start = time.time()
    df.to_csv(OUTPUT_TICKETS_FILE, index=False)
    logger.info(f"Saved clustered dataset to: {OUTPUT_TICKETS_FILE}")
    logger.info(f"Step 8 completed in {time.time() - step_start:.2f}s")

    # =========================================================================
    # Pipeline complete
    # =========================================================================
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total execution time: {total_time:.2f}s ({total_time / 60:.1f} min)")
    logger.info(f"Tickets processed: {len(df):,}")
    logger.info(f"Topics discovered: {len(topic_model.get_topic_info()) - 1}")
    logger.info(f"Outliers: {(df['topic'] == -1).sum():,}")

    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"  Clustered dataset: {OUTPUT_TICKETS_FILE}")
    print(f"  Topic insights: {OUTPUT_TICKETS_FILE.parent / 'topic_insights.csv'}")
    print(f"  Saved model: {OUTPUT_MODEL_DIR}")
    print(f"  Visualizations: {OUTPUT_TICKETS_FILE.parent.parent / 'visualizations'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)
