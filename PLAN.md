# Implementation Plan: Customer Support Ticket Theme Discovery

This document outlines the step-by-step implementation plan for building a semantic clustering and LLM analysis pipeline for customer support tickets using BERTopic and OpenAI.

---

## Project Overview

| Attribute | Value |
|-----------|-------|
| Dataset | 8,469 customer support tickets (17 columns) |
| Primary Goal | Identify common customer issues, themes, and pain points |
| Core Tech | BERTopic + sentence-transformers + OpenAI |
| Output | Clustered dataset, topic insights, visualizations |

---

## Project Structure

```
bert-embedding-llm-insights/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Constants, paths, model settings
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py             # Data loading & preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── topic_model.py        # BERTopic configuration & training
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── insights.py           # Cross-tabulations & queries
│   └── visualization/
│       ├── __init__.py
│       └── charts.py             # HTML visualization generation
├── main.py                       # Pipeline orchestration
├── input/
│   └── customer_support_tickets.csv
├── output/
│   ├── tickets_with_topics.csv
│   ├── topic_insights.csv
│   └── bertopic_model/
└── visualizations/
    ├── topic_map.html
    ├── barchart.html
    ├── heatmap.html
    └── topics_per_class.html
```

---

## Implementation Phases

### Phase 1: Configuration Module

**File:** `src/config.py`

**Tasks:**
- [ ] Define project paths (INPUT_DIR, OUTPUT_DIR, VIZ_DIR)
- [ ] Define model constants (EMBEDDING_MODEL, MIN_TOPIC_SIZE, OPENAI_MODEL)
- [ ] Define processing constants (MIN_TEXT_LENGTH, OPENAI_DELAY)
- [ ] Load environment variables (OPENAI_API_KEY)
- [ ] Implement GPU/CPU detection for torch

**Constants:**
```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_TOPIC_SIZE = 30
MIN_TEXT_LENGTH = 20
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_DOCS_PER_TOPIC = 10
OPENAI_DELAY_SECONDS = 2
```

---

### Phase 2: Data Loading Module

**File:** `src/data/loader.py`

**Tasks:**
- [ ] Implement `load_tickets()` function to read CSV
- [ ] Implement `preprocess_tickets()` function:
  - Combine `Ticket Subject` + `Ticket Description` → `question` field
  - Filter tickets where `question` < 20 characters
  - Remove duplicate `question` entries
  - Retain all metadata columns for downstream analysis
- [ ] Return cleaned DataFrame with shape info logged

**Input Columns (from dataset):**
- `Ticket ID`, `Customer Name`, `Customer Email`, `Customer Age`, `Customer Gender`
- `Product Purchased`, `Date of Purchase`, `Ticket Type`, `Ticket Subject`
- `Ticket Description`, `Ticket Status`, `Resolution`, `Ticket Priority`
- `Ticket Channel`, `First Response Time`, `Time to Resolution`, `Customer Satisfaction Rating`

**Output Columns Added:**
- `question` — Combined text field for topic modeling

---

### Phase 3: Topic Modeling Module

**File:** `src/models/topic_model.py`

**Tasks:**
- [ ] Implement `create_embedding_model()`:
  - Load `all-MiniLM-L6-v2` from sentence-transformers
  - Auto-detect GPU/CPU and set device accordingly
- [ ] Implement `create_representation_models()`:
  - KeyBERTInspired for keyword extraction (no API cost)
  - OpenAI representation with custom business insights prompt
- [ ] Implement `create_topic_model()`:
  - Configure BERTopic with min_topic_size=30
  - Attach embedding model and representation models
- [ ] Implement `fit_and_reduce_outliers()`:
  - Fit model on documents
  - Reduce outliers (reassign -1 topics to nearest cluster)
  - Return topics and probabilities
- [ ] Implement `save_model()`:
  - Save trained BERTopic model to `output/bertopic_model/`
- [ ] Implement `extract_topic_insights()`:
  - Extract OpenAI-generated insights per topic
  - Save to `output/topic_insights.csv`

**OpenAI Prompt Template:**
```
Based on the following customer support tickets, provide a structured analysis:

Main Theme: <one sentence summary of the core issue>

Key Pain Points:
- <bullet points describing customer frustrations>

Common Root Causes:
- <bullet points identifying underlying causes>

Business Recommendations:
- <actionable items to reduce support tickets>
```

---

### Phase 4: Analysis Module

**File:** `src/analysis/insights.py`

**Tasks:**
- [ ] Implement `get_topic_frequency()`:
  - Count tickets per topic
  - Return sorted DataFrame
- [ ] Implement `get_topics_by_product()`:
  - Group by topic + `Product Purchased`
  - Identify products causing most issues
- [ ] Implement `get_topics_by_type()`:
  - Crosstab of topics vs `Ticket Type`
- [ ] Implement `get_topics_by_channel()`:
  - Distribution of topics across `Ticket Channel`
- [ ] Implement `print_analysis_summary()`:
  - Print key findings to console

---

### Phase 5: Visualization Module

**File:** `src/visualization/charts.py`

**Essential Visualizations (4 of 6):**

| # | Visualization | Method | Output File |
|---|---------------|--------|-------------|
| 1 | Topic Map | `visualize_topics()` | `topic_map.html` |
| 2 | Bar Chart | `visualize_barchart()` | `barchart.html` |
| 3 | Heatmap | `visualize_heatmap()` | `heatmap.html` |
| 4 | Topics per Class | `visualize_topics_per_class()` | `topics_per_class.html` |

**Tasks:**
- [ ] Implement `generate_topic_map()`:
  - 2D projection showing intertopic distances
- [ ] Implement `generate_barchart()`:
  - Top words per topic visualization
- [ ] Implement `generate_heatmap()`:
  - Topic similarity matrix
- [ ] Implement `generate_topics_per_class()`:
  - Distribution across `Product Purchased` and `Ticket Type`
- [ ] Implement `save_all_visualizations()`:
  - Generate all charts and save to `visualizations/` directory

---

### Phase 6: Main Pipeline

**File:** `main.py`

**Tasks:**
- [ ] Implement main orchestration function
- [ ] Add logging throughout pipeline
- [ ] Add timing metrics for each phase
- [ ] Handle errors gracefully with informative messages

**Pipeline Steps:**
```python
def main():
    # 1. Load and preprocess data
    df = load_and_preprocess_tickets()
    
    # 2. Create and train topic model
    topic_model = create_and_fit_model(df["question"].tolist())
    
    # 3. Reduce outliers and get final topics
    df["topic"] = get_topics_with_outlier_reduction(topic_model, df)
    
    # 4. Extract and save LLM insights
    save_topic_insights(topic_model)
    
    # 5. Run analysis queries
    run_analysis(df, topic_model)
    
    # 6. Generate visualizations
    generate_all_visualizations(topic_model, df)
    
    # 7. Save final dataset
    save_clustered_dataset(df)
```

---

## File Creation Order

| Order | File | Purpose |
|-------|------|---------|
| 1 | `src/__init__.py` | Package marker |
| 2 | `src/config.py` | Configuration constants |
| 3 | `src/data/__init__.py` | Package marker |
| 4 | `src/data/loader.py` | Data loading functions |
| 5 | `src/models/__init__.py` | Package marker |
| 6 | `src/models/topic_model.py` | BERTopic + OpenAI integration |
| 7 | `src/analysis/__init__.py` | Package marker |
| 8 | `src/analysis/insights.py` | Analysis functions |
| 9 | `src/visualization/__init__.py` | Package marker |
| 10 | `src/visualization/charts.py` | Visualization generation |
| 11 | `main.py` | Update with full pipeline |
| 12 | `.env` | OpenAI API key (user-provided) |

---

## Dependencies

Already installed in `pyproject.toml`:
- `bertopic>=0.17.4`
- `sentence-transformers>=5.2.3`
- `torch>=2.10.0`
- `openai>=2.26.0`
- `pandas>=3.0.1`
- `numpy>=2.4.2`
- `python-dotenv>=1.2.2`

---

## Environment Setup

**Required:** Create `.env` file in project root:
```
OPENAI_API_KEY=sk-your-api-key-here
```

**Directories to create:**
- `visualizations/` (for HTML output)

---

## Expected Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Clustered dataset | `output/tickets_with_topics.csv` | Original data + topic assignments |
| Topic insights | `output/topic_insights.csv` | LLM-generated insights per topic |
| Saved model | `output/bertopic_model/` | Serialized BERTopic model |
| Topic map | `visualizations/topic_map.html` | Interactive 2D topic projection |
| Bar chart | `visualizations/barchart.html` | Top words per topic |
| Heatmap | `visualizations/heatmap.html` | Topic similarity matrix |
| Topics per class | `visualizations/topics_per_class.html` | Distribution by product/type |

---

## Estimated Topic Themes

Based on the dataset structure, expected themes include:
- Login / authentication failures
- Payment / billing confusion
- Product activation problems
- Software crashes or bugs
- Shipping or delivery issues
- Peripheral compatibility issues
- Network connectivity problems

---

## Success Criteria

1. All 8,469 tickets processed and assigned to topics
2. Outlier topics (-1) reduced to < 5% of total
3. LLM insights generated for all discovered topics
4. 4 visualizations saved as interactive HTML
5. Analysis queries executed and findings printed
6. Full pipeline completes without errors

---

## Notes

- GPU will be used if available, with automatic fallback to CPU
- OpenAI calls include 2-second delay to avoid rate limiting
- Model is saved for future reuse without retraining
- All visualizations are interactive HTML (can open in browser)
