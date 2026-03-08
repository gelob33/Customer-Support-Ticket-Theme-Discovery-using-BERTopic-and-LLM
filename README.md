# Customer Support Ticket Theme Discovery using BERTopic and LLM

Automated discovery of customer issues, pain points, and business improvement opportunities from support ticket data using semantic clustering (BERTopic) and AI-powered insights (OpenAI GPT).

---

## Purpose

This project solves a critical business challenge: **understanding what customers actually need help with** by analyzing thousands of support tickets at scale.

### Problems Solved

- **Manual analysis doesn't scale**: Reviewing 8,000+ tickets manually is impractical
- **Hidden patterns**: Common themes and pain points buried in unstructured text
- **Actionable insights**: Identify which products, features, or processes need improvement
- **Resource optimization**: Understand where to focus support and development efforts

### Key Questions Answered

1. What are the **most common customer problems**?
2. Which **products or features** generate the most support requests?
3. What **business improvements** could reduce support load and improve satisfaction?
4. How do issues vary by **ticket type, priority, and channel**?

---

## Tech Stack

| Component           | Technology                           | Purpose                               |
| ------------------- | ------------------------------------ | ------------------------------------- |
| Topic Modeling      | BERTopic                             | Semantic clustering of tickets        |
| Embeddings          | sentence-transformers (all-MiniLM-L6-v2) | Convert text to vector representations |
| LLM Analysis        | OpenAI GPT-4o-mini                   | Generate business insights per topic  |
| Data Processing     | pandas, numpy                        | Data manipulation and analysis        |
| Visualization       | BERTopic + plotly                    | Interactive HTML charts               |

---

## Workflow

The pipeline consists of four main phases:

### Phase 1: Data Preparation

**Purpose**: Clean and prepare ticket data for analysis

**Steps**:
1. Load customer support tickets from CSV (`input/customer_support_tickets.csv`)
2. Combine `Ticket Subject` + `Ticket Description` into a single `question` field
3. Filter out empty or very short tickets (< 20 characters)
4. Remove duplicate tickets
5. Retain all metadata columns (ticket type, product, priority, channel, etc.)

**Output**: Clean DataFrame ready for topic modeling

---

### Phase 2: Topic Modeling with BERTopic

**Purpose**: Automatically discover semantic themes and group similar tickets together

**Configuration**:
- **Embedding model**: `all-MiniLM-L6-v2` (efficient semantic embeddings)
- **Minimum topic size**: 30 tickets per cluster
- **Outlier reduction**: Enabled (reassigns outlier tickets to nearest topics)

**What it does**:
- Converts each ticket into a semantic vector representation
- Uses UMAP for dimensionality reduction
- Applies HDBSCAN clustering to group similar tickets
- Generates topic representations using c-TF-IDF
- Creates multiple representation types:
  - **KeyBERTInspired**: Fast keyword extraction (no API costs)
  - **OpenAI**: Structured business insights via custom prompts

**Output**: Topic assignments for each ticket + topic models

---

### Phase 3: LLM-Powered Insights Generation

**Purpose**: Generate human-readable business insights for each discovered topic

**Process**:
- For each topic, send representative tickets + keywords to OpenAI
- Use structured prompt to extract:
  - **Main Theme**: One-sentence summary of the topic
  - **Key Pain Points**: Specific customer frustrations
  - **Common Patterns**: Frequency, product involvement, sentiment
  - **Business Impact**: Severity, affected customers
  - **Recommendations**: Actionable improvements to reduce issues

**Output**: Structured insights for every topic cluster

---

### Phase 4: Visualization Generation

**Purpose**: Create interactive visualizations for exploring discovered themes

**Generated Visualizations**:

1. **Topic Map** (`topic_map.html`)
   - Interactive 2D scatter plot of all topics
   - Shows semantic relationships between themes
   - Hover to see topic keywords and document count

2. **Topic Bar Chart** (`barchart.html`)
   - Frequency distribution of top topics
   - Identifies most common support issues
   - Sortable and filterable

3. **Topic Heatmap** (`heatmap.html`)
   - Cross-tabulation of topics vs. metadata
   - Shows which products/ticket types correlate with topics
   - Helps identify patterns (e.g., "Product X → Login Issues")

**Output**: Interactive HTML visualizations for exploration and reporting

---

## Output Layout

All outputs are organized into two directories:

### `output/` Directory

Contains structured data files:

```
output/
├── tickets_with_topics.csv       # Original tickets + topic assignments
├── topic_insights.csv            # LLM-generated insights per topic
└── bertopic_model/               # Saved BERTopic model
    ├── config.json
    ├── topics.json
    ├── topic_embeddings.safetensors
    └── ... (model artifacts)
```

#### `tickets_with_topics.csv`

Enhanced ticket dataset with topic assignments.

**Key Columns**:
- All original columns (Ticket ID, Customer Name, Product, etc.)
- `question`: Combined ticket subject + description
- `Topic`: Assigned topic cluster ID (-1 for outliers)
- `Topic_Name`: Human-readable topic label
- `Topic_Probability`: Confidence score for assignment

**Use Cases**: 
- Filter tickets by topic
- Analyze topic distribution by product/channel/priority
- Export specific topics for team review

---

#### `topic_insights.csv`

LLM-generated business insights for each topic.

**Columns**:
- `Topic`: Topic ID
- `Count`: Number of tickets in topic
- `Name`: Auto-generated topic name
- `Representation`: Top keywords
- `Main_Theme`: One-sentence summary
- `Key_Pain_Points`: Bullet list of customer frustrations
- `Common_Patterns`: Usage frequency, products, sentiment
- `Business_Impact`: Severity and affected customer count
- `Recommendations`: Actionable improvements

**Use Cases**:
- Executive summaries of top issues
- Prioritization for product/engineering teams
- Support team training materials

---

#### `bertopic_model/`

Saved BERTopic model artifacts for reuse and updates.

**Use Cases**:
- Predict topics for new incoming tickets
- Update model with additional data
- Reproduce analysis results

---

### `visualizations/` Directory

Interactive HTML charts for exploration:

```
visualizations/
├── topic_map.html           # 2D topic space visualization
├── barchart.html            # Topic frequency distribution
└── heatmap.html             # Topic × metadata correlations
```

All visualizations are self-contained HTML files that can be:
- Opened directly in a browser
- Embedded in reports or dashboards
- Shared with stakeholders

---

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- OpenAI API key

### Setup

```bash
# Clone repository
git clone https://github.com/gelob33/Customer-Support-Ticket-Theme-Discovery-using-BERTopic-and-LLM.git
cd Customer-Support-Ticket-Theme-Discovery-using-BERTopic-and-LLM

# Install dependencies
uv sync

# Configure OpenAI API key
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Download dataset (requires Kaggle CLI)
kaggle datasets download -d suraj520/customer-support-ticket-dataset
unzip customer-support-ticket-dataset.zip -d ./input
```

---

## Usage

### Run Full Pipeline

```bash
uv run python main.py
```

This executes all phases:
1. Data loading and preprocessing
2. BERTopic model training
3. LLM insights generation
4. Visualization creation

### Expected Runtime

- Data preparation: < 1 minute
- Topic modeling: 2-5 minutes (depending on GPU availability)
- LLM insights: 5-10 minutes (depends on topic count and API rate limits)
- Visualizations: < 1 minute

**Total**: ~10-15 minutes for full pipeline

---

## Output Examples

### Top Discovered Themes (Example)

| Topic | Count | Main Theme | Business Impact |
|-------|-------|------------|-----------------|
| 0 | 847 | Login and authentication failures | High - blocking user access |
| 1 | 623 | Payment processing errors | Critical - revenue impact |
| 2 | 512 | Product delivery delays | High - customer satisfaction |
| 3 | 387 | Refund request processing | Medium - churn risk |
| 4 | 301 | Technical bugs in mobile app | Medium - UX degradation |

### Sample Insight (Topic 0: Login Issues)

```
Main Theme: 
Customers unable to log in due to password reset failures and account lockouts

Key Pain Points:
- Password reset emails not arriving or expired
- Account locked after failed attempts with no clear unlock process
- Two-factor authentication codes not working
- Support wait times > 2 hours for critical access issues

Common Patterns:
- Peak incidents on Monday mornings and after password policy updates
- 65% involve "Product Purchased: Premium Subscription"
- Average customer satisfaction rating: 2.1/5

Business Impact:
- Severity: HIGH - Blocking revenue-generating users from access
- Estimated affected users: 847 unique customers
- Average resolution time: 4.2 hours

Recommendations:
1. Implement self-service account unlock feature
2. Improve password reset email deliverability and extend token validity
3. Add SMS backup for 2FA codes
4. Create dedicated "account access" support queue with faster SLA
5. Add proactive monitoring for authentication service health
```

---

## Project Structure

```
.
├── main.py                          # Pipeline orchestration
├── src/
│   ├── config.py                    # Configuration and constants
│   ├── data/
│   │   └── loader.py                # Data loading and preprocessing
│   ├── models/
│   │   └── topic_model.py           # BERTopic model configuration
│   ├── analysis/
│   │   └── insights.py              # LLM insight generation
│   └── visualization/
│       └── charts.py                # Visualization generation
├── input/
│   └── customer_support_tickets.csv # Source data
├── output/                          # Generated data files
└── visualizations/                  # Generated HTML charts
```

---

## License

MIT

---

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

## Acknowledgments

- Dataset: [Kaggle Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)
- BERTopic: [Maarten Grootendorst](https://github.com/MaartenGr/BERTopic)
- Embeddings: sentence-transformers by UKPLab
