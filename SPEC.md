# Customer Support Ticket Theme Discovery

Semantic clustering + LLM analysis of customer support tickets using BERTopic and OpenAI.

---

## Goal

Identify **common customer issues, themes, and pain points** from support tickets.

The final output should answer:

- What are the **most common customer problems?**
- What **products or systems** cause the most issues?
- What **business improvements** could reduce support load?

---

## Dataset

Source: Kaggle - Customer Support Ticket Dataset (~8,460 tickets)

Download:
```bash
kaggle datasets download -d suraj520/customer-support-ticket-dataset
unzip customer-support-ticket-dataset.zip -d ./input
```

Key fields:

| Column             | Purpose                          |
| ------------------ | -------------------------------- |
| Ticket Subject     | Short issue title                |
| Ticket Description | Detailed problem description     |
| Ticket Type        | Technical, Billing, Product, etc |
| Product Purchased  | Product involved                 |
| Ticket Channel     | Web / Chat / Phone               |
| Ticket Priority    | Urgency level                    |
| Resolution         | Support response                 |

Primary analysis field: `Ticket Subject + Ticket Description`

---

## Tech Stack

| Component           | Tool/Library                    |
| ------------------- | ------------------------------- |
| Topic Modeling      | BERTopic                        |
| Embeddings          | sentence-transformers (all-MiniLM-L6-v2) |
| LLM Insights        | OpenAI (gpt-4o-mini)            |
| Data Processing     | pandas, numpy                   |

Install:
```bash
uv add bertopic sentence-transformers torch pandas numpy openai python-dotenv
```

Hardware: 
Initial implementation will be in a local workstation with GPU support but allow fallback to CPU if GPU is unavailable.

---

## Workflow

### Phase 1 — Data Preparation

1. Load CSV from `./input/`
2. Combine `Ticket Subject` + `Ticket Description` into `question` field
3. Remove empty tickets (< 20 chars)
4. Remove duplicates
5. Retain metadata columns for analysis

---

### Phase 2 — Topic Modeling with BERTopic

**Model Configuration:**

- Embedding model: `all-MiniLM-L6-v2`
- Minimum topic size: 30
- Outlier reduction: enabled (reassign -1 topics to nearest cluster)

**Multi-Aspect Representations:**

1. **KeyBERTInspired** — Fast keyword extraction, no API cost
2. **OpenAI** — Structured business insights via custom prompt

---

### Phase 3 — Custom LLM Insights

Use OpenAI representation model with structured prompt.

**Required output format for each topic:**

```
Main Theme: <one sentence summary>

Key Pain Points:
- <bullet points>

Common Root Causes:
- <bullet points>

Business Recommendations:
- <actionable items>
```

**Parameters:**
- Model: gpt-4o-mini
- Documents per topic: 10
- Delay between calls: 2 seconds

---

### Phase 4 — Analysis

**Queries to answer:**

- Topic frequency distribution
- Products causing most issues (group by topic + Product Purchased)
- Ticket types per topic (crosstab)
- Topics per channel

**Topics per class analysis:**
- Analyze topic distribution across Product Purchased
- Analyze topic distribution across Ticket Type

---

### Phase 5 — Visualization

Generate BERTopic built-in visualizations:

| Visualization       | Purpose                              |
| ------------------- | ------------------------------------ |
| Topic map           | Intertopic distance (2D projection)  |
| Hierarchy           | Topic tree structure                 |
| Barchart            | Top words per topic                  |
| Documents           | Document clusters (2D projection)    |
| Heatmap             | Topic similarity matrix              |
| Topics per class    | Distribution across categories       |

Output format: Interactive HTML files in `./visualizations/`

---

## Deliverables

| Output                      | Location                          |
| --------------------------- | --------------------------------- |
| Clustered dataset           | `output/tickets_with_topics.csv`  |
| Topic insights              | `output/topic_insights.csv`       |
| Saved BERTopic model        | `output/bertopic_model/`          |
| Visualizations              | `visualizations/*.html`           |

---

## Expected Insights

Likely topic themes:

- Login / authentication failures
- Payment / billing confusion
- Product activation problems
- Software crashes or bugs
- Shipping or delivery issues

Each topic maps to a **customer issue theme** with actionable recommendations.

---

## Possible Extensions

- **Sentiment analysis** — Add frustration level as topic aspect
- **Dynamic topics** — Track topic evolution over time using ticket dates
- **Hierarchical topics** — Discover parent/child topic relationships
- **Zero-shot guidance** — Seed with known categories (billing, technical, shipping)
