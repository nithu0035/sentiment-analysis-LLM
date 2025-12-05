
# Employee Sentiment Analysis – Final LLM Assessment (CSV Dataset Included)

This project analyzes employee email data to:
- Label sentiment (Positive, Neutral, Negative) using **TextBlob**
- Perform **EDA** and data visualizations
- Compute **monthly sentiment scores**
- Produce **employee rankings**
- Identify **flight risk employees**
- Fit a **linear regression model** on sentiment trends over time

The dataset (`data/employee_feedback.csv`) is already included and contains:
- `Subject`: email subject line
- `body`: email content (used as feedback text)
- `date`: date of the email
- `from`: sender address (used as employee ID)

---

## 1. Project Structure

```text
.
├─ data/
│   └─ employee_feedback.csv         # already included
├─ notebooks/
│   └─ employee_sentiment_analysis.ipynb
├─ src/
│   └─ sentiment_pipeline.py
├─ outputs/                          # generated when you run the code
├─ .env.example
├─ requirements.txt
└─ README.md
```

---

## 2. Setup Instructions

1. **Install Python 3.9+** (if not already installed).

2. **Unzip the project** somewhere on your computer, then open a terminal / command prompt
   in the unzipped folder, e.g. `employee-sentiment-llm-final-csv`.

3. **(Recommended) Create and activate a virtual environment**

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Configure environment variables**

Copy `.env.example` to `.env` (optional but recommended):

```bash
cp .env.example .env   # On Windows you can manually duplicate the file
```

Ensure `.env` contains:

```env
DATA_PATH=./data/employee_feedback.csv
OUTPUT_DIR=./outputs
```

---

## 3. Usage

### Option A – Jupyter Notebook (recommended)

```bash
jupyter notebook
```

Then:

1. In the browser that opens, go into the `notebooks/` folder.
2. Open `employee_sentiment_analysis.ipynb`.
3. Run all cells in order (`Kernel` → `Restart & Run All`).

The notebook will:

- Load `employee_feedback.csv`
- Do EDA on the feedback
- Compute TextBlob sentiment scores and labels
- Aggregate sentiment monthly
- Rank employees
- Flag high flight-risk employees
- Fit and plot a linear regression trend
- Save CSV outputs into the `outputs/` folder.

### Option B – Python Script

```bash
python src/sentiment_pipeline.py
```

The script runs the same pipeline and writes:

- `outputs/feedback_with_sentiment.csv`
- `outputs/employee_ranking_and_flight_risk.csv`
- `outputs/monthly_sentiment_trend.csv`

---

## 4. Dataset Column Mapping

In this project, we map the original CSV columns as:

- `from`   → `employee_id`
- `body`   → `feedback_text`
- `date`   → `date`
- `Subject` is kept for possible additional analysis but not required for sentiment.

---

## 5. Methodology Overview

1. **Data Cleaning & EDA**
   - Parse dates and keep only relevant columns.
   - Add derived features like text length.
   - Inspect distributions and basic statistics.

2. **Sentiment Analysis**
   - Use TextBlob to compute a polarity score in the range [-1, 1].
   - Map polarity to three labels:
     - `score > 0.05`  → Positive
     - `score < -0.05` → Negative
     - otherwise       → Neutral

3. **Monthly Sentiment Scoring**
   - Group by `year_month` and compute average sentiment and feedback counts.
   - Plot monthly sentiment as a time-series line chart.

4. **Employee Ranking**
   - Group by `employee_id` and compute:
     - Average sentiment
     - Feedback count
   - Rank employees by average sentiment.

5. **Flight Risk Identification**
   - Simple rule-based heuristic:
     - Average sentiment below a negative threshold
     - Minimum number of feedback entries
   - Mark these employees as **High** flight risk; others as **Low**.

6. **Linear Regression Trend**
   - Encode months as an integer time index.
   - Fit a scikit-learn `LinearRegression` model on monthly average sentiment.
   - Use the slope to interpret whether sentiment is improving or declining over time.

---

## 6. Requirement Checklist (from assignment)

- [x] Sentiment labelling (Positive, Negative, Neutral) using TextBlob
- [x] EDA and visualizations
- [x] Monthly sentiment scoring
- [x] Employee ranking
- [x] Flight risk identification
- [x] Linear regression model for sentiment trends
- [x] `README.md` with setup, usage and methodology
- [x] `.env.example`
- [x] Dataset file included (`data/employee_feedback.csv`)
