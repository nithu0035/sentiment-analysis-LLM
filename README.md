# 📊 Employee Sentiment Analysis — LLM & NLP Pipeline

An end-to-end NLP pipeline that analyzes employee email feedback to detect sentiment, rank employees, identify flight risks, and model sentiment trends over time using TextBlob and Scikit-learn.

---

## ✨ What This Project Does

- 🏷️ **Sentiment Labelling** — TextBlob polarity scoring maps each feedback to Positive / Neutral / Negative
- 📈 **Monthly Sentiment Scoring** — Aggregates average sentiment over time with a line chart
- 🏆 **Employee Ranking** — Ranks all employees by their average sentiment score
- ⚠️ **Flight Risk Detection** — Rule-based heuristic flags high-risk employees (avg sentiment ≤ -0.1 with ≥ 3 feedbacks)
- 📉 **Linear Regression Trend** — Fits a trend line over monthly sentiment to detect improving or declining patterns
- 🔍 **EDA & Visualizations** — Text length distributions, feedback counts per year, sentiment distribution charts

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Sentiment Engine | TextBlob |
| Data Processing | Pandas, NumPy |
| ML / Trend Model | Scikit-learn (LinearRegression) |
| Visualizations | Matplotlib, Seaborn |
| Notebook | Jupyter |
| Config | python-dotenv |

---

## 📁 Project Structure
```
sentiment-analysis-LLM/
├── data/
│   └── employee_feedback.csv       # Email dataset (included)
├── notebooks/
│   └── employee_sentiment_analysis.ipynb  # Full interactive pipeline
├── src/
│   └── sentiment_pipeline.py       # Standalone Python script
├── outputs/                        # Generated on run
│   ├── feedback_with_sentiment.csv
│   ├── employee_ranking_and_flight_risk.csv
│   └── monthly_sentiment_trend.csv
├── .env.example
├── requirements.txt
└── README.md
```

---

## 📂 Dataset

The dataset `data/employee_feedback.csv` contains employee email data with these columns:

| Column | Used As | Description |
|---|---|---|
| `from` | `employee_id` | Sender email (employee identifier) |
| `body` | `feedback_text` | Email content (analyzed for sentiment) |
| `date` | `date` | Date of the feedback |
| `Subject` | — | Email subject (kept for reference) |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/nithu0035/sentiment-analysis-LLM.git
cd sentiment-analysis-LLM
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
```

`.env` should contain:
```env
DATA_PATH=./data/employee_feedback.csv
OUTPUT_DIR=./outputs
```

---

## ▶️ Usage

### Option A — Jupyter Notebook (recommended)
```bash
jupyter notebook
```

Open `notebooks/employee_sentiment_analysis.ipynb` and run all cells (`Kernel → Restart & Run All`).

### Option B — Python Script
```bash
python src/sentiment_pipeline.py
```

Both options produce the same 3 output CSVs saved to the `outputs/` folder.

---

## 🧠 Methodology

**1. Data Cleaning** — Parse dates, drop nulls, rename columns, add text length feature.

**2. Sentiment Analysis** — TextBlob polarity score in range [-1, 1] mapped to three labels:
- `score > 0.05` → **Positive**
- `score < -0.05` → **Negative**
- otherwise → **Neutral**

**3. Monthly Aggregation** — Group by year-month, compute average sentiment and feedback count, plot time-series.

**4. Employee Ranking** — Group by employee ID, rank by average sentiment score descending.

**5. Flight Risk Flag** — Employees with avg sentiment ≤ -0.1 AND ≥ 3 feedback entries are flagged as **High** risk.

**6. Linear Regression Trend** — Encodes months as integer time index, fits `LinearRegression`, plots actual vs predicted trend line. A positive slope = improving sentiment over time.

---

## 📤 Output Files

| File | Description |
|---|---|
| `feedback_with_sentiment.csv` | All feedback rows with polarity score and label |
| `employee_ranking_and_flight_risk.csv` | Per-employee avg sentiment, feedback count, rank, and flight risk flag |
| `monthly_sentiment_trend.csv` | Monthly avg sentiment with linear regression predictions |

---

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)

## 👤 Author

**Gudipatoju Nitesh**  
GitHub: [@nithu0035](https://github.com/nithu0035)
