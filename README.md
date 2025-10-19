# Index-analysis-repo-Gurbax

# Enriching Stock Market Data using the OpenAI API

This is a project I completed, where I used the **OpenAI API** to classify companies from the **Nasdaq-100** (headquartered in California) into sectors and summarize their year-to-date (YTD) performance.  
It demonstrates how large language models can enrich and automate financial data analysis.

---

## 📊 Project Overview
- **Goal:** Classify Nasdaq-100 companies into sectors and generate insights about sector-level stock performance using AI.  
- **Input Data:**  
  - `nasdaq100_CA.csv` — Nasdaq-100 companies headquartered in California.  
  - `nasdaq100_price_change.csv` — YTD price change data for each ticker.  
- **Process:**
  1. Merge the two datasets on the company ticker symbol.
  2. Use the OpenAI `gpt-3.5-turbo` model to classify each company into one of ten sectors:
     *Technology, Consumer Cyclical, Industrials, Utilities, Healthcare, Communication, Energy, Consumer Defensive, Real Estate, Financial.*
  3. Summarize overall sector performance and identify top-performing sectors and companies.

---

## 🧠 Tools & Libraries
| Category | Technologies |
|-----------|--------------|
| Programming | Python 3 |
| Data Handling | `pandas` |
| API Integration | `openai`, `python-dotenv` |
| Utilities | `tqdm`, `tenacity` |
| Environment | GitHub Codespaces / VS Code |

---

## ⚙️ Project Structure
