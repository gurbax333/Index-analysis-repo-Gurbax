# Enriching Stock Market Data using OpenAI

This repo journals my completed DataCamp project: **Enriching Stock Market Data using the OpenAI API**.  
It classifies Nasdaq‑100 companies (headquartered in California) into a sector using an LLM and summarizes YTD performance.

## Repo Structure
```
datacamp-nasdaq-openai/
├── src/
│   └── enrich_nasdaq.py
├── data/
│   ├── nasdaq100_CA.csv
│   ├── nasdaq100_price_change.csv
│   └── output/                # enriched CSVs & text summaries
├── notebooks/
│   └── datacamp_project.ipynb
├── .gitignore
├── LICENSE
└── requirements.txt
```

## Quickstart

1) Create a virtual environment and install deps
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) Set your OpenAI API key
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

3) Run the script
```bash
python src/enrich_nasdaq.py   --input-ca data/nasdaq100_CA.csv   --price-change data/nasdaq100_price_change.csv   --out-csv data/output/enriched.csv   --out-summary data/output/summary.txt   --model gpt-3.5-turbo
```

The script:
- Merges the CA‑headquartered Nasdaq‑100 list with YTD returns
- Classifies each company into one of 10 sectors using a chat-completions model
- Writes an enriched CSV plus a text summary of recommended sectors and tickers

> Note: For journaling/portfolio purposes, commit the code and README. If you can’t share datasets, keep `data/` out of version control (already ignored via `.gitignore`) and include a few sample rows in the README instead.

## Citation / Credit
Original concept from a DataCamp guided project.
