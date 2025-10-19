import os
import argparse
import json
import time
from typing import Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from openai import OpenAI
from openai import APIError, RateLimitError, InternalServerError

SECTORS = [
    "Technology", "Consumer Cyclical", "Industrials", "Utilities", "Healthcare",
    "Communication", "Energy", "Consumer Defensive", "Real Estate", "Financial"
]

CLASSIFY_SYSTEM = (
    "You are a careful classifier. "
    "Given a public company name and ticker, return ONLY one sector label "
    "from this exact set: " + ", ".join(SECTORS) + "."
)

SUMMARY_SYSTEM = (
    "You are an equity strategy assistant. Using the provided table, "
    "summarize YTD performance for California-based Nasdaq-100 companies. "
    "Recommend the two best sectors and two or more companies per sector. "
    "Be concise and factual."
)

def read_data(path_ca: str, path_price: str) -> pd.DataFrame:
    df_ca = pd.read_csv(path_ca)
    df_price = pd.read_csv(path_price)
    # merge YTD by symbol
    if "symbol" not in df_ca.columns or "symbol" not in df_price.columns:
        raise ValueError("Both CSVs must have a 'symbol' column.")
    if "ytd" not in df_price.columns:
        raise ValueError("price_change CSV must have a 'ytd' column.")
    merged = df_ca.merge(df_price[["symbol", "ytd"]], on="symbol", how="inner")
    return merged

def build_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Set it in environment or .env.")
    return OpenAI(api_key=api_key)

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((APIError, RateLimitError, InternalServerError))
)
def chat_complete(client: OpenAI, model: str, system: str, user: str, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

def classify_row(client: OpenAI, model: str, name: str, symbol: str, cache: Dict[str, str]) -> str:
    key = f"{name}|{symbol}"
    if key in cache:
        return cache[key]
    prompt = (
        f"Company: {name} (Ticker: {symbol}). "
        f"Return only one of: {', '.join(SECTORS)}."
    )
    label = chat_complete(client, model, CLASSIFY_SYSTEM, prompt)
    # sanity check
    if label not in SECTORS:
        # try once more with stricter phrasing
        prompt2 = (
            f"Return only the sector label (no punctuation). "
            f"Company: {name} (Ticker: {symbol})."
        )
        label = chat_complete(client, model, CLASSIFY_SYSTEM, prompt2)
        if label not in SECTORS:
            label = "Technology"  # fall back (most common), or set to 'Unknown'
    cache[key] = label
    return label

def main(args: Optional[argparse.Namespace] = None):
    parser = argparse.ArgumentParser(description="Enrich Nasdaq-100 CA companies with sector labels via OpenAI.")
    parser.add_argument("--input-ca", required=True)
    parser.add_argument("--price-change", required=True)
    parser.add_argument("--out-csv", default="data/output/enriched.csv")
    parser.add_argument("--out-summary", default="data/output/summary.txt")
    parser.add_argument("--model", default="gpt-3.5-turbo")
    ns = parser.parse_args(args)

    client = build_client()
    df = read_data(ns.input_ca, ns.price_change)

    # choose name column if present
    name_col = "name" if "name" in df.columns else "company" if "company" in df.columns else None
    if name_col is None:
        # fallback to symbol for prompt
        df[name_col := "name_for_prompt"] = df["symbol"]

    cache_path = "data/output/sector_cache.json"
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    sectors = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        name = str(row[name_col])
        sym = str(row["symbol"])
        sector = classify_row(client, ns.model, name, sym, cache)
        sectors.append(sector)

    df["Sector"] = sectors

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    # Save enriched CSV
    os.makedirs(os.path.dirname(ns.out_csv), exist_ok=True)
    df.to_csv(ns.out_csv, index=False)

    # Build summary via model
    # For cost control, only pass minimal columns
    mini = df[["symbol", name_col, "ytd", "Sector"]].head(101)  # cap to 101 rows
    table_str = mini.to_csv(index=False)
    summary_prompt = f"Here is a CSV table with columns symbol,{name_col},ytd,Sector:\n\n{table_str}\n"
    summary = chat_complete(client, ns.model, SUMMARY_SYSTEM, summary_prompt)
    with open(ns.out_summary, "w") as f:
        f.write(summary)

    print(f"Wrote: {ns.out_csv}")
    print(f"Wrote: {ns.out_summary}")

if __name__ == "__main__":
    main()
