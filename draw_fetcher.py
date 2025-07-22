
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

SAVE_PATH = "data/historical_draws.csv"

def fetch_draws_from_lottonet(limit=300):
    """
    Scrapes recent Oz Lotto draws from lotto.net (or similar),
    parses the 7 main numbers, and returns a DataFrame.
    """
    draws = []
    page = 1

    while len(draws) < limit:
        url = f"https://www.lotto.net/oz-lotto/numbers/{page}"
        response = requests.get(url)
        if response.status_code != 200:
            break

        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.select(".resultsTable .resultsRow")

        for row in rows:
            balls = row.select(".balls .ball")
            if len(balls) >= 7:
                nums = [int(ball.text.strip()) for ball in balls[:7]]
                draws.append(nums)
            if len(draws) >= limit:
                break
        page += 1

    df = pd.DataFrame(draws, columns=[f"N{i+1}" for i in range(7)])
    os.makedirs("data", exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)
    return df

def load_local_draws():
    """Loads locally cached historical draws"""
    if os.path.exists(SAVE_PATH):
        return pd.read_csv(SAVE_PATH)
    return pd.DataFrame()
