import os
import pandas as pd


RAW_INPUT = "marvel_dc_raw.csv" if os.path.exists("marvel_dc_raw.csv") else "character.csv"
OUTPUT_FILE = "character.csv"

raw_df = pd.read_csv(RAW_INPUT)


raw_df.columns = [c.strip().replace('\ufeff', '').lower() for c in raw_df.columns]

vectorized_rows = []
