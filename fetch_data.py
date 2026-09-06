import os
import pandas as pd


RAW_INPUT = "marvel_dc_raw.csv" if os.path.exists("marvel_dc_raw.csv") else "character.csv"
OUTPUT_FILE = "character.csv"

raw_df = pd.read_csv(RAW_INPUT)


raw_df.columns = [c.strip().replace('\ufeff', '').lower() for c in raw_df.columns]

vectorized_rows = []

for _,row in raw_df.iterrows():
    def get_val(keys):
        for k in keys:
            if k in row and pd.notna(row[k]):
                return str(row[k]).strip().lower()

            return ""

    name= get_val(['character_name','name'])
    if not name:
        continue
