import pandas as pd

df = pd.read_feather("data/processed/laps_2024.feather")

print(df.head())           # See first few rows
print(df.shape)            # See total rows/columns
print(df.columns.tolist()) # See what columns are in it
