import pandas as pd

df = pd.read_csv("sentimentdataset.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
