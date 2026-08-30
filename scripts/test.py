import pandas as pd

df = pd.read_csv(
    "G:/My Drive/DeepfakeResearch/dataset_inventory.csv"
)

print(df.columns)
print(df.head())