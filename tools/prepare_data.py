import pandas as pd
import sys
from pathlib import Path

# Get dataset folder from CMD argument
data_folder = Path(sys.argv[1])

train_file = data_folder / "train_data.txt"

# Load data (tab separated file expected: text \t label)
df = pd.read_csv(train_file, sep="\t", header=None, names=["plot", "genre"])

# Clean text
df["plot"] = df["plot"].astype(str).str.strip()
df["genre"] = df["genre"].astype(str).str.strip()

# Save as CSV
output_path = data_folder / "movies.csv"
df.to_csv(output_path, index=False)

print("✅ Movies CSV Created At:", output_path)
print(df.head())
