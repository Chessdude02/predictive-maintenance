import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("../data/raw")

# Load one file to understand structure
sample_file = sorted(DATA_DIR.iterdir())[0]
df = pd.read_csv(sample_file, sep="\t", header=None)
df.columns = ["bearing_1", "bearing_2", "bearing_3", "bearing_4"]

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic stats:")
print(df.describe())