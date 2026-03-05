from ucimlrepo import fetch_ucirepo
import pandas as pd
import time

def load_dataset():
    print("\nStarting dataset download from UCI repository...")
    start_time = time.time()
    dataset = fetch_ucirepo(id=235)
    print("Dataset downloaded successfully.")
    print("Converting to pandas dataframe...")
    df = dataset.data.features
    end_time = time.time()
    print("\nDataset Loaded Successfully")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Total time:", round(end_time - start_time, 2), "seconds")
    return df

if __name__ == "__main__":
    df = load_dataset()
    print("\nFirst 5 rows of dataset:\n")
    print(df.head())