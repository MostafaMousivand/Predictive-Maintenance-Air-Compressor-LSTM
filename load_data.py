#1
import pandas as pd


def load_data():
    df = pd.read_csv("air_compressor_metro.csv")

    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = df.set_index("timestamp")

    df["failure"] = 0

    return df





