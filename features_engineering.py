#3
import pandas as pd

sensor_cols = [
    "TP2_mean",
    "TP3_mean",
    "H1_mean",
    "DV_pressure_mean",
    "Reservoirs_mean",
    "Oil_temperature_mean",
    "Motor_current_mean"
]


def feature_engineer1(df):
    df_10min = df.resample("10min").agg({
        "TP2": ["mean", "std", "min", "max"],
        "TP3": ["mean", "std", "min", "max"],
        "H1": ["mean", "std", "min", "max"],
        "DV_pressure": ["mean", "std", "min", "max"],
        "Reservoirs": ["mean", "std", "min", "max"],
        "Oil_temperature": ["mean", "std", "min", "max"],
        "Motor_current": ["mean", "std", "min", "max"],
        "COMP": ["mean", "std"],
        "DV_eletric": ["mean", "std"],
        "Towers": ["mean", "std"],
        "failure": "max"
    })

    df_10min.columns = ["_".join(col).strip("_")for col in df_10min.columns]

    df_10min = df_10min.dropna().copy()
    df_10min = df_10min.sort_index()

    return df_10min


def feature_engineer2(df_10min, sensor_cols):
    df_10min = df_10min.copy()

    for col in sensor_cols:
        df_10min[f"{col}_1h_mean"] = (df_10min[col].rolling(window=6, min_periods=6).mean())

        df_10min[f"{col}_1h_std"] = (df_10min[col].rolling(window=6, min_periods=6).std())

    df_10min = df_10min.dropna().copy()

    return df_10min
