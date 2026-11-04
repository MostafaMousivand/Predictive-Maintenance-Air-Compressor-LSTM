#2
import pandas as pd

def failure_event():
    failure_events = [
    {"failure_id": 1,"start": pd.Timestamp("2020-04-18 00:00:00"),"end": pd.Timestamp("2020-04-18 23:59:00")},
    {"failure_id": 2,"start": pd.Timestamp("2020-05-29 23:30:00"),"end": pd.Timestamp("2020-05-30 06:00:00")},
    {"failure_id": 3,"start": pd.Timestamp("2020-06-05 10:00:00"),"end": pd.Timestamp("2020-06-07 14:30:00")},
    {"failure_id": 4,"start": pd.Timestamp("2020-07-15 14:30:00"),"end": pd.Timestamp("2020-07-15 19:00:00")}]
    return failure_events

def failure_intervals(failure_events, df):
    df = df.copy()
    df["failure"] = 0
    for event in failure_events:
        start = pd.to_datetime(event["start"])
        end = pd.to_datetime(event["end"])
        mask = ((df.index >= start) & (df.index <= end))
        df.loc[mask, "failure"] = 1
    return df