import os
import pandas as pd

def export_csv(dataframe: pd.DataFrame, filename: str):
    dataframe.to_csv(f"{filename}.csv", index=False)
