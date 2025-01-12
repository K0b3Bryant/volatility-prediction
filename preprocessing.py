import os
import pandas as pd
import numpy as np

def csv_reader(file_path: str):
  """ Cleans CSV files """
    if os.path.exists(file_path):
        temp_df = pd.read_csv(file_path, nrows=0)
        date_column = None

        for col in temp_df.columns:
            if col.lower() == 'date':
                date_column = col
                break

        if date_column:
            return pd.read_csv(file_path, parse_dates=[date_column], index_col=date_column)
            print('file read with date column as index')
        else:
            print('file read raw')
            return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def export_csv(dataframe: pd.DataFrame, filename: str):
    dataframe.to_csv(f"{filename}.csv", index=False)

def cleaner():
  pass
