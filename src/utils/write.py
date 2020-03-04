import pandas as pd

def write_csv(df, filename):
    """
    Args:
        df: pandas dataframe to write
        filename (str): path to store the dataframe
    """
    df.to_csv(filename, index=False)