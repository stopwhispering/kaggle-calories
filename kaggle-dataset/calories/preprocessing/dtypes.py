import pandas as pd


def convert_sex(df: pd.DataFrame) -> pd.DataFrame:
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    return df
