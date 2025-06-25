import pandas as pd

EXCLUDE_COLUMNS = [
    'LapNumber','LapTime','Sector1Time','Sector2Time','Sector3Time',
    'PitInTime','PitOutTime','LapTimeSeconds','LapTime_s',
    'Sector1SessionTime','Sector2SessionTime','Sector3SessionTime',
    'SpeedI1','SpeedI2','SpeedFL','SpeedST','TyreLife','LapStartTime','LapStartDate',
    'DeletedReason','TyreAge','Time'
]

def explore_uniques(df: pd.DataFrame, exclude: list[str] = EXCLUDE_COLUMNS) -> None:
    """Print unique values for every column not in `exclude`."""
    for col in df.columns:
        if col not in exclude:
            vals = df[col].unique()
            print(f"\nColumn: {col}")
            print(f"Unique Values ({len(vals)}): {vals!r}")

def explore_missing(df: pd.DataFrame) -> pd.Series:
    """
    Compute and return the proportion of nulls per column,
    sorted descending—but do NOT print inside this function.
    """
    return df.isnull().mean().sort_values(ascending=False)
