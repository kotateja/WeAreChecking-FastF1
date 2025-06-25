import pandas as pd
from typing import Tuple, Set

VALID_COMPOUNDS: Set[str] = {
    "SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"
}

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()
    df["Season"] = df["LapStartDate"].dt.year
    df["Round"] = (
        df.groupby("Season")["GrandPrix"]
          .transform(lambda s: pd.Categorical(s, ordered=True).codes + 1)
    )
    df["stint_lap"] = (
        df.groupby(["GrandPrix","Driver","Stint"])
          .cumcount() + 1
    )
    df["RacePct"] = df["LapNumber"] / df.groupby("GrandPrix")["LapNumber"].transform("max")

    return df

def add_trackstatus_flags(
    laps: pd.DataFrame,
    status_col: str = "TrackStatus"
) -> pd.DataFrame:
    laps = laps.copy()
    stat_str = laps[status_col].fillna(0).astype(int).astype(str)
    for digit in map(str, range(1, 8)):
        laps[f"TS_{digit}"] = stat_str.str.contains(digit)
    return laps

def drop_sc_and_pit(df: pd.DataFrame) -> pd.DataFrame:
    mask_bad = df[["TS_4","TS_5","TS_6","TS_7"]].any(axis=1)
    mask_pit = df["PitInTime"].notna() | df["PitOutTime"].notna()
    n = (mask_bad | mask_pit).sum()
    print(f"Dropping {n:,} laps flagged SC/VSC/Red or Pit in/out")
    return df.loc[~(mask_bad | mask_pit)].copy()

def drop_slow_outliers(df: pd.DataFrame) -> pd.DataFrame:
    GROUP = ["Driver", "GrandPrix", "Stint"]
    g = df.groupby(GROUP)["LapTimeSeconds"]
    thr = 1.10 * g.transform("median")
    mask = df["LapTimeSeconds"] > thr
    n = mask.sum()
    print(f"Dropping {n:,} laps >110% of stint‐median pace")
    return df.loc[~mask].copy()

def drop_deleted_and_fastf1(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["Deleted"] | df["FastF1Generated"]
    n = mask.sum()
    print(f"Dropping {n:,} rows Deleted or FastF1‐generated")
    return df.loc[~mask].copy()

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "PitOutTime","PitInTime",
        "Deleted","DeletedReason","FastF1Generated",
        "TS_3","TS_4","TS_5","TS_6","TS_7"
    ]
    return df.drop(columns=cols, errors="ignore")

def drop_unknown_compounds(df: pd.DataFrame) -> pd.DataFrame:
    unknown_compounds = df.loc[~df["Compound"].isin(VALID_COMPOUNDS)].copy()
    print(f"Dropping {len(unknown_compounds):,} rows with invalid compounds")
    df = df.loc[df["Compound"].isin(VALID_COMPOUNDS)].copy()
    return df, unknown_compounds

def drop_missing_laptimes(df: pd.DataFrame) -> pd.DataFrame:
    missing_laptime = df.loc[df["LapTime"].isna()].copy()
    print(f"Dropping {len(missing_laptime):,} rows missing LapTime")
    df = df.loc[df["LapTime"].notna()]
    return df, missing_laptime

def drop_missing_positions(df: pd.DataFrame) -> pd.DataFrame:
    missing_position = df.loc[df["Position"].isna()].copy()
    print(f"Dropping {len(missing_position):,} rows missing Position")
    df = df.loc[df["Position"].notna()]
    return df, missing_position

def compute_laps_lost_stats(
    raw: pd.DataFrame,
    clean: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns per‐driver summary of TotalLaps, CleanLaps, DeletedLaps and %Deleted.
    """
    total = (
        raw.groupby("Driver")["LapTime"]
           .count()
           .reset_index(name="TotalLaps")
    )
    clean_ = (
        clean.groupby("Driver")["LapTime"]
             .count()
             .reset_index(name="CleanLaps")
    )

    stats = (
        total
        .merge(clean_, on="Driver", how="left")
        .assign(
            CleanLaps   = lambda df: df["CleanLaps"].fillna(0).astype(int),
            DeletedLaps = lambda df: df["TotalLaps"] - df["CleanLaps"],
            **{"%Deleted": lambda df: 100 * df["DeletedLaps"] / df["TotalLaps"]}
        )
        .sort_values("%Deleted", ascending=False)
        .reset_index(drop=True)
    )
    return stats


def preprocess(
    all_laps: pd.DataFrame
) -> Tuple[
    pd.DataFrame,  # clean laps
    pd.DataFrame,  # unknown compounds
    pd.DataFrame,  # missing LapTime
    pd.DataFrame   # missing Position
]:
    df = all_laps.copy()   
    # standard pipeline
    df = add_basic_features(df)
    df = add_trackstatus_flags(df)
    all_laps = df.copy()  # keep raw laps for stats
    df = drop_sc_and_pit(df)
    df, unknown_compounds = drop_unknown_compounds(df)
    df, missing_laptime = drop_missing_laptimes(df)
    df, missing_position = drop_missing_positions(df)
    df = drop_slow_outliers(df)
    df = drop_deleted_and_fastf1(df)
    df = drop_unused_columns(df)

    print(f"✔ Final clean laps: {len(df):,}")
    return df, unknown_compounds, missing_laptime, missing_position, all_laps

