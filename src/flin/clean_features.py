import pandas as pd
import numpy as np

def add_advanced_features(
    df: pd.DataFrame,
    dirty_air_thr: float = 2.0
) -> pd.DataFrame:
    """
    Given a cleaned laps DataFrame (with LapTimeSeconds, Season, Round, etc. already added),
    compute:
      - LapTime_Scaled_per_GP
      - GapToBest_s
      - LastRacePace (3‐race moving average of previous races)
      - DirtyAir flags per lap
      - PrevLapDT, PackCount_S3, LeadLapTime, GapToLeader
      - Lagged LapTime, TyreLife_lag1, DirtyAir_lag1
      - Interaction TyreLife_Compound
    """
    df = df.copy()
    
    # 1) scale lap‐time within each GP
    gp = df.groupby("GrandPrix")["LapTimeSeconds"]
    rng = gp.transform("max") - gp.transform("min")
    df["LapTime_Scaled_per_GP"] = (df["LapTimeSeconds"] - gp.transform("min")) / rng.replace(0, 1)

    # 2) gap to each driver’s best lap of that GP
    best = df.groupby(["GrandPrix", "Driver"])["LapTimeSeconds"].transform("min")
    df["GapToBest_s"] = df["LapTimeSeconds"] - best

    # 3) 3‐race moving average of *previous* race pace
    df = df.sort_values(["Driver", "Season", "Round"])
    df["LastRacePace"] = (
        df.groupby("Driver")["LapTime_Scaled_per_GP"]
          .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    # 4) Dirty‐air flag helper
    def mark_dirty_air(gp: pd.DataFrame) -> pd.DataFrame:
        g = gp.copy()
        for s in (1, 2, 3):
            t = g[f"Sector{s}SessionTime"].dt.total_seconds()
            g = g.assign(_t=t).sort_values("_t")
            g[f"GapAhead_S{s}"] = g["_t"].diff().fillna(np.inf).reindex(g.index)
            g = g.drop(columns=["_t"])
        g["DirtyAir"] = (
            (g["GapAhead_S1"] <= dirty_air_thr) |
            (g["GapAhead_S2"] <= dirty_air_thr) |
            (g["GapAhead_S3"] <= dirty_air_thr)
        ).astype(int)
        return g

    df = (
        df.groupby(["GrandPrix", "LapNumber"], group_keys=False)
          .apply(mark_dirty_air)
          .reset_index(drop=True)
    )

    # 5) Previous Lap delta time
    df = df.sort_values(["Driver", "GrandPrix", "LapNumber"])
    df["PrevLapDT"] = df.groupby(["Driver", "GrandPrix"])["LapTimeSeconds"].diff().fillna(0)

    # Number of cars within 2 seconds ahead 
    df["PackCount_S3"] = (
    df.groupby(["GrandPrix", "LapNumber"])["GapAhead_S3"]
      .transform(lambda x: (x <= dirty_air_thr).sum())
      .astype(int)
)
    # 6) Lead lap time and gap to leader
    df["LeadLapTime"] = df.groupby(["GrandPrix", "LapNumber"])["LapTimeSeconds"].transform("min")
    df["GapToLeader"] = df["LapTimeSeconds"] - df["LeadLapTime"]

    # 7) Create lagged features for LapTimeSeconds, TyreLife, and DirtyAir
    grp = df.groupby(["GrandPrix", "Driver"])
    for lag in (1, 2, 3):
        df[f"LapTime_lag{lag}"] = grp["LapTimeSeconds"].shift(lag)
    df["TyreLife_lag1"] = grp["TyreLife"].shift(1)
    df["DirtyAir_lag1"] = grp["DirtyAir"].shift(1)

    return df
