import os
import fastf1 as f1
from fastf1 import Cache, get_session
import pandas as pd
from pathlib import Path

# 0) enable cache from your FASTF1_CACHE env var, if set
cache_dir = os.environ.get("FASTF1_CACHE")
if cache_dir:
    Cache.enable_cache(cache_dir)

def fetch_season(year: int) -> pd.DataFrame:
    records = []
    schedule = f1.get_event_schedule(year)
    for rnd in schedule.index:
        if schedule.loc[rnd, "EventFormat"] == "testing":
            continue
        ses = f1.get_session(year, rnd, "R")
        try:
            ses.load()
            laps = ses.laps.copy()
            laps["GrandPrix"] = ses.event["EventName"]
            # ★ create LapTimeSeconds once here
            laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
            records.append(laps)
        except Exception as e:
            print(f"⚠️ Could not load session for round {rnd}: {e}")
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

def build_best_lap_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    # a) choose drivers with ≥15 GPs
    gp_counts     = df.groupby("Driver")["GrandPrix"].nunique()
    drivers_min15 = gp_counts[gp_counts >= 15].index

    # b) subset to those drivers
    df2 = df[df["Driver"].isin(drivers_min15)]

    # c) GPs where all these drivers appear
    gp_driver_counts = df2.groupby("GrandPrix")["Driver"].nunique()
    full_gps         = gp_driver_counts[gp_driver_counts == len(drivers_min15)].index
    df3 = df2[df2["GrandPrix"].isin(full_gps)].copy()

    # d) pick each driver’s index of fastest lap by LapTimeSeconds
    best_idx  = df3.groupby(["GrandPrix","Driver"])["LapTimeSeconds"].idxmin().dropna().astype(int)
    best_laps = df3.loc[best_idx].reset_index(drop=True)

    rows = []
    for _, lap in best_laps.iterrows():
        year    = lap["LapStartDate"].year
        gp_code = lap["GrandPrix"].split()[0]
        try:
            ses = get_session(year, gp_code, "R"); ses.load()
        except Exception:
            continue

        cond   = (
            (ses.laps["LapNumber"] == lap["LapNumber"]) &
            (ses.laps["Driver"]    == lap["Driver"])
        )
        subset = ses.laps.loc[cond]
        if subset.empty:
            continue

        tel = subset.iloc[0].get_telemetry().dropna()
        rows.append({
            "GrandPrix"      : lap["GrandPrix"],
            "Driver"         : lap["Driver"],
            "LapTimeSeconds" : lap["LapTimeSeconds"],  # numeric seconds ready for analysis
            "MeanBrake"      : tel["Brake"].mean(),
            "MeanThrottle"   : tel["Throttle"].mean(),
            "AvgGear"        : tel["nGear"].mean(),
            "CornerExitSpd"  : tel.query("Throttle>0 & Speed<150")["Speed"].mean()
        })

    # ★ return after processing all best laps
    return pd.DataFrame(rows)

if __name__ == "__main__":
    year = 2024
    # 1) fetch & save raw laps
    df = fetch_season(year)
    raw_out = Path(f"data/raw/all_laps_{year}.feather")
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(raw_out)
    print(f"Wrote {len(df):,} rows ➜ {raw_out}")

    # 2) build & save best-lap telemetry
    telemetry_df = build_best_lap_telemetry(df)
    tel_out = Path(f"data/raw/best_lap_telemetry_{year}.feather")
    telemetry_df.to_feather(tel_out)
    print(f"Wrote {len(telemetry_df):,} rows ➜ {tel_out}")