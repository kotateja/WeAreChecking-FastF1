# src/flin/plots.py

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
from statsmodels.robust.norms import HuberT
from statsmodels.stats.outliers_influence import variance_inflation_factor

# bring in your viz defaults
from viz_config import sns, plt, dark_params

# ---- helpers -----------------------------------------------------------
_slicks = ['SOFT', 'MEDIUM', 'HARD']


# ------------------------------------------------------------------------
# compute once: classification threshold & per-driver rounds
# ------------------------------------------------------------------------
def _compute_classification_metrics(all_laps_clean: pd.DataFrame):
    """
    Compute:
      threshold = 2/3 of max Round
      rounds    = number of unique rounds per driver
    """
    threshold = all_laps_clean["Round"].max() * 2/3
    rounds = all_laps_clean.groupby("Driver")["Round"].nunique()
    return threshold, rounds

# ------------------------------------------------------------------------
# Average finishing position for each driver
# ------------------------------------------------------------------------
def plot_average_finishing_position(
    all_laps: pd.DataFrame,
    all_laps_clean: pd.DataFrame
):
    # Grab each driver’s final lap row
    final_rows = (
        all_laps
        .sort_values("LapNumber")
        .groupby(["GrandPrix", "Driver"])
        .tail(1)
        .reset_index(drop=True)
    )

    # compute threshold for number of races
    threshold, rounds = _compute_classification_metrics(all_laps_clean)

    # Mark who is classified (≥ 90% rule)
    final_rows["Classified"] = final_rows["RacePct"] >= 0.90

    # Keep only classified cars
    classified = final_rows[final_rows["Classified"]].copy()

    # Rank them within each GP by laps completed and crossing time
    time_key = "LapStartTime" if "LapStartTime" in classified.columns else "LapStartDate"
    def rank_finishers(gp):
        gp = gp.sort_values(
            by=["LapNumber", time_key],
            ascending=[False, True]      # most laps first; earlier time first
        ).reset_index(drop=True)
        gp["FinalPos"] = gp.index + 1
        return gp

    ranked = (
        classified
        .groupby("GrandPrix", group_keys=False)
        .apply(rank_finishers)
    )

    # Compute average finishing position per driver
    pos_summary = (
        ranked.groupby("Driver")["FinalPos"]
              .mean()
              .sort_values()
    )

    # Plotting
    cmap  = mpl.cm.coolwarm           
    norm  = mpl.colors.Normalize(pos_summary.min(), pos_summary.max())
    colors = cmap(norm(pos_summary.values))
    fig, ax = plt.subplots(figsize=(8, max(4, len(pos_summary)*0.25)))
    pos_summary.index = pos_summary.index.map(
        lambda d: f"{d}*" if rounds[d] <= threshold else d
    )
    pos_summary.plot.barh(ax=ax, color=colors, edgecolor="none")
    ax.invert_yaxis()
    ax.set_xlabel("Mean classified position (lower = better)")
    ax.set_title("Average Finishing Position – Season to Date")
    smap = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])
    plt.colorbar(smap, ax=ax, pad=0.015, label="Avg. Pos")
    plt.tight_layout()
    return fig, ax

# ------------------------------------------------------------------------
# Driver LapTime Percentiles
# ------------------------------------------------------------------------
def plot_lap_time_percentiles(all_laps_clean: pd.DataFrame):
    threshold, rounds = _compute_classification_metrics(all_laps_clean)

    # Group by Driver, compute the five quantiles, then unstack so each percentile is its own column.
    pctiles_df = (
        all_laps_clean
        .groupby("Driver")["LapTime_Scaled_per_GP"]
        .quantile([0.10, 0.25, 0.50, 0.90])
        .unstack(level=1)
    )

    # Rename the columns so they’re easier to handle:
    pctiles_df.columns = ["P10", "P25", "P50", "P90"]
    selected_pcts = ["P10", "P25", "P50"]

    # Sort drivers by P50
    ordered = pctiles_df.sort_values(by="P50", ascending=True)
    ordered.index = ordered.index.map(
        lambda d: f"{d}*" if rounds[d] <= threshold else d
    )
    drivers = ordered.index.tolist()
    x = np.arange(len(drivers))  # the label locations

    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]  # positions for P10, P50, P90

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each percentile
    for pct, offset in zip(selected_pcts, offsets):
        ax.bar(
            x + offset,
            ordered[pct],
            width=bar_width,
            label=pct,
            edgecolor='black'
        )

    # Add vertical reference lines for each percentile's fastest value
    for pct in selected_pcts:
        fastest_time = ordered[pct].min()
        fastest_driver = ordered[ordered[pct] == fastest_time].index[0]
        driver_index = drivers.index(fastest_driver)
        ax.axhline(fastest_time, color='white', linestyle='--', linewidth=1)
        ax.text(
            len(drivers) + 0.3,
            fastest_time,
            f"{pct} → {fastest_driver}",
            va="center", ha="left", fontsize=9,
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
        )

    # Tidy up
    ax.set_xticks(x)
    ax.set_xticklabels(drivers, rotation=90)
    ax.set_ylabel("Lap Time (s)")
    ax.set_title("Driver Lap Time Percentiles (P10, P50, P90)")
    ax.legend(title="Percentile")
    plt.tight_layout()
    return fig, ax

# ------------------------------------------------------------------------
# Driver lap-time consistency (full season)
# ------------------------------------------------------------------------
def plot_driver_consistency_cv(all_laps_clean: pd.DataFrame):
    threshold, rounds = _compute_classification_metrics(all_laps_clean)

    # Compute per-(Driver, GP, Stint) CV
    group_stats = (
        all_laps_clean
        .groupby(["Driver", "GrandPrix", "Stint"])["LapTimeSeconds"]
        .agg(mean_lap_s="mean", std_lap_s="std")
        .reset_index()
    )
    group_stats["cv_percent"] = 100 * group_stats["std_lap_s"] / group_stats["mean_lap_s"]

    # Sort so “best consistency” (lowest CV) is on top if you prefer
    cv_driver = group_stats.groupby("Driver")["cv_percent"].mean().sort_values(ascending=False)

    # Plotting
    cmap  = mpl.cm.coolwarm           
    norm  = mpl.colors.Normalize(cv_driver.min(), cv_driver.max())
    colors = cmap(norm(cv_driver.values))
    cv_driver.index = cv_driver.index.map(
        lambda d: f"{d}*" if rounds[d] <= threshold else d
    )
    fig, ax = plt.subplots(figsize=(8,6))
    cv_driver.plot.barh(ax=ax, color=colors, edgecolor="none")
    ax.set_xlabel("Mean coefficient of variation (%)")
    ax.set_title("Driver lap-time consistency (lower is better)")
    smap = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])
    cbar = plt.colorbar(smap, ax=ax, pad=0.02)
    cbar.set_label("CV %")
    plt.tight_layout()
    return fig, ax

# ------------------------------------------------------------------------
# Within-stint Variability
# ------------------------------------------------------------------------
def plot_within_stint_variability(all_laps_clean: pd.DataFrame):
    threshold, rounds = _compute_classification_metrics(all_laps_clean)

    # Group by (Driver, GrandPrix, Stint) and compute delta of each lap to the median
    consistency_df = (
        all_laps_clean
        .assign(
            stint_delta = lambda df: (
                df.groupby(["Driver", "GrandPrix", "Stint"])["LapTime_Scaled_per_GP"]
                  .transform(lambda x: x - x.median())
            )
        )
    )

    # Compute std-dev for each driver:
    consistency_score = (
        consistency_df
        .groupby("Driver")["stint_delta"]
        .std()
        .sort_values(ascending=True)
    )
    consistency_score.index = consistency_score.index.map(
        lambda d: f"{d}*" if rounds[d] <= threshold else d
    )

    # Plotting
    cmap  = mpl.cm.coolwarm           
    norm  = mpl.colors.Normalize(consistency_score.min(), consistency_score.max())
    colors = cmap(norm(consistency_score.values))
    fig, ax = plt.subplots(figsize=(6, max(4, len(consistency_score)*0.25)))
    consistency_score.plot.barh(ax=ax, color=colors, edgecolor="none")
    ax.invert_yaxis()
    ax.set_xlabel("Std-Dev(LapTime − stint median)  [seconds]")
    ax.set_title("Within-Stint Pace Variability  (lower = steadier)")
    smap = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])
    cbar = plt.colorbar(smap, ax=ax, pad=0.015)
    cbar.set_label("Std-dev (s)")
    plt.tight_layout()
    return fig, ax

# ------------------------------------------------------------------------
# Lap Time Distribution by Tyre Compound
# ------------------------------------------------------------------------
def plot_lap_time_distribution_by_compound(all_laps_clean: pd.DataFrame):
    slicks = ['SOFT','MEDIUM','HARD']  # choosing only slicks 
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(
        data=all_laps_clean,
        x='Compound',
        y='LapTime_Scaled_per_GP',
        order=slicks,
        palette=['#ff6666','#ffcc33','#ffffff']  # soft=red, med=yellow, hard=grey
    )
    plt.title("Lap Time Distribution by Tyre Compound")
    plt.xlabel("Tyre Compound")
    plt.ylabel("Lap Time (seconds)")
    plt.tight_layout()
    return fig, ax

# ------------------------------------------------------------------------
# Average Stint Length by Compound
# ------------------------------------------------------------------------
def plot_average_stint_length_by_compound(all_laps: pd.DataFrame):
    slicks = ['SOFT','MEDIUM','HARD']

    # Filter to slicks and then for each (Driver, GP, Stint, Compound) take max stint_lap as the stint length
    stint_lengths = (
        all_laps[all_laps["Compound"].isin(slicks)]
          .groupby(["Driver", "GrandPrix", "Stint", "Compound"], as_index=False)
          ["stint_lap"]
          .max()
          .rename(columns={"stint_lap": "stint_length"})
    )

    # Compute the average stint length by compound
    avg_stint_by_compound = (
        stint_lengths
          .groupby("Compound")["stint_length"]
          .mean()
          .reindex(slicks)
    )

    # Plotting
    fig, ax = plt.subplots(figsize=(6,4))
    avg_stint_by_compound.plot.bar(
        ax=ax,
        color=['#ff6666','#ffcc33','#ffffff'],  # SOFT, MEDIUM, HARD
        edgecolor="black"
    )
    ax.set_xticklabels(avg_stint_by_compound.index, rotation=0)
    ax.set_ylabel("Average Stint Length (laps)")
    ax.set_xlabel("Tyre Compound")
    ax.set_title("Average Stint Length by Slick Compound")
    plt.tight_layout()
    return fig, ax

# ------------------------------------------------------------------------
# Median Lap Time Trend Across Race
# ------------------------------------------------------------------------
def plot_lap_time_trend_vs_race_pct(all_laps_clean: pd.DataFrame, frac=0.05):
    # take just the two columns and sort by RacePct
    df_pct = all_laps_clean[['RacePct', 'LapTime_Scaled_per_GP']].sort_values('RacePct')

    # LOWESS smoothing
    lowess_smoothed = lowess(
        endog=df_pct['LapTime_Scaled_per_GP'],
        exog=df_pct['RacePct'],
        frac=frac,
        it=0
    )

    # plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(lowess_smoothed[:,0], lowess_smoothed[:,1], color='red', label=f'LOWESS (frac={frac})')
    ax.set_title('Smoothed Lap Time Trend vs RacePct')
    ax.set_xlabel('RacePct')
    ax.set_ylabel('Scaled Lap Time')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig, ax

# ------------------------------------------------------------------------
# Tyre-compound performance deltas by race (fuel-adjusted)
# ------------------------------------------------------------------------
def plot_compound_deltas_by_race(all_laps_clean: pd.DataFrame, min_stint_len: int = 10):
    """
    Per-GP bar-chart of median fuel-adjusted lap-time gaps:
       Soft−Medium   &   Medium−Hard
    """
    # keep only stints ≥ `min_stint_len`
    stint_len = (
        all_laps_clean
        .groupby(["GrandPrix", "Driver", "Stint"])["stint_lap"]
        .transform("max")
    )
    laps = all_laps_clean.loc[stint_len >= min_stint_len].copy()

    # β_pct  (fuel proxy slope)  per GP
    def _fit_beta_pct(gp_df):
        X = sm.add_constant(gp_df["RacePct"], has_constant="add")
        y = gp_df["LapTime_Scaled_per_GP"]
        return sm.OLS(y, X).fit().params["RacePct"]

    gp_coefs = (
        laps.groupby("GrandPrix")
            .apply(_fit_beta_pct, include_groups=False)
            .rename("β_pct")
            .to_frame()
    )

    # fuel-adjust the laps
    laps = (
        laps.merge(gp_coefs, left_on="GrandPrix", right_index=True)
            .assign(LapTime_fuelAdj=lambda d: d["LapTime_Scaled_per_GP"]
                                              - d["β_pct"] * d["RacePct"])
            .query("Compound in @_slicks")
    )

    # min per (GP, Compound)
    min_adj = (
        laps.groupby(["GrandPrix", "Compound"])["LapTime_fuelAdj"]
            .min()          
            .unstack()
            .loc[:, _slicks]   # ensure SOFT, MEDIUM, HARD order
    )

    # deltas (+ = right-hand compound faster)
    min_adj["Soft−Medium"]  = min_adj["SOFT"]  - min_adj["MEDIUM"]
    min_adj["Medium−Hard"] = min_adj["MEDIUM"] - min_adj["HARD"]

    # plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 10))
    min_adj[["Soft−Medium", "Medium−Hard"]].plot.bar(
        ax=ax, edgecolor="black"
    )
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_ylabel("Delta  (fuel-adjusted lap-time, s)")
    ax.set_xlabel("Grand Prix")
    ax.set_title("Tyre-compound performance deltas by race (fuel-adjusted)")
    plt.tight_layout()
    return fig, ax

# ------------------------------------------------------------------------
# Tyre degradation heat-maps  (track × compound  &  driver × compound)
# ------------------------------------------------------------------------
def plot_tyre_degradation_heatmaps(all_laps_clean: pd.DataFrame):
    """
    Two side-by-side heat-maps:

      • Track-level median β1 (TyreLife slope)  – rows = GP
      • Driver-level median β1 + SeasonAvg      – rows = Driver
    """
    # keep slicks & prune warm-up / marathon stints
    df = all_laps_clean.query(
        "Compound in @_slicks and 2 <= stint_lap < 50"
    ).copy()

    # robust slope extractor  (β1 wrt TyreLife)
    def _beta_sec(g):
        if len(g) < 8:
            return np.nan, len(g)
        X = sm.add_constant(
            np.c_[g["TyreLife"], g["RacePct"]], has_constant="add"
        )
        res = sm.RLM(g["LapTime_Scaled_per_GP"], X, M=HuberT()).fit()
        return res.params[1], len(g)   # (β1, N)

    slopes = (
        df.groupby(["GrandPrix", "Driver", "Compound"])
          .apply(_beta_sec, include_groups=False).dropna()
          .apply(lambda x: pd.Series({'beta_s': x[0], 'N': int(x[1])}))
          .reset_index()
    )

    # median per GP / Driver
    track_deg  = slopes.groupby(["GrandPrix", "Compound"])["beta_s"].median().unstack().sort_index()
    driver_deg = slopes.groupby(["Driver",    "Compound"])["beta_s"].median().unstack().sort_index()
    driver_deg["SeasonAvg"] = driver_deg.mean(axis=1)
    driver_deg = driver_deg.sort_values("SeasonAvg")

    total_laps = slopes.groupby("Driver")["N"].sum()

    # helper to over-write numeric + (N laps) annotation -----------------
    def heat(ax, data, index_field, title, ylabel="", xlabel=""):
        # pick grouping keys
        grp_keys = {
            "GrandPrix": ["GrandPrix", "Compound"],
            "Driver": ["Driver", "Compound"]
        }[index_field]

        # build and align N
        N = (
            slopes
            .groupby(grp_keys)["N"]
            .sum()
            .unstack(fill_value=0)
            .reindex(index=data.index, columns=data.columns, fill_value=0)
        )
        if index_field == "Driver" and "SeasonAvg" in data.columns:
            N["SeasonAvg"] = total_laps
        N = N.reindex(index=data.index, columns=data.columns, fill_value=0)

        # annotation: value + (N)
        annot = data.round(3).astype(str) + " (" + N.astype(int).astype(str) + ")"

        # draw heatmap
        sns.heatmap(
            data, ax=ax, cmap="RdYlGn_r", center=0,
            annot=annot, fmt="",
            cbar_kws={"label": "Beta1 (scaled_s / lap)"}
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=False)

    heat(
        ax1,
        track_deg,
        index_field="GrandPrix",
        title="Track tyre drop off",
        ylabel="Grand Prix",
        xlabel="Compound"
    )

    heat(
        ax2,
        driver_deg,
        index_field="Driver",
        title="Driver tyre drop off + SeasonAvg",
        ylabel="",
        xlabel="Compound / SeasonAvg"
    )

    plt.tight_layout()
    plt.show()

    return fig, (ax1, ax2)

# ------------------------------------------------------------------------
# Green-flag pit-stop durations by team  (box-plot)
# ------------------------------------------------------------------------
def plot_pitstop_durations_by_team(all_laps: pd.DataFrame):
    # isolate in-laps
    pit_in = (
        all_laps.loc[all_laps["PitInTime"].notna(),
                     ["GrandPrix","Driver","Team","LapNumber",
                      "PitInTime","TS_4","TS_5","TS_6","TS_7"]]
        .rename(columns={"LapNumber":"in_lap",
                         "TS_4":"TS_4_in","TS_5":"TS_5_in",
                         "TS_6":"TS_6_in","TS_7":"TS_7_in"})
    )
    pit_in["out_lap"] = pit_in["in_lap"] + 1

    pit = pit_in.merge(
        all_laps[["GrandPrix","Driver","LapNumber",
                  "PitOutTime","TS_4","TS_5","TS_6","TS_7"]],
        left_on=["GrandPrix","Driver","out_lap"],
        right_on=["GrandPrix","Driver","LapNumber"],
        how="left",
        suffixes=("", "_out")
    ).rename(columns={"TS_4":"TS_4_out","TS_5":"TS_5_out",
                      "TS_6":"TS_6_out","TS_7":"TS_7_out"})

    pit = pit.loc[pit["PitOutTime"].notna()]
    green_mask = ~(pit[["TS_4_in","TS_5_in","TS_6_in","TS_7_in",
                        "TS_4_out","TS_5_out","TS_6_out","TS_7_out"]].any(axis=1))
    pit = pit.loc[green_mask]
    pit["pit_duration"] = (pit["PitOutTime"] - pit["PitInTime"]).dt.total_seconds()
    pit = pit.query("0 < pit_duration < 60")

    # plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(
        data=pit,
        x="Team", y="pit_duration", hue="Team",
        palette="Set2",
        order=pit.groupby("Team")["pit_duration"].median().sort_values().index,
        legend=False, ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Pit-stop duration  (s)")
    ax.set_title("Green-flag pit-stop durations by Team")
    plt.tight_layout()
    return fig, ax

# ------------------------------------------------------------------------
# Median green-flag pit-stop duration by Grand Prix
# ------------------------------------------------------------------------
def plot_median_pit_duration_by_gp(all_laps: pd.DataFrame):
    # Recreate the pit DataFrame as in plot_pitstop_durations_by_team
    pit_in = (
        all_laps.loc[all_laps["PitInTime"].notna(),
                     ["GrandPrix","Driver","Team","LapNumber",
                      "PitInTime","TS_4","TS_5","TS_6","TS_7"]]
        .rename(columns={"LapNumber":"in_lap",
                         "TS_4":"TS_4_in","TS_5":"TS_5_in",
                         "TS_6":"TS_6_in","TS_7":"TS_7_in"})
    )
    pit_in["out_lap"] = pit_in["in_lap"] + 1

    pit = pit_in.merge(
        all_laps[["GrandPrix","Driver","LapNumber",
                  "PitOutTime","TS_4","TS_5","TS_6","TS_7"]],
        left_on=["GrandPrix","Driver","out_lap"],
        right_on=["GrandPrix","Driver","LapNumber"],
        how="left",
        suffixes=("", "_out")
    ).rename(columns={"TS_4":"TS_4_out","TS_5":"TS_5_out",
                      "TS_6":"TS_6_out","TS_7":"TS_7_out"})

    pit = pit.loc[pit["PitOutTime"].notna()]
    green_mask = ~(pit[["TS_4_in","TS_5_in","TS_6_in","TS_7_in",
                        "TS_4_out","TS_5_out","TS_6_out","TS_7_out"]].any(axis=1))
    pit = pit.loc[green_mask]
    pit["pit_duration"] = (pit["PitOutTime"] - pit["PitInTime"]).dt.total_seconds()
    pit = pit.query("0 < pit_duration < 60")

    gp_stats = (
        pit
        .groupby("GrandPrix")["pit_duration"]
        .median()
        .sort_values()
    )

    # plot median pit duration per GP:
    fig, ax = plt.subplots(figsize=(10,8))
    sns.barplot(
        data=gp_stats.reset_index(),
        y="GrandPrix", x="pit_duration",
        hue = "GrandPrix",
        palette="coolwarm",
        legend = False,
        order=gp_stats.index,
        ax=ax
    )   
    plt.xlabel("Median Pit Stop Duration (s)")
    plt.ylabel("Grand Prix")
    plt.title("Median Green-Flag Pit Duration by GP")
    plt.tight_layout()
    return fig, ax

