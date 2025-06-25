# src/flin/race_widget.py
"""
Interactive lap-by-lap comparison widget
----------------------------------------
Usage
-----
from flin.race_widget import show_race_comparison

show_race_comparison(all_laps, BG='#06003A', FG='white')
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import ipywidgets as wd
from IPython.display import display, clear_output
from viz_config import sns, plt, BG, FG, dark_params


# ----------------------------------------------------------------------
# Public entry-point ----------------------------------------------------
# ----------------------------------------------------------------------
def show_race_comparison(all_laps: pd.DataFrame):
    """
    Create & display the interactive driver-vs-driver race comparison
    widget.

    Parameters
    ----------
    all_laps : pd.DataFrame
        Raw ( *not* pre-filtered ) FastF1 laps dataframe.
    BG, FG : str
        Background / foreground colour hex codes that match your theme.

    Returns
    -------
    (controls, output) – the ipywidgets objects, in case you want to
    lay them out manually.
    """
    # ------------------------------------------------------------------
    # Source data & derived columns
    # ------------------------------------------------------------------
    df = (
        all_laps
        .copy()
        .sort_values(["GrandPrix", "Driver", "LapNumber"])
        .reset_index(drop=True)
    )

    COLOR_MAP = {
        "SOFT"        : "red",
        "MEDIUM"      : "yellow",
        "HARD"        : "white",
        "INTERMEDIATE": "green",
        "WET"         : "blue"
    }

    df["color"] = df["Compound"].map(COLOR_MAP)
    df.loc[df["Position"].isna(),       "color"] = "black"   # crashes
    df.loc[df["PitInTime"].notna(),     "color"] = "purple"  # pit events
    df.loc[df["PitOutTime"].notna(),    "color"] = "purple"

    # helper ----------------------------------------------------------------
    laps_for = lambda gp, drv: df[(df.GrandPrix == gp) & (df.Driver == drv)]

    # ------------------------------------------------------------------
    # Main plot routine
    # ------------------------------------------------------------------
    def _plot_laps(gp, d1, d2):
        clear_output(wait=True)
        if not d1 or not d2:
            print("Select two drivers to compare.")
            return

        subset = df[df.GrandPrix == gp]
        fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        ax.set_xlabel("Lap Number", color=FG)
        ax.set_ylabel("Lap Time (s)", color=FG)
        ax.set_title(f"{gp}  ({d1} vs {d2})", color=FG)

        # background shading for SC / VSC / red
        mask = subset[["TS_4", "TS_5", "TS_6", "TS_7"]].any(axis=1)
        spans, prev = [], None
        for lap in subset.loc[mask, "LapNumber"]:
            if prev is None or lap != prev + 1:
                spans.append([lap, lap])
            else:
                spans[-1][1] = lap
            prev = lap
        for start, end in spans:
            ax.axvspan(start - 0.5, end + 0.5, color="orange", alpha=0.02)

        # plot each driver -------------------------------------------------
        legend_lines = []
        for ls, drv in zip(["-", "--"], [d1, d2]):
            data = laps_for(gp, drv)
            for i in range(len(data) - 1):
                x0, y0 = data.LapNumber.iat[i],     data.LapTimeSeconds.iat[i]
                x1, y1 = data.LapNumber.iat[i + 1], data.LapTimeSeconds.iat[i + 1]
                pit_in  = pd.notna(data.PitInTime.iat[i])
                pit_out = pd.notna(data.PitOutTime.iat[i + 1])
                line_c = "purple" if pit_in and pit_out else data.color.iat[i]
                ax.plot([x0, x1], [y0, y1], linestyle=ls, color=line_c, lw=2)

            legend_lines.append(Line2D([0], [0], linestyle=ls,
                                       color="white", label=drv))

        # legend ----------------------------------------------------------------
        handles = legend_lines + [
            Patch(facecolor=c, edgecolor="k", label=k) for k, c in COLOR_MAP.items()
        ] + [
            Patch(facecolor="purple", edgecolor="k", label="Pit event"),
            Patch(facecolor="black",  edgecolor="k", label="Crash"),
            Patch(facecolor="orange", alpha=0.5,      label="SC/VSC/Red flag"),
        ]
        ax.legend(handles=handles, loc="upper right")
        plt.show()

        # text summary -----------------------------------------------------
        total = subset.LapNumber.max()
        print(f"Total laps: {total}")
        for drv in (d1, d2):
            data = laps_for(gp, drv)
            fastest = data.LapTimeSeconds.min()
            stints  = "; ".join(
                f"{g.Compound.iloc[0]}({len(g)} laps)"
                for _, g in data.groupby("Stint")
            )
            print(f"{drv} ({data.Team.iat[0]}): Fastest {fastest:.3f}s; {stints}")

    # ------------------------------------------------------------------
    # ipywidgets controls
    # ------------------------------------------------------------------
    gp_dd   = wd.Dropdown(options=sorted(df.GrandPrix.unique()),
                          description="Grand Prix")
    drv1_dd = wd.Dropdown(description="Driver 1")
    drv2_dd = wd.Dropdown(description="Driver 2")

    def _update_drivers(*_):
        drs = sorted(df[df.GrandPrix == gp_dd.value].Driver.unique())
        drv1_dd.options = drv2_dd.options = drs
        if drs:
            drv1_dd.value = drs[0]
            drv2_dd.value = drs[1] if len(drs) > 1 else drs[0]

    gp_dd.observe(_update_drivers, "value")
    _update_drivers()  # initialise

    controls = wd.VBox([gp_dd, wd.HBox([drv1_dd, drv2_dd])])
    out      = wd.interactive_output(_plot_laps,
                                     {"gp": gp_dd, "d1": drv1_dd, "d2": drv2_dd})

    print("Select GP and drivers:")
    display(controls, out)
