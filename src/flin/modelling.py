# ──────────────────────────────────────────────────────────────
#  src/flin/modeling.py
#  (full end-to-end modelling pipeline — mirrors notebook cells)
# ──────────────────────────────────────────────────────────────
import warnings
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from sklearn.preprocessing    import StandardScaler, OrdinalEncoder
from sklearn.compose          import ColumnTransformer
from sklearn.pipeline         import Pipeline
from sklearn.cluster          import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics          import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    mean_absolute_error, pairwise_distances,
)
from sklearn.decomposition    import PCA
from sklearn.model_selection  import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble         import HistGradientBoostingRegressor

import lightgbm as lgb
from lightgbm import LGBMRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split



# ──────────────────────────────────────────────────────────────
# CELL-1  · stats tests & rubber-in delta
# ──────────────────────────────────────────────────────────────
def run_stats_tests(all_laps_clean: pd.DataFrame):
    """Return (anova_results, MixedLM result, rubber-in Δ DataFrame)."""
    # 1) Two-way ANOVA: main + interaction effects of Compound & TyreAge on LapTime
    model = smf.ols(
        "LapTime_Scaled_per_GP ~ C(Compound) + TyreLife + C(Compound):TyreLife",
        data=all_laps_clean
    ).fit()
    anova_results = anova_lm(model, typ=2)
    print("Two-way ANOVA (Type II):\n", anova_results)

    # 2) Mixed-effects: per-driver random intercept, quantify driver variability
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    md_driver = sm.MixedLM.from_formula(
        "LapTime_Scaled_per_GP ~ TyreLife",
        groups="Driver",
        data=all_laps_clean
    )
    res_driver = md_driver.fit(method="lbfgs", reml=True, maxiter=2000)
    print("\nMixedLM (random intercept ~ Driver):\n", res_driver.summary())

    # 3) Track “rubber-in” δ: early vs penultimate lap
    rubber = []
    for gp, grp in all_laps_clean.groupby("GrandPrix"):
        if grp.LapNumber.nunique() < 3:
            delta = np.nan
        else:
            early   = grp.loc[grp.LapNumber == 2,
                              "LapTime_Scaled_per_GP"].mean()
            penult  = grp.loc[grp.LapNumber ==
                              grp.LapNumber.max()-1,
                              "LapTime_Scaled_per_GP"].mean()
            delta = early - penult
        rubber.append({"GrandPrix": gp, "RubberDelta_s": delta})
    rubber_df = pd.DataFrame(rubber)
    print("\nTrack rubber-in delta (Lap 2 minus penultimate):\n", rubber_df)

    return anova_results, res_driver, rubber_df


# ──────────────────────────────────────────────────────────────
# CELL-2  · clustering sweeps
# ──────────────────────────────────────────────────────────────
def sweep_clustering(all_laps_clean: pd.DataFrame):
    """
    Runs KMeans, Agglomerative, GMM, DBSCAN sweeps exactly as in notebook.
    Returns (metrics_df, driver_df, X, VI) so later cells can reuse them.
    """
    # Define features for clustering
    telemetry = ["MeanBrake", "MeanThrottle", "AvgGear", "CornerExitSpd"]

    # fastest lap per driver per GP – only drivers with ≥15 GPs
    gp_counts = all_laps_clean.groupby("Driver")["GrandPrix"].nunique()
    drivers_min15 = gp_counts[gp_counts >= 15].index
    df2 = all_laps_clean[all_laps_clean["Driver"].isin(drivers_min15)]

    gp_driver_counts = (
        df2.groupby("GrandPrix")["Driver"].nunique()
    )
    full_gps = gp_driver_counts[gp_driver_counts == len(drivers_min15)].index
    df3 = df2[df2["GrandPrix"].isin(full_gps)]

    best_idx = (
        df3.groupby(["GrandPrix", "Driver"])["LapTimeSeconds"].idxmin()
    )
    best_laps = df3.loc[best_idx].reset_index(drop=True)

    # Read telemetry data pre-built from our ETL file 
    # This was built on uncleaned data so we will cross-reference our filtering on the cleaned data
    feat = pd.read_feather("data/processed/best_lap_telemetry_2024.feather")
    feat = feat.merge(
        best_laps[["GrandPrix", "Driver"]],
        on=["GrandPrix", "Driver"],
        how="inner"
    )

    # per-GP scaling
    scaled_parts = []
    for gp, grp in feat.groupby("GrandPrix"):
        grp = grp.copy()
        grp[telemetry] = StandardScaler().fit_transform(grp[telemetry])
        scaled_parts.append(grp)
    feat_scaled = pd.concat(scaled_parts, ignore_index=True)
    
    # aggregate to one row per driver (mean of their per‐GP scaled values)
    driver_df = (
        feat_scaled.groupby("Driver")[telemetry]
                   .mean()
                   .sort_index()
    )
    X = driver_df[telemetry].values

    # precompute inverse covariance for Mahalanobis
    cov = np.cov(X, rowvar=False)
    VI  = np.linalg.pinv(cov)

    inertia, ks = [], range(2, 11)
    for k in ks:
        inertia.append(KMeans(n_clusters=k, random_state=42).fit(X).inertia_)
    plt.figure(figsize=(6, 4))
    plt.plot(ks, inertia, "o-")
    plt.xlabel("Number of clusters k"); plt.ylabel("Inertia")
    plt.title("Elbow Method for K-Means"); plt.show()

    distances = ["euclidean", "manhattan", "cosine", "mahalanobis"]
    results: List[Dict[str, Any]] = []

    # KMeans sweeps
    for k in ks:
        lbl = KMeans(n_clusters=k, random_state=42).fit_predict(X)
        results.append({
            "method":            "KMeans",
            "params":            f"k={k}",
            "n_clusters":        k,
            "silhouette":        silhouette_score(X, lbl),
            "calinski_harabasz": calinski_harabasz_score(X, lbl),
            "davies_bouldin":    davies_bouldin_score(X, lbl)
        })

    # Agglomerative sweeps
    for k in ks:
        for metric in distances:
            if metric == "mahalanobis":
                D = pairwise_distances(X, metric="mahalanobis", VI=VI)
                lbl = AgglomerativeClustering(
                    n_clusters=k, metric="precomputed", linkage="average"
                ).fit_predict(D)
                sil = silhouette_score(X, lbl, metric="mahalanobis", VI=VI)
            else:
                lbl = AgglomerativeClustering(
                    n_clusters=k, metric=metric, linkage="average"
                ).fit_predict(X)
                sil = silhouette_score(X, lbl, metric=metric)

            results.append({
                "method":            "Agglomerative",
                "params":            f"k={k}, metric={metric}",
                "n_clusters":        k,
                "silhouette":        sil,
                "calinski_harabasz": calinski_harabasz_score(X, lbl),
                "davies_bouldin":    davies_bouldin_score(X, lbl)
            })

    # Gaussian Mixture
    from sklearn.mixture import GaussianMixture
    for k in ks:
        lbl = GaussianMixture(n_components=k, random_state=42).fit(X).predict(X)
        results.append({
            "method":            "GaussianMixture",
            "params":            f"k={k}",
            "n_clusters":        k,
            "silhouette":        silhouette_score(X, lbl),
            "calinski_harabasz": calinski_harabasz_score(X, lbl),
            "davies_bouldin":    davies_bouldin_score(X, lbl)
        })

    # DBSCAN sweeps
    for metric in distances:
        kwargs = {"metric": metric}
        if metric == "mahalanobis":
            kwargs["metric_params"] = {"VI": VI}
        lbl = DBSCAN(eps=0.5, min_samples=5, **kwargs).fit_predict(X)
        mask = lbl >= 0
        nclus = len(np.unique(lbl[mask])) if mask.any() else 0
        if nclus > 1:
            sil = (silhouette_score(X[mask], lbl[mask], metric="mahalanobis", VI=VI)
                   if metric == "mahalanobis"
                   else silhouette_score(X[mask], lbl[mask], metric=metric))
            ch = calinski_harabasz_score(X[mask], lbl[mask])
            db = davies_bouldin_score(X[mask], lbl[mask])
        else:
            sil = ch = db = np.nan
        results.append({
            "method":            "DBSCAN",
            "params":            f"eps=0.5,min=5,metric={metric}",
            "n_clusters":        nclus,
            "silhouette":        sil,
            "calinski_harabasz": ch,
            "davies_bouldin":    db
        })

    metrics_df = (
        pd.DataFrame(results)
          .set_index(["method", "params"])
          .groupby(level="method", group_keys=False)
          .apply(lambda d: d.sort_values("silhouette", ascending=False))
    )
    return metrics_df, driver_df, X, VI


# ──────────────────────────────────────────────────────────────
# CELL-3  · PCA scree
# ──────────────────────────────────────────────────────────────
def pca_scree(driver_df: pd.DataFrame):
    X = driver_df[["MeanBrake", "MeanThrottle", "AvgGear", "CornerExitSpd"]].values
    pca_full = PCA().fit(X)
    var_ratios = pca_full.explained_variance_ratio_
    cum_var    = np.cumsum(var_ratios)

    fig, ax = plt.subplots()
    ax.bar(range(1, 5), var_ratios * 100, label="individual")
    ax.step(range(1, 5), cum_var * 100, where="mid", color="white",
            label="cumulative")
    ax.set_xticks(range(1, 5))
    ax.set_ylabel("% variance explained")
    ax.set_xlabel("Principal component")
    ax.set_title("Scree + cumulative variance")
    ax.legend()
    plt.show()

    print(f"Component 1–2 cumulative: {cum_var[1]*100:.1f}%")
    return fig, cum_var


# ──────────────────────────────────────────────────────────────
# CELL-4  · final Agglomerative plot & silhouette
# ──────────────────────────────────────────────────────────────
def final_clustering_plot(driver_df: pd.DataFrame,
                          X: np.ndarray,
                          *, k: int = 4):
    telemetry = ["MeanBrake", "MeanThrottle", "AvgGear", "CornerExitSpd"]

    pca = PCA(n_components=2, random_state=42)
    driver_df[["PC1", "PC2"]] = pca.fit_transform(driver_df[telemetry].values)

    agg4 = AgglomerativeClustering(
        n_clusters=k, metric="cosine", linkage="average"
    )
    labels = agg4.fit_predict(X)
    driver_df["cluster"] = labels

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        driver_df["PC1"], driver_df["PC2"],
        c=labels, cmap="tab10", s=150, edgecolor="k"
    )
    for i, drv in enumerate(driver_df.index):
        ax.text(
            driver_df["PC1"].iat[i] + 0.02,
            driver_df["PC2"].iat[i] + 0.02,
            drv, fontsize=9, color="white",
            path_effects=[pe.withStroke(linewidth=1, foreground="black")]
        )
    handles, _ = scatter.legend_elements()
    ax.legend(handles, [f"Cluster {i}" for i in range(k)],
              title="Cluster", loc="upper left")
    sil_full = silhouette_score(X, labels, metric="cosine")
    pcs = driver_df[["PC1", "PC2"]].values
    sil_pca  = silhouette_score(pcs, labels, metric="euclidean")
    ax.set_title(f"Agglomerative (k={k}, cosine)\nSilhouette = {sil_full:.2f}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    print(f"Silhouette (4D) : {sil_full:.3f}")
    print(f"Silhouette (2D) : {sil_pca :.3f}")
    interp = (
        "Clusters overlap badly"            if sil_full < 0 else
        "Very weak structure"               if sil_full < 0.25 else
        "Reasonable structure"              if sil_full < 0.50 else
        "Strong, well-separated clusters"
    )
    print("Interpretation :", interp)
    return fig, labels, sil_full, sil_pca


# ──────────────────────────────────────────────────────────────
# CELL-6 · strict training & tuning
# ──────────────────────────────────────────────────────────────
def strict_train_tune(all_laps_clean: pd.DataFrame):
    df = (all_laps_clean.copy()
            .dropna(subset=["LapTimeSeconds"])
            .assign(IsFirstRaceOfSeason=lambda x: (x["Round"] == 1).astype(int)))

    need = [f"LapTime_lag{i}" for i in (1, 2, 3)] + ["DirtyAir_lag1"]
    df   = df.dropna(subset=need).reset_index(drop=True)

    features = [
        "RacePct","Position","TyreLife","PrevLapDT", "stint_lap",
        "PackCount_S3","GapToLeader","LastRacePace","IsFirstRaceOfSeason",
        "LapTime_lag1","LapTime_lag2","LapTime_lag3","DirtyAir_lag1", "DirtyAir",
        "Driver","Compound","Team","LeadLapTime"
    ]
    num_cols = [c for c in features if c not in ("Driver","Compound","Team")]
    cat_cols = ["Driver","Compound","Team"]

    n_num = len(num_cols)
    n_cat = len(cat_cols)
    cat_idx = list(range(n_num, n_num + n_cat))


    X_full, y_full = df[features], df["LapTimeSeconds"]

    gp_order = (
        df[["Season","Round","GrandPrix"]]
          .drop_duplicates()
          .sort_values(["Season","Round"])
    )
    hold_gps  = gp_order.tail(3)["GrandPrix"]
    mask_hold = df["GrandPrix"].isin(hold_gps)

    X_tune, y_tune = (X_full.loc[~mask_hold].reset_index(drop=True),
                      y_full.loc[~mask_hold].reset_index(drop=True))

    X_tune_lap = X_tune.assign(
        LapNumber = df.loc[~mask_hold, "LapNumber"].reset_index(drop=True),
        GrandPrix = df.loc[~mask_hold, "GrandPrix"].reset_index(drop=True)
    )

    def strict_gp_ts_splits(X_lap: pd.DataFrame, n_splits=3):
        for _, gp_frame in X_lap.groupby("GrandPrix"):
            idx = gp_frame.index.to_numpy()
            idx = idx[np.argsort(gp_frame["LapNumber"].values)]
            for tr, va in TimeSeriesSplit(n_splits=n_splits).split(idx):
                yield idx[tr], idx[va]
    cv_iter = list(strict_gp_ts_splits(X_tune_lap, n_splits=3))

    full_pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), cat_cols)
    ])

    hgbr_pipe = Pipeline([
        ("pre", full_pre),
        ("reg", HistGradientBoostingRegressor(random_state=42, early_stopping=True))
    ])
    hg_grid = {
        "reg__learning_rate":    [0.05, 0.1],
        "reg__max_depth":        [4, 6],
        "reg__max_iter":         [200, 300],
        "reg__l2_regularization":[0.0, 1e-3, 1e-2]
    }
    hgbr_cv = GridSearchCV(hgbr_pipe, hg_grid,
                           cv=cv_iter, scoring="neg_mean_absolute_error",
                           n_jobs=-1, verbose=1)
    hgbr_cv.fit(X_tune, y_tune)
    best_hgbr = hgbr_cv.best_estimator_
    print(f"HGBR strict-CV MAE: {-hgbr_cv.best_score_:.3f}s")

    X_tune_pre = full_pre.fit_transform(X_tune)
    lgbm = LGBMRegressor(random_state=42, categorical_feature=cat_idx)
    lgb_grid = {
        "num_leaves":[31,63], "min_data_in_leaf":[20,50],
        "feature_fraction":[0.8,1.0]
    }
    lgb_cv = GridSearchCV(lgbm, lgb_grid,
                          cv=cv_iter, scoring="neg_mean_absolute_error",
                          n_jobs=-1, verbose=1)
    lgb_cv.fit(X_tune_pre, y_tune)
    best_lgb = lgb_cv.best_estimator_
    print(f"LGBM strict-CV MAE: {-lgb_cv.best_score_:.3f}s")

    # Seq-to-one LSTM
    df["Compound_code"], comp_categories = pd.factorize(df["Compound"])
    SEQ_LEN = 5
    SEQ_FEAT = ["RacePct","Position","TyreLife","PrevLapDT",
        "PackCount_S3","GapToLeader",
        "LapTime_lag1","LapTime_lag2","LapTime_lag3","DirtyAir_lag1", "DirtyAir", "Compound_code"]

    def make_seqs(sub):
        Xs, ys = [], []
        for (_, _), g in sub.groupby(["GrandPrix","Driver"]):
            arr, tar = g[SEQ_FEAT].values, g["LapTimeSeconds"].values
            for i in range(len(arr) - SEQ_LEN):
                Xs.append(arr[i:i+SEQ_LEN]); ys.append(tar[i+SEQ_LEN])
        return np.stack(Xs), np.array(ys)

    X_seq_tune, y_seq_tune = make_seqs(df.loc[~mask_hold])
    ns, t, nf = X_seq_tune.shape
    seq_scaler = StandardScaler().fit(X_seq_tune.reshape(-1, nf))
    X_seq_tune = seq_scaler.transform(X_seq_tune.reshape(-1, nf)).reshape(ns, t, nf)

    lstm_ds = TensorDataset(torch.from_numpy(X_seq_tune).float(),
                            torch.from_numpy(y_seq_tune).float())
    n_val = int(0.2 * len(lstm_ds))
    train_ds, val_ds = random_split(
        lstm_ds, [len(lstm_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_ld = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_ld   = DataLoader(val_ds, batch_size=32)

    class LapLSTM(nn.Module):
        def __init__(self, in_feat, hid=64, fc=32):
            super().__init__()
            self.lstm = nn.LSTM(in_feat, hid, batch_first=True)
            self.drop = nn.Dropout(0.2)
            self.fc1  = nn.Linear(hid, fc)
            self.out  = nn.Linear(fc, 1)
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            h = self.drop(h.squeeze(0))
            return self.out(torch.relu(self.fc1(h))).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm = LapLSTM(nf).to(device)
    opt  = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    best_v, patience, wait = np.inf, 10, 0
    for epoch in range(1, 101):
        lstm.train(); tloss = 0
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); out = lstm(xb)
            loss = loss_fn(out, yb); loss.backward(); opt.step()
            tloss += loss.item()
        lstm.eval(); vloss = 0
        with torch.no_grad():
            for xb, yb in val_ld:
                vloss += loss_fn(lstm(xb.to(device)), yb.to(device)).item()
        tloss /= len(train_ld); vloss /= len(val_ld)
        print(f"[{epoch:03}] train {tloss:.3f}  val {vloss:.3f}")
        if vloss < best_v:
            best_v, wait = vloss, 0
            torch.save(lstm.state_dict(), "best_lstm.pt")
        else:
            wait += 1
            if wait >= patience:
                print("→ early stopping"); break
    print(f"LSTM best val MAE: {best_v:.3f}s")

    return best_hgbr, best_lgb, lstm, seq_scaler, full_pre


# ──────────────────────────────────────────────────────────────
# CELL-7 · strict evaluation on hold-out races
# ──────────────────────────────────────────────────────────────
def evaluate_holdout(best_hgbr, best_lgb, lstm, seq_scaler, full_pre, all_laps_clean):
    df = (all_laps_clean.copy()
            .dropna(subset=["LapTimeSeconds"])
            .assign(IsFirstRaceOfSeason=lambda x: (x["Round"] == 1).astype(int)))

    need = [f"LapTime_lag{i}" for i in (1, 2, 3)] + ["DirtyAir_lag1"]
    df = df.dropna(subset=need).reset_index(drop=True)

    features = [
        "RacePct","Position","TyreLife","PrevLapDT", "stint_lap",
        "PackCount_S3","GapToLeader","LastRacePace","IsFirstRaceOfSeason",
        "LapTime_lag1","LapTime_lag2","LapTime_lag3","DirtyAir_lag1", "DirtyAir",
        "Driver","Compound","Team","LeadLapTime"
    ]
    

    X_full, y_full = df[features], df["LapTimeSeconds"]

    gp_order = (
        df[["Season","Round","GrandPrix"]]
          .drop_duplicates()
          .sort_values(["Season","Round"])
    )
    hold_gps  = gp_order.tail(3)["GrandPrix"]
    mask_hold = df["GrandPrix"].isin(hold_gps)

    
    X_hold, y_hold = X_full.loc[mask_hold], y_full.loc[mask_hold]

    h_pred = best_hgbr.predict(X_hold)
    l_pred    = best_lgb.predict(full_pre.transform(X_hold))
    blend  = 0.5 * h_pred + 0.5 * l_pred

    print(f"HGBR hold-out MAE  : {mean_absolute_error(y_hold, h_pred):.3f}s")
    print(f"LGBM hold-out MAE  : {mean_absolute_error(y_hold, l_pred):.3f}s")
    print(f"Blend hold-out MAE : {mean_absolute_error(y_hold, blend ):.3f}s")

    # Seq evaluation
    df["Compound_code"], comp_categories = pd.factorize(df["Compound"])
    SEQ_LEN = 5
    SEQ_FEAT = ["RacePct","Position","TyreLife","PrevLapDT",
        "PackCount_S3","GapToLeader",
        "LapTime_lag1","LapTime_lag2","LapTime_lag3","DirtyAir_lag1", "DirtyAir", "Compound_code"]

    def make_seqs(sub):
        Xs, ys = [], []
        for (_, _), g in sub.groupby(["GrandPrix","Driver"]):
            arr, tar = g[SEQ_FEAT].values, g["LapTimeSeconds"].values
            for i in range(len(arr) - SEQ_LEN):
                Xs.append(arr[i:i+SEQ_LEN]); ys.append(tar[i+SEQ_LEN])
        return np.stack(Xs), np.array(ys)

    X_seq_hold, y_seq_hold = make_seqs(df.loc[mask_hold])
    ns, t, nf = X_seq_hold.shape
    X_seq_hold = seq_scaler.transform(X_seq_hold.reshape(-1, nf)).reshape(ns, t, nf)

    hold_ds = TensorDataset(torch.from_numpy(X_seq_hold).float(),
                            torch.from_numpy(y_seq_hold).float())
    hold_ld = DataLoader(hold_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm.load_state_dict(torch.load("best_lstm.pt"))
    lstm.eval(); preds = []
    with torch.no_grad():
        for xb, _ in hold_ld:
            preds.append(lstm(xb.to(device)).cpu())
    preds = torch.cat(preds).numpy()
    mae = mean_absolute_error(y_seq_hold, preds)
    print(f"LSTM hold-out MAE  : {mae:.3f}s")

