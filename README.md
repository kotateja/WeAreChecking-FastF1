# Formula 1 Data Analysis & Modeling with FastF1

## 1 · Project Overview
This repository provides a comprehensive pipeline for Formula 1 race data analysis, visualisation, and modelling using the FastF1 API. Its name, "We Are Checking", is from Ferrari’s famously ambiguous radio communication from recent seasons, often announcing "we are checking" during strategic confusion without providing clear answers. Unlike Ferrari’s pit wall, this project actually does the checking thoroughly. 

![Picsart_25-06-26_21-28-32-408](https://github.com/user-attachments/assets/84abb323-b6cf-4612-a89b-b34c1f8a3e75)

It fetches official F1 timing and telemetry data via the FastF1 library, preprocesses the data, and explores various critical aspects of race performance. Included analyses cover statistical evaluations of tyre performance, driver consistency, and race pace, alongside machine learning models to predict lap times. Rich visualisations (with interactive widgets) illustrate key outcomes, such as race results and tyre degradation, insights from ANOVA and mixed-model tests on lap times, driver style clustering from telemetry data, and predictive modelling (gradient boosting, LightGBM, LSTM) evaluated on hold-out races. All data is retrieved dynamically via the FastF1 API—no manual downloads required—and the entire analysis pipeline is fully reproducible with the provided code and Jupyter notebook. 

---

## 2 · Repository Layout

```
FLIN-FASTF1/
├─ data/                 ← auto-generated raw & processed Feather files
├─ notebooks/
│  └─ Fast F1 Data analysis + Modelling.ipynb
├─ src/
│  └─ flin/
│     ├─ etl_all.py        # season download + telemetry builder
│     ├─ preprocess.py     # cleaning pipeline
│     ├─ clean_features.py # engineered racing features
│     ├─ initial_eda.py    # quick EDA helpers
│     ├─ modelling.py      # stats tests, clustering, ML models
│     ├─ plotting.py       # publication-ready charts
│     ├─ widgets.py        # ipywidgets interactive viewer
│     └─ viz_config.py     # dark theme + custom F1 font
└─ requirements.txt
```

---

## 3 · Quick Start
**Running the Jupyter Notebook**: Open the Fast F1 Data analysis + Modelling.ipynb notebook in Jupyter. Step through each section to reproduce the analysis. The notebook will call the modules in the correct order. You can adjust the year in etl-all.py or notebook to analyze a different season (by default it might be set to 2024). Running the notebook will:

- Fetch the season’s lap data and save it under data/raw/.
- Preprocess and feature-engineer the data.
- Generate various plots (they will display inline) for performance, consistency, tyre behavior, etc.
- Print out tables and metrics from statistical tests and model training.
- Train ML models and output their evaluation results.
- Provide an interactive widget at the end for your own driver-vs-driver comparisons.

This is the recommended approach to see all results with explanatory text. Note: The first data fetch can take some time (several minutes) since it downloads all race data from the FastF1 API; subsequent runs will be faster if caching is enabled.
### 3.1 Set-up

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export FASTF1_CACHE=~/.fastf1       # speeds up repeat runs (optional)
```

Python 3.9+ recommended.

### 3.2 Run the notebook

Launch Jupyter and open  
`notebooks/Fast F1 Data analysis + Modelling.ipynb`  
→ executes the full pipeline end-to-end with explanations, plots, and widgets.

---

## 4 · Script-Level Usage Examples
**Using the Python Modules Directly**: If you prefer to run parts of the analysis or integrate it into other projects, you can use the modules programmatically.
### 4.1 Season download + preprocessing

```python
from etl_all import fetch_season, build_best_lap_telemetry
from preprocess import preprocess
from clean_features import add_advanced_features

laps_raw = fetch_season(2024)
telemetry = build_best_lap_telemetry(laps_raw)

laps_clean, *_ = preprocess(laps_raw)
laps_full = add_advanced_features(laps_clean)
```

### 4.2 Statistical tests & track evolution

```python
from modelling import run_stats_tests
anova_tbl, mixed_res, rubber_df = run_stats_tests(laps_full)
print(anova_tbl, rubber_df.head())
```

### 4.3 Driver clustering workflow

```python
from modelling import sweep_clustering, pca_scree, final_clustering_plot

metrics, driver_df, X, VI = sweep_clustering(laps_full)
fig, cum_var = pca_scree(driver_df)
final_clustering_plot(driver_df, X, k=4)
```

### 4.4 Strict modelling pipeline (**matches `modelling.py` logic**)

```python
from modelling import strict_train_tune, evaluate_holdout

best_hgbr, best_lgb, lstm, seq_scaler, full_pre = strict_train_tune(laps_full)

evaluate_holdout(
    best_hgbr,
    best_lgb,
    lstm,
    seq_scaler,
    full_pre,
    laps_full
)
```

### 4.5 Interactive race widget (in Jupyter)

```python
from widgets import show_race_comparison
show_race_comparison(laps_raw)
```

---
## 5 · Outputs and Results
By following the pipeline, you will obtain a variety of outputs that provide insights into Formula 1 race dynamics:
- Cleaned data + Feature Engineering: A curated dataset of laps with rich features (suitable for your own analyses or machine learning tasks).
- Visualisations: High-quality plots illustrating:

 | Plot                                 | Insight                                               |
|--------------------------------------|--------------------------------------------------------|
| **Average finishing position**       | Season-to-date ranking of classified results          |
| **Lap-time percentiles**             | P10 / P25 / P50 pace spread per driver                |
| **Consistency Plots**                 | Lap-time variability per driver & stint |
| **Compound analysis Plots** | Slick compound perfomance comparisions |
| **Tyre degradation heat-maps**       | β₁ slope (sec / lap) across tracks & drivers          |
| **Compound delta by GP**            | Soft-Medium & Medium-Hard performance gaps (adjusted) |
| **Pit-stop Analysis**                | Green-flag pit-lane performance by team and pit duration at different GPs              |
| **Driver style clusters**            | PCA + agglomerative clustering on telemetry features  |

- Statistical Analysis: Console outputs/tables including:

       - ANOVA table showing the significance of tyre compound and age on lap time.
       - Mixed-effects model summary quantifying variance between drivers.
       - Calculated “rubber-in” improvements for each track.

- Clustering Results: Metrics comparing clustering algorithms and a chosen clustering of drivers (e.g. identifying distinct driver archetypes like “late brakers” vs “smooth drivers” based on telemetry). A plotted cluster chart in principal component space helps visualize these groupings.
- Model Performance: Training logs and summary metrics:

       - Cross-validated Mean Absolute Error (MAE) for the gradient boosting and LightGBM models, indicating how well we can predict lap times.
       - Best validation MAE for the LSTM model, and its hold-out test MAE on unseen races (for example, you might see an LSTM hold-out MAE on the order of a few tenths of a second, showing accuracy of sequential predictions).
       - These metrics help evaluate whether the models capture the important factors in lap performance and how much unpredictability remains.
- Interactive Exploration: The ability to interactively compare drivers for any race, enhancing the qualitative analysis (for example, you can visually confirm where a strategy or pit stop gained one driver an advantage over another).


---

## 6 · Credits & Licence

**Data source:**  
Timing and telemetry from the [FastF1 API](https://theoehrly.github.io/Fast-F1/)  


**Font:**  
"Formula 1 Regular Web 0" by FerrariFan08 (CC-BY-NC)

**Author:**  
© 2025 [kotateja](https://github.com/kotateja)  
Licensed under MIT.

PRs, forks, stars, questions, and suggestions welcome!
