from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from dash import Dash, Input, Output, State, dcc, html

from src.config import Paths
from src.data import DatasetSpec, add_time_features, coerce_types, load_dataset, validate_dataset
from src.modeling import TrainConfig, train_evaluate_baseline


def prepare_kaggle_billboard_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make kaggle_billboard_songs.csv compatible with the shared model/EDA code:
    - derive `is_hit` from `billboard_matched`
    - synthesize `release_date` from `release_year` (Jan 1 of that year)
    """
    out = df.copy()
    if "is_hit" not in out.columns and "billboard_matched" in out.columns:
        out["is_hit"] = pd.to_numeric(out["billboard_matched"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    if "release_date" not in out.columns and "release_year" in out.columns:
        yr = pd.to_numeric(out["release_year"], errors="coerce")
        out["release_date"] = pd.to_datetime(yr.astype("Int64").astype(str) + "-01-01", errors="coerce")
    return out


def kaggle_billboard_spec() -> DatasetSpec:
    return DatasetSpec(
        target_col="is_hit",
        id_cols=("track_name", "artist_name"),
        date_col="release_date",
        numeric_cols=(
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "spotify_popularity",
            "duration_ms",
        ),
        categorical_cols=("genre",),
    )


def load_default_df() -> pd.DataFrame:
    paths = Paths.default()
    # Prefer the larger merged dataset if present; fall back to sample.
    default_path = paths.data_dir / "kaggle_billboard_songs.csv"
    if not default_path.exists():
        default_path = paths.data_dir / "sample_songs.csv"
    df = load_dataset(default_path)
    if default_path.name == "kaggle_billboard_songs.csv":
        df = prepare_kaggle_billboard_df(df)
        spec = kaggle_billboard_spec()
    else:
        spec = DatasetSpec()
    df = add_time_features(coerce_types(df, spec), spec)
    return df


def make_histogram(df: pd.DataFrame, feature: str) -> px.histogram:
    fig = px.histogram(
        df,
        x=feature,
        color="is_hit",
        barmode="overlay",
        nbins=25,
        title=f"Distribution of {feature} (Hit vs Miss)",
        labels={"is_hit": "Hit (1) / Miss (0)"},
    )
    fig.update_layout(legend_title_text="is_hit")
    return fig


def make_scatter(df: pd.DataFrame) -> px.scatter:
    x = "danceability" if "danceability" in df.columns else df.select_dtypes("number").columns[0]
    y = "energy" if "energy" in df.columns else df.select_dtypes("number").columns[1]
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color="is_hit",
        hover_data=[c for c in ["track_name", "artist_name", "genre", "release_date"] if c in df.columns],
        title=f"{y} vs {x} (colored by hit)",
        labels={"is_hit": "Hit (1) / Miss (0)"},
    )
    fig.update_layout(legend_title_text="is_hit")
    return fig


def make_corr_heatmap(df: pd.DataFrame) -> ff.create_annotated_heatmap:
    num = df.select_dtypes("number").copy()
    # Drop target for correlation visuals (keeps focus on feature-feature structure)
    if "is_hit" in num.columns:
        num = num.drop(columns=["is_hit"])

    corr = num.corr(numeric_only=True).round(2)
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.astype(str).values,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        showscale=True,
    )
    fig.update_layout(title="Feature Correlation Heatmap (numeric features)")
    return fig


def make_hit_rate_over_time(df: pd.DataFrame) -> px.line:
    # Uses derived `release_year` if available; otherwise falls back to no-op.
    if "release_year" not in df.columns:
        return px.line(title="Hit rate over time (release_year missing)")

    tmp = (
        df.dropna(subset=["release_year", "is_hit"])
        .assign(release_year=lambda d: d["release_year"].astype(int))
        .groupby("release_year", as_index=False)["is_hit"]
        .mean()
        .rename(columns={"is_hit": "hit_rate"})
        .sort_values("release_year")
    )
    fig = px.line(tmp, x="release_year", y="hit_rate", markers=True, title="Hit rate over time")
    fig.update_yaxes(range=[0, 1])
    return fig


app = Dash(__name__)
app.title = "Popularity Predictor (Baseline)"

df0 = load_default_df()
spec0 = DatasetSpec()
issues0 = validate_dataset(df0, spec0)

numeric_candidates = [
    c
    for c in df0.columns
    if c in set(spec0.numeric_cols) | {"release_year", "release_month", "release_dow"} and pd.api.types.is_numeric_dtype(df0[c])
]
default_feature = "danceability" if "danceability" in numeric_candidates else (numeric_candidates[0] if numeric_candidates else "")


app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "24px auto", "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto"},
    children=[
        html.H2("Global Music Chart Success — Baseline Dashboard"),
        html.Div(
            children=[
                html.P(
                    "This intermediate scaffold uses a local dataset file (no API calls yet). "
                    "Swap in your final merged dataset later by pointing to it in the UI."
                ),
                html.Div(
                    style={"padding": "12px", "background": "#fafafa", "border": "1px solid #eee", "borderRadius": "8px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "12px", "alignItems": "end", "flexWrap": "wrap"},
                            children=[
                                html.Div(
                                    style={"minWidth": "420px", "flex": "1"},
                                    children=[
                                        html.Label("Dataset path (CSV/Parquet)"),
                                        dcc.Input(
                                            id="dataset-path",
                                            type="text",
                                            value=str(
                                                (Paths.default().data_dir / "kaggle_billboard_songs.csv")
                                                if (Paths.default().data_dir / "kaggle_billboard_songs.csv").exists()
                                                else (Paths.default().data_dir / "sample_songs.csv")
                                            ),
                                            style={"width": "100%"},
                                        ),
                                    ],
                                ),
                                html.Button("Load dataset", id="btn-load", n_clicks=0),
                                html.Button("Train baseline model", id="btn-train", n_clicks=0),
                            ],
                        ),
                        html.Div(id="load-status", style={"marginTop": "8px", "color": "#333"}),
                    ],
                ),
            ]
        ),
        dcc.Store(id="df-store"),
        html.H3("EDA"),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            children=[
                html.Div(
                    style={"border": "1px solid #eee", "borderRadius": "8px", "padding": "12px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "12px"},
                            children=[
                                html.H4("1) Feature distribution by hit", style={"margin": 0}),
                                dcc.Dropdown(
                                    id="feature-dropdown",
                                    options=[{"label": c, "value": c} for c in numeric_candidates],
                                    value=default_feature,
                                    clearable=False,
                                    style={"minWidth": "240px"},
                                ),
                            ],
                        ),
                        dcc.Graph(id="hist-graph"),
                    ],
                ),
                html.Div(
                    style={"border": "1px solid #eee", "borderRadius": "8px", "padding": "12px"},
                    children=[
                        html.H4("2) Danceability vs Energy (hit coloring)", style={"marginTop": 0}),
                        dcc.Graph(id="scatter-graph"),
                    ],
                ),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginTop": "16px"},
            children=[
                html.Div(
                    style={"border": "1px solid #eee", "borderRadius": "8px", "padding": "12px"},
                    children=[
                        html.H4("3) Correlation heatmap", style={"marginTop": 0}),
                        dcc.Graph(id="corr-graph"),
                    ],
                ),
                html.Div(
                    style={"border": "1px solid #eee", "borderRadius": "8px", "padding": "12px"},
                    children=[
                        html.H4("Bonus: Hit rate over time", style={"marginTop": 0}),
                        dcc.Graph(id="time-graph"),
                    ],
                ),
            ],
        ),
        html.H3("Baseline model output"),
        html.Pre(
            id="train-output",
            style={
                "whiteSpace": "pre-wrap",
                "background": "#0b1020",
                "color": "#e6edf3",
                "padding": "12px",
                "borderRadius": "8px",
                "border": "1px solid #111827",
                "minHeight": "120px",
            },
        ),
    ],
)


@app.callback(
    Output("df-store", "data"),
    Output("load-status", "children"),
    Input("btn-load", "n_clicks"),
    State("dataset-path", "value"),
    prevent_initial_call=True,
)
def load_dataset_callback(n_clicks: int, dataset_path: str):
    try:
        df = load_dataset(Path(dataset_path))
        # Auto-detect common dataset types.
        if "billboard_matched" in df.columns and "danceability" in df.columns:
            df = prepare_kaggle_billboard_df(df)
            spec = kaggle_billboard_spec()
        else:
            spec = DatasetSpec()
        df = add_time_features(coerce_types(df, spec), spec)
        issues = validate_dataset(df, spec)
        if issues:
            return None, "Loaded, but validation issues:\n- " + "\n- ".join(issues)
        return df.to_json(date_format="iso", orient="split"), f"Loaded dataset: {dataset_path} (rows={len(df)})"
    except Exception as e:
        return None, f"Failed to load dataset: {e}"


@app.callback(
    Output("hist-graph", "figure"),
    Output("scatter-graph", "figure"),
    Output("corr-graph", "figure"),
    Output("time-graph", "figure"),
    Input("df-store", "data"),
    Input("feature-dropdown", "value"),
)
def update_eda(df_json, feature: str):
    df = df0 if not df_json else pd.read_json(StringIO(df_json), orient="split")
    # Ensure datetime restored
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    if not feature or feature not in df.columns:
        numeric = df.select_dtypes("number").columns.tolist()
        feature = "danceability" if "danceability" in numeric else (numeric[0] if numeric else None)

    hist = make_histogram(df, feature) if feature else px.histogram(title="No numeric feature available")
    scatter = make_scatter(df)
    corr = make_corr_heatmap(df)
    time_fig = make_hit_rate_over_time(df)
    return hist, scatter, corr, time_fig


@app.callback(
    Output("train-output", "children"),
    Input("btn-train", "n_clicks"),
    State("df-store", "data"),
    prevent_initial_call=True,
)
def train_baseline_callback(n_clicks: int, df_json):
    try:
        df = df0 if not df_json else pd.read_json(StringIO(df_json), orient="split")
        if "release_date" in df.columns:
            df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

        if "is_hit" not in df.columns and "billboard_matched" in df.columns:
            df = prepare_kaggle_billboard_df(df)

        spec = kaggle_billboard_spec() if "billboard_matched" in df.columns else DatasetSpec()
        result = train_evaluate_baseline(
            df,
            spec=spec,
            cfg=TrainConfig(use_smote=True),
        )
        m = result["metrics"]
        out = []
        out.append("Baseline: Logistic Regression (with preprocessing + optional genre one-hot + SMOTE)")
        out.append("")
        for k in ["n_rows", "n_features_numeric", "n_features_categorical", "accuracy", "f1", "precision", "recall", "roc_auc"]:
            out.append(f"{k}: {m.get(k)}")
        out.append("")
        out.append(f"confusion_matrix: {m.get('confusion_matrix')}")
        out.append("")
        out.append("classification_report:")
        out.append(m.get("classification_report", ""))
        return "\n".join(out)
    except Exception as e:
        return f"Training failed: {e}"


if __name__ == "__main__":
    app.run(debug=True)

