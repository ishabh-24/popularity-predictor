from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from src.data import DatasetSpec, add_time_features, coerce_types, load_dataset, validate_dataset
from src.logreg_shap import shap_summary_for_logreg_pipeline
from src.modeling import RandomForestTrainConfig, TrainConfig, train_evaluate_baseline, train_evaluate_random_forest
from src.nn_modeling import NeuralNetTrainConfig, train_evaluate_neural_net
from src.nn_shap import shap_summary_for_hitnet_bundle
from src.random_forest_shap import shap_summary_for_random_forest_pipeline


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
            "duration_ms",
        ),
        categorical_cols=("genre",),
    )


def load_default_df() -> pd.DataFrame:
    # Prefer the larger merged dataset if present; fall back to sample.
    default_path = Path("data/kaggle_billboard_songs.csv")
    if not default_path.exists():
        default_path = Path("data/sample_songs.csv")
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
   
    AUDIO_FEATURES = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence",
        "tempo", "duration_ms",
    ]
    cols = [c for c in AUDIO_FEATURES if c in df.columns]
    num = df[cols].copy()

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

    n = len(corr.columns)
    font_size = max(6, min(10, int(180 / n)))

    fig.update_layout(
        height=max(500, n * 28),
        margin=dict(l=160, b=160, t=40, r=40),
        font=dict(size=font_size),
    )

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


def empty_shap_figure(message: str = "SHAP summary unavailable.") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=message,
        xaxis_title="mean |SHAP value|",
        yaxis_title="Feature",
        height=420,
    )
    return fig


def make_shap_bar_figure(shap_rows: list[dict[str, float | str]], title: str) -> go.Figure:
    if not shap_rows:
        return empty_shap_figure("No SHAP values available.")
    shap_df = pd.DataFrame(shap_rows).sort_values("mean_abs_shap", ascending=True)
    fig = px.bar(
        shap_df,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title=title,
        labels={"mean_abs_shap": "mean |SHAP value|", "feature": "Transformed feature"},
    )
    fig.update_layout(height=420, margin=dict(l=160, r=24, t=48, b=36))
    return fig


app = Dash(__name__)
app.title = "Popularity Predictor"
DEFAULT_DATASET_PATH = (
    Path("data/kaggle_billboard_songs.csv")
    if Path("data/kaggle_billboard_songs.csv").exists()
    else Path("data/sample_songs.csv")
)

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
        html.H2("Global Music Chart Success Predictor"),
        html.Div(
            children=[
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
                                            value=str(DEFAULT_DATASET_PATH),
                                            style={"width": "100%"},
                                        ),
                                    ],
                                ),
                                html.Button("Load dataset", id="btn-load", n_clicks=0),
                                html.Div(
                                    style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                    children=[
                                        html.Label("Model", style={"fontSize": "13px", "fontWeight": "600"}),
                                        dcc.RadioItems(
                                            id="train-model-type",
                                            options=[
                                                {"label": "Logistic regression (baseline)", "value": "logreg"},
                                                {"label": "Random forest", "value": "rf"},
                                                {"label": "Neural net (HitNet / Lightning)", "value": "nn"},
                                            ],
                                            value="logreg",
                                            labelStyle={"display": "block"},
                                        ),
                                    ],
                                ),
                                html.Button("Train model", id="btn-train", n_clicks=0),
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
                                html.H4("1) Feature distribution", style={"margin": 0}),
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
                        html.H4("2) Danceability vs Energy", style={"marginTop": 0}),
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
                        html.H4("3) Feature Correlation Heatmap", style={"marginTop": 0}),
                        dcc.Graph(id="corr-graph"),
                    ],
                ),
                html.Div(
                    style={"border": "1px solid #eee", "borderRadius": "8px", "padding": "12px"},
                    children=[
                        html.H4("Hit rate over time", style={"marginTop": 0}),
                        dcc.Graph(id="time-graph"),
                    ],
                ),
            ],
        ),
        html.H3("Model training output"),
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
        html.Div(
            style={"marginTop": "16px", "border": "1px solid #eee", "borderRadius": "8px", "padding": "12px"},
            children=[
                html.H4("Explainability (SHAP)", style={"marginTop": 0}),
                dcc.Graph(id="shap-summary-graph", figure=empty_shap_figure()),
            ],
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
    Output("shap-summary-graph", "figure"),
    Input("btn-train", "n_clicks"),
    State("df-store", "data"),
    State("train-model-type", "value"),
    prevent_initial_call=True,
)
def train_model_callback(n_clicks: int, df_json, model_type: str):
    try:
        df = df0 if not df_json else pd.read_json(StringIO(df_json), orient="split")
        if "release_date" in df.columns:
            df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

        if "is_hit" not in df.columns and "billboard_matched" in df.columns:
            df = prepare_kaggle_billboard_df(df)

        spec = kaggle_billboard_spec() if "billboard_matched" in df.columns else DatasetSpec()
        if model_type == "rf":
            result = train_evaluate_random_forest(
                df,
                spec=spec,
                cfg=RandomForestTrainConfig(use_smote=True),
            )
            header = "Random forest (all release years by default; more hits in train/test than logreg window)"
            extra_keys = ["n_estimators", "max_depth"]
            shap_rows = shap_summary_for_random_forest_pipeline(
                result["pipeline"],
                result["X_train"],
                result["X_test"],
                random_state=42,
                max_background=300,
                max_explain=400,
                top_k=15,
            )
            shap_fig = make_shap_bar_figure(shap_rows, "SHAP summary for random forest (top features)")
        elif model_type == "nn":
            result = train_evaluate_neural_net(df, spec=spec, cfg=NeuralNetTrainConfig())
            header = "HitNet (PyTorch Lightning; ColumnTransformer like baseline, no SMOTE)"
            extra_keys = ["batch_size", "epochs", "lr", "input_dim", "decision_threshold"]
            shap_rows = shap_summary_for_hitnet_bundle(
                result["pipeline"],
                result["X_train"],
                result["X_test"],
                random_state=42,
                max_background=200,
                max_explain=250,
                top_k=15,
            )
            shap_fig = make_shap_bar_figure(shap_rows, "SHAP summary for HitNet neural net (top features)")
        else:
            result = train_evaluate_baseline(
                df,
                spec=spec,
                cfg=TrainConfig(use_smote=True),
            )
            header = "Logistic regression (baseline, preprocessing, optional genre one-hot + SMOTE)"
            extra_keys = []
            shap_rows = shap_summary_for_logreg_pipeline(
                result["pipeline"],
                result["X_train"],
                result["X_test"],
                random_state=42,
                max_background=300,
                max_explain=400,
                top_k=15,
            )
            shap_fig = make_shap_bar_figure(shap_rows, "SHAP summary for logistic regression (top features)")

        m = result["metrics"]
        out = [header, ""]
        for k in [
            "n_rows",
            "recent_years_window",
            "n_features_numeric",
            "n_features_categorical",
            *extra_keys,
            "accuracy",
            "f1",
            "precision",
            "recall",
            "roc_auc",
        ]:
            if k in m:
                out.append(f"{k}: {m.get(k)}")
        out.append("")
        out.append(f"confusion_matrix: {m.get('confusion_matrix')}")
        out.append("")
        out.append("classification_report:")
        out.append(m.get("classification_report", ""))
        return "\n".join(out), shap_fig
    except Exception as e:
        return f"Training failed: {e}", empty_shap_figure("SHAP summary unavailable due to training error.")


if __name__ == "__main__":
    app.run(debug=True)

