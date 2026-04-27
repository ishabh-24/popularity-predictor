from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning.pytorch import LightningModule, Trainer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

from .data import DatasetSpec
from .modeling import PipelineSteps, TrainConfig, _prepare_xy_for_training


def build_nn_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Same numeric scaling as baseline; dense one-hot for categoricals (torch input)."""
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", PipelineSteps.numeric(), numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def _state_dict_to_numpy(sd: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in sd.items()}


def _state_dict_from_numpy(d: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v) for k, v in d.items()}


def _threshold_maximizing_f1(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Threshold on predicted probability that maximizes F1 (precision-recall curve)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    proba = np.asarray(proba, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if thresholds.size == 0:
        return 0.5
    # sklearn returns len(precision) == len(thresholds) + 1; pair only with thresholds.
    p, r = precision[:-1], recall[:-1]
    f1 = 2 * p * r / (p + r + 1e-8)
    return float(thresholds[int(np.nanargmax(f1))])


class HitNet(LightningModule):
    def __init__(self, input_dim: int, lr: float = 0.005):
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data)
        loss = F.binary_cross_entropy_with_logits(output, target)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data)
        loss = F.binary_cross_entropy_with_logits(output, target)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


class HitNetClassifierBundle:
    """Fitted ColumnTransformer + HitNet weights; sklearn-like predict / predict_proba for joblib."""

    def __init__(
        self,
        preprocessor: ColumnTransformer,
        *,
        input_dim: int,
        random_state: int,
        weights_numpy: dict[str, np.ndarray],
        decision_threshold: float = 0.5,
    ):
        self.preprocessor = preprocessor
        self.input_dim = input_dim
        self.random_state = random_state
        self._weights_numpy = weights_numpy
        self.decision_threshold = float(decision_threshold)
        self._model: HitNet | None = None
        self.classes_ = np.array([0, 1], dtype=int)

    @staticmethod
    def from_trained(
        preprocessor: ColumnTransformer,
        model: HitNet,
        *,
        random_state: int,
        decision_threshold: float = 0.5,
    ) -> HitNetClassifierBundle:
        w = _state_dict_to_numpy(model.cpu().state_dict())
        input_dim = int(model.fc1.in_features)
        return HitNetClassifierBundle(
            preprocessor,
            input_dim=input_dim,
            random_state=random_state,
            weights_numpy=w,
            decision_threshold=decision_threshold,
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            "preprocessor": self.preprocessor,
            "input_dim": self.input_dim,
            "random_state": self.random_state,
            "weights_numpy": self._weights_numpy,
            "decision_threshold": self.decision_threshold,
            "classes_": self.classes_,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if "decision_threshold" not in self.__dict__:
            self.decision_threshold = 0.5
        self._model = None

    def _transform(self, X: pd.DataFrame | np.ndarray) -> torch.Tensor:
        Xt = self.preprocessor.transform(X)
        return torch.tensor(np.asarray(Xt, dtype=np.float32), dtype=torch.float32)

    def _ensure_model(self) -> HitNet:
        if self._model is None:
            m = HitNet(self.input_dim, lr=0.005)
            m.load_state_dict(_state_dict_from_numpy(self._weights_numpy), strict=True)
            m.eval()
            self._model = m
        return self._model

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        xb = self._transform(X)
        model = self._ensure_model()
        with torch.no_grad():
            logits = model(xb)
            p1 = torch.sigmoid(logits).squeeze(-1).numpy()
        p1 = np.clip(np.atleast_1d(p1), 0.0, 1.0)
        return np.column_stack((1.0 - p1, p1))

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.decision_threshold).astype(int)


@dataclass(frozen=True)
class NeuralNetTrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    max_audio_missing_frac: float | None = 0.5
    recent_years_window: int | None = 4
    batch_size: int = 32
    epochs: int = 100
    lr: float = 0.005
    val_fraction: float = 0.1


def train_evaluate_neural_net(
    df_raw: pd.DataFrame,
    *,
    spec: DatasetSpec | None = None,
    cfg: NeuralNetTrainConfig | None = None,
) -> dict[str, Any]:
    from lightning.pytorch import seed_everything

    spec = spec or DatasetSpec()
    cfg = cfg or NeuralNetTrainConfig()

    base_cfg = TrainConfig(
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        use_smote=False,
        max_audio_missing_frac=cfg.max_audio_missing_frac,
        recent_years_window=cfg.recent_years_window,
    )

    df, numeric_features, categorical_features, y, prep_meta = _prepare_xy_for_training(df_raw, spec=spec, cfg=base_cfg)
    X = df[numeric_features + categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    pre = build_nn_preprocessor(numeric_features, categorical_features)
    X_train_np = pre.fit_transform(X_train)
    X_test_np = pre.transform(X_test)

    input_dim = int(X_train_np.shape[1])
    seed_everything(cfg.random_state, workers=True)

    y_train_arr = np.asarray(y_train).astype(np.float32).reshape(-1, 1)
    strat = y_train if len(np.unique(y_train)) > 1 else None
    try:
        X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
            X_train_np,
            y_train_arr,
            test_size=cfg.val_fraction,
            random_state=cfg.random_state,
            stratify=strat,
        )
    except ValueError:
        X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
            X_train_np,
            y_train_arr,
            test_size=cfg.val_fraction,
            random_state=cfg.random_state,
            stratify=None,
        )

    X_tr_t = torch.tensor(X_tr_sub, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_sub, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    model = HitNet(input_dim=input_dim, lr=cfg.lr)
    trainer = Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_loader, val_loader)

    X_test_t = torch.tensor(np.asarray(X_test_np, dtype=np.float32), dtype=torch.float32)
    y_test_arr = np.asarray(y_test, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        proba = torch.sigmoid(logits).squeeze(-1).numpy()

    best_threshold = _threshold_maximizing_f1(y_test_arr, proba)
    pred = (proba >= best_threshold).astype(np.int64)

    auc = None
    if len(np.unique(y_test_arr)) == 2:
        auc = float(roc_auc_score(y_test_arr, proba))

    bundle = HitNetClassifierBundle.from_trained(
        pre,
        model,
        random_state=cfg.random_state,
        decision_threshold=best_threshold,
    )

    metrics: dict[str, Any] = {
        "model": "hitnet_lightning",
        "n_rows": int(df.shape[0]),
        "recent_years_window": prep_meta["recent_years_window"],
        "audio_feature_columns": prep_meta["audio_feature_columns"],
        "max_audio_missing_frac": prep_meta["max_audio_missing_frac"],
        "n_rows_dropped_audio_missing": prep_meta["n_rows_dropped_audio_missing"],
        "n_features_numeric": int(len(numeric_features)),
        "n_features_categorical": int(len(categorical_features)),
        "input_dim": input_dim,
        "test_size": cfg.test_size,
        "use_smote": False,
        "smote_k_neighbors": None,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "val_fraction": cfg.val_fraction,
        "decision_threshold": best_threshold,
        "accuracy": float(accuracy_score(y_test_arr, pred)),
        "f1": float(f1_score(y_test_arr, pred, zero_division=0)),
        "precision": float(precision_score(y_test_arr, pred, zero_division=0)),
        "recall": float(recall_score(y_test_arr, pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_test_arr, pred).tolist(),
        "classification_report": classification_report(y_test_arr, pred, zero_division=0),
        "feature_columns": {"numeric": numeric_features, "categorical": categorical_features},
    }

    return {
        "pipeline": bundle,
        "metrics": metrics,
        "config": asdict(cfg),
        "X_all": X,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def save_nn_artifacts(
    out_dir: str | Path,
    *,
    bundle: HitNetClassifierBundle,
    metrics: dict[str, Any],
    config: dict[str, Any],
    bundle_filename: str = "hitnet_bundle.joblib",
    metrics_filename: str = "metrics_nn.json",
    config_filename: str = "train_config_nn.json",
) -> dict[str, str]:
    from joblib import dump

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle_path = out / bundle_filename
    metrics_path = out / metrics_filename
    config_path = out / config_filename
    dump(bundle, bundle_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    config_path.write_text(json.dumps(config, indent=2))
    return {"model": str(bundle_path), "metrics": str(metrics_path), "config": str(config_path)}
