# xgboost_model.py

from __future__ import annotations

import os
import sys
import yaml
import argparse
from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

# -------------------------------------------------------------------
# Local-project imports – ensure repo root is on PYTHONPATH
# -------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

from components.utils import split_data, preprocess_data  # noqa: E402
from components.metric import Metrics  # noqa: E402
from components.model import Models  # noqa: E402

# -------------------------------------------------------------------
# Dataset configuration
# -------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetConfig:
    file_path: str
    columns_to_remove: list[str]
    target_column: str
    description: str


DATASETS: Dict[str, DatasetConfig] = {
    "ASK_REDDIT": DatasetConfig(
        file_path=os.path.join(REPO_ROOT, "data", "preprocessed_reddit_data.csv"),
        columns_to_remove=["author"],
        target_column="active",
        description="AskReddit users labelled active vs inactive.",
    ),
    "STACK_OVERFLOW": DatasetConfig(
        file_path=os.path.join(REPO_ROOT, "data", "preprocessed_stackoverflow_data.csv"),
        columns_to_remove=["user_id", "display_name"],
        target_column="influential",
        description="Stack Overflow users labelled influential vs non-influential.",
    ),
}

# -------------------------------------------------------------------
# XGBoost training pipeline class
# -------------------------------------------------------------------
class XGBTrainer:
    """Handles data loading, preprocessing, tuning, training, and evaluation."""

    def __init__(
        self,
        cfg: DatasetConfig,
        set_index: int,
        config_path: str = os.path.join(REPO_ROOT, "components", "config.yaml"),
    ) -> None:
        self.cfg = cfg
        self.set_index = set_index
        self.config_path = config_path
        self.df: pd.DataFrame | None = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.best_model = None

    # --------------------------------------------------------------
    # Public workflow methods
    # --------------------------------------------------------------
    def run_full_pipeline(self) -> None:
        self._load_and_preprocess()
        self._split()
        self._tune_hyperparams()
        self._train_final()
        self._evaluate()

    # --------------------------------------------------------------
    # Internal steps
    # --------------------------------------------------------------
    def _load_and_preprocess(self) -> None:
        if not os.path.isfile(self.cfg.file_path):
            raise FileNotFoundError(f"Dataset not found: {self.cfg.file_path}")

        self.df = pd.read_csv(self.cfg.file_path)
        print(self.df.info())
        self.df = preprocess_data(self.df, self.cfg.columns_to_remove)
        print("Columns after preprocessing:", self.df.columns.tolist())

    def _split(self) -> None:
        X_train, X_test, y_train, y_test = split_data(
            self.df, target_column=self.cfg.target_column
        )
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    def _tune_hyperparams(self) -> None:
        # Handle class imbalance
        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count else 1
        print("Computed scale_pos_weight:", scale_pos_weight)

        # Load parameter grids
        with open(self.config_path, "r") as fh:
            config = yaml.safe_load(fh)

        try:
            param_dist = config["xgboost"]["sets"][self.set_index].copy()
        except IndexError:
            raise ValueError(
                f"Hyper-parameter set index {self.set_index} is out of range. "
                f"Available sets: 0-{len(config['xgboost']['sets']) - 1}"
            )

        param_dist["scale_pos_weight"] = [scale_pos_weight]
        print("Using param grid:", param_dist)

        xgb_model = Models.get_xgboost_model()
        search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_dist,
            n_iter=20,
            scoring="f1",
            cv=3,
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )
        search.fit(self.X_train, self.y_train)
        print("Best hyper-parameters:", search.best_params_)
        self.best_model = search.best_estimator_

    def _train_final(self) -> None:
        if self.best_model is None:
            raise RuntimeError("Hyper-parameter search must be run before training.")
        self.best_model.fit(self.X_train, self.y_train)

    def _evaluate(self) -> None:
        y_probs = self.best_model.predict_proba(self.X_test)[:, 1]
        # y_pred = self.best_model.predict(self.X_test)
        y_pred = (y_probs >= 0.5).astype(int)
        metrics: Dict[str, Any] = Metrics.compute_metrics(self.y_test, y_pred, y_probs)

        # Calculate loss
        loss = 1.0 - metrics["accuracy"]

        # Neatly display metrics
        print("\nTest Performance Metrics")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"F1_score:  {metrics['f1_score']:.3f}")
        print(f"Accuracy:  {metrics['accuracy']:.3f}")
        print(f"AUC:       {metrics['auc']:.3f}")
        print(f"Loss:      {loss:.3f}")



# -------------------------------------------------------------------
# CLI entry-point
# -------------------------------------------------------------------
def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model on the chosen dataset and parameter set."
    )
    parser.add_argument(
        "dataset",
        choices=DATASETS.keys(),
        help="Dataset key (e.g., ASK_REDDIT).",
    )
    parser.add_argument(
        "set_index",
        type=int,
        help="Index of the hyper-parameter set inside xgboost->sets in config.yaml",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=os.path.join(REPO_ROOT, "components", "config.yaml"),
        help="Optional path to YAML with XGBoost hyper-parameter grids.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli()
    cfg = DATASETS[args.dataset]
    print(f"\n▶ Selected dataset: {args.dataset} – {cfg.description}")

    trainer = XGBTrainer(cfg, args.set_index, args.config)
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()
