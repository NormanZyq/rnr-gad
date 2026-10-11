import csv
import gc
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
import numpy as np
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from torch.serialization import add_safe_globals
from omegaconf import DictConfig, OmegaConf, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pygod.nn.decoder import DotProductDecoder
from torch_geometric.nn import GCNConv, GATConv, GraphConv, SAGEConv
from torch_geometric.nn.aggr.basic import SumAggregation

# Checkpoint loading needs these PyG classes in the weights-only allowlist.
add_safe_globals(
    [
        DotProductDecoder,
        GCNConv,
        GATConv,
        GraphConv,
        SAGEConv,
        SumAggregation,
    ]
)

from src import utils

log = utils.get_pylogger(__name__)

METRIC_KEYS = (
    "test/auroc",
    "test/auprc",
    "test/ap",
    "test/spearman_pre_final",
    "test/auc_gain",
)
REQUIRED_PREDICTIONS = (
    "score_pre.npy",
    "score_final.npy",
    "score_graph_candidate.npy",
)


def _apply_phase_epoch_schedule(cfg: DictConfig) -> None:
    model_cfg = cfg.get("model", {}).get("cfg") if cfg.get("model") is not None else None
    if model_cfg is None or not model_cfg.get("use_phase_epochs", False):
        return

    pretrain_epochs = int(model_cfg.get("pretrain_epochs", 0))
    correction_epochs_cfg = model_cfg.get("correction_epochs", None)
    if correction_epochs_cfg is None and model_cfg.get("refinement") is not None:
        refinement_cfg = model_cfg.get("refinement")
        correction_epochs = int(refinement_cfg.get("outer_steps", 0)) * int(refinement_cfg.get("inner_epochs", 0))
    else:
        correction_epochs = int(correction_epochs_cfg or 0)
    max_epochs = pretrain_epochs + correction_epochs
    if max_epochs <= 0:
        return

    with open_dict(cfg):
        cfg.trainer.max_epochs = max_epochs
    log.info(
        "Using phase epoch schedule: "
        f"pretrain_epochs={pretrain_epochs}, "
        f"correction_epochs={correction_epochs}, "
        f"trainer.max_epochs={max_epochs}"
    )


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train and optionally test the current weights; return metrics and objects."""
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    _apply_phase_epoch_schedule(cfg)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    if cfg.get("seed") is not None and bool(cfg.get("reseed_after_model_init", True)):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info("Best ckpt path skipped (using current weights).")

    test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


def summarize(values, *, pct=True, ddof=0):
    arr = np.asarray(values, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=ddof)
    if pct:
        mean *= 100
        std *= 100
    return round(mean, 2), round(std, 2)


def to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _seed_prediction_dir(output_dir: Path, seed: int) -> Path:
    return output_dir / "predictions" / f"seed_{seed}"


def _seed_complete(output_dir: Path, seed: int) -> bool:
    seed_dir = _seed_prediction_dir(output_dir, seed)
    return (seed_dir / "seed_metrics.json").exists() and all(
        (seed_dir / filename).exists() for filename in REQUIRED_PREDICTIONS
    )


def _load_seed_metrics(output_dir: Path, seed: int) -> Dict[str, float]:
    path = _seed_prediction_dir(output_dir, seed) / "seed_metrics.json"
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    missing = [key for key in METRIC_KEYS if key not in payload]
    if missing:
        raise KeyError(f"Missing metric(s) {missing} in {path}")
    return {key: float(payload[key]) for key in METRIC_KEYS}


def _clear_current_predictions(output_dir: Path) -> None:
    prediction_dir = output_dir / "predictions"
    if not prediction_dir.exists():
        return
    for path in prediction_dir.glob("*.npy"):
        path.unlink()


def _save_seed_artifacts(
    output_dir: Path,
    seed: int,
    metric_dict: Dict[str, Any],
) -> None:
    prediction_dir = output_dir / "predictions"
    seed_dir = _seed_prediction_dir(output_dir, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        filename for filename in REQUIRED_PREDICTIONS if not (prediction_dir / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Seed {seed} did not save required prediction(s) {missing} in {prediction_dir}"
        )
    for path in prediction_dir.glob("*.npy"):
        shutil.copy2(path, seed_dir / path.name)

    scalar_metrics = {}
    for key, value in metric_dict.items():
        try:
            scalar_metrics[str(key)] = to_float(value)
        except (TypeError, ValueError, RuntimeError):
            pass
    with (seed_dir / "seed_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(scalar_metrics, stream, indent=2, sort_keys=True)


def _write_multi_seed_tables(output_dir: Path, rows: List[Dict[str, float]]) -> None:
    metric_path = output_dir / "multi_seed_metrics.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", *METRIC_KEYS])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "multi_seed_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["metric", "mean", "population_std", "n_seeds"],
        )
        writer.writeheader()
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in rows]
            writer.writerow(
                {
                    "metric": key,
                    "mean": float(np.mean(values)),
                    "population_std": float(np.std(values, ddof=0)),
                    "n_seeds": len(values),
                }
            )


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)

    seeds = list(range(10))
    output_dir = Path(str(cfg.paths.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = bool(cfg.get("multi_train_resume", True))

    metrics = {key: [] for key in METRIC_KEYS}
    rows = []

    for seed in seeds:
        if resume and _seed_complete(output_dir, seed):
            seed_metrics = _load_seed_metrics(output_dir, seed)
            print(f"seed {seed:5d}: reusing saved predictions", flush=True)
        else:
            _clear_current_predictions(output_dir)
            cfg_run = OmegaConf.merge(cfg, {"seed": seed})
            metric_dict, object_dict = train(cfg_run)

            missing = [key for key in METRIC_KEYS if key not in metric_dict]
            if missing:
                raise KeyError(
                    f"Missing metric(s) {missing}. Available keys: {list(metric_dict.keys())}"
                )
            _save_seed_artifacts(output_dir, seed, metric_dict)
            seed_metrics = {key: to_float(metric_dict[key]) for key in METRIC_KEYS}
            del object_dict
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for key in METRIC_KEYS:
            metrics[key].append(float(seed_metrics[key]))
        rows.append({"seed": seed, **seed_metrics})
        _write_multi_seed_tables(output_dir, rows)

    for i, seed in enumerate(seeds):
        parts = []
        for key in METRIC_KEYS:
            parts.append(f"{key.split('/')[-1]}: {metrics[key][i]:.4f}")
        print(f"seed {seed:5d}: " + ", ".join(parts))

    lines = []
    for key in METRIC_KEYS:
        m, s = summarize(metrics[key], pct=True)
        name = key.split("/")[-1]
        lines.append(f"{name}: {m:.2f} (±{s:.2f})")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
