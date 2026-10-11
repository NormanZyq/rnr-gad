from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

from src.models.components.metrics import compute_binary_eval_metrics, precision_at_anomaly_k
from src.models.components.rank_utils import spearman_corr
from src.models.components.rnr_gad import (
    CrossFittedResult,
    CrossFittedScorer,
    NormalityBackbone,
    RankSafeProjector,
    RobustNormalityRefiner,
    _cfg_get,
    _enabled_view_names,
    _section_get,
)


def _binary_evaluation_view(
    batch: Any,
    score_pre: torch.Tensor,
    score_final: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return label-bearing nodes for binary metrics.

    Most datasets evaluate every node. Sources such as Elliptic attach an
    ``eval_mask`` so unknown labels can participate in unsupervised inference
    without being silently treated as normal examples during evaluation.
    """
    score_pre = score_pre.reshape(-1)
    score_final = score_final.reshape(-1)
    y = y.reshape(-1)
    if not (score_pre.numel() == score_final.numel() == y.numel()):
        raise ValueError(
            "Evaluation tensors must have the same number of nodes: "
            f"score_pre={score_pre.numel()}, score_final={score_final.numel()}, y={y.numel()}."
        )

    eval_mask = getattr(batch, "eval_mask", None)
    if eval_mask is None:
        eval_mask = torch.ones(y.numel(), dtype=torch.bool, device=y.device)
    else:
        eval_mask = eval_mask.reshape(-1).to(device=y.device, dtype=torch.bool)
        if eval_mask.numel() != y.numel():
            raise ValueError(
                f"eval_mask has {eval_mask.numel()} entries for {y.numel()} nodes."
            )
    if not eval_mask.any():
        raise ValueError("eval_mask excludes every node; binary metrics are undefined.")

    score_mask = eval_mask.to(score_pre.device)
    return score_pre[score_mask], score_final[score_mask], y[eval_mask], eval_mask


class RNRGADLitModule(LightningModule):
    """Lightning module for Rank-Safe Robust Normality Refinement GAD."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        cfg: DictConfig,
        compile: bool = False,
    ) -> None:
        super().__init__()
        cfg_to_save = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
        self.save_hyperparameters({"cfg": cfg_to_save, "compile": compile}, logger=False)
        self.automatic_optimization = False

        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.model: Optional[NormalityBackbone] = None
        self.scorer: Optional[CrossFittedScorer] = None
        self.refiner: Optional[RobustNormalityRefiner] = None
        self.projector: Optional[RankSafeProjector] = None

        self.stage2_initialized = False
        self._active_outer_idx = -1
        self._test_outputs: list[Dict[str, torch.Tensor]] = []
        self._test_rank_outputs: list[torch.Tensor] = []

        self.stage1_result: Optional[CrossFittedResult] = None
        self.score_base: Optional[torch.Tensor] = None
        self.score_current: Optional[torch.Tensor] = None
        self.confidence: Optional[torch.Tensor] = None
        self.view_weights: Optional[torch.Tensor] = None
        self.view_scales: Optional[torch.Tensor] = None
        self.node_weights: Optional[torch.Tensor] = None
        self.current_objective: Optional[torch.Tensor] = None
        self.stage2_eligible: Optional[torch.Tensor] = None
        self.best_refine_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        self._pending_decoder_reference: Optional[Dict[str, torch.Tensor]] = None
        self._pending_best_refine_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

    @property
    def cfg(self) -> Any:
        return self.hparams.cfg

    def _build_model(self, in_dim: int) -> None:
        if self.model is not None:
            return
        # Keep the backbone unwrapped so Lightning hooks can call its task methods.
        self.model = NormalityBackbone(in_dim=in_dim, cfg=self.cfg)
        self.scorer = CrossFittedScorer(self.cfg)
        self.refiner = RobustNormalityRefiner(self.cfg)
        self.projector = RankSafeProjector(
            epsilon_max=float(_section_get(self.cfg, "rank_safety", "epsilon_max", 0.05)),
            enabled=bool(_section_get(self.cfg, "rank_safety", "enabled", True)),
            max_pairs=int(_section_get(self.cfg, "rank_safety", "max_pairs", 200_000)),
        )
        self._restore_loaded_runtime_state()

    def _restore_loaded_runtime_state(self) -> None:
        if self.stage2_initialized and self.model is not None:
            self.model.freeze_encoder()
        if self.refiner is not None and self._pending_decoder_reference is not None:
            self.refiner.decoder_reference = {
                key: value.detach().clone()
                for key, value in self._pending_decoder_reference.items()
            }
        if self._pending_best_refine_state is not None:
            self.best_refine_state = self._pending_best_refine_state

    def _infer_in_dim_from_datamodule(self) -> Optional[int]:
        if _cfg_get(self.cfg, "in_dim", None) is not None:
            return int(_cfg_get(self.cfg, "in_dim", None))
        datamodule = getattr(getattr(self, "trainer", None), "datamodule", None)
        num_features = getattr(datamodule, "num_features", None)
        if num_features:
            return int(num_features)
        data = getattr(datamodule, "data", None)
        if data is not None and getattr(data, "x", None) is not None:
            return int(data.x.size(-1))
        return None

    def setup(self, stage: str) -> None:
        in_dim = self._infer_in_dim_from_datamodule()
        if in_dim is not None:
            self._build_model(in_dim)

    def _ensure_model_for_batch(self, batch: Any) -> NormalityBackbone:
        if self.model is None:
            if getattr(batch, "x", None) is None:
                raise ValueError("Cannot infer model input dimension because batch.x is missing.")
            self._build_model(int(batch.x.size(-1)))
        assert isinstance(self.model, NormalityBackbone)
        assert self.scorer is not None
        assert self.refiner is not None
        assert self.projector is not None
        return self.model

    def _pretrain_epochs(self) -> int:
        return int(_cfg_get(self.cfg, "pretrain_epochs", 100))

    def _inner_epochs(self) -> int:
        return max(1, int(_section_get(self.cfg, "refinement", "inner_epochs", 10)))

    def _outer_steps(self) -> int:
        return max(1, int(_section_get(self.cfg, "refinement", "outer_steps", 3)))

    def _stage2_epoch_index(self) -> int:
        return max(0, int(self.current_epoch) - self._pretrain_epochs())

    def _outer_idx(self) -> int:
        return min(self._stage2_epoch_index() // self._inner_epochs(), self._outer_steps() - 1)

    def _stage2_is_eligible(self) -> bool:
        if self.stage2_eligible is None:
            return True
        return bool((self.stage2_eligible > 0.5).detach().cpu().item())

    def _maybe_clip_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        clip_value = _section_get(self.cfg, "optimization", "gradient_clip_val", None)
        if clip_value is None or float(clip_value) <= 0.0:
            return
        self.clip_gradients(
            optimizer,
            gradient_clip_val=float(clip_value),
            gradient_clip_algorithm=str(
                _section_get(self.cfg, "optimization", "gradient_clip_algorithm", "norm")
            ),
        )

    def _pretrain_loss_dict(self, data: Any) -> Dict[str, torch.Tensor]:
        model = self._ensure_model_for_batch(data)
        feature_mask_rate = float(_section_get(self.cfg, "crossfit", "feature_mask_rate", 0.3))
        edge_holdout_rate = float(_section_get(self.cfg, "crossfit", "edge_holdout_rate", 0.1))
        neg_ratio = float(_section_get(self.cfg, "crossfit", "neg_ratio", 1.0))
        task = model.sample_task_batch(
            data,
            feature_mask_rate=feature_mask_rate,
            edge_holdout_rate=edge_holdout_rate,
            neg_ratio=neg_ratio,
            eval_mode=False,
        )
        losses = model.nodewise_task_losses(data, task, use_adapter=False)
        view_names = _enabled_view_names(self.cfg)
        loss = losses["attribute"].sum() * 0.0
        log_dict = {}
        for name in view_names:
            if name == "structure" and losses["edge_loss"].numel() > 0:
                view_loss = losses["edge_loss"].mean()
            else:
                view_loss = losses[name].mean()
            weight = float(_section_get(self.cfg, "pretrain", f"lambda_{name}", 1.0))
            loss = loss + weight * view_loss
            log_dict[f"{name}_loss"] = view_loss.detach()
        log_dict["loss"] = loss
        return log_dict

    def _initialize_stage2(self, data: Any) -> None:
        model = self._ensure_model_for_batch(data)
        assert self.scorer is not None and self.refiner is not None

        model.freeze_encoder()
        self.refiner.snapshot_reference(model)
        result = self.scorer.compute(model, data, use_adapter=False)
        self.stage1_result = result
        self.score_base = result.score.detach()
        self.score_current = result.score.detach().clone()
        self.confidence = result.confidence.detach()
        self.view_weights = result.train_view_weights.detach()
        self.view_scales = result.train_view_scales.detach()
        self.stage2_eligible = result.diagnostics["stage2_eligible"].detach()
        self.current_objective = self.refiner.robust_objective(
            result,
            self.view_weights,
            self.view_scales,
            model,
        )
        self.best_refine_state = self.refiner.trainable_state(model)
        self.stage2_initialized = True
        self._active_outer_idx = -1

        self.log_dict(
            {f"stage1/{key}": value for key, value in result.diagnostics.items()},
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            "stage1/heldout_robust_objective",
            self.current_objective,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

    def _start_outer_step(self, data: Any, outer_idx: int) -> None:
        model = self._ensure_model_for_batch(data)
        assert self.scorer is not None and self.refiner is not None
        assert self.view_weights is not None and self.view_scales is not None

        result = self.scorer.compute(model, data, use_adapter=True)
        node_weights, node_loss, tau = self.refiner.node_weights_from_result(
            result,
            self.view_weights,
            self.view_scales,
        )
        self.node_weights = node_weights.detach()
        self._active_outer_idx = outer_idx
        self.log_dict(
            {
                "refine/outer_idx": torch.tensor(float(outer_idx), device=node_weights.device),
                "refine/robust_tau": tau,
                "refine/node_loss_median": node_loss.median(),
                "refine/effective_normal_weight": node_weights.mean(),
                "refine/weight_min": node_weights.min(),
                "refine/weight_p10": torch.quantile(node_weights.float(), 0.10),
            },
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

    def _finish_outer_step(self, data: Any, outer_idx: int) -> None:
        model = self._ensure_model_for_batch(data)
        assert self.scorer is not None and self.refiner is not None and self.projector is not None
        assert self.score_base is not None and self.score_current is not None
        assert self.confidence is not None and self.view_weights is not None and self.view_scales is not None

        previous_score = self.score_current.detach()
        candidate_result = self.scorer.compute(model, data, use_adapter=True)
        candidate_score = self.scorer.aggregate_with_weights(candidate_result, self.view_weights)
        projected = self.projector.project(self.score_base, candidate_score, self.confidence)
        candidate_objective = self.refiner.robust_objective(
            candidate_result,
            self.view_weights,
            self.view_scales,
            model,
        )
        tolerance = float(_section_get(self.cfg, "selection", "accept_tolerance", 1e-8))
        if self.current_objective is None:
            accept = True
        else:
            accept = bool((candidate_objective <= self.current_objective + tolerance).detach().cpu().item())

        if accept:
            self.current_objective = candidate_objective.detach()
            self.score_current = projected["score"].detach()
            self.best_refine_state = self.refiner.trainable_state(model)
        elif self.best_refine_state is not None:
            self.refiner.load_trainable_state(model, self.best_refine_state)

        final_score = self.score_current if accept else previous_score
        self.log_dict(
            {
                "refine/outer_accept": torch.tensor(float(accept), device=final_score.device),
                "refine/robust_objective": candidate_objective,
                "refine/spearman_prev": spearman_corr(previous_score, final_score),
                "refine/spearman_base": spearman_corr(self.score_base, final_score),
                "refine/rank_flip_ratio": projected["rank_flip_ratio"],
                "refine/max_rank_displacement": projected["max_rank_displacement"],
                "refine/protected_pair_violations": projected["protected_pair_violations"],
                "refine/outer_completed": torch.tensor(float(outer_idx + 1), device=final_score.device),
            },
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        self._active_outer_idx = -1

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        model = self._ensure_model_for_batch(batch)
        optimizer = self.optimizers()

        if self.current_epoch < self._pretrain_epochs():
            loss_dict = self._pretrain_loss_dict(batch)
            optimizer.zero_grad()
            self.manual_backward(loss_dict["loss"])
            self._maybe_clip_gradients(optimizer)
            optimizer.step()
            self.log_dict(
                {f"train/pretrain/{key}": value for key, value in loss_dict.items()},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=1,
            )
            return loss_dict["loss"]

        if not self.stage2_initialized:
            self._initialize_stage2(batch)

        if not self._stage2_is_eligible():
            zero = batch.x.float().sum() * 0.0
            self.log_dict(
                {
                    "train/refine/loss": zero.detach(),
                    "refine/skipped_by_eligibility": zero.detach() + 1.0,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=1,
            )
            return zero

        outer_idx = self._outer_idx()
        if self._active_outer_idx != outer_idx or self.node_weights is None:
            self._start_outer_step(batch, outer_idx)

        assert self.refiner is not None
        assert self.node_weights is not None and self.view_weights is not None and self.view_scales is not None
        loss, diagnostics = self.refiner.training_objective(
            model,
            batch,
            node_weights=self.node_weights,
            view_weights=self.view_weights,
            view_scales=self.view_scales,
        )
        optimizer.zero_grad()
        self.manual_backward(loss)
        self._maybe_clip_gradients(optimizer)
        optimizer.step()

        self.log_dict(
            {"train/refine/loss": loss, **{f"train/refine/{key}": value for key, value in diagnostics.items()}},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )

        is_outer_end = (self._stage2_epoch_index() + 1) % self._inner_epochs() == 0
        if is_outer_end:
            self._finish_outer_step(batch, outer_idx)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        if self.current_epoch < self._pretrain_epochs():
            with torch.no_grad():
                loss_dict = self._pretrain_loss_dict(batch)
            monitor_loss = loss_dict["loss"].detach().new_tensor(1_000_000.0 - float(self.current_epoch))
            self.log_dict(
                {
                    "val/loss": monitor_loss,
                    "val/pretrain_loss": loss_dict["loss"].detach(),
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=1,
            )
            return

        if not self.stage2_initialized:
            self._initialize_stage2(batch)
        assert self.current_objective is not None
        self.log("val/loss", self.current_objective, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log(
            "val/heldout_robust_objective",
            self.current_objective,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

    def _scores_for_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        model = self._ensure_model_for_batch(batch)
        assert self.scorer is not None and self.projector is not None
        num_nodes = int(batch.num_nodes)
        device = batch.x.device

        if self.score_base is not None and self.score_base.numel() == num_nodes:
            score_base = self.score_base.to(device)
            score_final = (
                self.score_current.to(device)
                if self.score_current is not None and self.score_current.numel() == num_nodes
                else score_base
            )
            confidence = (
                self.confidence.to(device)
                if self.confidence is not None and self.confidence.numel() == num_nodes
                else torch.ones_like(score_base)
            )
            return {
                "score_pre": score_base,
                "score_final": score_final,
                "confidence": confidence,
            }

        base_result = self.scorer.compute(model, batch, use_adapter=False)
        candidate_result = self.scorer.compute(model, batch, use_adapter=True)
        candidate_score = self.scorer.aggregate_with_weights(candidate_result, base_result.view_weights)
        projected = self.projector.project(base_result.score, candidate_score, base_result.confidence)
        return {
            "score_pre": base_result.score.detach(),
            "score_final": projected["score"].detach(),
            "confidence": base_result.confidence.detach(),
        }

    def on_test_start(self) -> None:
        self._test_outputs = []
        self._test_rank_outputs = []

    def _save_test_outputs_enabled(self) -> bool:
        return bool(_cfg_get(self.cfg, "save_test_outputs", True))

    def test_step(self, batch: Any, batch_idx: int) -> None:
        out = self._scores_for_batch(batch)
        all_node_spearman = spearman_corr(out["score_pre"], out["score_final"])
        diagnostics = {
            "spearman_stage1_final": all_node_spearman,
            "rank_plasticity_all_nodes": 1.0 - all_node_spearman,
            "confidence_mean": out["confidence"].mean(),
        }
        self.log_dict(
            {f"test/{key}": value for key, value in diagnostics.items()},
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

        if self._save_test_outputs_enabled():
            self._test_outputs.append(
                {
                    key: value.detach().cpu()
                    for key, value in out.items()
                    if isinstance(value, torch.Tensor)
                }
            )

        y = getattr(batch, "y", None)
        if y is None or y.numel() != out["score_final"].numel():
            return

        score_pre, score_final, y_eval, eval_mask = _binary_evaluation_view(
            batch,
            out["score_pre"].detach(),
            out["score_final"].detach(),
            y,
        )
        metrics = compute_binary_eval_metrics(score_pre, score_final, y_eval)
        pair_metrics = self._useful_harmful_flip_metrics(score_pre, score_final, y_eval)
        metrics = {
            **metrics,
            "recall_at_k": precision_at_anomaly_k(score_final, (y_eval.reshape(-1) > 0).long()),
            "eval_node_count": score_final.new_tensor(int(eval_mask.sum().item()), dtype=torch.float32),
            "eval_anomaly_count": score_final.new_tensor(
                int((y_eval.reshape(-1) > 0).sum().item()), dtype=torch.float32
            ),
            **pair_metrics,
        }
        self.log_dict(
            {f"test/{key}": value for key, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

        if self._save_test_outputs_enabled():
            y_bin = (y_eval.reshape(-1) > 0).long().detach().cpu()
            ranks = self._descending_ranks(score_final.detach().cpu())
            self._test_rank_outputs.append(ranks[y_bin > 0])

    def _descending_ranks(self, score: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(score.reshape(-1), descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(1, score.numel() + 1, dtype=torch.long, device=score.device)
        return ranks

    def _useful_harmful_flip_metrics(
        self,
        score_pre: torch.Tensor,
        score_final: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        y = (y.reshape(-1) > 0).long()
        anomaly_idx = torch.nonzero(y > 0, as_tuple=False).reshape(-1)
        normal_idx = torch.nonzero(y == 0, as_tuple=False).reshape(-1)
        if anomaly_idx.numel() == 0 or normal_idx.numel() == 0:
            nan = score_final.new_tensor(float("nan"))
            return {
                "useful_flip_count": nan,
                "harmful_flip_count": nan,
                "net_useful_flip": nan,
                "auc_gain_pair_decomp": nan,
            }

        total_pairs = int(anomaly_idx.numel() * normal_idx.numel())
        max_pairs = int(_cfg_get(self.cfg, "test_pair_sample_max", 200_000))
        if total_pairs <= max_pairs:
            anomaly = anomaly_idx.repeat_interleave(normal_idx.numel())
            normal = normal_idx.repeat(anomaly_idx.numel())
            scale = score_final.new_tensor(1.0)
        else:
            sample_count = max_pairs
            anomaly = anomaly_idx[torch.randint(anomaly_idx.numel(), (sample_count,), device=y.device)]
            normal = normal_idx[torch.randint(normal_idx.numel(), (sample_count,), device=y.device)]
            scale = score_final.new_tensor(float(total_pairs) / float(sample_count))

        pre_correct = score_pre[anomaly] > score_pre[normal]
        final_correct = score_final[anomaly] > score_final[normal]
        useful = (~pre_correct & final_correct).float().sum() * scale
        harmful = (pre_correct & ~final_correct).float().sum() * scale
        net = useful - harmful
        return {
            "useful_flip_count": useful,
            "harmful_flip_count": harmful,
            "net_useful_flip": net,
            "auc_gain_pair_decomp": net / float(total_pairs),
        }

    def on_test_epoch_end(self) -> None:
        if not self._save_test_outputs_enabled() or not self._test_outputs:
            return

        output_dir = os.path.join(self.trainer.default_root_dir, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        keys = self._test_outputs[0].keys()
        for key in keys:
            value = torch.cat([item[key].reshape(-1) for item in self._test_outputs], dim=0)
            np.save(os.path.join(output_dir, f"{key}.npy"), value.numpy())
        if self._test_rank_outputs:
            ranks = torch.cat([item.reshape(-1) for item in self._test_rank_outputs], dim=0)
            np.save(os.path.join(output_dir, "per_anomaly_rank.npy"), ranks.numpy())

    def predict_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        out = self._scores_for_batch(batch)
        return {key: value.detach().cpu() for key, value in out.items()}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state = {}
        state_keys = (
            "score_base",
            "score_current",
            "confidence",
            "view_weights",
            "view_scales",
            "node_weights",
            "current_objective",
            "stage2_eligible",
        )
        for key in state_keys:
            value = getattr(self, key, None)
            if isinstance(value, torch.Tensor):
                state[key] = value.detach().cpu()
        checkpoint["rnr_state"] = state
        checkpoint["stage2_initialized"] = self.stage2_initialized
        if self.refiner is not None and self.refiner.decoder_reference is not None:
            checkpoint["rnr_decoder_reference"] = {
                key: value.detach().cpu()
                for key, value in self.refiner.decoder_reference.items()
            }
        if self.best_refine_state is not None:
            checkpoint["rnr_best_refine_state"] = {
                module_name: {
                    key: value.detach().cpu()
                    for key, value in module_state.items()
                }
                for module_name, module_state in self.best_refine_state.items()
            }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state = checkpoint.get("rnr_state", {})
        for key, value in state.items():
            setattr(self, key, value)
        self.stage2_initialized = bool(checkpoint.get("stage2_initialized", False))
        self._pending_decoder_reference = checkpoint.get("rnr_decoder_reference")
        self._pending_best_refine_state = checkpoint.get("rnr_best_refine_state")
        self._active_outer_idx = -1
        self._restore_loaded_runtime_state()

    def configure_optimizers(self) -> Dict[str, Any]:
        in_dim = self._infer_in_dim_from_datamodule()
        if self.model is None:
            if in_dim is None:
                raise RuntimeError(
                    "RNRGADLitModule could not infer in_dim. Set model.cfg.in_dim or use a "
                    "datamodule exposing num_features."
                )
            self._build_model(in_dim)

        optimizer = self.optimizer_factory(params=self.parameters())
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
