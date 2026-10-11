from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from src.models.components.rnr_gad_v3 import RNRv3Result, RNRv3Scorer
from src.models.rnr_gad_module import RNRGADLitModule


class RNRGADV3LitModule(RNRGADLitModule):
    """RNRv3 with bipolar evidence-field propagation initialization."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        cfg: DictConfig,
        compile: bool = False,
    ) -> None:
        super().__init__(optimizer=optimizer, scheduler=scheduler, cfg=cfg, compile=compile)
        self.v3_scorer: Optional[RNRv3Scorer] = None
        self.stage2_result: Optional[RNRv3Result] = None

    def _build_model(self, in_dim: int) -> None:
        super()._build_model(in_dim)
        if self.v3_scorer is None:
            self.v3_scorer = RNRv3Scorer(self.cfg)

    def _initialize_stage2(self, data: Any) -> None:
        model = self._ensure_model_for_batch(data)
        assert self.scorer is not None
        if self.v3_scorer is None:
            self.v3_scorer = RNRv3Scorer(self.cfg)

        model.freeze_encoder()
        stage1 = self.scorer.compute(model, data, use_adapter=False)
        result = self.v3_scorer.compute(data, stage1)

        self.stage1_result = stage1
        self.stage2_result = result
        self.score_base = stage1.score.detach()
        self.score_current = result.score.detach().clone()
        self.confidence = result.confidence.detach()
        self.stage2_eligible = result.diagnostics["rnrv3/eligible"].detach()
        self.current_objective = (
            result.component_scores.to(result.component_weights)
            * result.component_weights.reshape(1, -1)
        ).sum(dim=1).mean().detach()
        self.stage2_initialized = True
        self._active_outer_idx = -1

        self.log_dict(
            {f"stage1/{key}": value for key, value in stage1.diagnostics.items()},
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        self.log_dict(
            result.diagnostics,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            "rnrv3/objective",
            self.current_objective,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

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

        zero = batch.x.float().sum() * 0.0
        self.log_dict(
            {
                "train/rnrv3/loss": zero.detach(),
                "rnrv3/nonparametric_stage2": zero.detach() + 1.0,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )
        return zero

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
        self.log("val/rnrv3_objective", self.current_objective, on_step=False, on_epoch=True, batch_size=1)

    def _scores_for_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        model = self._ensure_model_for_batch(batch)
        assert self.scorer is not None
        if self.v3_scorer is None:
            self.v3_scorer = RNRv3Scorer(self.cfg)

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
            return {"score_pre": score_base, "score_final": score_final, "confidence": confidence}

        stage1 = self.scorer.compute(model, batch, use_adapter=False)
        result = self.v3_scorer.compute(batch, stage1)
        return {
            "score_pre": stage1.score.detach(),
            "score_final": result.score.detach(),
            "confidence": result.confidence.detach(),
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        if self.stage2_result is not None:
            checkpoint["rnrv3_state"] = {
                "component_names": list(self.stage2_result.component_names),
                "component_weights": self.stage2_result.component_weights.detach().cpu(),
                "component_probabilities": self.stage2_result.component_probabilities.detach().cpu(),
                "raw_component_names": list(self.stage2_result.raw_component_names),
            }

    @property
    def cfg(self) -> Any:
        raw = self.hparams.cfg
        return OmegaConf.create(raw) if isinstance(raw, dict) else raw
