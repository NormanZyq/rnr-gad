from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from src.models.components.rnr_gad_v31 import (
    RNRv31AttentionInputs,
    RNRv31AttentionScorer,
    RNRv31Result,
    attention_self_supervised_loss,
)
from src.models.rnr_gad_v3_module import RNRGADV3LitModule


class RNRGADV31LitModule(RNRGADV3LitModule):
    """RNRv3.1 with trainable self-attention arbitration over stage evidence."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        cfg: DictConfig,
        compile: bool = False,
    ) -> None:
        super().__init__(optimizer=optimizer, scheduler=scheduler, cfg=cfg, compile=compile)
        self.v31_scorer: Optional[RNRv31AttentionScorer] = None
        self.stage31_result: Optional[RNRv31Result] = None
        self.attention_inputs: Optional[RNRv31AttentionInputs] = None
        self.attention_target_inputs: Optional[RNRv31AttentionInputs] = None
        self.attention_target_score: Optional[torch.Tensor] = None
        self.attention_target_confidence: Optional[torch.Tensor] = None
        self.stage1_confidence: Optional[torch.Tensor] = None

    def _build_model(self, in_dim: int) -> None:
        super()._build_model(in_dim)
        if self.v31_scorer is None:
            self.v31_scorer = RNRv31AttentionScorer(self.cfg)

    def _attention_epochs(self) -> int:
        return max(1, int(self.cfg_get("attention", "epochs", int(self.cfg_get_root("correction_epochs", 10)))))

    def cfg_get_root(self, key: str, default: Any) -> Any:
        cfg = self.cfg
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def cfg_get(self, section: str, key: str, default: Any) -> Any:
        cfg = self.cfg
        section_cfg = cfg.get(section, None) if hasattr(cfg, "get") else getattr(cfg, section, None)
        if section_cfg is None:
            return default
        if hasattr(section_cfg, "get"):
            return section_cfg.get(key, default)
        return getattr(section_cfg, key, default)

    def _initialize_stage2(self, data: Any) -> None:
        model = self._ensure_model_for_batch(data)
        assert self.scorer is not None
        if self.v3_scorer is None:
            from src.models.components.rnr_gad_v3 import RNRv3Scorer

            self.v3_scorer = RNRv3Scorer(self.cfg)
        if self.v31_scorer is None:
            self.v31_scorer = RNRv31AttentionScorer(self.cfg)

        model.freeze_encoder()
        stage1 = self.scorer.compute(model, data, use_adapter=False)
        stage2 = self.v3_scorer.compute(data, stage1)
        inputs = self.v31_scorer.build_inputs(stage1, stage2)

        if self.v31_scorer.reliability_enabled:
            target_masks = int(
                self.cfg_get("attention", "num_target_masks", self.cfg_get("crossfit", "num_eval_masks", 4))
            )
            target_stage1 = self.scorer.compute(model, data, num_masks=target_masks, use_adapter=False)
            target_mode = str(self.cfg_get("attention", "target_source", "heldout_consensus"))
            target_stage2 = None
            if target_mode != "heldout_stage1":
                target_stage2 = self.v3_scorer.compute(data, target_stage1)
            build_target_inputs = bool(self.cfg_get("attention", "build_target_inputs", True))
            if build_target_inputs and target_stage2 is None:
                target_stage2 = self.v3_scorer.compute(data, target_stage1)
            target_score = self.v31_scorer.target_score(target_stage1, target_stage2)
            target_confidence = target_stage1.confidence.detach()
            target_inputs = (
                self.v31_scorer.build_inputs(target_stage1, target_stage2) if target_stage2 is not None else None
            )
        else:
            target_score = self.v31_scorer.target_score(stage1, stage2)
            target_confidence = torch.ones_like(stage1.confidence)
            target_inputs = inputs

        result = self.v31_scorer.compute(data, stage1, stage2)
        self.stage1_result = stage1
        self.stage2_result = stage2
        self.stage31_result = result
        self.attention_inputs = inputs
        self.attention_target_inputs = target_inputs
        self.attention_target_score = target_score.detach()
        self.attention_target_confidence = target_confidence.detach()
        self.stage1_confidence = stage1.confidence.detach()
        self.score_base = stage1.score.detach()
        self.score_current = result.score.detach().clone()
        self.confidence = result.confidence.detach()
        self.stage2_eligible = stage2.diagnostics["rnrv3/eligible"].detach()
        self.current_objective = self._attention_objective_detached()
        self.stage2_initialized = True
        self._active_outer_idx = -1

        self.log_dict(
            {f"stage1/{key}": value for key, value in stage1.diagnostics.items()},
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        self.log_dict(stage2.diagnostics, on_step=False, on_epoch=True, batch_size=1)
        self.log_dict(result.diagnostics, on_step=False, on_epoch=True, batch_size=1)
        self.log("val/rnrv31_objective", self.current_objective, on_step=False, on_epoch=True, batch_size=1)

    def _attention_objective_detached(self) -> torch.Tensor:
        assert self.v31_scorer is not None
        assert self.attention_inputs is not None
        assert self.attention_target_score is not None
        assert self.attention_target_confidence is not None
        was_training = self.v31_scorer.training
        self.v31_scorer.eval()
        with torch.no_grad():
            out_a = self.v31_scorer.forward_inputs(self.attention_inputs)
            out_b = self.v31_scorer.forward_inputs(self.attention_inputs)
            loss_dict = attention_self_supervised_loss(
                out_a,
                out_b,
                self.attention_target_score,
                self.attention_target_confidence,
                self.cfg,
                stage1_score=(
                    self.stage1_result.score
                    if self.stage1_result is not None and self.v31_scorer.direct_anchor_enabled
                    else None
                ),
                stage2_score=(
                    self.stage2_result.score
                    if self.stage2_result is not None and self.v31_scorer.graph_evidence_enabled
                    else None
                ),
                inputs=self.attention_inputs,
                target_inputs=self.attention_target_inputs,
            )
        if was_training:
            self.v31_scorer.train()
        return loss_dict["loss"].detach()

    def _refresh_attention_score(self) -> Dict[str, torch.Tensor]:
        assert self.v31_scorer is not None
        assert self.attention_inputs is not None
        assert self.stage1_result is not None
        assert self.stage2_result is not None
        was_training = self.v31_scorer.training
        self.v31_scorer.eval()
        with torch.no_grad():
            result = self.v31_scorer.compute(
                data=None,
                stage1=self.stage1_result,
                stage2=self.stage2_result,
            )
        if was_training:
            self.v31_scorer.train()
        self.stage31_result = result
        self.score_current = result.score.detach().clone()
        self.confidence = result.confidence.detach()
        self.current_objective = self._attention_objective_detached()
        return result.diagnostics

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

        assert self.v31_scorer is not None
        assert self.attention_inputs is not None
        assert self.attention_target_score is not None
        assert self.attention_target_confidence is not None

        model.eval()
        self.v31_scorer.train()
        out_a = self.v31_scorer.forward_inputs(self.attention_inputs)
        out_b = self.v31_scorer.forward_inputs(self.attention_inputs)
        loss_dict = attention_self_supervised_loss(
            out_a,
            out_b,
            self.attention_target_score,
            self.attention_target_confidence,
            self.cfg,
            stage1_score=(
                self.stage1_result.score
                if self.stage1_result is not None and self.v31_scorer.direct_anchor_enabled
                else None
            ),
            stage2_score=(
                self.stage2_result.score
                if self.stage2_result is not None and self.v31_scorer.graph_evidence_enabled
                else None
            ),
            inputs=self.attention_inputs,
            target_inputs=self.attention_target_inputs,
        )
        loss = loss_dict["loss"]
        optimizer.zero_grad()
        self.manual_backward(loss)
        self._maybe_clip_gradients(optimizer)
        optimizer.step()

        diagnostics = self._refresh_attention_score()
        self.log_dict(
            {
                "train/rnrv31/loss": loss.detach(),
                **{f"train/rnrv31/{key}": value for key, value in loss_dict.items() if key != "loss"},
                **{f"train/{key}": value for key, value in diagnostics.items()},
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )
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
        self.log("val/rnrv31_objective", self.current_objective, on_step=False, on_epoch=True, batch_size=1)

    def _scores_for_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        model = self._ensure_model_for_batch(batch)
        assert self.scorer is not None
        if self.v3_scorer is None:
            from src.models.components.rnr_gad_v3 import RNRv3Scorer

            self.v3_scorer = RNRv3Scorer(self.cfg)
        if self.v31_scorer is None:
            self.v31_scorer = RNRv31AttentionScorer(self.cfg)

        num_nodes = int(batch.num_nodes)
        device = batch.x.device
        if self.score_base is not None and self.score_base.numel() == num_nodes:
            score_base = self.score_base.to(device)
            stage1_confidence = self._stage1_confidence_for_batch(
                model,
                batch,
                num_nodes=num_nodes,
                device=device,
            )
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
            out = {
                "score_pre": score_base,
                "score_final": score_final,
                "confidence": confidence,
                "stage1_confidence": stage1_confidence,
            }
            if (
                self.stage2_result is not None
                and self.stage2_result.candidate_score.numel() == num_nodes
            ):
                out["score_graph_candidate"] = self.stage2_result.candidate_score.to(device)
            return out

        stage1 = self.scorer.compute(model, batch, use_adapter=False)
        stage2 = self.v3_scorer.compute(batch, stage1)
        result = self.v31_scorer.compute(batch, stage1, stage2)
        self.stage1_confidence = stage1.confidence.detach()
        return {
            "score_pre": stage1.score.detach(),
            "score_final": result.score.detach(),
            "score_graph_candidate": stage2.candidate_score.detach(),
            "confidence": result.confidence.detach(),
            "stage1_confidence": stage1.confidence.detach(),
        }

    def _stage1_confidence_for_batch(
        self,
        model: Any,
        batch: Any,
        *,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the Stage-1 scorer confidence without altering model scores."""
        cached = self.stage1_confidence
        if cached is None and self.stage1_result is not None:
            cached = self.stage1_result.confidence
        if cached is not None and cached.numel() == num_nodes:
            return cached.detach().to(device)

        with torch.no_grad():
            stage1 = self.scorer.compute(model, batch, use_adapter=False)
        self.stage1_confidence = stage1.confidence.detach()
        return self.stage1_confidence.to(device)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        state = {}
        if self.stage31_result is not None:
            state.update(
                {
                    "token_names": list(self.stage31_result.token_names),
                    "attention_weights": self.stage31_result.attention_weights.detach().cpu(),
                    "source_ids": self.stage31_result.source_ids.detach().cpu(),
                    "orientation_ids": self.stage31_result.orientation_ids.detach().cpu(),
                }
            )
        stage1_confidence = self.stage1_confidence
        if stage1_confidence is None and self.stage1_result is not None:
            stage1_confidence = self.stage1_result.confidence
        if stage1_confidence is not None:
            state["stage1_confidence"] = stage1_confidence.detach().cpu()
        if state:
            checkpoint["rnrv31_state"] = state

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        state = checkpoint.get("rnrv31_state", {})
        stage1_confidence = state.get("stage1_confidence")
        if isinstance(stage1_confidence, torch.Tensor):
            self.stage1_confidence = stage1_confidence.detach()

    @property
    def cfg(self) -> Any:
        raw = self.hparams.cfg
        return OmegaConf.create(raw) if isinstance(raw, dict) else raw
