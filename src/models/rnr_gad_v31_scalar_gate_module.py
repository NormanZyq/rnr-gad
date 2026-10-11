from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig

from src.models.components.rnr_gad_v31_scalar_gate import (
    RNRv31ScalarGateInputs,
    RNRv31ScalarGateResult,
    RNRv31ScalarGateScorer,
    scalar_gate_self_supervised_loss,
)
from src.models.rnr_gad_v3_module import RNRGADV3LitModule
from src.models.rnr_gad_v31_module import RNRGADV31LitModule


class RNRGADV31ScalarGateLitModule(RNRGADV31LitModule):
    """Experimental low-capacity comparator for the v31 arbitration head."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        cfg: DictConfig,
        compile: bool = False,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            compile=compile,
        )
        self.stage31_result: Optional[RNRv31ScalarGateResult] = None
        self.attention_inputs: Optional[RNRv31ScalarGateInputs] = None
        self.attention_target_inputs: Optional[RNRv31ScalarGateInputs] = None

    def _build_model(self, in_dim: int) -> None:
        # Build the backbone without registering the Transformer scorer.
        RNRGADV3LitModule._build_model(self, in_dim)
        if not isinstance(self.v31_scorer, RNRv31ScalarGateScorer):
            self.v31_scorer = RNRv31ScalarGateScorer(self.cfg)

    def _scalar_loss_dict(
        self,
        output_a: Any,
        output_b: Any,
    ) -> Dict[str, torch.Tensor]:
        assert isinstance(self.v31_scorer, RNRv31ScalarGateScorer)
        assert isinstance(self.attention_inputs, RNRv31ScalarGateInputs)
        assert self.attention_target_score is not None
        assert self.attention_target_confidence is not None
        assert self.stage1_result is not None
        assert self.stage2_result is not None
        return scalar_gate_self_supervised_loss(
            output_a,
            output_b,
            self.attention_target_score,
            self.attention_target_confidence,
            self.cfg,
            stage1_score=self.stage1_result.score,
            stage2_score=self.stage2_result.score,
            inputs=self.attention_inputs,
            target_inputs=self.attention_target_inputs,
        )

    def _attention_objective_detached(self) -> torch.Tensor:
        assert isinstance(self.v31_scorer, RNRv31ScalarGateScorer)
        assert isinstance(self.attention_inputs, RNRv31ScalarGateInputs)
        was_training = self.v31_scorer.training
        self.v31_scorer.eval()
        with torch.no_grad():
            output_a = self.v31_scorer.forward_inputs(self.attention_inputs)
            output_b = self.v31_scorer.forward_inputs(self.attention_inputs)
            loss = self._scalar_loss_dict(output_a, output_b)["loss"].detach()
        if was_training:
            self.v31_scorer.train()
        return loss

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
                {
                    f"train/pretrain/{key}": value
                    for key, value in loss_dict.items()
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=1,
            )
            return loss_dict["loss"]

        if not self.stage2_initialized:
            self._initialize_stage2(batch)

        assert isinstance(self.v31_scorer, RNRv31ScalarGateScorer)
        assert isinstance(self.attention_inputs, RNRv31ScalarGateInputs)
        model.eval()
        self.v31_scorer.train()
        output_a = self.v31_scorer.forward_inputs(self.attention_inputs)
        output_b = self.v31_scorer.forward_inputs(self.attention_inputs)
        loss_dict = self._scalar_loss_dict(output_a, output_b)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        self.manual_backward(loss)
        self._maybe_clip_gradients(optimizer)
        optimizer.step()

        diagnostics = self._refresh_attention_score()
        self.log_dict(
            {
                "train/rnrv31_scalar_gate/loss": loss.detach(),
                **{
                    f"train/rnrv31_scalar_gate/{key}": value
                    for key, value in loss_dict.items()
                    if key != "loss"
                },
                **{f"train/{key}": value for key, value in diagnostics.items()},
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )
        return loss

    def _scores_for_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        model = self._ensure_model_for_batch(batch)
        assert self.scorer is not None
        if self.v3_scorer is None:
            from src.models.components.rnr_gad_v3 import RNRv3Scorer

            self.v3_scorer = RNRv3Scorer(self.cfg)
        if not isinstance(self.v31_scorer, RNRv31ScalarGateScorer):
            self.v31_scorer = RNRv31ScalarGateScorer(self.cfg)

        num_nodes = int(batch.num_nodes)
        device = batch.x.device
        if self.score_base is not None and self.score_base.numel() == num_nodes:
            score_base = self.score_base.to(device)
            score_final = (
                self.score_current.to(device)
                if self.score_current is not None
                and self.score_current.numel() == num_nodes
                else score_base
            )
            confidence = (
                self.confidence.to(device)
                if self.confidence is not None
                and self.confidence.numel() == num_nodes
                else torch.ones_like(score_base)
            )
            output = {
                "score_pre": score_base,
                "score_final": score_final,
                "confidence": confidence,
            }
            if self.stage1_result is not None:
                output["stage1_confidence"] = self.stage1_result.confidence.to(device)
            if self.stage2_result is not None:
                output["score_graph_candidate"] = self.stage2_result.candidate_score.to(device)
                output["graph_candidate_confidence"] = self.stage2_result.confidence.to(device)
            if self.stage31_result is not None:
                output["scalar_gate_lambda_stage1"] = self.stage31_result.lambda_stage1.to(device)
            return output

        stage1 = self.scorer.compute(model, batch, use_adapter=False)
        stage2 = self.v3_scorer.compute(batch, stage1)
        result = self.v31_scorer.compute(batch, stage1, stage2)
        return {
            "score_pre": stage1.score.detach(),
            "score_final": result.score.detach(),
            "score_graph_candidate": stage2.candidate_score.detach(),
            "confidence": result.confidence.detach(),
            "stage1_confidence": stage1.confidence.detach(),
            "graph_candidate_confidence": stage2.confidence.detach(),
            "scalar_gate_lambda_stage1": result.lambda_stage1.detach(),
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Save the base/v3 state; the token-checkpoint hook expects a Transformer.
        RNRGADV3LitModule.on_save_checkpoint(self, checkpoint)
        if self.stage31_result is not None:
            checkpoint["rnrv31_scalar_gate_state"] = {
                "lambda_stage1": self.stage31_result.lambda_stage1.detach().cpu(),
            }
