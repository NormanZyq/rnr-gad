#!/usr/bin/env python3
"""Empirical radius mapping fitted to the current study's selected budgets.

The inputs are graph counts; the branch values encode the fixed budgets
used by the current study.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json


RULE_VERSION = "rnr-gad-projection-radius-v3-20261006"


@dataclass(frozen=True)
class ProjectionRadiusDecision:
    rule_version: str
    epsilon_max: float
    regime: str
    mean_edges: float
    attribute_load: float
    relation_load: float


def resolve_projection_radius(
    *,
    nodes: int,
    stored_directed_edges: int,
    features: int,
    hidden_width: int = 64,
    propagation_depth: int = 4,
) -> ProjectionRadiusDecision:
    for name, value in (
        ("nodes", nodes),
        ("stored_directed_edges", stored_directed_edges),
        ("features", features),
        ("hidden_width", hidden_width),
        ("propagation_depth", propagation_depth),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    moderate_attribute = hidden_width <= 2 * features <= 2 * hidden_width
    strict_load = features > hidden_width or (
        moderate_attribute and stored_directed_edges > nodes * hidden_width
    )
    compact_low_attribute = (
        2 * features < hidden_width and nodes < 4 * hidden_width
    )
    topology_supported = (
        2 * features < hidden_width
        and nodes >= 4 * hidden_width
        and stored_directed_edges > nodes * 2 * propagation_depth
    )
    if strict_load:
        epsilon_max, regime = 0.05, "strict_attribute_or_joint_load"
    elif compact_low_attribute:
        epsilon_max, regime = 0.70, "compact_low_attribute_load"
    elif moderate_attribute or topology_supported:
        epsilon_max = 1.00
        regime = (
            "moderate_attribute_sparse_relation"
            if moderate_attribute else "topology_supported_correction"
        )
    else:
        epsilon_max, regime = 2.00, "low_load_wide_revision"

    mean_edges = stored_directed_edges / nodes
    return ProjectionRadiusDecision(
        rule_version=RULE_VERSION,
        epsilon_max=epsilon_max,
        regime=regime,
        mean_edges=mean_edges,
        attribute_load=features / hidden_width,
        relation_load=mean_edges / hidden_width,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--stored-directed-edges", type=int, required=True)
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--propagation-depth", type=int, default=4)
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    decision = resolve_projection_radius(
        nodes=args.nodes,
        stored_directed_edges=args.stored_directed_edges,
        features=args.features,
        hidden_width=args.hidden_width,
        propagation_depth=args.propagation_depth,
    )
    if args.explain:
        print(json.dumps(asdict(decision), indent=2))
    else:
        print(f"{decision.epsilon_max:.2f}")


if __name__ == "__main__":
    main()
