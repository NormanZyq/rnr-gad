#!/usr/bin/env python3
"""Run the canonical RNR-GAD ten-seed release matrix.

This launcher intentionally delegates training to ``src/multi_train.py``.  Its
responsibilities are scheduling, provenance capture, resumability, and strict
validation of the per-seed release artifacts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "rnrgad_multiseed"
METHOD_NAME = "RNR-GAD"
SEEDS = tuple(range(10))

DATASET_EXPERIMENTS = {
    "books": "rnrgad_books",
    "disney": "rnrgad_disney",
    "enron": "rnrgad_enron",
    "reddit": "rnrgad_reddit",
    "inj_cora": "rnrgad_inj_cora",
    "inj_flickr": "rnrgad_inj_flickr",
    "weibo": "rnrgad_weibo",
    "yelpchi": "rnrgad_yelpchi",
    "tfinance": "rnrgad_tfinance",
}
DATASET_SCHEDULE = (
    "yelpchi",
    "inj_flickr",
    "reddit",
    "weibo",
    "enron",
    "inj_cora",
    "books",
    "disney",
    "tfinance",
)

# Prediction filenames follow the multi_train output format.
PREDICTION_ARTIFACTS = (
    ("stage1 score", "score_pre.npy"),
    ("final score", "score_final.npy"),
    ("graph score", "score_graph_candidate.npy"),
)
REQUIRED_PREDICTIONS = tuple(filename for _, filename in PREDICTION_ARTIFACTS)
REQUIRED_METRICS = (
    "test/auroc",
    "test/auprc",
    "test/ap",
    "test/spearman_pre_final",
    "test/auc_gain",
)

# Reject experimental identifiers in release metadata and run paths.
NON_RELEASE_TOKENS = (
    "temporary",
    "pruned",
    "candidate",
)

SOURCE_FILES = (
    '.project-root',
    'configs/ablation/scalar_gate.yaml',
    'configs/ablation/wo_direct_anchor.yaml',
    'configs/ablation/wo_graph_evidence.yaml',
    'configs/ablation/wo_reliability.yaml',
    'configs/callbacks/early_stopping.yaml',
    'configs/callbacks/gad_unsup.yaml',
    'configs/callbacks/model_checkpoint.yaml',
    'configs/callbacks/model_summary.yaml',
    'configs/callbacks/rich_progress_bar.yaml',
    'configs/callbacks/rnrgad.yaml',
    'configs/data/gad.yaml',
    'configs/eval.yaml',
    'configs/experiment/rnrgad_books.yaml',
    'configs/experiment/rnrgad_disney.yaml',
    'configs/experiment/rnrgad_enron.yaml',
    'configs/experiment/rnrgad_inj_cora.yaml',
    'configs/experiment/rnrgad_inj_flickr.yaml',
    'configs/experiment/rnrgad_reddit.yaml',
    'configs/experiment/rnrgad_tfinance.yaml',
    'configs/experiment/rnrgad_weibo.yaml',
    'configs/experiment/rnrgad_yelpchi.yaml',
    'configs/extras/default.yaml',
    'configs/hydra/default.yaml',
    'configs/model/rnr_gad_v31_global_flat.yaml',
    'configs/model/rnrgad.yaml',
    'configs/paths/default.yaml',
    'configs/train.yaml',
    'configs/trainer/default.yaml',
    'configs/trainer/gpu.yaml',
    'scripts/resolve_projection_radius.py',
    'scripts/run_rnrgad_multiseed.py',
    'src/__init__.py',
    'src/data/gad_datamodule.py',
    'src/eval.py',
    'src/models/__init__.py',
    'src/models/components/__init__.py',
    'src/models/components/decoders.py',
    'src/models/components/encoders.py',
    'src/models/components/graph_utils.py',
    'src/models/components/metrics.py',
    'src/models/components/rank_utils.py',
    'src/models/components/rnr_gad.py',
    'src/models/components/rnr_gad_v3.py',
    'src/models/components/rnr_gad_v31.py',
    'src/models/components/rnr_gad_v31_scalar_gate.py',
    'src/models/rnr_gad_module.py',
    'src/models/rnr_gad_v31_module.py',
    'src/models/rnr_gad_v31_scalar_gate_module.py',
    'src/models/rnr_gad_v3_module.py',
    'src/models/rnrgad_module.py',
    'src/multi_train.py',
    'src/utils/__init__.py',
    'src/utils/instantiators.py',
    'src/utils/logging_utils.py',
    'src/utils/pylogger.py',
    'src/utils/rich_utils.py',
    'src/utils/utils.py',
)

SUMMARY_FIELDS = (
    "dataset",
    "experiment",
    "gpu",
    "status",
    "returncode",
    "wall_seconds",
    "run_dir",
    "stdout_path",
    "command",
    "validation_error",
)


def _assert_release_text(label: str, value: str) -> None:
    lowered = value.lower()
    hits = [token for token in NON_RELEASE_TOKENS if token in lowered]
    if hits:
        raise ValueError(f"{label} contains a non-release identifier: {hits[0]}")


def _split_values(values: Sequence[str]) -> tuple[str, ...]:
    parts = []
    for value in values:
        parts.extend(item.strip() for item in value.split(",") if item.strip())
    return tuple(parts)


def _normalize_seeds(values: Sequence[str]) -> tuple[int, ...]:
    parts = _split_values(values)
    try:
        seeds = tuple(int(item) for item in parts)
    except ValueError as error:
        raise ValueError("--seeds must contain integers") from error
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must not contain duplicates")
    normalized = tuple(sorted(seeds))
    if normalized != SEEDS:
        raise ValueError(
            "src/multi_train.py currently owns the seed loop; the release runner "
            "therefore requires exactly seeds 0 through 9"
        )
    return normalized


def _normalize_gpus(values: Sequence[str]) -> tuple[str, ...]:
    gpus = _split_values(values)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU index")
    if len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must not contain duplicates")
    if any(re.fullmatch(r"[0-9]+", gpu) is None for gpu in gpus):
        raise ValueError("--gpus accepts non-negative numeric GPU indices")
    return gpus


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_EXPERIMENTS),
        default=list(DATASET_EXPERIMENTS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=[",".join(str(seed) for seed in SEEDS)],
        help="Release seed set; currently required to be exactly 0,...,9.",
    )
    parser.add_argument("--gpus", nargs="+", default=["0,1"])
    parser.add_argument(
        "--python",
        default=os.environ.get("RNRGAD_PYTHON", sys.executable),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse only seed groups that pass the complete release gate.",
    )
    args = parser.parse_args(argv)

    try:
        args.seeds = _normalize_seeds(args.seeds)
        args.gpus = _normalize_gpus(args.gpus)
        args.output_root = args.output_root.expanduser().resolve()
        selected = set(args.datasets)
        args.datasets = tuple(name for name in DATASET_SCHEDULE if name in selected)
    except ValueError as error:
        parser.error(str(error))
    return args


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_module_files(module: str) -> tuple[str, ...]:
    if not module or module.split(".", 1)[0] not in {"src", "scripts"}:
        return ()
    stem = Path(*module.split("."))
    candidates = (stem.with_suffix(".py"), stem / "__init__.py")
    return tuple(
        path.as_posix() for path in candidates if (REPO_ROOT / path).is_file()
    )


def _imported_local_modules(relative: str, tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    package_parts = list(Path(relative).parent.parts)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        if node.level:
            keep = len(package_parts) - (node.level - 1)
            if keep < 0:
                continue
            base_parts = package_parts[:keep]
            if node.module:
                base_parts.extend(node.module.split("."))
            module = ".".join(base_parts)
        else:
            module = node.module or ""

        if module:
            modules.add(module)
        for alias in node.names:
            if alias.name == "*":
                continue
            modules.add(f"{module}.{alias.name}" if module else alias.name)
    return modules


def _local_python_dependency_closure(entries: Sequence[str]) -> set[str]:
    pending = list(entries)
    closure: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in closure:
            continue
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required release source file is missing: {path}")
        closure.add(relative)
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _imported_local_modules(relative, tree):
            for dependency in _local_module_files(module):
                if dependency not in closure:
                    pending.append(dependency)
    return closure


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".new")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _validate_existing_json(path: Path, payload: Mapping[str, Any]) -> None:
    if not path.exists():
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Existing release manifest is unreadable: {path}") from error
    if existing != payload:
        raise RuntimeError(
            f"Existing release manifest does not match the current configuration/source "
            f"fingerprints: {path}. Choose a new --output-root."
        )


def _compose_configuration(experiment: str) -> str:
    # Keep --help usable without training dependencies.
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"experiment={experiment}",
                "extras.print_config=false",
                "extras.enforce_tags=false",
            ],
        )
    text = OmegaConf.to_yaml(cfg, resolve=False, sort_keys=True)
    _assert_release_text(f"composed configuration {experiment}", text)
    return text


def _source_fingerprints() -> dict[str, str]:
    relative_paths = _local_python_dependency_closure(SOURCE_FILES)
    relative_paths.update(
        f"configs/experiment/{experiment}.yaml"
        for experiment in DATASET_EXPERIMENTS.values()
    )
    fingerprints = {}
    for relative in sorted(relative_paths):
        _assert_release_text("source filename", relative)
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required release source file is missing: {path}")
        fingerprints[relative] = _sha256_file(path)
    return fingerprints


def _tree_fingerprint(fingerprints: Mapping[str, str]) -> str:
    canonical = "".join(
        f"{relative}\0{fingerprints[relative]}\n" for relative in sorted(fingerprints)
    )
    return _sha256_text(canonical)


def _output_contract() -> dict[str, Any]:
    return {
        "checkpoint": "checkpoints/seed_<seed>/last.ckpt",
        "metrics": "predictions/seed_<seed>/seed_metrics.json",
        "prediction_directory": "predictions/seed_<seed>",
        "prediction_roles": ["stage1_score", "graph_score", "final_score"],
        "required_metrics": list(REQUIRED_METRICS),
    }


def _release_manifests(
    configurations: Mapping[str, str],
    source_fingerprints: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    config_fingerprints = {
        dataset: _sha256_text(configurations[dataset])
        for dataset in DATASET_EXPERIMENTS
    }
    source_tree = _tree_fingerprint(source_fingerprints)
    root_manifest = {
        "schema_version": 1,
        "method": METHOD_NAME,
        "formal_version": True,
        "release_stage": "pre_release",
        "runner": "scripts/run_rnrgad_multiseed.py",
        "seed_policy": {
            "seeds": list(SEEDS),
            "implementation": "src/multi_train.py",
        },
        "dataset_experiments": dict(DATASET_EXPERIMENTS),
        "configuration_files": {
            dataset: f"configurations/{dataset}.yaml"
            for dataset in DATASET_EXPERIMENTS
        },
        "configuration_sha256": config_fingerprints,
        "source_files_sha256": dict(source_fingerprints),
        "source_tree_sha256": source_tree,
        "output_contract": _output_contract(),
    }
    group_manifests = {}
    for dataset, experiment in DATASET_EXPERIMENTS.items():
        group_manifests[dataset] = {
            "schema_version": 1,
            "method": METHOD_NAME,
            "formal_version": True,
            "release_stage": "pre_release",
            "dataset": dataset,
            "experiment": experiment,
            "seeds": list(SEEDS),
            "configuration_file": f"../configurations/{dataset}.yaml",
            "configuration_sha256": config_fingerprints[dataset],
            "source_files_sha256": dict(source_fingerprints),
            "source_tree_sha256": source_tree,
            "output_contract": _output_contract(),
        }
    return root_manifest, group_manifests


def _prepare_release_metadata(output_root: Path) -> None:
    configurations = {
        dataset: _compose_configuration(experiment)
        for dataset, experiment in DATASET_EXPERIMENTS.items()
    }
    source_fingerprints = _source_fingerprints()
    root_manifest, group_manifests = _release_manifests(
        configurations, source_fingerprints
    )

    root_manifest_path = output_root / "run_manifest.json"
    if not root_manifest_path.exists():
        unprovenanced = [
            dataset
            for dataset in DATASET_EXPERIMENTS
            if _has_training_artifacts(output_root / dataset)
        ]
        if unprovenanced:
            raise RuntimeError(
                "Training artifacts exist without a release manifest for dataset(s) "
                f"{','.join(unprovenanced)}. Choose a new --output-root."
            )
    _validate_existing_json(root_manifest_path, root_manifest)
    for dataset, payload in group_manifests.items():
        group_manifest_path = output_root / dataset / "run_manifest.json"
        if (
            not group_manifest_path.exists()
            and _has_training_artifacts(output_root / dataset)
        ):
            raise RuntimeError(
                "Training artifacts exist without a dataset release manifest for "
                f"{dataset}. Choose a new --output-root."
            )
        _validate_existing_json(group_manifest_path, payload)

    for dataset, text in configurations.items():
        _atomic_write_text(output_root / "configurations" / f"{dataset}.yaml", text)
    for dataset, payload in group_manifests.items():
        manifest_text = _json_text(payload)
        _assert_release_text(f"dataset manifest {dataset}", manifest_text)
        _atomic_write_text(output_root / dataset / "run_manifest.json", manifest_text)
    root_manifest_text = _json_text(root_manifest)
    _assert_release_text("root manifest", root_manifest_text)
    _atomic_write_text(root_manifest_path, root_manifest_text)


def _seed_has_artifacts(run_dir: Path, seed: int) -> bool:
    locations = (
        run_dir / "predictions" / f"seed_{seed}",
        run_dir / "checkpoints" / f"seed_{seed}",
    )
    return any(
        location.exists() and any(path.is_file() for path in location.rglob("*"))
        for location in locations
    )


def _seed_metric_errors(metrics_path: Path, seed: int) -> list[str]:
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        return [f"seed {seed}: missing or empty metrics"]

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [f"seed {seed}: metrics are not valid JSON"]

    if not isinstance(metrics, dict):
        return [f"seed {seed}: metrics must be a JSON object"]

    errors = []
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        errors.append(f"seed {seed}: metrics missing {','.join(missing)}")
    for key in REQUIRED_METRICS:
        if key not in metrics:
            continue
        try:
            valid = math.isfinite(float(metrics[key]))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            errors.append(f"seed {seed}: metric {key} is not finite")
    return errors


def _group_validation_errors(run_dir: Path, seeds: Sequence[int]) -> list[str]:
    errors = []
    for seed in seeds:
        seed_dir = run_dir / "predictions" / f"seed_{seed}"
        for role, filename in PREDICTION_ARTIFACTS:
            path = seed_dir / filename
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"seed {seed}: missing or empty {role} array")

        metrics_path = seed_dir / "seed_metrics.json"
        errors.extend(_seed_metric_errors(metrics_path, seed))

        checkpoint = run_dir / "checkpoints" / f"seed_{seed}" / "last.ckpt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            errors.append(f"seed {seed}: missing or empty last checkpoint")
    return errors


def group_complete(run_dir: Path, seeds: Sequence[int] = SEEDS) -> bool:
    return not _group_validation_errors(run_dir, seeds)


def _has_training_artifacts(run_dir: Path) -> bool:
    for directory in (run_dir / "predictions", run_dir / "checkpoints"):
        if directory.exists() and any(path.is_file() for path in directory.rglob("*")):
            return True
    return False


def _resume_validation_error(run_dir: Path, seeds: Sequence[int]) -> str:
    invalid = set()
    for error in _group_validation_errors(run_dir, seeds):
        match = re.match(r"seed ([0-9]+):", error)
        if match:
            invalid.add(int(match.group(1)))
    blocked = sorted(
        seed
        for seed in invalid
        if _seed_has_artifacts(run_dir, seed)
    )
    if not blocked:
        return ""
    joined = ",".join(str(seed) for seed in blocked)
    return (
        f"saved seed(s) {joined} contain incomplete release artifacts that cannot "
        "be provenance-safely resumed; rerun into a fresh --output-root with "
        "--no-resume"
    )


def _build_command(
    python: str,
    experiment: str,
    run_dir: Path,
    *,
    resume: bool,
) -> list[str]:
    command = [
        python,
        "src/multi_train.py",
        f"experiment={experiment}",
        "extras.print_config=false",
        "extras.enforce_tags=false",
        f"hydra.run.dir={run_dir}",
        f"+multi_train_resume={'true' if resume else 'false'}",
    ]
    return command


def _run_group(
    dataset: str,
    gpu: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    experiment = DATASET_EXPERIMENTS[dataset]
    run_dir = args.output_root / dataset
    stdout_path = run_dir / "multi_train.log"
    command = _build_command(
        args.python,
        experiment,
        run_dir,
        resume=bool(args.resume),
    )
    started = time.monotonic()
    returncode = 0
    validation_error = ""

    try:
        if args.resume and group_complete(run_dir, args.seeds):
            status = "reused"
        elif args.resume and (validation_error := _resume_validation_error(run_dir, args.seeds)):
            status = "failed"
            returncode = 2
        elif not args.resume and _has_training_artifacts(run_dir):
            status = "failed"
            returncode = 2
            validation_error = (
                "--no-resume requires a fresh dataset output directory; choose a new "
                "--output-root"
            )
        elif args.dry_run:
            status = "planned"
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            with stdout_path.open("a", encoding="utf-8") as output:
                output.write("\nCOMMAND " + shlex.join(command) + "\n")
                output.flush()
                process = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            returncode = int(process.returncode)
            errors = _group_validation_errors(run_dir, args.seeds)
            if returncode == 0 and not errors:
                status = "completed"
            else:
                status = "failed"
                validation_error = "; ".join(errors[:8])
                if len(errors) > 8:
                    validation_error += f"; plus {len(errors) - 8} more error(s)"
                if returncode != 0 and not validation_error:
                    validation_error = "src/multi_train.py returned a non-zero status"
    except Exception as error:  # keep all groups auditable even if one worker fails
        status = "failed"
        returncode = 1
        validation_error = f"{type(error).__name__}: {error}"

    return {
        "dataset": dataset,
        "experiment": experiment,
        "gpu": gpu,
        "status": status,
        "returncode": returncode,
        "wall_seconds": round(time.monotonic() - started, 3),
        "run_dir": str(run_dir),
        "stdout_path": str(stdout_path),
        "command": shlex.join(command),
        "validation_error": validation_error,
    }


def _read_summary(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_summary(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    by_dataset = {str(row["dataset"]): dict(row) for row in rows}
    ordered = [
        by_dataset[dataset]
        for dataset in DATASET_EXPERIMENTS
        if dataset in by_dataset
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".new")
    with staging.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    os.replace(staging, path)


def _upsert_rows(
    existing: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_dataset = {str(row["dataset"]): dict(row) for row in existing}
    for row in updates:
        by_dataset[str(row["dataset"])] = dict(row)
    return list(by_dataset.values())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _prepare_release_metadata(args.output_root)

    tasks = Queue()
    for dataset in args.datasets:
        tasks.put(dataset)

    current_rows = []
    summary_path = args.output_root / "run_summary.csv"
    merged_rows = _read_summary(summary_path)
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        nonlocal merged_rows
        while True:
            try:
                dataset = tasks.get_nowait()
            except Empty:
                return
            try:
                print(f"[start gpu={gpu}] {dataset}", flush=True)
                row = _run_group(dataset, gpu, args)
                with lock:
                    current_rows.append(row)
                    merged_rows = _upsert_rows(merged_rows, [row])
                    _write_summary(merged_rows, summary_path)
                print(
                    f"[done gpu={gpu}] {dataset} status={row['status']} "
                    f"seconds={row['wall_seconds']}",
                    flush=True,
                )
            finally:
                tasks.task_done()

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in args.gpus]
        for future in as_completed(futures):
            future.result()

    _write_summary(merged_rows, summary_path)
    failures = [row for row in current_rows if row["status"] == "failed"]
    if failures:
        print(
            f"RNRGAD_MULTISEED_FAILED groups={len(failures)} summary={summary_path}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(f"RNRGAD_MULTISEED_DONE {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
